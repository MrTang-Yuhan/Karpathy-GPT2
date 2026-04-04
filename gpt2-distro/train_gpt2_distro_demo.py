"""
复现 GPT-2。
整体上，GPT-2 的模型和 Transformer 类似，只是有一些改动。
"""

import inspect
import math
import os
import time
from dataclasses import dataclass

import numpy as np
import tiktoken
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed import destroy_process_group, init_process_group
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

# import torch_xla.core.xla_model as xm # google TPU v5e1
"""
google tpu手册：
https://docs.cloud.google.com/tpu/docs/run-calculation-pytorch?hl=zh-cn
"""


# -----------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert (
            config.n_embd % config.n_head == 0
        ), f"Embedding dimension {config.n_embd} should be divisible by number of heads {config.n_head}"

        # 为所有注意力头一次性计算 key、query、value 投影，并在批次中统一处理
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # 输出投影
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1  # 标记是否需要进行 1/sqrt(dk) 缩放

        # 正则化相关
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # 严格来说这不算“偏置”，更像是一个掩码，不过这里沿用了 OpenAI/HF 的命名方式
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x):
        B, T, C = x.size()  # 批大小、序列长度、嵌入维度（n_embd）
        # 在一个批次中为所有注意力头计算 query、key、value，并将 head 维前移，作为批处理维度的一部分
        # nh 表示“头的数量”，hs 表示“每个头的大小”，而 C（通道数）= nh * hs
        # 例如在 GPT-2（124M）中，n_head=12，hs=64，因此在 Transformer 中 nh*hs=C=768 个通道
        qkv = self.c_attn(x)  # (B, T, 3*C)
        q, k, v = qkv.split(self.n_embd, dim=2)  # q、k、v 的形状都是 (B, T, C)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # Flash Attention 算法
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # 将所有头的输出重新拼接在一起
        y = self.c_proj(y)  # 输出投影
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1  # 标记是否需要进行 1/sqrt(dk) 缩放

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024  # 最大序列长度
    # 关于 vocab_size 有一个小技巧：由于我们希望刚开始训练时，模型的输出 logits 接近于均匀分布（即每个 token 的概率相等）
    # 这样在训练开始时，模型的输出 logits 应该要接近于 1/vocab_size 的概率分布。这有助于稳定训练过程，并且在训练初期不会过于偏向某些 token
    # 由于损失使用的是交叉熵损失函数，所以如果第一次训练的损失接近 -ln(1/vocab_size), 那么说明初始 logits 分布比较均匀
    vocab_size: int = 50257  # token 数量：50,000 个 BPE merge + 256 个字节 token + 1 个 <|endoftext|> token
    n_layer: int = 12  # 层数
    n_head: int = 12  # 注意力头数
    n_embd: int = 768  # 嵌入维度


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),  # token 嵌入权重
                wpe=nn.Embedding(config.block_size, config.n_embd),  # 位置嵌入权重
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(
            config.n_embd, config.vocab_size, bias=False
        )  # language model head

        # 权重共享机制: 将 lm_head 的权重与 token 嵌入权重 wte 共享。这是 GPT 模型中的一个重要设计选择，可以减少模型参数数量并提高训练效率
        # 注意，在 Pytorch 中，Linear 层 weight 的 shape 是 (out_features, in_features)
        self.transformer.wte.weight = self.lm_head.weight

        # 初始化参数
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """参考 GPT2 论文进行设置"""
        if isinstance(module, nn.Linear):
            std = 0.02  # 这里如果设置为0.2，训练会非常不稳定
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                # 每一层有两个残差分支加回主干：
                #   x_{l+1} = x_l + a_l + m_l
                # 为了防止标准差会随着 sqrt(n_layer) 增长，我们在这里进行缩放
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx 的形状为 (B, T)
        B, T = idx.size()
        assert (
            T <= self.config.block_size
        ), f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        # 前向计算 token 和位置嵌入
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # 形状为 (T)
        pos_emb = self.transformer.wpe(pos)  # 位置嵌入，形状为 (T, n_embd)
        tok_emb = self.transformer.wte(idx)  # token 嵌入，形状为 (B, T, n_embd)
        x = tok_emb + pos_emb

        # 前向通过 Transformer 的各个 block
        for block in self.transformer.h:
            x = block(x)

        # 前向通过最后的 layernorm 和分类器
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """从 huggingface 加载预训练 GPT-2 模型权重"""
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel

        print(f"loading weights from pretrained gpt: {model_type}")

        # n_layer、n_head 和 n_embd 由 model_type 决定
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M 参数
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M 参数
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M 参数
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M 参数
        }[model_type]
        config_args["vocab_size"] = 50257  # GPT 模型检查点始终为 50257
        config_args["block_size"] = 1024  # GPT 模型检查点始终为 1024

        # 创建一个从零初始化的 minGPT 模型
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = [k for k in sd.keys() if not k.endswith(".attn.bias")]

        # 初始化一个 huggingface/transformers 模型
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # 在确保所有参数名称和形状都对齐匹配的前提下进行拷贝
        sd_keys_hf = [k for k in sd_hf.keys() if not k.endswith(".attn.masked_bias")]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.bias")]
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]

        # 本质上 openai 的检查点使用的是一个“Conv1D”模块，但我们这里只想使用普通的 Linear
        # 这意味着在导入这些权重时，我们必须对它们进行转置
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for key in sd_keys_hf:
            if any(key.endswith(name) for name in transposed):
                # 对需要转置的 Conv1D 权重做特殊处理
                assert sd_hf[key].shape[::-1] == sd[key].shape
                with torch.no_grad():
                    sd[key].copy_(sd_hf[key].t())
            else:
                # 普通方式拷贝其他参数
                assert sd_hf[key].shape == sd[key].shape
                with torch.no_grad():
                    sd[key].copy_(sd_hf[key])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # 首先收集所有候选参数（即需要计算梯度的参数）
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # 创建优化器参数组。
        # 任何二维及以上的参数都会施加权重衰减，否则不施加。
        # 也就是说：所有矩阵乘法中的权重张量 + embedding 权重 (一般二维) 会做衰减，
        # 而所有偏置项和 LayerNorm 参数 (一般一维) 不做衰减。
        # 权重衰减是一种正则化手段，它迫使优化过程更均衡地利用所有参数，从而避免单个权重值过大
        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)

        # if master_process:
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )

        # 创建 AdamW 优化器，如果 fused 版本可用，则使用 fused 版本
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"  # fused 优化一般主要针对 GPU 内核加速

        # if master_process:
        print(f"using fused AdamW: {use_fused}")

        adamw_kwargs = dict(
            lr=learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
        )
        if fused_available:
            adamw_kwargs["fused"] = use_fused

        optimizer = torch.optim.AdamW(optim_groups, **adamw_kwargs)
        return optimizer


# -----------------------------------------------------------------------------

class DataLoaderLite:
    def __init__(self, B, T, ddp_rank, num_processes):
        self.B = B
        self.T = T
        self.process_rank = ddp_rank
        self.num_processes = num_processes

        # 在初始化时从磁盘加载 tokens 并将其存储到内存中
        data_file = os.path.join(os.path.dirname(__file__), "input.txt")
        with open(data_file, "r", encoding="utf-8") as f:
            text = f.read()

        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens, dtype=torch.long)

        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        # 状态: 不同的进程会从不同的位置开始处理数据，从而避免多个进程处理相同的数据
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        # 得到一个批次数据
        # 一种常用的处理数据处理方式：将输入文本转换为 token ID 的序列，并将其分成输入（x）和目标（y）
        # x 是输入序列，y 是输入序列向右偏移一个位置的版本。模型在训练时会学习根据 x 预测 y 中的下一个 token
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T)  # 输入
        y = (buf[1:]).view(B, T)  # 目标

        # 在张量中推进位置
        self.current_position += B * T * self.num_processes

        # 如果加载下一个 batch 会越界，则重置
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_position = self.B * self.T * self.process_rank
        return x, y


def setup_ddp():
    # 设置 DDP（分布式数据并行）
    # 简单启动方式：
    # python train_gpt2_distro_demo.py
    # DDP 启动方式，例如使用 8 张 GPU：
    # torchrun --standalone --nproc_per_node=8 train_gpt2_distro_demo.py
    # 运行训练循环
    ddp = int(os.environ.get("RANK", -1)) != -1  # 这是否是一次 ddp 运行？

    if ddp:
        # 使用 DDP 目前要求 CUDA，因此我们根据 rank 来设置对应的设备
        assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
        init_process_group(backend="nccl")

        ddp_rank = int(os.environ["RANK"])  # 'RANK': 当前进程在全局中的编号. 范围 0, 1, 2, ..., WORLD_SIZE - 1
        ddp_local_rank = int(os.environ["LOCAL_RANK"]) # 'LOCAL_RANK': 当前进程在本机上的编号, 这个值通常对应当前进程应该绑定本机的第几张 GPU
        ddp_world_size = int(os.environ["WORLD_SIZE"]) # 'WORLD_SIZE': 总共有多少个进程参与训练

        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        # 普通的、非 DDP 的运行
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True

        # 尝试自动检测设备
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        print(f"using device: {device}")

    return ddp, ddp_rank, ddp_local_rank, ddp_world_size, device, master_process


def get_lr(it, warmup_steps, max_steps, max_lr, min_lr):
    """
    学习率的余弦衰减方法:
    根据当前训练迭代步数 it，动态返回当前应该使用的学习率 lr
    """
    # 1）在 warmup 迭代步数内进行线性预热
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps  # +1 避免 lr 从 0 开始. 不喜欢0

    # 2）如果 it > lr_decay_iters，则返回最小学习率
    if it > max_steps:
        return min_lr

    # 3）在中间阶段，使用余弦衰减下降到最小学习率
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # 系数从 1 开始下降到 0

    # 当 coeff 为 1，lr=max_lr. 当coeff 为 0 时，lr=min_lr.
    # 当 coeff 在 0 和 1 之间时： lr 在 max_lr 和 min_lr 之间平滑变化
    return min_lr + coeff * (max_lr - min_lr)


def main():
    # 简单启动方式：
    # python train_gpt2_distro_demo.py
    # DDP 启动方式，例如使用 8 张 GPU：
    # torchrun --standalone --nproc_per_node=8 train_gpt2_distro_demo.py
    # 运行训练循环
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device, master_process = setup_ddp()

    # 将随机种子设置为 42
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # 为什么可以使用“梯度累加”：
    #
    # 1. 反向传播得到的梯度，对“loss 的求和/求平均”是线性的。
    #    假设一个大 batch 被拆成 N 个 micro-batch，
    #    总损失可以写成：
    #        L = (L1 + L2 + ... + LN) / N
    #    那么总梯度就是：
    #        dL/dθ = (dL1/dθ + dL2/dθ + ... + dLN/dθ) / N
    #    也就是说：
    #    “先分别算每个 micro-batch 的梯度，再把它们加起来取平均”
    #    和
    #    “直接对整个大 batch 算一次梯度”
    #    在数学上是等价的。
    #
    # 2. PyTorch 的 .backward() 默认就是把梯度累加到 param.grad 中，
    #    而不是每次覆盖，所以天然支持 gradient accumulation。
    #
    # 3. 这样做的核心好处是：显存不够时，也能模拟更大的 batch size。
    #    例如：
    #        micro_batch_size = 8
    #        grad_accum_steps = 4
    #    那等效 batch size 就是 32
    #    但每次前向/反向只需要容纳 8 个样本的显存。
    #
    # 4. 注意：为了和“大 batch 的平均 loss”完全等价，
    #    每个 micro-batch 的 loss 必须先除以 grad_accum_steps，
    #    否则累加后的梯度会大 N 倍。
    #
    # 5. 所以 gradient accumulation 本质上不是“近似技巧”，
    #    而是利用了梯度对 loss 的线性可加性，
    #    在不改变数学目标的前提下，把一个大 batch 拆成多个小 batch 来算。
    total_batch_size = 524288  # 2**19，约 0.5M，以 token 数量计
    B = 2  # 微批大小
    T = 256  # 序列长度
    # B = 16  # 微批大小
    # T = 1024  # 序列长度
    assert (
        total_batch_size % (B * T * ddp_world_size) == 0
    ), "make sure total_batch_size is divisible by B * T * ddp_world_size"
    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)

    # 限制只有主进程进行打印，避免多个进程同时打印导致日志混乱
    if master_process:
        print(f"total desired batch size: {total_batch_size}")
        print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    train_loader = DataLoaderLite(
        B=B,
        T=T,
        ddp_rank=ddp_rank,
        num_processes=ddp_world_size,
    )

    # torch.set_float32_matmul_precision("high") # 选择 F32 矩阵乘的精度
    # get logits
    model = GPT(GPTConfig(vocab_size=50304))  # 传入一个能被 2 整除的 vocab，可以提高运行速度
    model = model.to(device)

    # model = torch.compile(model) # 编译代码，使代码运行更快. 在非调试模式下，建议开启.
    #                            # 但是它对一些功能的支持暂时还不完善
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    # 如果使用 DDP 封装，那么真正的原始模型在 model.module 内；否则，如果没用 DDP，model 本身就是原始模型。
    raw_model = model.module if ddp else model  # 始终包含“原始的”、未包装的模型

    max_lr = 3e-4
    min_lr = max_lr * 0.1
    warmup_steps = 10
    max_steps = 1

    # 优化!
    # optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
    optimizer = raw_model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=6e-4,
        device_type=device,
    )

    for step in range(max_steps):
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        loss_accum = torch.zeros(1, device=device)

        for micro_step in range(grad_accum_steps):
            print(f"step {step}, micro_step {micro_step}")
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)

            # 我们需要对 loss 进行缩放，以适配梯度累积，
            # 因为梯度会在每次 successive backward() 中直接累加。
            # 梯度的相加对应于目标函数的“求和”，
            # 但我们想要的不是 sum，而是 mean。
            # 因此这里要把 loss 缩放一下，使最终结果正确。
            if ddp and micro_step < grad_accum_steps - 1:
                with model.no_sync():
                    # 在梯度累积时，前几个 micro step 只在本地累加梯度，不做多卡同步；
                    # 因为梯度求和满足交换律/结合律，先本地累积、最后一次再 all-reduce，
                    # 与每个 micro step 都同步在数学上等价，但通信开销更小。
                    # 如果不使用 model.no_sync()，那么每次 backward() 都会自动触发 DDP 梯度同步。
                    _, loss = model(x, y)
                    loss = loss / grad_accum_steps
                    loss_accum += loss.detach()
                    loss.backward()
            else:
                _, loss = model(x, y)
                loss = loss / grad_accum_steps
                loss_accum += loss.detach()
                loss.backward()

        if ddp:
            # 把所有 GPU 上的 loss_accum 聚合起来，并取平均值，
            # 得到“所有卡共同的平均 loss”，主要用于日志记录。
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

        # 计算所有参数梯度的全局范数（平方和开根号）。
        # 如果大于 1.0，就把所有梯度等比例缩小，防止梯度过大导致训练不稳定。
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        lr = get_lr(
            it=step,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            max_lr=max_lr,
            min_lr=min_lr,
        )
        for param_group in optimizer.param_groups:
            # PyTorch 允许遍历优化器中的所有参数组，并修改它们的学习率
            param_group["lr"] = lr

        optimizer.step()

        # torch.cuda.synchronize()  # 如需精确统计 GPU 时间可打开
        t1 = time.time()
        dt = (t1 - t0) * 1000  # 毫秒
        tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
        tokens_per_sec = tokens_processed / (t1 - t0)

        if master_process:
            print(
                f"step: {step} | "
                f"loss: {loss_accum.item():.6f} | "
                f"lr: {lr:.4e} | "
                f"norm: {norm:.4f} | "
                f"dt: {dt:.2f}ms | "
                f"tok/sec: {tokens_per_sec:.2f}"
            )

    # 销毁进程组释放资源
    if ddp:
        destroy_process_group()

    # import sys; sys.exit(0)
    # prefix tokens
    num_return_sequences = 5
    max_length = 30

    model = raw_model   # 要用原始模型
    model.eval()
    model = model.to(device)

    enc = tiktoken.get_encoding("gpt2")
    valid_vocab_size = enc.n_vocab
    tokens = enc.encode("Hello, I'm a language model,")
    tokens = torch.tensor(tokens, dtype=torch.long)  # (8,)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)  # (5, 8)
    x = tokens.to(device)

    # 开始生成！当前 x 的形状是 (B, T)，其中 B = 5，T = 8
    while x.size(1) < max_length:
        # 前向运行模型以获得 logits
        with torch.no_grad():
            logits, _ = model(x)  # (B, T, vocab_size)
            # 取最后一个位置的 logits
            logits = logits[:, -1, :]  # (B, vocab_size)
            logits = logits[:, :valid_vocab_size]  # 仅保留 tokenizer 支持的词表范围，避免 decode 越界

            # 获取概率
            probs = F.softmax(logits, dim=-1)
            # 执行 top-k=50 采样（huggingface pipeline 默认值）
            # 这里 topk_probs 变成 (5, 50)，topk_indices 是 (5, 50)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            # 从 top-k 概率中选择一个 token
            ix = torch.multinomial(topk_probs, 1)  # (B, 1)
            # 收集对应的索引. 这里是从 topk_indices 的 -1 维中，按照 ix 给出的位置，取出对应的元素。
            xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
            # 将其追加到序列中
            x = torch.cat((x, xcol), dim=1)

    # 打印生成的文本
    for i in range(num_return_sequences):
        tokens = x[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        print(">", decoded)


if __name__ == "__main__":
    main()
