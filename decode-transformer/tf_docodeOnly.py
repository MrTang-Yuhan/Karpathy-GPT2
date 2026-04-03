import torch
import torch.nn as nn
from torch.nn import functional as F

"""
Decoder-Only Transformer（字符级语言模型）

网络结构：
1. token embedding：把字符 id 映射到向量
2. position embedding：加入位置信息
3. N 个 Transformer Block 串联
   - LayerNorm
   - Masked Multi-Head Self-Attention
   - 残差连接
   - LayerNorm
   - FeedForward
   - 残差连接
4. final LayerNorm
5. lm_head：映射到词表大小，输出下一个字符的 logits

这是一个自回归模型：
- 训练时：输入长度为 T 的序列，预测每个位置的下一个字符
- 生成时：每次只取最后一个位置的预测结果，采样出新 token 后拼回输入
"""

# =========================
# 超参数
# =========================
batch_size = 16  # 每次并行处理的序列数
block_size = 32  # 最大上下文长度
max_iters = 5000
eval_interval = 100
learning_rate = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 64  # token / position embedding 维度
n_head = 4  # 注意力头数
n_layer = 4  # Transformer Block 层数
dropout = 0.2

torch.manual_seed(1337)

# =========================
# 读取数据
# =========================
# 数据集来源：
# https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# 构建字符级词表
chars = sorted(list(set(text)))
vocab_size = len(chars)

# 字符 <-> 索引 映射
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s: str):
    """字符串编码为 token id 列表"""
    return [stoi[c] for c in s]


def decode(ids):
    """token id 列表解码为字符串"""
    return "".join([itos[i] for i in ids])

# 划分训练集 / 验证集
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# =========================
# 数据采样
# =========================
def get_batch(split):
    """
    随机采样一个 batch 的训练样本

    返回：
    x: (B, T) 输入序列
    y: (B, T) 目标序列，等于 x 整体右移一位
    """
    data_source = train_data if split == "train" else val_data

    # 每个起点取一个长度为 block_size 的片段
    ix = torch.randint(
        len(data_source) - block_size, (batch_size,)
    )  # randint能取到下界，取不到上界
    x = torch.stack([data_source[i:i + block_size] for i in ix])
    y = torch.stack([data_source[i + 1:i + block_size + 1] for i in ix])

    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss():
    """在训练集和验证集上估计平均 loss"""
    out = {}
    model.eval()

    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()

    model.train()
    return out

class Head(nn.Module):
    """单个 masked self-attention 头"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # 下三角 mask：保证当前位置只能看见自己和之前的位置
        self.register_buffer(
            "tril", torch.tril(torch.ones(block_size, block_size))
        )  # tril会注册模块，但是不会进行参数训练

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: (B, T, n_embd)
        return: (B, T, head_size)
        """
        B, T, _ = x.shape

        # 映射到 Q / K / V
        k = self.key(x)    # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        v = self.value(x)  # (B, T, head_size)

        # 计算 attention score
        # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        # 缩放因子应使用 head_size，而不是 n_embd
        wei = (q @ k.transpose(-2, -1)) * (k.shape[-1] ** -0.5)

        # 因果 mask：未来位置不可见
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))

        # softmax 得到注意力权重
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        # 用注意力权重对 V 做加权求和
        # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """多头 masked self-attention"""

    def __init__(self, num_heads, head_size):
        super().__init__()

        # 多个注意力头并行计算
        self.heads = nn.ModuleList(
            [Head(head_size) for _ in range(num_heads)]
        )  # ModuleList才会注册模块模块

        # 拼接后投影回 n_embd 维
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 每个头输出 (B, T, head_size)
        # 拼接后得到 (B, T, num_heads * head_size)
        out = torch.cat([head(x) for head in self.heads], dim=-1)

        # 线性投影回残差分支的维度
        out = self.proj(out)  # (B, T, num_heads * head_size) -> (B, T, n_embd)
        out = self.dropout(out)  # (B, T, n_embd)
        return out  # (B, T, n_embd)

class FeedForward(nn.Module):
    """位置独立的前馈网络：MLP"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """
    一个 Transformer Block（Pre-Norm 结构）

    结构：
    x -> x + SelfAttention(LN(x))
      -> x + FeedForward(LN(x))
    """

    def __init__(self, n_embd, n_head):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd 必须能被 n_head 整除"

        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)  # 对最后 1 个维度做归一化，并且最后一维大小必须是 n_embd
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Pre-Norm + 残差
        # 这里展示的Pre-Norm是现在更常用的方法，和transformer论文不同
        x = x + self.sa(self.ln1(x))

        # Pre-Norm + 残差
        x = x + self.ffwd(self.ln2(x))
        return x

class DecoderOnlyTransformer(nn.Module):
    """
    字符级 Decoder-Only Transformer

    输入：
        idx: (B, T)
    输出：
        logits: (B, T, vocab_size)
    """

    def __init__(self):
        super().__init__()

        # token embedding：字符 id -> 向量
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

        # position embedding：位置 id -> 向量
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # 多层 Transformer Block
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])  # * 解包

        # 最后一层归一化
        self.ln_f = nn.LayerNorm(n_embd)

        # 输出层：映射到词表大小
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        """
        idx: (B, T)
        targets: 训练： (B, T)
                 生成： None
        """
        B, T = idx.shape

        # token embedding
        tok_emb = self.token_embedding_table(idx)  # (B, T, n_embd)

        # position embedding
        pos = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding_table(pos)  # (T, n_embd)

        # token 向量 + 位置向量
        x = tok_emb + pos_emb  # (B, T, n_embd)

        # 经过多层 Transformer Block
        x = self.blocks(x)  # (B, T, n_embd)

        # 最后归一化 + 输出层
        x = self.ln_f(x)  # (B, T, n_embd)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # 计算交叉熵时，将 (B, T, C) 展平成 (B*T, C)
            B, T, C = logits.shape
            logits_flat = logits.reshape(B * T, C)
            targets_flat = targets.reshape(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        # logits = (B, T, vocab_size)
        # loss = 标量
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """
        自回归生成：
        每次只根据当前上下文预测下一个 token
        """
        for _ in range(max_new_tokens):
            # 只保留最近 block_size 个 token
            idx_cond = idx[:, -block_size:]

            # 前向计算
            logits, _ = self(idx_cond)

            # 只取最后一个位置的预测结果
            logits = logits[:, -1, :]  # (B, vocab_size)

            # 转成概率分布
            probs = F.softmax(logits, dim=-1)

            # 按概率采样下一个 token
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # 拼接到生成序列末尾
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

        return idx


if __name__ == '__main__':
    # =========================
    # 创建模型
    # =========================
    model = DecoderOnlyTransformer().to(device)

    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.3f} M parameters")

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # =========================
    # 训练
    # =========================
    for step in range(max_iters):
        # 定期评估 train / val loss
        if step % eval_interval == 0 or step == max_iters - 1:
            losses = estimate_loss()
            print(
                f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )

        # 采样一个 batch
        xb, yb = get_batch("train")

        # 前向 + 反向 + 参数更新
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # =========================
    # 生成文本
    # =========================
    # 生成文本前切换到评估模式，关闭 dropout
    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=2000)[0].tolist()
    print(decode(generated))