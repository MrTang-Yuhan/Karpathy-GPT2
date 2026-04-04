# GPT-2 Distro（分布式训练与生成）

## 运行命令
`torchrun --standalone --nproc_per_node=8 train_gpt2_distro_demo.py`

8 张 GPU 运行。

## 总体功能

本目录围绕 `train_gpt2_distro_demo.py`，在 `gpt2-single` 版本基础上加入 DDP（DistributedDataParallel）能力，实现多卡并行训练：
- 保留 GPT-2 单文件实现的核心结构（模型、优化器、学习率调度、采样）
- 新增分布式初始化、进程角色划分、按 rank 的数据切分与梯度同步控制
- 在多卡场景下保持与单卡近似等价的训练目标，同时提升吞吐

## 文件说明

| 文件 | 说明 |
|---|---|
| [`train_gpt2_distro_demo.py`](./train_gpt2_distro_demo.py) | 分布式核心脚本：模型定义 + DDP 初始化 + 分布式训练循环 + 文本生成 |
| [`input.txt`](./input.txt) | 训练语料，`DataLoaderLite` 读取后进行 token 化与批次切分 |

## 与 `gpt2-single` 的核心差异（重点）

### 1) 运行与设备管理：单进程 → 多进程多卡
- **single 版本**：`setup_device()` 自动选 `cpu/cuda/mps`，单进程训练。
- **distro 版本**：`setup_ddp()` 读取 `RANK/LOCAL_RANK/WORLD_SIZE`，并在 DDP 场景调用 `init_process_group(backend="nccl")`。
- **作用**：为每个进程绑定专属 GPU（`cuda:{LOCAL_RANK}`），建立多进程通信拓扑。

### 2) 模型封装：普通 `nn.Module` → `DDP(model)`
- **single 版本**：直接对 `model` 前向与反向。
- **distro 版本**：在 DDP 场景使用 `model = DDP(model, device_ids=[ddp_local_rank])`。
- **关键点**：
	- 训练时使用包装后的 `model` 做反向传播（触发分布式梯度逻辑）
	- 配置优化器与推理时使用 `raw_model = model.module if ddp else model` 获取原始模型

### 3) 数据读取：顺序取 batch → 按进程分片取 batch
- **single 版本 `next_batch()`**：`current_position += B*T`，越界后回到 0。
- **distro 版本 `next_batch()`**：
	- 初始偏移：`current_position = B*T*process_rank`
	- 步进跨度：`current_position += B*T*num_processes`
	- 越界重置：回到该 rank 对应起点 `B*T*process_rank`
- **作用**：避免多进程重复消费同一片数据，实现近似数据并行分片。

### 4) 梯度累加：增加 DDP 通信优化（`no_sync`）
- **共同点**：都使用 `loss /= grad_accum_steps`，保证累加后梯度尺度与大 batch 等价。
- **distro 新增点**：在非最后一个 micro-step 使用 `with model.no_sync(): ...`。
- **原理与作用**：前几个 micro-step 只本地累加梯度，不立即 all-reduce；最后一步再同步，减少通信次数并提升效率。

### 5) loss 统计：单卡日志 → 跨卡聚合日志
- **single 版本**：直接打印本进程 `loss`。
- **distro 版本**：`dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)` 聚合所有卡的平均 loss。
- **作用**：日志反映全局训练状态，而不是某一张卡的局部波动。

### 6) 全局 batch 计算方式变化
- **single 版本**：`grad_accum_steps = total_batch_size // (B*T)`。
- **distro 版本**：`grad_accum_steps = total_batch_size // (B*T*ddp_world_size)`。
- **作用**：把 world size 纳入等效 batch 计算，保证扩展到多卡后总 token 预算仍可控。

### 7) 打印与资源收尾
- **single 版本**：统一打印。
- **distro 版本**：仅 `master_process` 打印关键日志，避免多进程输出混杂；训练结束后调用 `destroy_process_group()` 释放分布式资源。

## 与 single 共享但在 distro 中仍然关键的机制

### 1) 学习率余弦衰减（`get_lr`）
- warmup 线性升温，随后余弦衰减到 `min_lr`；在多卡训练中同样用于稳定前期更新与后期收敛。

### 2) 参数组权重衰减（`configure_optimizers`）
- 二维及以上参数做衰减，一维参数不衰减；在分布式中依旧遵循同一优化策略。

### 3) 梯度裁剪（`clip_grad_norm_`）
- 对全局梯度范数超阈值时执行等比例缩放，降低梯度爆炸风险，保证多卡训练稳定。

### 4) Top-k 采样生成
- 生成阶段保持 `topk + multinomial + gather` 流程；`distro` 额外通过 `valid_vocab_size` 截断 logits，避免 decode 越界。

## 快速理解：为什么这些改动是“分布式必需”

- **并行计算需要通信协议**：`init_process_group` + `DDP` 负责定义“谁和谁同步、何时同步”。
- **并行数据需要去重分片**：rank 偏移与跨进程步进，确保每卡看到不同子序列。
- **并行训练需要控制通信成本**：`no_sync` 将多次同步压缩为一次，提升吞吐。
- **并行日志需要全局视角**：`all_reduce` 把各卡指标汇总成可解释的全局值。

