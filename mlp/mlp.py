import random

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# =========================
# 读取数据并构建词表
# =========================
with open("names.txt", "r", encoding="utf-8") as f:
    words = f.read().splitlines()

# 约定：
# - '.' 表示词结束符，索引固定为 0
# - 其余字符从 1 开始编号
chars = sorted(list(set("".join(words))))
stoi = {ch: i + 1 for i, ch in enumerate(chars)}
stoi["."] = 0
itos = {i: ch for ch, i in stoi.items()}
vocab_size = len(itos)

print("字符表：", itos)
print("词表大小：", vocab_size)

# =========================
# 固定随机种子
# =========================
random.seed(42)
torch.manual_seed(42)
g = torch.Generator().manual_seed(42)

random.shuffle(words)

# =========================
# 构建数据集
# =========================
block_size = 8  # 用前 8 个字符预测下一个字符


def build_dataset(word_list):
    X_data = []
    Y_data = []

    for w in word_list:
        context = [0] * block_size  # 初始上下文为 "..."
        for ch in w + ".":
            ix = stoi[ch]
            X_data.append(context)
            Y_data.append(ix)
            # 上下文窗口左移一位，并拼上当前字符
            context = context[1:] + [ix]

    X = torch.tensor(X_data, dtype=torch.long)
    Y = torch.tensor(Y_data, dtype=torch.long)
    print(X.shape, Y.shape)
    return X, Y


n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

Xtr, Ytr = build_dataset(words[:n1])  # 训练集 80%
Xdev, Ydev = build_dataset(words[n1:n2])  # 验证集 10%
Xte, Yte = build_dataset(words[n2:])  # 测试集 10%

print("\n部分训练样本：")
for x, y in zip(Xtr[:20], Ytr[:20]):
    context = "".join(itos[i.item()] for i in x)
    target = itos[y.item()]
    print(f"上下文: '{context}' -> 目标: '{target}'")

# =========================
# 定义网络层
# =========================
class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        # 简单随机初始化
        self.weight = torch.randn((fan_in, fan_out), generator=g) / fan_in**0.5
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x):
        # 仿射变换：xW + b
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        self.out = out
        return self.out

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])


class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True

        # 可训练参数
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

        # 运行时统计量：评估阶段使用，不参与梯度更新
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        if self.training:
            # 按 batch 维统计，每个特征各自求均值和方差
            xmean = x.mean(dim=0, keepdim=True)
            xvar = x.var(dim=0, keepdim=True)
            with torch.no_grad():
                # 动量更新运行统计量
                self.running_mean = (
                    (1 - self.momentum) * self.running_mean + self.momentum * xmean
                )
                self.running_var = (
                    (1 - self.momentum) * self.running_var + self.momentum * xvar
                )
        else:
            # 推理时使用训练过程中积累的运行统计量
            xmean = self.running_mean
            xvar = self.running_var

        # 标准化，再做缩放和平移
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]


class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        return []


class Embedding:
    def __init__(self, num_embeddings, embedding_dim):
        # 每个字符对应一个可学习向量
        self.weight = torch.randn((num_embeddings, embedding_dim), generator=g)

    def __call__(self, ix):
        # 查表：输入索引 -> 嵌入向量
        self.out = self.weight[ix]
        return self.out

    def parameters(self):
        return [self.weight]


class Flatten:
    def __call__(self, x):
        # 将 (B, block_size, n_embd) 拉平成 (B, block_size * n_embd)
        self.out = x.view(x.shape[0], -1)
        return self.out

    def parameters(self):
        return []


class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def train(self):
        for layer in self.layers:
            if hasattr(layer, "training"):
                layer.training = True

    def eval(self):
        for layer in self.layers:
            if hasattr(layer, "training"):
                layer.training = False


# =========================
# 定义模型
# =========================
n_embd = 10
n_hidden = 100

# 当前网络结构：
# 1) Embedding:
#    输入 (B, 3) 的字符索引
#    输出 (B, 3, 10) 的字符嵌入
#
# 2) Flatten:
#    将 3 个字符嵌入拼接为一个向量
#    (B, 3, 10) -> (B, 30)
#
# 3) Linear(30 -> 100)
# 4) BatchNorm1d(100)
# 5) Tanh
# 6) Linear(100 -> vocab_size)
#
# 最终输出 logits 的形状为 (B, vocab_size)，
# 每一行表示“下一个字符”的类别分数。
model = Sequential(
    [
        Embedding(vocab_size, n_embd),
        Flatten(),
        Linear(n_embd * block_size, n_hidden, bias=False),
        BatchNorm1d(n_hidden),
        Tanh(),
        Linear(n_hidden, vocab_size, bias=True),
    ]
)

parameters = model.parameters()
print("\n参数总数：", sum(p.nelement() for p in parameters))

for p in parameters:
    p.requires_grad = True

# =========================
# 训练
# =========================
max_steps = 10000
batch_size = 32
lossi = []

model.train()

for i in range(max_steps):
    # 采样一个小批量
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
    Xb, Yb = Xtr[ix], Ytr[ix]

    # 核心训练流程：前向 -> 损失 -> 反向
    # 前向传播：得到每个类别的 logits
    logits = model(Xb)

    # 交叉熵损失：
    # 输入 shape: (B, vocab_size)
    # 目标 shape: (B,)
    loss = F.cross_entropy(logits, Yb)

    # 清空旧梯度
    for p in parameters:
        p.grad = None

    # 反向传播
    loss.backward()

    # 学习率：前 80% 步数用较大学习率，后 20% 缩小
    lr = 0.1 if i < int(max_steps * 0.8) else 0.01

    # 核心参数更新：使用 SGD 按学习率更新所有可训练参数
    for p in parameters:
        p.data -= lr * p.grad

    if i % 1000 == 0:
        print(f"{i:7d}/{max_steps:7d}: loss={loss.item():.4f}")

    lossi.append(loss.log10().item())

# =========================
# 绘制损失曲线
# =========================
plt.figure(figsize=(8, 4))
smoothed_loss = torch.tensor(lossi).view(-1, 1000).mean(dim=1)
plt.plot(smoothed_loss)
plt.xlabel("Step (x1000)")
plt.ylabel("log10(loss)")
plt.title("Training Loss")
plt.show()

# =========================
# 评估
# =========================
model.eval()


@torch.no_grad()
def split_loss(split: str):
    # 核心评估流程：在固定数据切分上计算整集交叉熵
    x, y = {
        "train": (Xtr, Ytr),
        "val": (Xdev, Ydev),
        "test": (Xte, Yte),
    }[split]

    logits = model(x)
    loss = F.cross_entropy(logits, y)
    print(f"{split}: {loss.item():.4f}")

print()
split_loss("train")
split_loss("val")
split_loss("test")

# =========================
# 采样生成名字
# =========================
print("\n生成样例：")
for _ in range(20):
    out = []
    context = [0] * block_size  # 初始上下文为 "..."

    # 核心生成流程：自回归采样，直到遇到结束符
    while True:
        x = torch.tensor([context], dtype=torch.long)
        logits = model(x)  # shape: (1, vocab_size)
        probs = F.softmax(logits, dim=1)  # 转为概率分布

        # 按概率采样下一个字符
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()

        # 更新上下文窗口
        context = context[1:] + [ix]
        out.append(ix)

        # 采样到结束符则停止
        if ix == 0:
            break

    print("".join(itos[i] for i in out))
