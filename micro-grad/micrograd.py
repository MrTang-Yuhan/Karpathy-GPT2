import math
import random

import numpy as np
import matplotlib.pyplot as plt
from graphviz import Digraph
import networkx as nx
from pyvis.network import Network

def trace(root):
    """
    遍历整个计算图，收集所有节点和边的信息。

    从根节点开始，通过 v._prev 递归找到所有父节点，
    记录所有节点及它们之间的连接关系（边）。

    参数:
        root: 计算图的根节点（通常是 loss）

    返回:
        (nodes, edges): 节点集合和边集合
    """
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for parent in v._prev:
                edges.add((parent, v))
                build(parent)

    build(root)
    return nodes, edges

def draw_dot_nx_pyvis(root, out_html="ComputationGraph.html", open_browser=True):
    """
    使用 networkx + pyvis 绘制计算图的交互式可视化。

    节点类型：
    - 数值节点(Value): 矩形，显示 label / data / grad
    - 操作节点(_op): 圆形，显示操作符

    边的方向：
    parent_value -> op_node -> result_value
    """
    nodes, edges = trace(root)
    g = nx.DiGraph()

    # 1) 添加 Value 节点与 op 节点
    for n in nodes:
        # id(n) 返回对象的唯一内存地址（整数），用于生成唯一节点 ID
        val_id = f"val_{id(n)}"
        g.add_node(
            val_id,
            kind="value",
            label=f"{n.label} | data {n.data:.4f} | grad {n.grad:.4f}",
            shape="box",
            color="lightgrey",
        )

        # 如果该节点是由某个操作产生的，添加对应的操作节点
        if n._op:
            op_id = f"op_{id(n)}_{n._op}"
            g.add_node(
                op_id,
                kind="op",
                label=n._op,
                shape="circle",
                color="lightblue",
            )
            # 操作节点 -> 结果值节点
            g.add_edge(op_id, val_id)

    # 2) 添加 父值节点 -> 操作节点 的边
    for n1, n2 in edges:
        src = f"val_{id(n1)}"
        if n2._op:
            dst = f"op_{id(n2)}_{n2._op}"
            g.add_edge(src, dst)
        else:
            # 兜底：没有操作符时直接连到目标值节点
            dst = f"val_{id(n2)}"
            g.add_edge(src, dst)

    # 3) 用 pyvis 渲染为交互式 HTML
    net = Network(height="700px", width="100%", directed=True)
    net.from_nx(g)

    # 设置为类似 Graphviz 的从左到右层级布局
    net.set_options(
        """
        {
          "layout": {
            "hierarchical": {
              "enabled": true,
              "direction": "LR",
              "sortMethod": "directed",
              "levelSeparation": 160,
              "nodeSpacing": 140
            }
          },
          "physics": {"enabled": false},
          "edges": {
            "arrows": {"to": {"enabled": true}},
            "smooth": false
          },
          "nodes": {
            "font": {"size": 16}
          }
        }
        """
    )

    net.write_html(out_html, notebook=False, open_browser=open_browser)
    return g, net

class Value:
    """
    自动微分引擎的核心类。

    每个 Value 对象存储：
    - data: 前向传播的数值
    - grad: 反向传播计算得到的梯度（dL/d_self）
    - _prev: 产生该节点的父节点集合（用于构建计算图）
    - _op: 产生该节点的操作符（用于可视化）
    - _backward: 反向传播时计算梯度的闭包函数

    注意：默认参数使用不可变对象 tuple 和 str，
    避免使用 list/set/dict 等可变对象作为默认参数。
    """

    def __init__(self, data, _parent=(), _op='', label=''):
        self.data = data
        self._prev = set(_parent)
        self._backward = lambda: None  # 默认空操作，叶子节点无需反向传播
        self._op = _op
        self.label = label
        self.grad = 0.0

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        """加法：out = self + other"""
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # 加法的局部导数均为 1
            # dL/d_self = dL/d_out * d_out/d_self = out.grad * 1
            # 使用 += 是因为同一个节点可能被多条路径使用（扇出 > 1）
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __sub__(self, other):
        """减法：out = self - other，复用 __add__ 和 __neg__"""
        other = other if isinstance(other, Value) else Value(other)
        return self + (-other)

    def __mul__(self, other):
        """乘法：out = self * other"""
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            # 乘法的局部导数：d(a*b)/da = b, d(a*b)/db = a
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, power, modulo=None):
        """幂运算：out = self ** power（power 须为 int 或 float）"""
        assert isinstance(power, (int, float)), "目前仅支持 int 或 float 的幂运算"
        out = Value(self.data ** power, (self,), f'**{power}')

        def _backward():
            # d(x^n)/dx = n * x^(n-1)
            self.grad += (power * self.data ** (power - 1)) * out.grad

        out._backward = _backward
        return out

    def __truediv__(self, other):
        """除法：out = self / other，复用 __mul__ 和 __pow__"""
        return self * other ** -1

    def __neg__(self):
        """取负：out = -self，复用 __mul__"""
        return self * (-1)

    def __radd__(self, other):
        """反向加法：当 other + self 时，other 不是 Value 类型时调用"""
        return self + other

    def __rmul__(self, other):
        """反向乘法：当 other * self 时，other 不是 Value 类型时调用"""
        return self * other

    def __rsub__(self, other):
        """反向减法：other - self"""
        other = other if isinstance(other, Value) else Value(other)
        return other + (-self)

    def exp(self):
        """指数函数：out = e^self"""
        out = Value(math.exp(self.data), (self,), 'exp')

        def _backward():
            # d(e^x)/dx = e^x = out.data
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        """双曲正切激活函数：out = tanh(self)"""
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')

        def _backward():
            # d(tanh(x))/dx = 1 - tanh²(x)
            # 直接使用 out.data（即 t），避免重复计算 tanh
            self.grad += (1 - out.data ** 2) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        """ReLU 激活函数：out = max(0, self)"""
        out = Value(max(0, self.data), (self,), 'relu')

        def _backward():
            # d(relu(x))/dx = 1 if x > 0 else 0
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        """
        从当前节点开始反向传播，计算所有可达节点的梯度。

        步骤：
        1. 通过 DFS 构建拓扑排序（确保子节点排在父节点之后）
        2. 设置当前节点（输出节点）的梯度为 1.0（dL/dL = 1）
        3. 按拓扑排序的逆序调用每个节点的 _backward()

        注意：此方法应只对最终输出节点（如 loss）调用。
        """
        topo = []
        visited = set()

        def build_topo(v):
            """DFS 构建拓扑排序：先递归处理所有父节点，再将自身加入列表"""
            if v not in visited:
                visited.add(v)
                for parent in v._prev:
                    build_topo(parent)
                topo.append(v)

        build_topo(self)

        # 输出节点对自身的导数为 1
        self.grad = 1.0
        # 逆拓扑序遍历，确保每个节点的梯度在传播给父节点之前已经完全计算
        for node in reversed(topo):
            node._backward()

# ============================================================
# 神经网络模块
# ============================================================

class Neuron:
    """
    单个神经元：out = tanh(w · x + b)

    参数:
        nin: 输入维度（权重数量）
    """

    def __init__(self, nin):
        # 权重初始化为 [-1, 1] 的随机数
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        # 偏置初始化为 [-1, 1] 的随机数
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        """
        前向传播：计算加权和并通过 tanh 激活。

        使用 sum(..., self.b) 将 self.b 作为起始值，
        避免 sum 默认从 int 0 开始导致额外的类型转换。
        """
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out

    def parameters(self):
        """返回该神经元的所有可训练参数（权重 + 偏置）"""
        return self.w + [self.b]

class Layer:
    """
    神经网络的一层，包含多个神经元。

    参数:
        nin:  每个神经元的输入维度
        nout: 该层的神经元数量（输出维度）
    """

    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        """
        前向传播：对每个神经元计算输出。

        如果该层只有 1 个神经元，直接返回 Value 对象（而非列表），
        方便后续直接对标量 loss 调用 backward()。
        """
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        """返回该层所有神经元的可训练参数（扁平列表）"""
        return [p for n in self.neurons for p in n.parameters()]

class MLP:
    """
    多层感知机（Multi-Layer Perceptron）。

    参数:
        nin:   输入维度
        nouts: 列表，每个元素表示对应层的神经元数量
               例如 [4, 4, 1] 表示两个隐藏层（各4个神经元）+ 1个输出层

    示例:
        model = MLP(3, [4, 4, 1])
        构建的网络结构：3 -> 4 -> 4 -> 1
    """

    def __init__(self, nin, nouts):
        # sz = [输入维度, 第1层输出, 第2层输出, ..., 最后一层输出]
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

    def __call__(self, x):
        """前向传播：数据依次通过每一层"""
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        """返回整个网络的所有可训练参数（扁平列表）"""
        return [p for layer in self.layers for p in layer.parameters()]

# ============================================================
# 训练
# ============================================================

# 训练数据：4 个样本，每个样本 3 个特征
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]  # 目标值

# 创建网络：输入 3 维，隐藏层 4-4，输出 1 维
n = MLP(3, [4, 4, 1])

# 训练循环
for k in range(100):

    # 1. 梯度清零（必须在 backward 之前，防止梯度累加）
    for p in n.parameters():
        p.grad = 0.0

    # 2. 前向传播：对每个样本计算预测值
    y_pred = [n(x) for x in xs]

    # 3. 计算损失：均方误差（MSE）的总和
    loss = sum(((y_hat - y) ** 2 for y, y_hat in zip(ys, y_pred)), Value(0.0))

    # 4. 反向传播：计算所有参数的梯度
    loss.backward()

    # 5. 参数更新：梯度下降
    learning_rate = 0.05
    for p in n.parameters():
        p.data -= learning_rate * p.grad

    if k % 10 == 0 or k == 99:
        print(f"step {k:3d}, loss = {loss.data:.6f}")

# 查看最终预测结果
y_pred = [n(x) for x in xs]
print("\n最终预测值:")
for i, (y, yp) in enumerate(zip(ys, y_pred)):
    print(f"  样本 {i}: 目标={y:+.1f}, 预测={yp.data:+.4f}")

# 可视化计算图（可选）
draw_dot_nx_pyvis(loss)


