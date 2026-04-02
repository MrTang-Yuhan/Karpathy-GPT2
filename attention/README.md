# Attention（从 Bigram 到 Self-Attention）

## 总体功能

本目录围绕字符级语言建模，演示从 **Bigram 语言模型** 到 **Self-Attention 模型** 的完整学习路径。项目以 `input.txt`（tiny shakespeare）为语料，包含：
- Bigram 的训练与文本生成
- 三种“因果前缀均值聚合”实现（循环 / 下三角矩阵 / masked softmax）
- Self-Attention 模型的训练与文本生成

## 文件说明

| 文件 | 说明 |
|---|---|
| [`自注意力机制的由来.md`](./自注意力机制的由来.md) | 理论文档：系统说明 Bigram 局限、三种均值等价性与自注意力引入动机 |
| [`attention.ipynb`](./attention.ipynb) | 主教学 Notebook：包含 Bigram、三种均值版本、自注意力模型训练与生成代码 |
| [`input.txt`](./input.txt) | 训练语料（tiny shakespeare 字符级文本） |
| [`img/`](./img) | 文档配图资源（如交叉熵示意图） |

## 功能介绍

- **数据处理与采样**：构建字符词表，完成编码/解码，并按 `(B, T)` 批次采样训练数据。
- **Bigram 建模**：使用 `Embedding(vocab_size, vocab_size)` 做高效查表参数化，配合交叉熵进行训练与生成。
- **均值聚合三种实现**：展示前缀均值在循环实现、矩阵实现、masked softmax 实现下的等价性。
- **Self-Attention 建模**：通过 `Query/Key/Value`、因果 mask 与 softmax 动态分配注意力权重，实现基于上下文的生成。


