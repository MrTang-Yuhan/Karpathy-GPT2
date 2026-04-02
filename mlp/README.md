# MLP（姓名生成）

## 总体功能

本目录实现了一个字符级 MLP 姓名生成项目：
- 用 `names.txt` 构建训练数据（根据前文预测下一个字符）
- 训练模型并评估效果
- 采样生成新的姓名
- 提供模块化教学示例，便于理解每个层的作用

## 文件说明

| 文件 | 说明 |
|---|---|
| [`mlp.py`](./mlp.py) | 完整训练脚本：数据处理、模型训练、评估与采样生成 |
| [`module.ipynb`](./module.ipynb) | 教学 Notebook：按模块演示类的功能与使用方式 |
| [`names.txt`](./names.txt) | 训练数据 |

## 功能介绍

- **数据构建**：`build_dataset` 将字符序列转换成 `(context -> next_char)` 样本。
- **核心层实现**：`Linear`、`Embedding`、`Flatten`、`Tanh`。
- **归一化层**：`BatchNorm1d` 与 `LayerNorm1d`，展示两者统计维度差异。
- **模型组装**：`Sequential` 用于按顺序组合网络层。
- **教学演示**：`module.ipynb` 为每个模块提供独立演示用例。


