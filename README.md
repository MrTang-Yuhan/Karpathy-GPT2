# Karpathy-GPT2

基于 Andrej Karpathy《Neural Networks: Zero to Hero》系列视频的学习与复现项目，内容包括自动微分、MLP、Attention、Tokenizer，以及 GPT-2 的单卡/多卡训练 Demo。

## 参考资料

- [YouTube 英文视频](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
- [Bilibili 中文汉化](https://www.bilibili.com/video/BV1mqrTBvEaf/?spm_id_from=333.788.player.switch&vd_source=e98b669ccbafff4b5aa59dd6303b722f&p=11)

---

## 各部分内容简介

### 手动搭建自动微分器
从零实现一个简单的自动微分系统 (为了简便，不包含张量处理)，理解反向传播的核心机制。  
👉 [进入目录](./micro-grad/README.md)

### BatchNorm 和 LayerNorm 原理
整理并解释 BatchNorm 与 LayerNorm 的作用、公式与区别。  
👉 [进入目录](./batchnorm&layernorm/README.md)

### 手动搭建神经网络
从基础模块出发实现一个简单神经网络，并理解训练流程。  
👉 [进入目录](./mlp/README.md)

### 解释 Attention 机制
介绍 Attention 的基本思想、计算过程及其在 Transformer 中的作用。  
👉 [进入目录](./attention/README.md)

### Decode-only Transformer
实现并分析 Decoder-only Transformer 的结构。  
👉 [进入目录](./decode-transformer/README.md)

### 分词器
实现或整理分词器相关内容，理解文本到 token 的映射过程。  
👉 [进入目录](./tokenizer/README.md)

### GPT-2 单卡 Demo
单卡环境下的 GPT-2 训练或推理示例。  
👉 [进入目录](./gpt2-single/README.md)

### GPT-2 多卡 Demo
多卡训练相关实现与实验记录。  
👉 [进入目录](./gpt2-distro/README.md)

---

