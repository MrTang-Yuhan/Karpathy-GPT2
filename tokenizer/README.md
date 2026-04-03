# tokenizer

字节级 BPE（Byte Pair Encoding）Tokenizer 教学实现，展示从训练到编码/解码的完整流程。

## 文件说明
| [`basic_tokenizer_demo.py`](./basic_tokenizer_demo.py) | 核心示例：实现 `BasicTokenizer`，包含训练、编码、解码与最小可运行演示 |

| [`input.txt`](./input.txt) | 可选训练语料文件，可替换脚本中的默认训练文本 |

## 功能介绍

### BasicTokenizer（字节级 BPE）
- 初始词表固定为 0~255，共 256 个单字节 token
- 通过训练文本统计高频相邻 token pair，迭代学习 merges 规则
- 根据 merges 自动构建扩展词表（新 token 对应组合后的 bytes）

### 训练流程（train）
- 将输入文本先编码为 UTF-8 字节序列
- 按 `vocab_size - 256` 次迭代执行 pair 合并
- 每轮选择当前频率最高的 pair，并分配新的 token id
- 支持提前停止：当序列无法继续形成相邻 pair 时结束

### 编码与解码（bt_encode / bt_decode）
- 编码时按训练优先级应用 merges，得到压缩后的 token id 序列
- 解码时将 token id 还原为 bytes，并拼接后按 UTF-8 转回文本
- 对未知 token id 进行显式报错，保证解码行为可控

### 运行示例（脚本主程序）
- 默认训练文本为 `abababab`，目标词表大小为 `259`
- 运行后打印 merges、部分新词表项，以及编码/解码结果


