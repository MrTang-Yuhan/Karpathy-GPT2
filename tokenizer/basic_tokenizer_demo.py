class BasicTokenizer:
    """
    一个基础的字节级 BPE（Byte Pair Encoding）Tokenizer 示例。

    实现流程：
    1. 使用训练文本学习合并规则（merges）
    2. 根据 merges 构建词表（vocab）
    3. 使用训练好的规则对新文本进行编码
    4. 将 token 序列解码还原为文本

    说明：
    - 本实现是“字节级 BPE”，训练和编码的基本单位是 UTF-8 字节，而不是字符。
    - 初始词表固定包含 0~255 共 256 个基础 token，每个 token 对应一个单字节。
    """

    def __init__(self):
        """
        初始化 tokenizer。

        成员变量：
        - self.merges:
            合并规则字典。
            键为二元组 (token1, token2)，值为合并后生成的新 token id。
        - self.vocab:
            词表字典。
            键为 token id，值为该 token 对应的 bytes。
        """
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}

    def get_stats(self, ids):
        """
        统计 token 序列中所有相邻 token pair 的出现次数。

        参数：
        - ids: token 整数列表

        返回：
        - 一个字典：
            键：相邻 token 对，例如 (97, 98)
            值：该 pair 在当前序列中的出现次数
        """
        counts = {}

        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1

        return counts

    def merge(self, ids, pair, new_id):
        """
        将 token 列表中所有连续出现的指定 pair 合并为一个新 token。

        例如：
        ids = [1, 2, 1, 2, 3]
        pair = (1, 2)
        new_id = 256

        合并后得到：
        [256, 256, 3]

        参数：
        - ids: 原始 token 列表
        - pair: 需要合并的相邻 token 对
        - new_id: 合并后生成的新 token id

        返回：
        - 合并后的 token 列表
        """
        new_ids = []
        i = 0

        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                new_ids.append(new_id)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1

        return new_ids

    def build_vocab(self):
        """
        根据当前的 merges 重新构建完整词表。
        """
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]

    def train(self, text, vocab_size):
        """
        使用训练文本学习 BPE 合并规则，并构建词表。

        参数：
        - text: 训练文本（字符串）
        - vocab_size: 目标词表大小，必须大于等于 256

        返回：
        - 无返回值
        """
        if vocab_size < 256:
            raise ValueError("vocab_size 必须大于等于 256")

        # 每次重新训练时，都应重置 merges 和 vocab，
        # 避免同一个实例多次训练时发生状态污染。
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}

        # 将训练文本编码为 UTF-8 字节，再转成整数列表
        ids = list(text.encode("utf-8"))

        # 需要执行的合并次数
        num_merges = vocab_size - 256

        for i in range(num_merges):
            stats = self.get_stats(ids)

            # 如果当前序列已无法再统计出相邻 pair (仅有一个 token)，则提前结束
            if not stats:
                break

            # 选择出现频率最高的 pair 进行合并
            pair = max(stats, key=stats.get)    # max() 默认遍历字典时，会比较键的大小
                                                # 加上 key=stats.get 后，每次比较前会先调用 stats.get(key) 获取对应的值
                                                # 最终返回值最大的那个键

            # 为本轮合并分配新的 token id
            new_id = 256 + i

            # 执行合并
            ids = self.merge(ids, pair, new_id)

            # 记录合并规则
            self.merges[pair] = new_id

        # 根据最终学到的 merges 构建词表
        self.build_vocab()

    def bt_encode(self, text):
        """
        将输入文本编码为 token id 列表。

        编码过程：
        1. 先将文本转为 UTF-8 字节序列
        2. 再根据训练阶段学到的 merges，按优先级不断执行合并
        3. 直到没有可继续合并的 pair 为止

        参数：
        - text: 输入文本（字符串）

        返回：
        - 编码后的 token id 列表
        """
        tokens = list(text.encode("utf-8"))

        while len(tokens) >= 2:
            stats = self.get_stats(tokens)
            if not stats:
                break

            # 在当前所有相邻 pair 中，选择“训练时最早学到”的那个 pair。
            # 这里通过比较 self.merges 中对应的新 token id 实现：
            # token id 越小，说明该 merge 学到得越早，优先级越高。
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf"))) # 如果 pair p 在 self.merges 里，就取它对应的新 token id；如果不在，就返回无穷大，表示优先级最低。

            # 如果当前所有 pair 都不在 merges 中，说明无法继续合并
            if pair not in self.merges:
                break

            new_id = self.merges[pair]
            tokens = self.merge(tokens, pair, new_id)

        return tokens

    def bt_decode(self, ids):
        """
        将 token id 列表解码为字符串。

        解码过程：
        1. 先将每个 token id 映射回对应的 bytes
        2. 再将所有 bytes 拼接
        3. 最后使用 UTF-8 解码为字符串

        参数：
        - ids: token id 列表

        返回：
        - 解码后的字符串
        """
        byte_pieces = []

        for idx in ids:
            if idx not in self.vocab:
                raise ValueError("无法解码，未知 token id: {}".format(idx))
            byte_pieces.append(self.vocab[idx])

        text_bytes = b"".join(byte_pieces)
        return text_bytes.decode("utf-8", errors="replace")

if __name__ == "__main__":
    # =========================
    # 训练文本
    # =========================
    train_text = "abababab"
    # with open("input.txt", "r", encoding="utf-8") as f:
    #     train_text = f.read()

    print("train text =", train_text)

    # =========================
    # 训练 Tokenizer
    # =========================
    tokenizer = BasicTokenizer()
    tokenizer.train(train_text, vocab_size=259)

    print("merges =", tokenizer.merges)
    print("vocab[256] =", tokenizer.vocab.get(256))
    print("vocab[257] =", tokenizer.vocab.get(257))
    print("vocab[258] =", tokenizer.vocab.get(258))

    # =========================
    # 编码与解码测试
    # =========================
    text = "abababab"

    encoded = tokenizer.bt_encode(text)
    decoded = tokenizer.bt_decode(encoded)

    print("原文本:", text)
    print("编码后:", encoded)
    print("解码后:", decoded)