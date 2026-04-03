import torch

def educational_dropout(X, drop_prob, training=True):
    """
    教学版 Inverted Dropout
    加入了详细的 print 输出，方便观察内部变量的变化情况。
    """
    print("【步骤 1】进入 Dropout 层，原始输入 X:")
    print(X)
    
    assert 0 <= drop_prob <= 1, "丢弃概率必须在0和1之间"
    keep_prob = 1.0 - drop_prob
    print(f"\n设定丢弃概率 p={drop_prob}, 意味着保留概率 keep_prob={keep_prob}")

    # Dropout 只在训练(training)时才会生效
    if not training:
        print("\n[推理模式] 直接放行，不改变输入。")
        return X

    # 极端情况
    if keep_prob == 0:
        return torch.zeros_like(X) # 生成一个和张量 X 属性完全相同，但里面所有元素全为 0 的新张量

    # 生成 0~1 之间的均匀分布随机数矩阵
    random_matrix = torch.rand(X.shape)
    print("\n【步骤 2】生成和 X 相同形状的随机矩阵 torch.rand(X.shape):")
    print(random_matrix)

    # 制作 Mask 掩码：随机值小于保留率的变为 True(1.0)，否则为 False(0.0)
    mask = (random_matrix < keep_prob).float()
    print(f"\n【步骤 3】通过 < {keep_prob} 比较并转为float，生成掩码 Mask (0或1):")
    print(mask)

    # 应用掩码，并且除以 keep_prob（倒置机制/Inverted）
    # 为什么要除以 keep_prob？ 
    # 因为有 drop_prob 比例的神经元被归零了，剩下的神经元需要“被拉伸/放大”，以保证总的期望值与原始输入相同。
    Y = (mask * X) / keep_prob
    print("\n【步骤 4】生成最终输出 Y = (mask * X) / keep_prob :")
    print(Y)

    return Y

# =======================
# 教学测试：运行以下代码观察结果
# =======================
if __name__ == '__main__':
    # 设置随机数种子，以便重现相同的结果
    torch.manual_seed(42)
    
    # 假设这是某一层神经元的输出，为了方便观察，我们给一个全 1 的张量
    # 你会发现在 Dropout 后，保留下来的值不再是 1，而是 1 / keep_prob
    sample_inputs = torch.ones(100, 50) # 尺寸要大，从而符合大数定律
    
    print("============== 演示开始 ==============")
    # 设定丢弃概率 40% (即保留 60%)
    output = educational_dropout(sample_inputs, drop_prob=0.4, training=True)
    
    print("\n============== 期望值验证 (Inverted Dropout的精髓) ==============")
    print(f"原始输入 X 的期望值 (平均值): {sample_inputs.mean().item():.4f}")
    print(f"输出结果 Y 的期望值 (平均值): {output.mean().item():.4f}")
    print("结论: 尽管有部分神经元被置为了0，但因为我们在代码中 /keep_prob 进行了缩放，")
    print("使得输出数据的整体分布期望值和输入保持了大致相等。这也就是文章中 E(h'_i) = h_i 的原理！")