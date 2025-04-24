import os
from collections import defaultdict
import random

def split_dataset(txt_file, train_output, val_output, split_ratio=0.9):
    """
    按照每个类别的比例划分数据集，并保存到两个新的txt文件中。

    Args:
        txt_file (str): 原始txt文件路径。
        train_output (str): 训练集输出文件路径。
        val_output (str): 验证集输出文件路径。
        split_ratio (float): 训练集所占比例，默认0.9。
    """
    # 按类别存储数据
    data_by_label = defaultdict(list)
    with open(txt_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:  # 确保不是空行
                filename, label = line.split("\t")
                data_by_label[label].append(f"{filename}\t{label}")

    # 按比例划分数据
    train_data = []
    val_data = []
    for label, items in data_by_label.items():
        random.shuffle(items)  # 打乱数据
        split_idx = int(len(items) * split_ratio)
        train_data.extend(items[:split_idx])
        val_data.extend(items[split_idx:])

    # 保存到新的txt文件
    with open(train_output, "w", encoding="utf-8") as f:
        f.write("\n".join(train_data))
    with open(val_output, "w", encoding="utf-8") as f:
        f.write("\n".join(val_data))

if __name__ == "__main__":
    # 原始txt文件路径
    txt_file = "dataset/train.txt"
    # 输出文件路径
    train_output = "dataset/train_split.txt"
    val_output = "dataset/val_split.txt"

    # 调用函数进行划分
    split_dataset(txt_file, train_output, val_output, split_ratio=0.9)
    print(f"数据集已划分完成，训练集保存到 {train_output}，验证集保存到 {val_output}")