# ELE AI算法大赛“赛道二：智慧骑士—消防隐患识别”

# 数据集下载

| 名称 | 链接 |
|:--:|:--:|
|train images|[train images](https://tianchi-race-prod-sh.oss-cn-shanghai.aliyuncs.com/file/race/documents/prod/532324/1489/public/%E6%99%BA%E6%85%A7%E9%AA%91%E5%A3%AB_train.zip?Expires=1745360208&OSSAccessKeyId=LTAI5t7fj2oKqzKgLGz6kGQc&Signature=ixXim6rGFJkiKjtsVcWRncBFiXg%3D&response-content-disposition=attachment%3B%20)|
|labels|[labels](https://tianchi-race-prod-sh.oss-cn-shanghai.aliyuncs.com/file/race/documents/prod/532324/1489/public/%E6%99%BA%E6%85%A7%E9%AA%91%E5%A3%AB_label.zip?Expires=1745360214&OSSAccessKeyId=LTAI5t7fj2oKqzKgLGz6kGQc&Signature=JfxlxDieeZu7B8%2F%2BuaK8eYtif9o%3D&response-content-disposition=attachment%3B%20) |
|A榜|[A榜](https://tianchi-race-prod-sh.oss-cn-shanghai.aliyuncs.com/file/race/documents/prod/532324/1489/public/%E6%99%BA%E6%85%A7%E9%AA%91%E5%A3%AB_A.zip?Expires=1745360215&OSSAccessKeyId=LTAI5t7fj2oKqzKgLGz6kGQc&Signature=9uDWeaSzGCSIyzI3CLLyOfLTx2M%3D&response-content-disposition=attachment%3B%20) |

下载数据集后解压到 `dataset` 目录下，目录结构如下：

```bash
dataset
├── A
│   ├── id1.jpg
│   ├── id2.jpg
│   ├── ...
│   └── idN.jpg
├── train
│   ├── id1.jpg
│   ├── id2.jpg
│   ├── ...
│   └── idN.jpg
├── A.txt
├── submit_example.txt
└── train.txt
```
# 数据集划分
运行 `split_dataset.py` 脚本将数据集划分为训练集和验证集，划分比例为 9:1。划分后的数据集标注文件为 `train_split.txt` 和 `val_split.txt`

# 训练