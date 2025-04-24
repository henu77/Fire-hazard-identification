import os
from PIL import Image
from torch.utils.data import Dataset
import torch
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, root_dir, txt_file, transform=None):
        """
        Args:
            root_dir (str): 数据集的根目录路径。
            txt_file (str): 包含文件名和标签的txt文件路径。
            transform (callable, optional): 对图像进行的变换。
        """
        self.root_dir = root_dir
        self.txt_file = txt_file
        self.transform = transform
        self.ori_img_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),  # 调整大小
                transforms.ToTensor(),  # 转为张量
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # 将 [0, 1] 映射到 [-1, 1]
            ]
        )

        label_map = {
            "非楼道": 0,
            "无风险": 1,
            "低风险": 2,
            "中风险": 3,
            "高风险": 4,
        }

        # 读取txt文件，解析文件名和标签
        self.data = []
        with open(txt_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:  # 确保不是空行
                    filename, label = line.split("\t")
                    self.data.append((filename, label_map[label]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        # 获取文件名和标签
        img_name, label = self.data[idx]

        # 加载图像
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        ori_image = image.copy()

        # 应用变换（如果有）
        if self.transform:
            image = self.transform(image)
        ori_image = self.ori_img_transform(ori_image)
        return image, ori_image, label


if __name__ == "__main__":
    # 示例用法
    from torchvision import transforms

    root_dir = "dataset/train"
    txt_file = "dataset/train.txt"

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)), # 调整大小
            transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
            transforms.RandomRotation(degrees=15),  # 随机旋转
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),  # 随机颜色抖动
            transforms.ToTensor(),  # 转为张量
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # 将 [0, 1] 映射到 [-1, 1]
        ]
    )

    dataset = CustomDataset(root_dir=root_dir, txt_file=txt_file, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    import time
    start_time = time.time()

    for images, labels in dataloader:
        print(images.size(), labels)
        end_time = time.time()
        break
    print("Time taken for one batch:", end_time - start_time)
