import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from datetime import datetime
from dataset import CustomDataset
import logging
import argparse


def setup_logging(output_dir):
    """设置日志记录"""
    log_file = os.path.join(output_dir, "training.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def evaluate(model, dataloader, device, num_classes=None):
    """验证模型性能（不使用 sklearn，支持动态类别数）"""
    import torch

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, ori_images, labels in dataloader:
            images,  labels = images.to(device), labels.to(device)
            outputs = model(images)
            outputs = outputs.softmax(dim=1)
            _, preds = torch.max(outputs, 1)
            all_preds.append(preds)
            all_labels.append(labels)

    # 拼接所有批次
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # 如果未指定类别数，自动根据模型输出推测
    if num_classes is None:
        num_classes = model(images).shape[1]

    # 初始化指标
    class_metrics = {cls: {"precision": 0.0, "recall": 0.0, "f1": 0.0, "acc": 0.0} for cls in range(num_classes)}

    for cls in range(num_classes):
        # 预测为 cls 的位置
        pred_pos = (all_preds == cls)
        # 实际为 cls 的位置
        true_pos = (all_labels == cls)

        # TP, FP, FN
        tp = (pred_pos & true_pos).sum().item()
        fp = (pred_pos & ~true_pos).sum().item()
        fn = (~pred_pos & true_pos).sum().item()
        tn = (~pred_pos & ~true_pos).sum().item()

        cls_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        cls_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        cls_f1 = (
            2 * cls_precision * cls_recall / (cls_precision + cls_recall)
            if (cls_precision + cls_recall) > 0 else 0.0
        )
        cls_acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

        class_metrics[cls]["precision"] = cls_precision
        class_metrics[cls]["recall"] = cls_recall
        class_metrics[cls]["f1"] = cls_f1
        class_metrics[cls]["acc"] = cls_acc

    return class_metrics

def create_transform(transform_config):
    """根据配置创建transform"""
    transform_list = []
    
    # 添加配置中定义的所有变换
    for transform_name, params in transform_config.items():
        if transform_name == "Resize":
            transform_list.append(transforms.Resize(tuple(params)))
        elif transform_name == "RandomCrop":
            transform_list.append(transforms.RandomCrop(tuple(params)))
        elif transform_name == "RandomHorizontalFlip":
            transform_list.append(transforms.RandomHorizontalFlip(p=params))
        elif transform_name == "RandomVerticalFlip":
            transform_list.append(transforms.RandomVerticalFlip(p=params))
        elif transform_name == "ColorJitter":
            transform_list.append(transforms.ColorJitter(**params))
        elif transform_name == "RandomRotation":
            transform_list.append(transforms.RandomRotation(params))
        elif transform_name == "ToTensor":
            transform_list.append(transforms.ToTensor())
        elif transform_name == "Normalize":
            transform_list.append(transforms.Normalize(mean=params["mean"], std=params["std"]))
    
    return transforms.Compose(transform_list)

def main(config_path):
    # 加载配置
    config = load_config(config_path)

    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config["output_dir"], timestamp)
    os.makedirs(output_dir, exist_ok=True)
    setup_logging(output_dir)

    # 复制配置文件到输出目录
    config_save_path = os.path.join(output_dir, "config.yaml")
    os.system(f"cp {config_path} {config_save_path}")
    logging.info(f"Saved configuration to {config_save_path}")

    # 日志记录
    logging.info("Loaded configuration:")
    logging.info(config)

    # 设置设备
    device = torch.device(config["device"])
    logging.info(f"Using device: {device}")

    # 数据集和数据加载器 - 使用不同的transform
    train_transform = create_transform(config["transform"]["train"])
    val_transform = create_transform(config["transform"]["val"])
    
    train_dataset = CustomDataset(
        root_dir=config["dataset"]["data_root_path"],
        txt_file=config["dataset"]["train_txt"],
        transform=train_transform
    )
    val_dataset = CustomDataset(
        root_dir=config["dataset"]["data_root_path"],
        txt_file=config["dataset"]["val_txt"],
        transform=val_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True, num_workers=config["training"]["num_workers"])
    val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"], shuffle=False, num_workers=config["training"]["num_workers"])

    # 模型
    model = getattr(models, config["model"]["name"])(pretrained=config["model"]["pretrained"])
    model.classifier = nn.Linear(model.classifier[1].in_features, config["model"]["num_classes"])
    model = model.to(device)

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # 优化器
    optimizer = getattr(optim, config["optimizer"]["type"])(
        model.parameters(), **config["optimizer"]["params"]
    )

    # 学习率调度器
    scheduler = getattr(optim.lr_scheduler, config["scheduler"]["type"])(
        optimizer, **config["scheduler"]["params"]
    )

    # 训练
    best_f1 = 0.0
    for epoch in range(config["training"]["epochs"]):
        model.train()
        running_loss = 0.0
        for images, ori_images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # 验证
        class_metrics = evaluate(model, val_loader, device, num_classes=config["model"]["num_classes"])
        logging.info(f"Epoch {epoch+1}/{config['training']['epochs']}, Loss: {running_loss/len(train_loader):.4f}")
        for cls, metrics in class_metrics.items():
            logging.info(
                f"Class {cls} - Acc: {metrics['acc']:.4f}, F1: {metrics['f1']:.4f}, "
                f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}"
            )

        # 保存最佳模型
        avg_f1 = sum(metrics["f1"] for metrics in class_metrics.values()) / len(class_metrics)
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            best_model_path = os.path.join(output_dir, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"Saved best model to {best_model_path}")

        # 更新学习率
        scheduler.step()

    logging.info("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='模型训练脚本')
    parser.add_argument('--config', type=str, default='config.yaml', 
                        help='配置文件路径 (默认: config.yaml)')
    args = parser.parse_args()
    main(args.config)