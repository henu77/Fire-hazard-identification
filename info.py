import os
import torch
import argparse
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn
from tqdm import tqdm
import yaml
import logging
from dataset import CustomDataset

def setup_logging():
    """设置日志记录"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def create_transform(transform_config):
    """根据配置创建transform"""
    transform_list = []
    
    for transform_name, params in transform_config.items():
        if transform_name == "Resize":
            transform_list.append(transforms.Resize(tuple(params)))
        elif transform_name == "ToTensor":
            transform_list.append(transforms.ToTensor())
        elif transform_name == "Normalize":
            transform_list.append(transforms.Normalize(mean=params["mean"], std=params["std"]))
    
    return transforms.Compose(transform_list)

def load_model(model_path, config):
    """加载模型"""
    model = getattr(models, config["model"]["name"])(pretrained=False)
    # 替换分类器
    model.classifier = nn.Linear(model.classifier[1].in_features, config["model"]["num_classes"])
    model.load_state_dict(torch.load(model_path))
    return model

def infer(args):
    # 设置日志
    setup_logging()
    path = args.path
    # 加载配置
    config = load_config(os.path.join(path, "config.yaml"))
    logging.info(f"加载配置文件: {config}")
    
    # 设置设备
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    logging.info(f"使用设备: {device}")
    
    # 创建测试用transform
    transform = create_transform(config["transform"]["val"])
    
    # 加载模型
    model = load_model(os.path.join(path, "best_model.pth"), config)
    model = model.to(device)
    model.eval()
    logging.info(f"加载模型: model")
    
    # 读取图片列表
    with open(args.img_list, "r") as f:
        img_list = [line.strip() for line in f.readlines()]
    logging.info(f"从 {args.img_list} 读取了 {len(img_list)} 张图片")
    
    # 类别映射（从索引到类别名）
    label_map = {
        0: "非楼道",
        1: "无风险",
        2: "低风险",
        3: "中风险",
        4: "高风险"
    }
    
    # 创建结果文件
    results = []
    
    # 开始推理
    logging.info("开始推理...")
    with torch.no_grad():
        for img_name in tqdm(img_list):
            try:
                # 构建图片路径
                img_path = os.path.join(args.img_dir, img_name)
                
                # 加载并预处理图片
                image = Image.open(img_path).convert("RGB")
                image = transform(image)
                image = image.unsqueeze(0).to(device)  # 添加批次维度
                
                # 推理
                outputs = model(image)
                prob = torch.softmax(outputs, dim=1)
                confidence, pred = torch.max(prob, 1)
                
                # 获取预测结果
                pred_class = pred.item()
                pred_label = label_map[pred_class]
                conf_score = confidence.item()
                
                # 保存结果
                results.append((img_name, pred_label, pred_class, conf_score))
                
            except Exception as e:
                logging.error(f"处理图片 {img_name} 时发生错误: {e}")
                results.append((img_name, "错误", -1, 0.0))
    output_dir = os.path.join(path, "result.txt")
    # 保存结果
    with open(output_dir, "w", encoding="utf-8") as f:
        f.write("图片名\t预测类别\t类别ID\t置信度\n")
        for img_name, label, class_id, conf in results:
            f.write(f"{img_name}\t{label}\t{class_id}\t{conf:.4f}\n")
    
    logging.info(f"推理完成，结果已保存到: {output_dir}")

    submit_dir = os.path.join(path, "submit.txt")
    with open(submit_dir, "w", encoding="utf-8") as f:
        for img_name, label, class_id, conf in results:
            f.write(f"{img_name}\t{label}\n")
    
    # 统计每个类别的数量
    class_counts = {}
    for _, label, _, _ in results:
        if label not in class_counts:
            class_counts[label] = 0
        class_counts[label] += 1
    
    logging.info("类别统计:")
    for label, count in class_counts.items():
        logging.info(f"  {label}: {count} ({count/len(results)*100:.2f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='模型推理')
    parser.add_argument('--path', type=str, help='配置文件路径')
    parser.add_argument('--img_list', type=str, required=True, help='图片列表文件路径(A.txt)')
    parser.add_argument('--img_dir', type=str, required=True, help='图片所在目录')
    
    args = parser.parse_args()
    infer(args)