import torch
import argparse
from torch.utils.data import DataLoader
import torch.nn as nn
from model import AntBeeClassifier
from dataset import ClassDirectoryDataset

def evaluate(model, dataloader, criterion, device="cuda"):
    """
    对给定模型和数据集进行评估
    返回平均 loss 和准确率
    """
    model.eval()  # 切换到评估模式
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # 验证时不需要计算梯度
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            predicted = outputs.argmax(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print("correct:",correct)
    print("total:",total)

    # avg_loss = total_loss / total
    accuracy = correct / total
    return total_loss, accuracy


def get_args():
    parser = argparse.ArgumentParser(description="Evaluation script for AntBeeClassifier")
    parser.add_argument("-dp", "--data_path", type=str, default="data/hymenoptera_data/val", help="validation dataset path", metavar="")
    parser.add_argument("-bs", "--batch_size", type=int, default=16, help="batch size", metavar="")
    parser.add_argument("-nw", "--num_workers", type=int, default=0, help="dataloader num_workers", metavar="")
    parser.add_argument("-ckpt", "--checkpoint", type=str, default="", help="path to model checkpoint (.pth)", metavar="")
    parser.add_argument("--use_gpu", action="store_true", help="use GPU if available")

    return parser.parse_args()


def main():
    args = get_args()
    # ==== Device Selection ====
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # ==== Create dataset and dataloader ====
    val_dataset = ClassDirectoryDataset(args.data_path, ["jpg"],is_train=False)  # 自定义 Dataset
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,num_workers=args.num_workers)

    # ==== Model Instance ====
    model = AntBeeClassifier(num_classes=2).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device,weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    # ==== Loss Function & Optimizer ====
    criterion = nn.CrossEntropyLoss(reduction="sum")

    val_loss, val_acc = evaluate(model, val_loader, criterion, str(device))
    print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")


if __name__ == "__main__":
    main()
