import torch
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


def main():
    # ==== Device Selection ====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ==== Create dataset and dataloader ====
    val_dataset = ClassDirectoryDataset("data/hymenoptera_data/val", ["jpg"],is_train=False)  # 自定义 Dataset
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False,num_workers=0)

    # ==== Model Instance ====
    model = AntBeeClassifier(num_classes=2).to(device)
    checkpoint = torch.load("experiments/checkpoints/exp-20250823-020050/best_model.pth", map_location=device,weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    # ==== Loss Function & Optimizer ====
    criterion = nn.CrossEntropyLoss(reduction="sum")

    val_loss, val_acc = evaluate(model, val_loader, criterion, str(device))
    print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")


if __name__ == "__main__":
    main()
