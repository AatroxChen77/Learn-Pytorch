"""
蚂蚁蜜蜂图像分类模型

该模块包含用于蚂蚁和蜜蜂图像分类的自定义CNN模型。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AntBeeClassifier(nn.Module):
    
    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.5) -> None:

        super().__init__()
        
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # 第一个卷积块
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 第二个卷积块
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 第三个卷积块
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 第四个卷积块
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 自适应平均池化
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 全连接分类器
        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(128, self.num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入图像张量，形状为 (batch_size, 3, height, width)
            
        Returns:
            分类logits，形状为 (batch_size, num_classes)
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)  # 展平，保持batch维度
        x = self.classifier(x)
        return x
    
    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """
        获取特征图
        
        Args:
            x: 输入图像张量
            
        Returns:
            特征图张量
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x


def count_parameters(model: nn.Module) -> int:
    """
    计算模型参数数量
    
    Args:
        model: PyTorch模型
        
    Returns:
        可训练参数的总数
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary(model: nn.Module) -> str:
    """
    获取模型摘要信息
    
    Args:
        model: PyTorch模型
        
    Returns:
        模型摘要字符串
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = count_parameters(model)
    
    summary = f"""
模型摘要:
- 总参数数量: {total_params:,}
- 可训练参数数量: {trainable_params:,}
- 模型类型: {model.__class__.__name__}
"""
    return summary


if __name__ == "__main__":
    # 测试代码
    print("测试模型创建...")
    
    # 创建模型
    model = AntBeeClassifier(dropout_rate=0.5)
    print(get_model_summary(model))
    print(model)
    
    # 测试前向传播
    test_input = torch.randn(1, 3, 224, 224)
    
    with torch.no_grad():
        output = model(test_input)
        print(f"\n前向传播测试:")
        print(f"输入形状: {test_input.shape}")
        print(f"输出形状: {output.shape}")
        
        # 测试特征图提取
        feature_maps = model.get_feature_maps(test_input)
        print(f"特征图形状: {feature_maps.shape}")