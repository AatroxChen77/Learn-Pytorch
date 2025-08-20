from torch.utils.data import Dataset, DataLoader
from PIL import Image
from os import listdir
import os
from os.path import isdir,join,splitext,basename,exists
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
IMG_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
)

def _pil_loader(path):
    with Image.open(path) as img:
        return img.convert("RGB")

def _is_image_file(filename,extensions):
    return filename.lower().endswith(tuple(ext.lower() for ext in extensions))

def _my_transforms(image_size=224, is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

def _find_classes(directory):
    """
    从指定文件夹中读取所有的类别
    :param directory:
    :return:
    """
    classes = [class_dir for class_dir in listdir(directory) if isdir(join(directory, class_dir))]
    classes.sort()
    if len(classes) == 0:
        raise RuntimeError(f"目录{directory}中找不到任何类别子目录")
    return classes


class ClassDirectoryDataset(Dataset):
    """
    适用于 ImageFolder 风格目录结构的数据集，例如：

    root/
        class_a/
            img1.jpg
            ...
        class_b/
            img2.jpg
            ...

    - 自动按子文件夹名建立 `class_to_idx`
    - 收集 (image_path, label_idx) 样本列表
    - 使用给定 `transform` 和 `target_transform`
    """

    def __init__(self, root, extensions, loader=_pil_loader,
                 transform_builder=_my_transforms, target_transform=None,
                 is_train=True):
        self.root = root
        self.extensions = tuple(ext.lower() for ext in extensions)
        self.loader = loader

        self.classes = _find_classes(self.root)
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.classes)}
        self.samples = self._make_dataset(self.root)
        self.transform = transform_builder(is_train=is_train)
        self.target_transform = target_transform

    def _make_dataset(self, directory):
        instances = []
        for class_name in sorted(self.classes):
            class_dir = join(directory, class_name)
            if not isdir(class_dir):
                continue
            for root_dir, _, file_names in os.walk(class_dir):
                for file_name in sorted(file_names):
                    if _is_image_file(file_name, self.extensions):
                        path = join(root_dir, file_name)
                        item = (path, self.class_to_idx[class_name])  # 一个item是 (地址 , 类别id)的元组
                        instances.append(item)
        return instances

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """
        根据id，返回真实的数据
        :param index:
        :return:
        """
        path, class_idx = self.samples[index]
        img_rgb = self.loader(path)
        if self.transform is not None:
            img_rgb = self.transform(img_rgb)
        if self.target_transform is not None:
            class_idx = self.target_transform(class_idx)
        return img_rgb, class_idx

if __name__ == "__main__":
    myClassDirectoryDataset = ClassDirectoryDataset("../data/hymenoptera_data/train", ["jpg"])
    img0, label = myClassDirectoryDataset[5]
    # img0.show()
    print(img0.shape)
    print(myClassDirectoryDataset.classes[label])