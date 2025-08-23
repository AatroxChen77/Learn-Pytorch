from time import time
import multiprocessing as mp
import torch
import torchvision
from torchvision import transforms
from dataset import ClassDirectoryDataset
from torch.utils.data import DataLoader,random_split
 
transform = transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,))
])

dataset = ClassDirectoryDataset("data/hymenoptera_data/train", ["jpg"])
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size],
                                  generator=torch.Generator().manual_seed(42)
)

if __name__ == "__main__":
 
    print(f"num of CPU: {mp.cpu_count()}")
    for num_workers in range(0, mp.cpu_count(), 1):  
        train_loader = DataLoader(train_set,
                            batch_size=16,
                            num_workers=num_workers,
                            shuffle=True,
                            pin_memory=True  # Enable this option to accelerate data transfer if using GPU
    )
        start = time()
        for epoch in range(1, 3):
            for i, data in enumerate(train_loader, 0):
                pass
        end = time()
        print("Finish with:{} second, num_workers={}".format(end - start, num_workers))