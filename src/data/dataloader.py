import os
from torch.utils.data import DataLoader, Dataset
import torch
from PIL import Image
from pathlib import Path


class UnpairedImageDataset(Dataset):
    """Simple unpaired image dataset."""
    def __init__(self, dir_A, dir_B, transform_A=None, transform_B=None, mode="train"):
        self.dir_A = dir_A
        self.dir_B = dir_B
        self.transform_A = transform_A
        self.transform_B = transform_B
        self.mode = mode
        
        # Get list of images
        self.img_A = list(Path(dir_A).glob("*.*")) if os.path.exists(dir_A) else []
        self.img_B = list(Path(dir_B).glob("*.*")) if os.path.exists(dir_B) else []
    
    def __len__(self):
        return max(len(self.img_A), len(self.img_B))
    
    def __getitem__(self, idx):
        img_A_path = self.img_A[idx % len(self.img_A)] if self.img_A else None
        img_B_path = self.img_B[idx % len(self.img_B)] if self.img_B else None
        
        # Load images
        if img_A_path and img_B_path:
            img_A = Image.open(img_A_path).convert('RGB')
            img_B = Image.open(img_B_path).convert('RGB')
            
            if self.transform_A:
                img_A = self.transform_A(img_A)
            if self.transform_B:
                img_B = self.transform_B(img_B)
                
            return {"A": img_A, "B": img_B}
        else:
            raise ValueError("Missing dataset directories")


def get_dataloader(config, train=True):
    """Get data loader for training or validation."""
    if config is None:
        raise ValueError("Config is None")
    
    # Get paths from config
    paths = config.get("paths", {})
    trainA_path = paths.get("trainA", "datasets/horse2zebra/trainA")
    trainB_path = paths.get("trainB", "datasets/horse2zebra/trainB")
    valA_path = paths.get("valA", "data/valA")
    valB_path = paths.get("valB", "data/valB")
    
    image_size = config.get("data", {}).get("image_size", 256)
    batch_size = config.get("training", {}).get("batch_size", 1)

    dir_A = trainA_path if train else valA_path
    dir_B = trainB_path if train else valB_path

    # Simple transform (resize to image_size)
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = UnpairedImageDataset(
        dir_A=dir_A,
        dir_B=dir_B,
        transform_A=transform,
        transform_B=transform,
        mode="train" if train else "val"
    )

    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=0)