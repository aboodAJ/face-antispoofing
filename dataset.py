import os
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

class AntiSpoofDataset(Dataset):
    def __init__(self, lcc_fasd_root, siw_root, split='train', transform=None):
        """
        Args:
            lcc_fasd_root (str): Root directory for LCC_FASD dataset.
            siw_root (str): Root directory for SiW dataset.
            split (str): 'train', 'val', or 'test'.
            transform (albumentations.Compose, optional): Albumentations transformations.
        """
        self.split = split
        self.transform = transform
        self.samples = []
        
        # 0 for Real, 1 for Spoof
        
        # Load LCC_FASD
        self._load_lcc_fasd(lcc_fasd_root)
        
        # Load SiW
        self._load_siw(siw_root)
        
        if not self.samples:
            print(f"Warning: No samples found for split '{split}'")

    def _load_lcc_fasd(self, root):
        if not os.path.exists(root):
            print(f"LCC_FASD root not found: {root}")
            return
            
        # Mapping for LCC_FASD splits
        folder_map = {
            'train': 'LCC_FASD_training',
            'val': 'LCC_FASD_development',
            'test': 'LCC_FASD_evaluation'
        }
        
        if self.split not in folder_map:
            return

        split_folder = folder_map[self.split]
        full_path = os.path.join(root, split_folder)
        
        if not os.path.exists(full_path):
            return
            
        self._add_samples_from_dir(full_path)

    def _load_siw(self, root):
        if not os.path.exists(root):
            print(f"SiW root not found: {root}")
            return
            
        # Mapping for SiW splits (folder names match split names)
        # Assuming structure: SiW/train, SiW/val, SiW/test
        full_path = os.path.join(root, self.split)
        
        if not os.path.exists(full_path):
            return
            
        self._add_samples_from_dir(full_path)

    def _add_samples_from_dir(self, dir_path):
        # Expecting 'real' and 'spoof' subdirectories
        real_path = os.path.join(dir_path, 'real')
        spoof_path = os.path.join(dir_path, 'spoof')
        
        # Add Real (Label 0)
        if os.path.exists(real_path):
            for fname in os.listdir(real_path):
                fpath = os.path.join(real_path, fname)
                if os.path.isfile(fpath):
                    self.samples.append((fpath, 0))
                    
        # Add Spoof (Label 1)
        if os.path.exists(spoof_path):
            for fname in os.listdir(spoof_path):
                fpath = os.path.join(spoof_path, fname)
                if os.path.isfile(fpath):
                    self.samples.append((fpath, 1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Read image with OpenCV
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
            
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        else:
            # Default transform if none provided: Resize and Normalize
             default_transform = A.Compose([
                A.Resize(224, 224),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
             augmented = default_transform(image=image)
             image = augmented['image']
            
        return image, torch.tensor(label, dtype=torch.long)

def get_transforms(split='train'):
    if split == 'train':
        return A.Compose([
            A.Resize(224, 224),
            # Total chance of seeing the perfectly clean original: 0.5 x 0.8 x 0.5 = 0.2 (20%)
            A.HorizontalFlip(p=0.5), # 50% chance of flipping
            A.RandomBrightnessContrast(p=0.2), # 20% chance of changing brightness and contrast
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5), # 50% chance of shifting, scaling, and rotating
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
