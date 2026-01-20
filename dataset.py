import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class VesselDataset(Dataset):
    def __init__(self, root_dir, split='train', img_size=1024):
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        
        self.images_dir = os.path.join(root_dir, split, 'images')
        self.masks_dir = os.path.join(root_dir, split, 'masks')
        
        self.image_files = sorted([f for f in os.listdir(self.images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        # Transforms
        self.img_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.mask_transform = T.Compose([
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, img_name) # Assuming mask has same name
        
        # If mask extension is different, we might need to handle it. 
        # But usually it's same name or we need to search.
        # Let's assume same name for now. If not found, try replacing extension.
        if not os.path.exists(mask_path):
            base, _ = os.path.splitext(img_name)
            for ext in ['.png', '.jpg', '.jpeg']:
                temp_path = os.path.join(self.masks_dir, base + ext)
                if os.path.exists(temp_path):
                    mask_path = temp_path
                    break
        
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L') # Grayscale

        # Store original dimensions before transform
        original_size = image.size[::-1]  # height, width from PIL (width, height) -> (height, width)

        image = self.img_transform(image)
        mask = self.mask_transform(mask)

        # Binarize mask (0 or 1)
        mask = (mask > 0.5).float()

        return image, mask, img_name, original_size
