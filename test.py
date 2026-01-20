import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from fusion_model import FusionModel
from dataset import VesselDataset
import os
import json
from tqdm import tqdm
import torchvision.utils as vutils
import torchvision.transforms.functional as TF

def test():
    # Config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Paths
    sam_checkpoint = 'sam/sam_pth/sam_vit_h_4b8939.pth'
    sam2_checkpoint = 'sam2/sam2_pth/sam2.1_hiera_large.pt'
    sam2_config = 'configs/sam2.1/sam2.1_hiera_l.yaml'
    sam3_checkpoint = 'sam3/sam3_pth/sam3.safetensors'
    dinov3_checkpoint = 'DINOv3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'
    data_root = 'Datasets/XCAD'
    model_path = 'best_fusion_model.pth'
    pruning_config_path = 'pruning_config.json'
    output_dir = 'results/XCAD'
    os.makedirs(output_dir, exist_ok=True)

    print("Initializing model...")
    model = FusionModel(sam_checkpoint, sam2_checkpoint, sam2_config, sam3_checkpoint, dinov3_checkpoint, fusion_type='coord')
    
    if os.path.exists(pruning_config_path):
        print(f"Loading pruning configuration from {pruning_config_path}...")
        with open(pruning_config_path, "r") as f:
            cfg = json.load(f)
        sam_idx = cfg["sam_indices"]
        sam2_idx = cfg["sam2_indices"]
        sam3_idx = cfg["sam3_indices"]
        dino_idx = cfg["dinov3_indices"]
    else:
        print("Warning: pruning_config.json not found, using manual fallback indices.")
        # auto detect the index
        sam_idx = [9, 10, 11]
        sam2_idx = [0, 1, 2, 3, 4, 18, 19, 20, 21, 22, 23]
        sam3_idx = [29, 30, 31]
        dino_idx = [9, 10, 11]
    
    print(f"Applying pruning: SAM={sam_idx}, SAM2={sam2_idx}, SAM3={sam3_idx}, DINO={dino_idx}")
    model.prune_to_blocks(sam_indices=sam_idx, sam2_indices=sam2_idx, sam3_indices=sam3_idx, dino_indices=dino_idx)

    print(f"Loading weights from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    print("Loading data...")
    val_dataset = VesselDataset(data_root, split='val')
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
    
    print("Starting inference...")
    with torch.no_grad():
        for i, (image, mask, img_name, original_size) in enumerate(tqdm(val_loader)):
            image = image.to(device)

            with torch.amp.autocast('cuda'):
                output = model(image)
            
            pred = torch.sigmoid(output)
            pred = (pred > 0.5).float()

            orig_height, orig_width = original_size[0].item(), original_size[1].item()
            pred_resized = TF.resize(pred.cpu(), (orig_height, orig_width), interpolation=TF.InterpolationMode.NEAREST)

            save_name = img_name[0]
            vutils.save_image(pred_resized, os.path.join(output_dir, save_name))
            
    print(f"Results saved to {output_dir}")

if __name__ == '__main__':
    test()

