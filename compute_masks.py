
import torch
import torch.nn.functional as F
from fusion_model import FusionModel
from dataset import VesselDataset
from torch.utils.data import DataLoader
import os

def compute_masks():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Paths (copied from train.py)
    sam_checkpoint = "sam/sam_pth/sam_vit_h_4b8939.pth"
    sam2_checkpoint = "sam2/sam2_pth/sam2.1_hiera_large.pt"
    sam2_config = "configs/sam2.1/sam2.1_hiera_l.yaml"
    dinov3_checkpoint = "DINOv3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
    data_root = "Datasets/aaa"

    # Model
    model = FusionModel(sam_checkpoint, sam2_checkpoint, sam2_config, dinov3_checkpoint)
    model = model.to(device)
    model.eval()

    # Data
    dataset = VesselDataset(data_root, split="train")
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Accumulators for activation intensities
    sam_acts = []
    sam2_acts = []
    dinov3_acts = []

    window_size = 3
    
    print("Computing activations for first 3 images...")
    with torch.no_grad():
        for i, (images, _, _, _) in enumerate(loader):
            if i >= 3:
                break
            
            images = images.to(device)
            
            # Get features through encoders
            # We copy parts of forward pass
            sam_feat = model.sam_encoder(images)
            sam2_out = model.sam2_encoder(images)
            sam2_feat = sam2_out["vision_features"]
            
            dinov3_feats = model.dinov3_encoder.get_intermediate_layers(
                images, n=[11], reshape=True, return_class_token=False, norm=True
            )
            dinov3_feat = dinov3_feats[0]
            
            # Resize sam2 and dinov3 to match sam if needed (usually 64x64)
            if sam2_feat.shape[-2:] != sam_feat.shape[-2:]:
                sam2_feat = F.interpolate(sam2_feat, size=sam_feat.shape[-2:], mode="bilinear")
            if dinov3_feat.shape[-2:] != sam_feat.shape[-2:]:
                dinov3_feat = F.interpolate(dinov3_feat, size=sam_feat.shape[-2:], mode="bilinear")

            # Compute A(w) for each
            def compute_aw(feat):
                mag = torch.mean(torch.abs(feat), dim=1, keepdim=True)
                aw = F.avg_pool2d(mag, kernel_size=window_size, stride=1, padding=window_size // 2)
                return aw

            sam_acts.append(compute_aw(sam_feat))
            sam2_acts.append(compute_aw(sam2_feat))
            dinov3_acts.append(compute_aw(dinov3_feat))

    # Average activations
    avg_sam = torch.stack(sam_acts).mean(0)
    avg_sam2 = torch.stack(sam2_acts).mean(0)
    avg_dinov3 = torch.stack(dinov3_acts).mean(0)

    # Determine thresholds (tau)
    # Use a simple heuristic: threshold at 0.5 * max or mean
    # Or just keep top 80% of active area?
    # Let's use a threshold that keeps significantly active regions.
    def get_mask(aw):
        # Normalize to 0-1 for easier thresholding
        aw_min = aw.min()
        aw_max = aw.max()
        aw_norm = (aw - aw_min) / (aw_max - aw_min + 1e-8)
        # Threshold at 0.1 (task specific)
        return (aw_norm > 0.1).float()

    sam_mask = get_mask(avg_sam)
    sam2_mask = get_mask(avg_sam2)
    dinov3_mask = get_mask(avg_dinov3)

    print(f"SAM mask active ratio: {sam_mask.mean().item():.2f}")
    print(f"SAM2 mask active ratio: {sam2_mask.mean().item():.2f}")
    print(f"DINOv3 mask active ratio: {dinov3_mask.mean().item():.2f}")

    # Save masks
    torch.save({
        'sam_mask': sam_mask.cpu(),
        'sam2_mask': sam2_mask.cpu(),
        'dinov3_mask': dinov3_mask.cpu()
    }, 'pruning_masks.pth')
    print("Masks saved to pruning_masks.pth")

if __name__ == "__main__":
    compute_masks()
