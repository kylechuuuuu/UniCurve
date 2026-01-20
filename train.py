import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from fusion_model import FusionModel
from dataset import VesselDataset
import os
import json
from tqdm import tqdm


def calculate_dice(pred, target, smooth=1e-5):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.item()


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum()
        union = probs.sum() + targets.sum()
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1. - dice


def compute_pruning_masks(model, loader, device, k=3, num_images=30):
    print(f"Computing similarity-based pruning (Sequential GPU mode) for {num_images} images...")
    model.eval()
    
    test_images = []
    test_masks = []
    collected = 0
    for imgs, msks, _, _ in loader:
        needed = min(imgs.size(0), num_images - collected)
        test_images.append(imgs[:needed].to(device))
        test_masks.append(msks[:needed].to(device))
        collected += needed
        if collected >= num_images:
            break

    mask_cache = {}

    def calculate_sim(feat, masks, batch_idx):
        # Handle different output shapes (B, C, H, W) or (B, L, D)
        if len(feat.shape) == 3:
            B, L, D = feat.shape
            possible_grids = [14, 16, 32, 40, 64, 70, 72]
            found = False
            for s in possible_grids:
                if s * s == L:
                    feat = feat.transpose(1, 2).view(B, D, s, s)
                    found = True; break
                elif s * s == L - 1: # 1 CLS
                    feat = feat[:, 1:, :].transpose(1, 2).view(B, D, s, s)
                    found = True; break
                elif s * s == L - 4: # 4 reg
                    feat = feat[:, 4:, :].transpose(1, 2).view(B, D, s, s)
                    found = True; break
                elif s * s == L - 5: # 1 CLS + 4 reg
                    feat = feat[:, 5:, :].transpose(1, 2).view(B, D, s, s)
                    found = True; break
            if not found: return 0.0

        B, C, H, W = feat.shape
        act_map = torch.abs(feat).mean(dim=1, keepdim=True)
        cache_key = (H, W, batch_idx)
        if cache_key not in mask_cache:
            m = torch.nn.functional.interpolate(masks.float(), size=(H, W), mode="bilinear", align_corners=False)
            m_flat = m.view(B, -1)
            m_min = m_flat.min(1, keepdim=True)[0].view(B, 1, 1, 1)
            m_max = m_flat.max(1, keepdim=True)[0].view(B, 1, 1, 1)
            mask_cache[cache_key] = (m - m_min) / (m_max - m_min + 1e-8)
        target_masks = mask_cache[cache_key]
        
        act_flat = act_map.view(B, -1)
        act_min = act_flat.min(1, keepdim=True)[0].view(B, 1, 1, 1)
        act_max = act_flat.max(1, keepdim=True)[0].view(B, 1, 1, 1)
        act_map = (act_map - act_min) / (act_max - act_min + 1e-8)
        
        sim = torch.nn.functional.cosine_similarity(act_map.view(B, -1), target_masks.view(B, -1), dim=1)
        return sim.mean().item()

    def get_capture_hook(storage):
        def hook(m, i, o):
            if isinstance(o, (list, tuple)): storage.append(o[0].detach())
            else: storage.append(o.detach())
        return hook

    def process_encoder(encoder_name):
        print(f"Processing {encoder_name}...")
        scores = None
        if encoder_name == "sam":
            enc = model.sam_encoder
        elif encoder_name == "sam2":
            enc = model.sam2_encoder
        elif encoder_name == "sam3":
            enc = model.sam3_encoder.backbone.vision_backbone
        else:
            enc = model.dinov3_encoder
        
        enc.to(device)
        mask_cache.clear()
        
        autocast_ctx = torch.amp.autocast('cuda') if device.type == 'cuda' else torch.no_grad()
        
        for b_idx, (imgs, msks) in enumerate(zip(test_images, test_masks)):
            res = []
            hooks = []
            blocks = enc.blocks if hasattr(enc, 'blocks') else enc.trunk.blocks
            for b in blocks:
                hooks.append(b.register_forward_hook(get_capture_hook(res)))
            
            with torch.no_grad(), autocast_ctx:
                if encoder_name == "sam3":
                    s3_img = torch.nn.functional.interpolate(imgs, size=(1008, 1008), mode="bilinear")
                    _ = enc(s3_img)
                else:
                    _ = enc(imgs)
            
            for h in hooks: h.remove()
            
            batch_sims = torch.tensor([calculate_sim(f, msks, b_idx) for f in res])
            scores = batch_sims if scores is None else scores + batch_sims
            del res
        
        enc.to("cpu")
        torch.cuda.empty_cache()
        return scores

    sam_scores = process_encoder("sam")
    sam2_scores = process_encoder("sam2")
    sam3_scores = process_encoder("sam3")
    dino_scores = process_encoder("dinov3")

    def find_best_window(avg_scores, k, restrict_range=None):
        L = len(avg_scores)
        start_search, end_search = 0, L
        if restrict_range:
            start_search, end_search = restrict_range
        
        best_val, best_start = -1.0, start_search
        for i in range(start_search, min(end_search, L - k + 1)):
            curr_val = avg_scores[i : i + k].sum()
            if curr_val > best_val:
                best_val = curr_val
                best_start = i
        return list(range(best_start, best_start + k))

    sam_idx = find_best_window(sam_scores, k)
    
    def get_hieradet_prune_indices(scores, k, stage_ends):
        must_keep = [0] + [e + 1 for e in stage_ends[:-1]]
        
        L = len(scores)
        best_val, best_start = -1.0, 0
        for i in range(L - k + 1):
            curr_val = scores[i : i + k].sum()
            if curr_val > best_val:
                best_val = curr_val
                best_start = i
        
        window = list(range(best_start, best_start + k))
        final_idx = sorted(list(set(must_keep + window)))
        return final_idx

    num_sam2_blocks = len(sam2_scores)
    sam2_trunk = model.sam2_encoder.trunk
    if num_sam2_blocks > 24:
        sam2_idx = get_hieradet_prune_indices(sam2_scores, k, sam2_trunk.stage_ends)
    else:
        sam2_idx = get_hieradet_prune_indices(sam2_scores, k, sam2_trunk.stage_ends)
        
    sam3_idx = find_best_window(sam3_scores, k)
    dino_idx = find_best_window(dino_scores, k)
    
    model.prune_to_blocks(sam_indices=sam_idx, sam2_indices=sam2_idx, sam3_indices=sam3_idx, dino_indices=dino_idx)

    pruning_config = {
        "sam_indices": sam_idx, "sam2_indices": sam2_idx, "sam3_indices": sam3_idx, "dinov3_indices": dino_idx, "window_size": k
    }
    with open("pruning_config.json", "w") as f:
        json.dump(pruning_config, f, indent=4)
    
    return sam_idx, sam2_idx, sam3_idx, dino_idx


def train():
    batch_size = 4
    num_epochs = 400
    learning_rate = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths
    sam_checkpoint = "sam/sam_pth/sam_vit_h_4b8939.pth"
    sam2_checkpoint = "sam2/sam2_pth/sam2.1_hiera_large.pt"
    sam2_config = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam3_checkpoint = "sam3/sam3_pth/sam3.safetensors"
    dinov3_checkpoint = "DINOv3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"  # DINOv3 Vit-L checkpoint
    data_root = "Datasets/DSCA"

    # Model
    print("Initializing model...")
    model = FusionModel(sam_checkpoint, sam2_checkpoint, sam2_config, sam3_checkpoint, dinov3_checkpoint, fusion_type="coord")
    
    # Data
    print("Loading data...")
    train_dataset = VesselDataset(data_root, split="train")
    val_dataset = VesselDataset(data_root, split="val")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    # Compute and set pruning blocks (Run on GPU for speed)
    sam_idx, sam2_idx, sam3_idx, dino_idx = compute_pruning_masks(model, train_loader, device, k=3, num_images=10)
    print(f"Hard pruning complete. Kept blocks: SAM={len(sam_idx)}, SAM2={len(sam2_idx)}, SAM3={len(sam3_idx)}, DINO={len(dino_idx)}")

    model = model.to(device)
    torch.cuda.empty_cache()

    # Optimizer & Loss
    # Train decoder and adapters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    optimizer = optim.AdamW(trainable_params, lr=learning_rate, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = torch.amp.GradScaler('cuda') # AMP
    bce_criterion = nn.BCEWithLogitsLoss()
    dice_criterion = DiceLoss()
    mse_criterion = nn.MSELoss()
    consistency_weight = 0.01

    # Training Loop
    best_dice = 0.0
    log_file = open("train_log.txt", "w")
    log_file.write("Epoch,Train_Loss,Val_Loss,Val_Dice\n")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for images, masks, _, _ in pbar:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'): # AMP
                outputs, bottlenecks = model(images)
                
                # Segmentation Loss (BCE + Dice)
                seg_loss = bce_criterion(outputs, masks) + dice_criterion(outputs, masks)
                
                # Consistency Loss
                s_b, s2_b, s3_b, d_b = bottlenecks
                cons_loss = (mse_criterion(s_b, s2_b) + mse_criterion(s2_b, s3_b) + 
                             mse_criterion(s3_b, d_b) + mse_criterion(s_b, d_b))
                
                loss = seg_loss + consistency_weight * cons_loss
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            pbar.set_postfix({"loss": loss.item(), "seg": seg_loss.item(), "cons": cons_loss.item()})

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        val_dice = 0
        with torch.no_grad():
            for images, masks, _, _ in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                
                loss = bce_criterion(outputs, masks) + dice_criterion(outputs, masks)
                val_loss += loss.item()

                dice = calculate_dice(outputs, masks)
                val_dice += dice

        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)

        log_str = f"{epoch + 1},{avg_train_loss:.4f},{avg_val_loss:.4f},{avg_val_dice:.4f}\n"
        log_file.write(log_str)
        log_file.flush()

        print(
            f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Dice: {avg_val_dice:.4f}"
        )

        if avg_val_dice > best_dice:
            best_dice = avg_val_dice
            torch.save(model.state_dict(), "best_fusion_model.pth")
            print(f"Saved best model with Dice: {best_dice:.4f}")

        scheduler.step()

    log_file.close()

if __name__ == "__main__":
    train()
