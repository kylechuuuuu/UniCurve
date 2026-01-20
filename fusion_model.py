import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "sam2"))
from sam2.build_sam import build_sam2

from sam.segment_anything import sam_model_registry

sys.path.append(os.path.join(os.path.dirname(__file__), "DINOv3/dinov3"))

sys.path.append(os.path.join(os.path.dirname(__file__), "sam3"))
from sam3.model_builder import build_sam3_image_model


class FusionModel(nn.Module):
    def __init__(
        self,
        sam_checkpoint,
        sam2_checkpoint,
        sam2_config,
        sam3_checkpoint,
        dinov3_checkpoint,
        fusion_type="coord",
    ):
        super().__init__()
        self.fusion_type = fusion_type

        # Load SAM
        self.sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
        self.sam_encoder = self.sam.image_encoder

        # Load SAM2
        # We load to cpu first to avoid OOM during init if multiple models are large
        self.sam2 = build_sam2(sam2_config, sam2_checkpoint, device="cpu")
        self.sam2_encoder = self.sam2.image_encoder

        # Load DINOv3
        # Import DINOv3 model
        import dinov3.hub.backbones as dinov3_backbones
        self.dinov3 = dinov3_backbones.dinov3_vitl16(pretrained=False)  # Initialize without pretrained weights first
        if dinov3_checkpoint:
            # Load the state dict from checkpoint
            dinov3_state_dict = torch.load(dinov3_checkpoint, map_location="cpu")
            self.dinov3.load_state_dict(dinov3_state_dict, strict=False)
        self.dinov3_encoder = self.dinov3

        # Load SAM3
        # Loading to CPU first to avoid OOM
        self.sam3 = build_sam3_image_model(checkpoint_path=None, device="cpu", eval_mode=True, load_from_HF=False)
        if sam3_checkpoint:
            if sam3_checkpoint.endswith(".safetensors"):
                from safetensors.torch import load_file
                sam3_sd = load_file(sam3_checkpoint, device="cpu")
            else:
                sam3_sd = torch.load(sam3_checkpoint, map_location="cpu")
            
            # If the checkpoint has a 'model' key, use it
            if "model" in sam3_sd:
                sam3_sd = sam3_sd["model"]
            
            # Handle potential prefixing from different saving methods
            # Based on user's inspection, some keys might be model.sam_model...
            # But the Sam3Image model expects its own state dict.
            # We'll try loading with strict=False first.
            self.sam3.load_state_dict(sam3_sd, strict=False)
        self.sam3_encoder = self.sam3

        # Freeze encoders
        for param in self.sam_encoder.parameters():
            param.requires_grad = False

        for param in self.sam2_encoder.parameters():
            param.requires_grad = False

        for param in self.sam3_encoder.parameters():
            param.requires_grad = False

        for param in self.dinov3_encoder.parameters():
            param.requires_grad = False

        # Dimensions
        self.sam_dim = 256  # SAM vit_h/l/b output channels are all 256 (via neck)
        self.sam2_dim = 256  # SAM2 hiera_l output channels 256
        self.sam3_dim = 256  # SAM3 neck output dimension
        self.dinov3_dim = 1024  # DINOv3 ViT-L output dimension (1024)

        # Shared Bottleneck Adapter components
        self.adapter_common_dim = 256
        self.adapter_bottleneck_dim = 64
        
        # Project to common dimension for shared bottleneck
        self.sam_to_adapter = nn.Identity()
        self.sam2_to_adapter = nn.Identity()
        self.sam3_to_adapter = nn.Identity()
        self.dinov3_to_adapter = nn.Conv2d(self.dinov3_dim, self.adapter_common_dim, 1)
        
        # Shared down-projection
        self.shared_down = nn.Conv2d(self.adapter_common_dim, self.adapter_bottleneck_dim, 1)
        
        # Individual up-projections to maintain residual connection
        self.sam_up = nn.Conv2d(self.adapter_bottleneck_dim, self.sam_dim, 1)
        self.sam2_up = nn.Conv2d(self.adapter_bottleneck_dim, self.sam2_dim, 1)
        self.sam3_up = nn.Conv2d(self.adapter_bottleneck_dim, self.sam3_dim, 1)
        self.dinov3_up = nn.Conv2d(self.adapter_bottleneck_dim, self.dinov3_dim, 1)

        if fusion_type == "cat":
            cat_input_dim = self.sam_dim + self.sam2_dim + self.sam3_dim + self.dinov3_dim  # Total concatenated dimension
            self.fusion_dim = max(self.sam_dim, self.sam2_dim, self.sam3_dim, self.dinov3_dim)  # Use max dimension for decoder input
            # Create projection for concatenation to reduce dimensionality
            self.cat_proj = nn.Conv2d(cat_input_dim, self.fusion_dim, 1)
        elif fusion_type == "coord":
            # Each feature map gets 2 extra channels (normalized x and y coordinates)
            cat_input_dim = (self.sam_dim + 2) + (self.sam2_dim + 2) + (self.sam3_dim + 2) + (self.dinov3_dim + 2)
            self.fusion_dim = max(self.sam_dim, self.sam2_dim, self.sam3_dim, self.dinov3_dim)
            self.cat_proj = nn.Conv2d(cat_input_dim, self.fusion_dim, 1)
        elif fusion_type == "add":
            # Project all models to the same dimension (choose the highest for information preservation)
            self.fusion_dim = max(self.sam_dim, self.sam2_dim, self.sam3_dim, self.dinov3_dim)
            self.sam_proj = nn.Conv2d(self.sam_dim, self.fusion_dim, 1)
            self.sam2_proj = nn.Conv2d(self.sam2_dim, self.fusion_dim, 1)
            self.sam3_proj = nn.Conv2d(self.sam3_dim, self.fusion_dim, 1)
            self.dinov3_proj = nn.Conv2d(self.dinov3_dim, self.fusion_dim, 1)  # Add projection for DINOv3
        else:
            raise ValueError("fusion_type must be 'cat', 'add' or 'coord'")

        # Decoder
        # We need to upsample from H/16 to H.
        # 4 upsampling blocks: 16x -> 8x -> 4x -> 2x -> 1x

        self.decoder = HighResDecoder(self.fusion_dim)

        # Block pruning window (number of blocks to keep)
        self.pruning_window = 3

    def prune_to_blocks(self, sam_indices=None, sam2_indices=None, sam3_indices=None, dino_indices=None):
        """
        Physically reduce the number of blocks in the encoders.
        For SAM2 (Hiera), we also need to update stage_ends.
        """
        if sam_indices is not None:
            self.sam_encoder.blocks = nn.ModuleList([self.sam_encoder.blocks[i] for i in sam_indices])
            print(f"SAM hard-pruned to blocks: {sam_indices}")
        
        if sam2_indices is not None:
            # SAM2 Hiera has stages and stage_ends. Pruning needs caution.
            # We assume sam2_indices is a sorted list of indices to KEEP.
            old_blocks = self.sam2_encoder.trunk.blocks
            old_ends = self.sam2_encoder.trunk.stage_ends
            
            # Physically prune the blocks
            self.sam2_encoder.trunk.blocks = nn.ModuleList([old_blocks[i] for i in sam2_indices])
            
            # Map original stage ends to the new indices.
            new_ends = []
            for oe in old_ends:
                # Count how many kept blocks are from original stages up to index 'oe'
                # These will be the blocks at the end of the new 'stage' in the pruned model.
                num_kept = sum(1 for idx in sam2_indices if idx <= oe)
                if num_kept > 0:
                    new_ends.append(num_kept - 1)
            
            # Ensure the trunk always returns exactly 4 scales if that's what the neck expects.
            # If some stages ended up having no blocks (though our prune logic prevents this),
            # this logic might need padding, but with must_keep it should be len(old_ends).
            self.sam2_encoder.trunk.stage_ends = new_ends
            # Force return_interm_layers to True so all stage outputs are collected
            self.sam2_encoder.trunk.return_interm_layers = True
            print(f"SAM2 hard-pruned to blocks: {sam2_indices}, new stage_ends: {new_ends}")

        if sam3_indices is not None:
            # SAM3 trunk is at sam3.backbone.vision_backbone.trunk
            trunk = self.sam3_encoder.backbone.vision_backbone.trunk
            trunk.blocks = nn.ModuleList([trunk.blocks[i] for i in sam3_indices])
            # Update full_attn_ids to match the new blocks
            trunk.full_attn_ids = [len(trunk.blocks) - 1]
            print(f"SAM3 hard-pruned to blocks: {sam3_indices}")
        
        if dino_indices is not None:
            self.dinov3_encoder.blocks = nn.ModuleList([self.dinov3_encoder.blocks[i] for i in dino_indices])
            print(f"DINOv3 hard-pruned to blocks: {dino_indices}")
        
        # Note: We keep original neck/head layers if any, 
        # but the heavy transformer blocks are reduced.

    def add_coords(self, x):
        # x: (B, C, H, W)
        batch_size, _, h, w = x.size()
        # Create normalized coordinates from -1 to 1
        y_coords = torch.linspace(-1, 1, h, device=x.device).view(1, 1, h, 1).expand(batch_size, 1, h, w)
        x_coords = torch.linspace(-1, 1, w, device=x.device).view(1, 1, 1, w).expand(batch_size, 1, h, w)
        return torch.cat([x, x_coords, y_coords], dim=1)

    def forward(self, x):
        # x: (B, 3, H, W)
        input_size = x.shape[-2:]

        # SAM Encoder
        # SAM expects 1024x1024 usually, but can handle other sizes if they are compatible with patch size.
        # We assume input is resized appropriately before passing here or we let the encoder handle it.
        # SAM encoder forward:
        sam_feat = self.sam_encoder(x)  # (B, 256, H/16, W/16)

        # SAM2 Encoder
        # SAM2 image encoder returns a dict
        sam2_out = self.sam2_encoder(x)
        sam2_feat = sam2_out["vision_features"]  # (B, 256, H/16, W/16)

        if sam2_feat.shape[-2:] != sam_feat.shape[-2:]:
            sam2_feat = F.interpolate(
                sam2_feat,
                size=sam_feat.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        # SAM3 Encoder
        # SAM3 neck returns a tuple (sam3_out, sam3_pos, sam2_out, sam2_pos)
        # Resize to 1008 for SAM3's 72x72 grid requirement (patch size 14)
        x_sam3 = F.interpolate(x, size=(1008, 1008), mode="bilinear", align_corners=False)
        sam3_outs, _, _, _ = self.sam3_encoder.backbone.vision_backbone(x_sam3)
        sam3_feat = sam3_outs[2] # Scale 1.0 (trunk resolution)
        
        if sam3_feat.shape[-2:] != sam_feat.shape[-2:]:
            sam3_feat = F.interpolate(
                sam3_feat,
                size=sam_feat.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        # DINOv3 Encoder
        # DINOv3 produces features in a different format, need to extract intermediate layers
        # Using get_intermediate_layers to get patch embeddings reshaped
        # Use the last block index (dynamically adjusted for pruning)
        last_blk_idx = len(self.dinov3_encoder.blocks) - 1
        dinov3_feats = self.dinov3_encoder.get_intermediate_layers(
            x, n=[last_blk_idx], reshape=True, return_class_token=False, norm=True
        )  # Get the last layer features, reshape to (B, C, H, W)
        dinov3_feat = dinov3_feats[0]  # (B, 768, H/16, W/16) - assuming patch size 16

        # Resize to match SAM features if needed
        if dinov3_feat.shape[-2:] != sam_feat.shape[-2:]:
            dinov3_feat = F.interpolate(
                dinov3_feat,
                size=sam_feat.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        # --- Shared Bottleneck Adapter ---
        # Project to common dimension
        s_c = self.sam_to_adapter(sam_feat)
        s2_c = self.sam2_to_adapter(sam2_feat)
        s3_c = self.sam3_to_adapter(sam3_feat)
        d_c = self.dinov3_to_adapter(dinov3_feat)

        # Shared down-projection to bottleneck
        s_bottleneck = self.shared_down(s_c)
        s2_bottleneck = self.shared_down(s2_c)
        s3_bottleneck = self.shared_down(s3_c)
        d_bottleneck = self.shared_down(d_c)

        # Individual up-projection and residual connection
        sam_feat = sam_feat + self.sam_up(F.relu(s_bottleneck))
        sam2_feat = sam2_feat + self.sam2_up(F.relu(s2_bottleneck))
        sam3_feat = sam3_feat + self.sam3_up(F.relu(s3_bottleneck))
        dinov3_feat = dinov3_feat + self.dinov3_up(F.relu(d_bottleneck))

        # Fusion
        if self.fusion_type == "cat":
            fused = torch.cat([sam_feat, sam2_feat, sam3_feat, dinov3_feat], dim=1)
            fused = self.cat_proj(fused)  # Project concatenated features to manageable dimension
        elif self.fusion_type == "coord":
            sam_feat = self.add_coords(sam_feat)
            sam2_feat = self.add_coords(sam2_feat)
            sam3_feat = self.add_coords(sam3_feat)
            dinov3_feat = self.add_coords(dinov3_feat)
            fused = torch.cat([sam_feat, sam2_feat, sam3_feat, dinov3_feat], dim=1)
            fused = self.cat_proj(fused)
        else:
            sam_proj = self.sam_proj(sam_feat)
            sam2_proj = self.sam2_proj(sam2_feat)
            sam3_proj = self.sam3_proj(sam3_feat)
            dinov3_proj = self.dinov3_proj(dinov3_feat)
            fused = sam_proj + sam2_proj + sam3_proj + dinov3_proj

        # Decode
        logits = self.decoder(fused)

        # Upsample to original input size if needed (though decoder should handle it)
        if logits.shape[-2:] != input_size:
            logits = F.interpolate(
                logits, size=input_size, mode="bilinear", align_corners=False
            )

        if self.training:
            return logits, (s_bottleneck, s2_bottleneck, s3_bottleneck, d_bottleneck)
        return logits


class HighResDecoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.up1 = PixelShuffleBlock(in_channels, 512)
        self.up2 = PixelShuffleBlock(512, 256)
        self.up3 = PixelShuffleBlock(256, 128)
        self.up4 = PixelShuffleBlock(128, 64)

        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.final_conv(x)
        return x


class PixelShuffleBlock(nn.Module):
    def __init__(self, in_ch, out_ch, upscale_factor=2):
        super().__init__()
        # To get out_ch after pixel shuffle, we need out_ch * upscale_factor^2 channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch * (upscale_factor**2), kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)
