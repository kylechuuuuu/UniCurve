import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
LOCAL_IMPORTS = (
    os.path.join(PROJECT_ROOT, "sam"),
    os.path.join(PROJECT_ROOT, "sam2"),
    os.path.join(PROJECT_ROOT, "DINOv3", "dinov3"),
)
for import_path in LOCAL_IMPORTS:
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from segment_anything.build_sam import build_sam_vit_h
from sam2.build_sam import build_sam2


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _sliding_window_starts(size: int, window: int, stride: int) -> List[int]:
    if window <= 0:
        raise ValueError("window must be > 0")
    if stride <= 0:
        raise ValueError("stride must be > 0")
    if size <= window:
        return [0]

    starts = list(range(0, size - window + 1, stride))
    last_start = size - window
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def _group_norm(num_channels: int, max_groups: int = 8) -> nn.GroupNorm:
    groups = min(max_groups, num_channels)
    while groups > 1 and num_channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


def _normalize(images: torch.Tensor, mean: Tuple[float, ...], std: Tuple[float, ...]) -> torch.Tensor:
    mean_t = images.new_tensor(mean).view(1, -1, 1, 1)
    std_t = images.new_tensor(std).view(1, -1, 1, 1)
    return (images - mean_t) / std_t


def _masked_average(features: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    masked = features * masks
    denom = masks.sum(dim=(-1, -2), keepdim=False).clamp_min(1e-6)
    return masked.sum(dim=(-1, -2)) / denom


def _bank_similarity(normalized_query: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    normalized_prototypes = F.normalize(prototypes, dim=-1)
    return torch.einsum("bchw,bpc->bphw", normalized_query, normalized_prototypes)


def _compute_dice_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return ((2 * intersection + eps) / (union + eps)).mean()


class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1.0) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice = (2 * intersection + self.eps) / (union + self.eps)
        return 1 - dice.mean()


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            _group_norm(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _group_norm(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = _group_norm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = _group_norm(out_channels)
        self.act = nn.GELU()
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + residual)


class SparseEpisodeRouter(nn.Module):
    def __init__(self, num_encoders: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(13, 32, kernel_size=3, stride=2, padding=1, bias=False),
            _group_norm(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            _group_norm(64),
            nn.GELU(),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1, bias=False),
            _group_norm(96),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(96, num_encoders)

    def forward(
        self,
        query_images: torch.Tensor,
        support_images: torch.Tensor,
        support_masks: torch.Tensor,
    ) -> torch.Tensor:
        support_masks = support_masks.float()
        fg_rgb = support_images * support_masks
        bg_masks = 1.0 - support_masks
        bg_rgb = support_images * bg_masks

        fg_norm = support_masks.sum(dim=(1, 3, 4), keepdim=True).clamp_min(1e-6)
        bg_norm = bg_masks.sum(dim=(1, 3, 4), keepdim=True).clamp_min(1e-6)
        fg_proto = (fg_rgb.sum(dim=(1, 3, 4), keepdim=True) / fg_norm).mean(dim=1)
        bg_proto = (bg_rgb.sum(dim=(1, 3, 4), keepdim=True) / bg_norm).mean(dim=1)
        fg_bg_contrast = fg_proto - bg_proto
        mean_mask = support_masks.mean(dim=1)
        if mean_mask.shape[-2:] != query_images.shape[-2:]:
            mean_mask = F.interpolate(mean_mask, size=query_images.shape[-2:], mode="nearest")
        route_input = torch.cat(
            [
                query_images,
                fg_proto.expand_as(query_images),
                bg_proto.expand_as(query_images),
                fg_bg_contrast.expand_as(query_images),
                mean_mask,
            ],
            dim=1,
        )
        features = self.stem(route_input).flatten(1)
        return self.fc(features)


class ProtoFusionDecoder(nn.Module):
    def __init__(self, coarse_channels: int) -> None:
        super().__init__()
        self.stem_2x = ResidualConvBlock(3, 32, stride=2)
        self.stem_4x = ResidualConvBlock(32, 64, stride=2)
        self.stem_8x = ResidualConvBlock(64, 128, stride=2)
        self.coarse_proj = nn.Sequential(
            nn.Conv2d(coarse_channels, 128, kernel_size=1, bias=False),
            _group_norm(128),
            nn.GELU(),
        )
        self.fuse_8x = ResidualConvBlock(256, 192)
        self.fuse_4x = ResidualConvBlock(192 + 64, 128)
        self.fuse_2x = ResidualConvBlock(128 + 32, 64)
        self.out_head = nn.Sequential(
            ResidualConvBlock(64, 32),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    def forward(self, query_image: torch.Tensor, coarse_feature: torch.Tensor) -> torch.Tensor:
        x2 = self.stem_2x(query_image)
        x4 = self.stem_4x(x2)
        x8 = self.stem_8x(x4)
        coarse = self.coarse_proj(coarse_feature)
        coarse_up = F.interpolate(coarse, size=x8.shape[-2:], mode="bilinear", align_corners=False)
        x = self.fuse_8x(torch.cat([coarse_up, x8], dim=1))
        x = F.interpolate(x, size=x4.shape[-2:], mode="bilinear", align_corners=False)
        x = self.fuse_4x(torch.cat([x, x4], dim=1))
        x = F.interpolate(x, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.fuse_2x(torch.cat([x, x2], dim=1))
        x = F.interpolate(x, size=query_image.shape[-2:], mode="bilinear", align_corners=False)
        return self.out_head(x)


class BaseEncoderAdapter(nn.Module):
    def __init__(self, name: str, image_size: int) -> None:
        super().__init__()
        self.name = name
        self.image_size = image_size
        self.module: nn.Module | None = None
        self.runtime_device = torch.device("cpu")

    def set_runtime_device(self, device: torch.device) -> None:
        self.runtime_device = device

    def freeze(self) -> None:
        if self.module is None:
            return
        self.module.eval()
        for parameter in self.module.parameters():
            parameter.requires_grad = False

    def _move_module(self, device: torch.device) -> None:
        assert self.module is not None
        self.module.to(device)

    def extract(self, images: torch.Tensor, offload_to_cpu: bool = True) -> torch.Tensor:
        device = self.runtime_device
        self._move_module(device)
        with torch.no_grad():
            features = self._forward_impl(images.to(device, non_blocking=True))
        if offload_to_cpu and device.type == "cuda":
            self._move_module(torch.device("cpu"))
            torch.cuda.empty_cache()
        return features

    def _forward_impl(self, images: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class SAMEncoderAdapter(BaseEncoderAdapter):
    def __init__(self, checkpoint_path: str) -> None:
        super().__init__(name="sam_vit_h", image_size=1024)
        self.module = build_sam_vit_h(checkpoint=checkpoint_path)
        self.freeze()

    def _forward_impl(self, images: torch.Tensor) -> torch.Tensor:
        resized = F.interpolate(images, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        assert self.module is not None
        preprocessed = self.module.preprocess(resized)
        return self.module.image_encoder(preprocessed)


class SAM2EncoderAdapter(BaseEncoderAdapter):
    def __init__(self, checkpoint_path: str, config_file: str) -> None:
        super().__init__(name="sam2_hiera_l", image_size=1024)
        self.module = build_sam2(config_file=config_file, ckpt_path=checkpoint_path, device="cpu", mode="eval")
        self.freeze()

    def _forward_impl(self, images: torch.Tensor) -> torch.Tensor:
        resized = F.interpolate(images, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        normalized = _normalize(resized, IMAGENET_MEAN, IMAGENET_STD)
        assert self.module is not None
        backbone_out = self.module.forward_image(normalized)
        _, vision_feats, _, feat_sizes = self.module._prepare_backbone_features(backbone_out)
        top_feat = vision_feats[-1].permute(1, 2, 0).reshape(images.shape[0], -1, *feat_sizes[-1])
        return top_feat


class DINOv3EncoderAdapter(BaseEncoderAdapter):
    def __init__(self, name: str, hub_name: str, weights_path: str, image_size: int) -> None:
        super().__init__(name=name, image_size=image_size)
        repo_dir = os.path.join(PROJECT_ROOT, "DINOv3", "dinov3")
        self.module = torch.hub.load(repo_dir, hub_name, source="local", weights=weights_path)
        self.freeze()

    def _forward_impl(self, images: torch.Tensor) -> torch.Tensor:
        resized = F.interpolate(images, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        normalized = _normalize(resized, IMAGENET_MEAN, IMAGENET_STD)
        assert self.module is not None
        return self.module.get_intermediate_layers(normalized, n=1, reshape=True)[0]


@dataclass
class FewShotModelConfig:
    encoder_topk: int = 2
    feature_dim: int = 128
    common_stride_size: int = 64
    spatial_keep_ratio: float = 0.35
    dense_encoder_epochs: int = 4
    sparse_ramp_epochs: int = 8
    spatial_dense_epochs: int = 4
    spatial_ramp_epochs: int = 8
    offload_to_cpu: bool = False
    dino_input_size: int = 448
    sam_checkpoint: str = os.path.join(PROJECT_ROOT, "sam", "sam_pth", "sam_vit_h_4b8939.pth")
    sam2_checkpoint: str = os.path.join(PROJECT_ROOT, "sam2", "sam2_pth", "sam2.1_hiera_large.pt")
    sam2_config: str = "configs/sam2.1/sam2.1_hiera_l.yaml"
    dinov3_vitb_checkpoint: str = os.path.join(
        PROJECT_ROOT, "DINOv3", "dinov3", "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
    )
    dinov3_vitl_checkpoint: str = os.path.join(
        PROJECT_ROOT, "DINOv3", "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
    )


class SparseFewShotSegmentor(nn.Module):
    def __init__(self, config: FewShotModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or FewShotModelConfig()
        self.encoder_names = ["sam_vit_h", "sam2_hiera_l", "dinov3_vitb16", "dinov3_vitl16"]

        self.encoders = nn.ModuleDict(
            {
                "sam_vit_h": SAMEncoderAdapter(self.config.sam_checkpoint),
                "sam2_hiera_l": SAM2EncoderAdapter(self.config.sam2_checkpoint, self.config.sam2_config),
                "dinov3_vitb16": DINOv3EncoderAdapter(
                    "dinov3_vitb16", "dinov3_vitb16", self.config.dinov3_vitb_checkpoint, self.config.dino_input_size
                ),
                "dinov3_vitl16": DINOv3EncoderAdapter(
                    "dinov3_vitl16", "dinov3_vitl16", self.config.dinov3_vitl_checkpoint, self.config.dino_input_size
                ),
            }
        )

        self.router = SparseEpisodeRouter(num_encoders=len(self.encoder_names))
        self.projections = nn.ModuleDict(
            {
                "sam_vit_h": nn.Sequential(nn.Conv2d(256, self.config.feature_dim, 1), nn.GELU()),
                "sam2_hiera_l": nn.Sequential(nn.Conv2d(256, self.config.feature_dim, 1), nn.GELU()),
                "dinov3_vitb16": nn.Sequential(nn.Conv2d(768, self.config.feature_dim, 1), nn.GELU()),
                "dinov3_vitl16": nn.Sequential(nn.Conv2d(1024, self.config.feature_dim, 1), nn.GELU()),
            }
        )
        shot_selector_in_channels = self.config.feature_dim * 3 + 2
        self.shot_selector = nn.Sequential(
            nn.Linear(shot_selector_in_channels, self.config.feature_dim),
            nn.GELU(),
            nn.Linear(self.config.feature_dim, 1),
        )
        evidence_in_channels = self.config.feature_dim + 4
        self.evidence_heads = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Conv2d(evidence_in_channels, self.config.feature_dim, kernel_size=3, padding=1, bias=False),
                    _group_norm(self.config.feature_dim),
                    nn.GELU(),
                    nn.Conv2d(self.config.feature_dim, self.config.feature_dim, kernel_size=3, padding=1, bias=False),
                    _group_norm(self.config.feature_dim),
                    nn.GELU(),
                )
                for name in self.encoder_names
            }
        )
        self.decoder = ProtoFusionDecoder(coarse_channels=self.config.feature_dim)
        self.loss_bce = nn.BCEWithLogitsLoss()
        self.loss_dice = DiceLoss()
        self.runtime_device = torch.device("cpu")
        self.training_epoch = 0
        self.total_epochs = 1

    def train(self, mode: bool = True):
        super().train(mode)
        for encoder in self.encoders.values():
            encoder.eval()
        return self

    def set_runtime_device(self, device: torch.device) -> None:
        self.runtime_device = device
        self.router.to(device)
        self.projections.to(device)
        self.shot_selector.to(device)
        self.evidence_heads.to(device)
        self.decoder.to(device)
        self.loss_bce.to(device)
        self.loss_dice.to(device)

        encoder_device = torch.device("cpu") if self.config.offload_to_cpu else device
        for encoder in self.encoders.values():
            encoder.set_runtime_device(device)
            encoder.eval()
            encoder.to(encoder_device)

    def set_training_progress(self, epoch_index: int, total_epochs: int) -> None:
        self.training_epoch = max(0, epoch_index)
        self.total_epochs = max(1, total_epochs)

    def _current_encoder_topk(self) -> int:
        full_count = len(self.encoder_names)
        target_topk = min(self.config.encoder_topk, full_count)
        if self.training_epoch < self.config.dense_encoder_epochs:
            return full_count
        if self.config.sparse_ramp_epochs <= 0:
            return target_topk
        progress = min(
            1.0,
            (self.training_epoch - self.config.dense_encoder_epochs + 1) / self.config.sparse_ramp_epochs,
        )
        current = round(full_count - (full_count - target_topk) * progress)
        return max(target_topk, min(full_count, current))

    def _current_keep_ratio(self) -> float:
        target_ratio = min(1.0, max(0.05, self.config.spatial_keep_ratio))
        if self.training_epoch < self.config.spatial_dense_epochs:
            return 1.0
        if self.config.spatial_ramp_epochs <= 0:
            return target_ratio
        progress = min(
            1.0,
            (self.training_epoch - self.config.spatial_dense_epochs + 1) / self.config.spatial_ramp_epochs,
        )
        return 1.0 - (1.0 - target_ratio) * progress

    def _compute_sparse_gates(
        self,
        query_images: torch.Tensor,
        support_images: torch.Tensor,
        support_masks: torch.Tensor,
    ):
        router_logits = self.router(query_images, support_images, support_masks)
        topk = self._current_encoder_topk()
        topk_indices = torch.topk(router_logits, k=topk, dim=1).indices
        active_mask = torch.zeros_like(router_logits)
        active_mask.scatter_(1, topk_indices, 1.0)
        gates = F.softmax(router_logits, dim=1) * active_mask
        gates = gates / gates.sum(dim=1, keepdim=True).clamp_min(1e-6)
        encoder_assignments: List[torch.Tensor] = []
        for encoder_idx in range(active_mask.shape[1]):
            encoder_assignments.append(torch.nonzero(active_mask[:, encoder_idx] > 0, as_tuple=False).flatten())
        return gates, topk_indices, encoder_assignments

    def _support_conditioned_sparse_map(
        self,
        fg_similarity: torch.Tensor,
        bg_similarity: torch.Tensor,
    ) -> torch.Tensor:
        current_keep_ratio = self._current_keep_ratio()
        if current_keep_ratio >= 1.0:
            return torch.ones_like(fg_similarity)

        score = (fg_similarity - bg_similarity).abs().flatten(1)
        keep_tokens = max(1, int(score.shape[1] * current_keep_ratio))
        topk_values = torch.topk(score, k=keep_tokens, dim=1).values[:, -1:]
        return (score >= topk_values).float().view_as(fg_similarity)

    def _compute_shot_weights(
        self,
        projected_query: torch.Tensor,
        fg_proto_per_shot: torch.Tensor,
        bg_proto_per_shot: torch.Tensor,
        resized_masks: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, shots = resized_masks.shape[:2]
        query_context = projected_query.mean(dim=(-1, -2)).unsqueeze(1).expand(-1, shots, -1)
        fg_coverage = resized_masks.mean(dim=(-1, -2, -3), keepdim=False).unsqueeze(-1)
        bg_coverage = (1.0 - resized_masks).mean(dim=(-1, -2, -3), keepdim=False).unsqueeze(-1)
        shot_features = torch.cat(
            [fg_proto_per_shot, bg_proto_per_shot, query_context, fg_coverage, bg_coverage],
            dim=-1,
        )
        shot_logits = self.shot_selector(shot_features).squeeze(-1)
        valid_shots = resized_masks.sum(dim=(2, 3, 4)) > 0
        shot_logits = shot_logits.masked_fill(~valid_shots, -1e4)
        shot_weights = F.softmax(shot_logits, dim=1)
        shot_weights = torch.where(valid_shots, shot_weights, torch.zeros_like(shot_weights))
        return shot_weights / shot_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)

    def _aggregate_similarity_bank(
        self,
        normalized_query: torch.Tensor,
        prototypes: torch.Tensor,
        prototype_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        similarity_bank = _bank_similarity(normalized_query, prototypes)
        weighted_similarity = (similarity_bank * prototype_weights.unsqueeze(-1).unsqueeze(-1)).sum(
            dim=1, keepdim=True
        )
        max_similarity = similarity_bank.max(dim=1, keepdim=True).values
        dispersion = similarity_bank.std(dim=1, keepdim=True, unbiased=False)
        return 0.5 * (weighted_similarity + max_similarity), dispersion

    def _prototype_evidence(
        self,
        encoder_name: str,
        query_feature: torch.Tensor,
        support_feature: torch.Tensor,
        support_masks: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, shots = support_masks.shape[:2]
        projected_query = self.projections[encoder_name](query_feature)
        projected_support = self.projections[encoder_name](support_feature)

        projected_support = projected_support.view(batch_size, shots, *projected_support.shape[1:])
        resized_masks = F.interpolate(
            support_masks.flatten(0, 1), size=projected_support.shape[-2:], mode="nearest"
        ).view(batch_size, shots, 1, *projected_support.shape[-2:])

        fg_proto_per_shot = _masked_average(projected_support, resized_masks)
        bg_proto_per_shot = _masked_average(projected_support, 1.0 - resized_masks)
        shot_weights = self._compute_shot_weights(projected_query, fg_proto_per_shot, bg_proto_per_shot, resized_masks)
        fg_proto = (fg_proto_per_shot * shot_weights.unsqueeze(-1)).sum(dim=1)
        bg_proto = (bg_proto_per_shot * shot_weights.unsqueeze(-1)).sum(dim=1)

        normalized_query = F.normalize(projected_query, dim=1)
        fg_bank = torch.cat([fg_proto.unsqueeze(1), fg_proto_per_shot], dim=1)
        bg_bank = torch.cat([bg_proto.unsqueeze(1), bg_proto_per_shot], dim=1)
        prototype_weights = torch.cat([torch.ones_like(shot_weights[:, :1]), shot_weights], dim=1)
        prototype_weights = prototype_weights / prototype_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        fg_similarity, _ = self._aggregate_similarity_bank(normalized_query, fg_bank, prototype_weights)
        bg_similarity, bg_dispersion = self._aggregate_similarity_bank(normalized_query, bg_bank, prototype_weights)
        sparse_mask = self._support_conditioned_sparse_map(fg_similarity, bg_similarity)
        sparse_query = projected_query * sparse_mask
        evidence = torch.cat(
            [sparse_query, fg_similarity, bg_similarity, fg_similarity - bg_similarity, bg_dispersion],
            dim=1,
        )
        return self.evidence_heads[encoder_name](evidence)

    def forward(self, query_images: torch.Tensor, support_images: torch.Tensor, support_masks: torch.Tensor) -> dict:
        batch_size = support_images.shape[0]
        gates, topk_indices, encoder_assignments = self._compute_sparse_gates(query_images, support_images, support_masks)

        query_device = query_images.device
        fused_feature = None
        support_masks = support_masks.to(query_device)

        for encoder_idx, sample_indices in enumerate(encoder_assignments):
            if sample_indices.numel() == 0:
                continue
            encoder_name = self.encoder_names[encoder_idx]
            encoder = self.encoders[encoder_name]
            query_subset = query_images.index_select(0, sample_indices)
            support_subset = support_images.index_select(0, sample_indices)
            support_mask_subset = support_masks.index_select(0, sample_indices)
            support_flat = support_subset.flatten(0, 1)

            query_feature = encoder.extract(query_subset, offload_to_cpu=self.config.offload_to_cpu).to(query_device)
            support_feature = encoder.extract(support_flat, offload_to_cpu=self.config.offload_to_cpu).to(query_device)
            evidence = self._prototype_evidence(encoder_name, query_feature, support_feature, support_mask_subset)
            evidence = F.interpolate(
                evidence,
                size=(self.config.common_stride_size, self.config.common_stride_size),
                mode="bilinear",
                align_corners=False,
            )
            if fused_feature is None:
                fused_feature = torch.zeros(
                    batch_size,
                    evidence.shape[1],
                    evidence.shape[2],
                    evidence.shape[3],
                    device=query_device,
                    dtype=evidence.dtype,
                )
            encoder_gate = gates.index_select(0, sample_indices)[:, encoder_idx].view(-1, 1, 1, 1)
            fused_feature.index_add_(0, sample_indices, evidence * encoder_gate)

        if fused_feature is None:
            raise RuntimeError("No active encoder was selected by the sparse router.")

        logits = self.decoder(query_images, fused_feature)
        return {"logits": logits, "gates": gates, "topk_indices": topk_indices}

    def predict_logits(
        self,
        query_images: torch.Tensor,
        support_images: torch.Tensor,
        support_masks: torch.Tensor,
        tile_size: int | None = None,
        tile_overlap: float = 0.25,
    ) -> torch.Tensor:
        if tile_size is None:
            return self.forward(query_images, support_images, support_masks)["logits"]
        if query_images.shape[0] != 1:
            raise ValueError("Sliding-window prediction requires batch_size=1.")
        if not 0.0 <= tile_overlap < 1.0:
            raise ValueError("tile_overlap must be in [0.0, 1.0).")

        stride = max(1, int(round(tile_size * (1.0 - tile_overlap))))
        height, width = query_images.shape[-2:]
        padded_height = max(height, tile_size)
        padded_width = max(width, tile_size)
        query_padded = F.pad(query_images, (0, padded_width - width, 0, padded_height - height))

        accumulated_logits = query_images.new_zeros((1, 1, padded_height, padded_width))
        accumulated_weights = query_images.new_zeros((1, 1, padded_height, padded_width))
        blend_weights = query_images.new_ones((1, 1, tile_size, tile_size))

        top_starts = _sliding_window_starts(padded_height, tile_size, stride)
        left_starts = _sliding_window_starts(padded_width, tile_size, stride)
        for top in top_starts:
            for left in left_starts:
                query_patch = query_padded[:, :, top : top + tile_size, left : left + tile_size]
                patch_logits = self.forward(query_patch, support_images, support_masks)["logits"]
                accumulated_logits[:, :, top : top + tile_size, left : left + tile_size] += patch_logits * blend_weights
                accumulated_weights[:, :, top : top + tile_size, left : left + tile_size] += blend_weights

        merged_logits = accumulated_logits / accumulated_weights.clamp_min(1e-6)
        return merged_logits[:, :, :height, :width]

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        bce = self.loss_bce(logits, targets)
        dice = self.loss_dice(logits, targets)
        loss = bce + dice
        return {"loss": loss, "bce": bce, "dice": dice}

    @torch.no_grad()
    def compute_metrics(self, logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        dice = _compute_dice_from_logits(logits, targets).item()
        probs = (torch.sigmoid(logits) > 0.5).float()
        intersection = (probs * targets).sum(dim=(1, 2, 3))
        union = (probs + targets - probs * targets).sum(dim=(1, 2, 3)).clamp_min(1e-6)
        iou = (intersection / union).mean().item()
        return {"dice": dice, "iou": iou}


def build_model_from_config(config_dict: dict | None = None) -> SparseFewShotSegmentor:
    config = FewShotModelConfig(**config_dict) if config_dict is not None else FewShotModelConfig()
    return SparseFewShotSegmentor(config=config)


def load_compatible_state_dict(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> Dict[str, List[str] | int]:
    model_state = model.state_dict()
    compatible_state = {
        key: value
        for key, value in state_dict.items()
        if key in model_state and model_state[key].shape == value.shape
    }
    skipped_keys = sorted(key for key in state_dict.keys() if key not in compatible_state)
    missing_keys = sorted(key for key in model_state.keys() if key not in compatible_state)
    model.load_state_dict(compatible_state, strict=False)
    return {
        "loaded_count": len(compatible_state),
        "skipped_keys": skipped_keys,
        "missing_keys": missing_keys,
    }


def load_model_state_dict(
    model: nn.Module, state_dict: Dict[str, torch.Tensor], allow_partial: bool = False
) -> Dict[str, List[str] | int | bool]:
    if not allow_partial:
        model.load_state_dict(state_dict, strict=True)
        return {
            "loaded_count": len(state_dict),
            "skipped_keys": [],
            "missing_keys": [],
            "partial": False,
        }
    compatible_result = load_compatible_state_dict(model, state_dict)
    compatible_result["partial"] = True
    return compatible_result
