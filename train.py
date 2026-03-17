import argparse
import json
import os
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from fewshot_dataset import FewShotSegDataset
from fewshot_model import FewShotModelConfig, SparseFewShotSegmentor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the sparse few-shot segmentation model.")
    parser.add_argument("--data-root", default="Datasets/DRIVE")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--support-split", default="train")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--shots", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output-dir", default="runs/fewshot_sparse_fusion")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--encoder-topk", type=int, default=2)
    parser.add_argument("--spatial-keep-ratio", type=float, default=0.35)
    parser.add_argument("--dense-encoder-epochs", type=int, default=4)
    parser.add_argument("--sparse-ramp-epochs", type=int, default=8)
    parser.add_argument("--spatial-dense-epochs", type=int, default=4)
    parser.add_argument("--spatial-ramp-epochs", type=int, default=8)
    parser.add_argument("--dino-input-size", type=int, default=448)
    parser.add_argument("--val-interval", type=int, default=5, help="Run validation every N epochs and on the final epoch.")
    parser.add_argument(
        "--train-episodes-per-epoch",
        type=int,
        default=0,
        help="Override the number of training episodes per epoch. 0 keeps the dataset length.",
    )
    offload_group = parser.add_mutually_exclusive_group()
    offload_group.add_argument(
        "--offload-to-cpu",
        dest="offload_to_cpu",
        action="store_true",
        help="Offload frozen encoders back to CPU after each feature extraction to reduce GPU memory usage.",
    )
    offload_group.add_argument(
        "--no-offload-to-cpu",
        dest="offload_to_cpu",
        action="store_false",
        help="Keep frozen encoders on the runtime device for faster training/inference when memory allows.",
    )
    parser.set_defaults(offload_to_cpu=False)
    parser.add_argument("--max-train-steps", type=int, default=0, help="0 means full epoch.")
    parser.add_argument("--max-val-steps", type=int, default=0, help="0 means full validation.")
    parser.add_argument(
        "--full-res-val",
        action="store_true",
        help="Run validation on full-resolution queries with sliding-window inference instead of fixed-size crops.",
    )
    parser.add_argument(
        "--eval-tile-overlap",
        type=float,
        default=0.25,
        help="Overlap ratio for sliding-window validation when --full-res-val is enabled.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    return {
        "query_image": batch["query_image"].to(device, non_blocking=True),
        "query_mask": batch["query_mask"].to(device, non_blocking=True),
        "support_images": batch["support_images"].to(device, non_blocking=True),
        "support_masks": batch["support_masks"].to(device, non_blocking=True),
        "query_name": batch["query_name"],
        "original_size": batch["original_size"],
    }


def run_epoch(
    model: SparseFewShotSegmentor,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler | None,
    max_steps: int,
    eval_tile_size: int | None = None,
    eval_tile_overlap: float = 0.25,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    running = {"loss": 0.0, "bce": 0.0, "dice_loss": 0.0, "dice": 0.0, "iou": 0.0}
    steps = 0

    amp_enabled = device.type == "cuda"
    iterator = tqdm(loader, desc="train" if is_train else "val")
    for batch in iterator:
        batch = move_batch_to_device(batch, device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                if is_train or eval_tile_size is None:
                    logits = model(batch["query_image"], batch["support_images"], batch["support_masks"])["logits"]
                else:
                    logits = model.predict_logits(
                        batch["query_image"],
                        batch["support_images"],
                        batch["support_masks"],
                        tile_size=eval_tile_size,
                        tile_overlap=eval_tile_overlap,
                    )
                losses = model.compute_loss(logits, batch["query_mask"])

        if is_train:
            assert optimizer is not None
            if scaler is not None and amp_enabled:
                scaler.scale(losses["loss"]).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                losses["loss"].backward()
                optimizer.step()

        metrics = model.compute_metrics(logits.detach(), batch["query_mask"])
        running["loss"] += losses["loss"].item()
        running["bce"] += losses["bce"].item()
        running["dice_loss"] += losses["dice"].item()
        running["dice"] += metrics["dice"]
        running["iou"] += metrics["iou"]
        steps += 1

        iterator.set_postfix(loss=losses["loss"].item(), dice=metrics["dice"], iou=metrics["iou"])
        if max_steps > 0 and steps >= max_steps:
            break

    if steps == 0:
        raise RuntimeError("No batches were processed.")
    return {key: value / steps for key, value in running.items()}


def main() -> None:
    args = parse_args()
    if args.val_interval <= 0:
        raise ValueError("--val-interval must be >= 1.")
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device(args.device)
    train_dataset = FewShotSegDataset(
        root_dir=args.data_root,
        split=args.train_split,
        shots=args.shots,
        image_size=args.image_size,
        support_split=args.support_split,
        deterministic=False,
        seed=args.seed,
        episodes_per_epoch=args.train_episodes_per_epoch,
    )
    if args.train_episodes_per_epoch == 0 and len(train_dataset.query_records) == 1:
        print(
            "warning: detected a single training image with --train-episodes-per-epoch=0; "
            "each epoch will contain only one episode. For single-image generalization, set "
            "--train-episodes-per-epoch to a larger value such as 128 or 256."
        )
    val_dataset = FewShotSegDataset(
        root_dir=args.data_root,
        split=args.val_split,
        shots=args.shots,
        image_size=args.image_size,
        support_split=args.support_split,
        deterministic=True,
        seed=args.seed,
        keep_query_size=args.full_res_val,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    model_config = FewShotModelConfig(
        encoder_topk=args.encoder_topk,
        spatial_keep_ratio=args.spatial_keep_ratio,
        dense_encoder_epochs=args.dense_encoder_epochs,
        sparse_ramp_epochs=args.sparse_ramp_epochs,
        spatial_dense_epochs=args.spatial_dense_epochs,
        spatial_ramp_epochs=args.spatial_ramp_epochs,
        dino_input_size=args.dino_input_size,
        offload_to_cpu=args.offload_to_cpu,
    )
    model = SparseFewShotSegmentor(model_config)
    model.set_runtime_device(device)

    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_dice = -1.0
    last_val_metrics = None
    last_validated_epoch = 0
    metrics_path = os.path.join(args.output_dir, "metrics.jsonl")
    model_config_path = os.path.join(args.output_dir, "model_config.json")
    with open(model_config_path, "w", encoding="utf-8") as config_file:
        json.dump(model_config.__dict__, config_file, indent=2)

    for epoch in range(1, args.epochs + 1):
        model.set_training_progress(epoch - 1, args.epochs)
        train_metrics = run_epoch(model, train_loader, device, optimizer, scaler, args.max_train_steps)
        should_validate = epoch % args.val_interval == 0 or epoch == args.epochs
        val_metrics = None
        if should_validate:
            val_metrics = run_epoch(
                model,
                val_loader,
                device,
                None,
                None,
                args.max_val_steps,
                eval_tile_size=args.image_size if args.full_res_val else None,
                eval_tile_overlap=args.eval_tile_overlap,
            )
            last_val_metrics = val_metrics
            last_validated_epoch = epoch

        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        with open(metrics_path, "a", encoding="utf-8") as metrics_file:
            metrics_file.write(json.dumps(record, ensure_ascii=False) + "\n")

        if val_metrics is None:
            print(
                f"epoch={epoch} "
                f"train_loss={train_metrics['loss']:.4f} train_dice={train_metrics['dice']:.4f} "
                f"val=skipped next_val_epoch={min(args.epochs, ((epoch // args.val_interval) + 1) * args.val_interval)}"
            )
        else:
            print(
                f"epoch={epoch} "
                f"train_loss={train_metrics['loss']:.4f} train_dice={train_metrics['dice']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} val_dice={val_metrics['dice']:.4f} val_iou={val_metrics['iou']:.4f}"
            )

        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "model_config": model_config.__dict__,
            "args": vars(args),
            "val_metrics": last_val_metrics,
            "last_validated_epoch": last_validated_epoch,
        }
        torch.save(checkpoint, os.path.join(args.output_dir, "last.pth"))
        if val_metrics is not None and val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            torch.save(checkpoint, os.path.join(args.output_dir, "best.pth"))
            print(f"saved best checkpoint with dice={best_dice:.4f}")


if __name__ == "__main__":
    main()
