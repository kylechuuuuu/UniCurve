import argparse
import os

import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from fewshot_dataset import FewShotSegDataset
from fewshot_model import build_model_from_config, load_model_state_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate or infer with the sparse few-shot segmentation model.")
    parser.add_argument("--data-root")
    parser.add_argument("--query-split")
    parser.add_argument("--support-split")
    parser.add_argument("--shots", type=int)
    parser.add_argument("--checkpoint", default="runs/fewshot_sparse_fusion/best.pth")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--save-dir")
    parser.add_argument("--save-preds", default=True, action="store_true")
    parser.add_argument(
        "--tile-overlap",
        type=float,
        default=0.25,
        help="Overlap ratio for sliding-window inference when the query image is larger than the training image size.",
    )
    parser.add_argument(
        "--allow-partial-load",
        action="store_true",
        help="Allow loading only shape-compatible checkpoint weights. By default evaluation fails on mismatched checkpoints.",
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
        help="Keep frozen encoders on the runtime device for faster evaluation when memory allows.",
    )
    parser.set_defaults(offload_to_cpu=False)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _resolve_eval_arg(cli_value, checkpoint_args: dict, key: str, fallback_key: str | None = None, default=None):
    if cli_value is not None:
        return cli_value
    if key in checkpoint_args:
        return checkpoint_args[key]
    if fallback_key is not None and fallback_key in checkpoint_args:
        return checkpoint_args[fallback_key]
    return default


def save_prediction(prediction: torch.Tensor, path: str) -> None:
    image = Image.fromarray((prediction.squeeze(0).cpu().numpy() * 255).astype("uint8"))
    image.save(path)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    checkpoint_args = checkpoint.get("args", {})

    data_root = _resolve_eval_arg(args.data_root, checkpoint_args, "data_root", default="Datasets/DRIVE")
    query_split = _resolve_eval_arg(args.query_split, checkpoint_args, "val_split", fallback_key="query_split", default="val")
    support_split = _resolve_eval_arg(args.support_split, checkpoint_args, "support_split", default="train")
    image_size = checkpoint_args.get("image_size", 1024)
    shots = _resolve_eval_arg(args.shots, checkpoint_args, "shots", default=1)
    save_dir = args.save_dir or os.path.join(os.path.dirname(args.checkpoint), "predictions")

    model_config = dict(checkpoint.get("model_config") or {})
    model_config["offload_to_cpu"] = args.offload_to_cpu
    model = build_model_from_config(model_config)
    load_result = load_model_state_dict(model, checkpoint["model_state"], allow_partial=args.allow_partial_load)
    model.set_training_progress(checkpoint.get("epoch", checkpoint_args.get("epochs", 1)) - 1, checkpoint_args.get("epochs", 1))
    model.set_runtime_device(device)
    model.eval()
    if load_result["skipped_keys"]:
        print(
            f"loaded {load_result['loaded_count']} compatible parameters; "
            f"skipped {len(load_result['skipped_keys'])} mismatched parameters from checkpoint"
        )
    print(
        "eval_config "
        f"data_root={data_root} query_split={query_split} support_split={support_split} "
        f"image_size={image_size} shots={shots}"
    )

    dataset = FewShotSegDataset(
        root_dir=data_root,
        split=query_split,
        shots=shots,
        image_size=image_size,
        support_split=support_split,
        deterministic=True,
        seed=args.seed,
        keep_query_size=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    os.makedirs(save_dir, exist_ok=True)
    total_dice = 0.0
    total_iou = 0.0
    steps = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="test"):
            query_image = batch["query_image"].to(device, non_blocking=True)
            query_mask = batch["query_mask"].to(device, non_blocking=True)
            support_images = batch["support_images"].to(device, non_blocking=True)
            support_masks = batch["support_masks"].to(device, non_blocking=True)

            logits = model.predict_logits(
                query_image,
                support_images,
                support_masks,
                tile_size=image_size,
                tile_overlap=args.tile_overlap,
            )
            refined_prediction = (torch.sigmoid(logits) > 0.5).float()
            metrics = model.compute_metrics(logits, query_mask)
            total_dice += metrics["dice"]
            total_iou += metrics["iou"]
            steps += 1

            if args.save_preds:
                save_prediction(
                    refined_prediction[0],
                    os.path.join(save_dir, batch["query_name"][0]),
                )

            if args.max_steps > 0 and steps >= args.max_steps:
                break

    if steps == 0:
        raise RuntimeError("No samples were evaluated.")
    print(f"dice={total_dice / steps:.4f} iou={total_iou / steps:.4f}")


if __name__ == "__main__":
    main()
