"""Training loop for SimCLR pretraining."""

from __future__ import annotations

import csv
import math
import shutil
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.simclr_dataset import SimCLRDataset
from src.losses.nt_xent_loss import NTXentLoss
from src.models.backbone import build_backbone
from src.models.projection_head import ProjectionHead
from src.models.simclr import SimCLR
from src.transforms.simclr_transforms import get_simclr_transform
from src.utils.checkpoint import encoder_checkpoint, full_model_checkpoint, save_checkpoint
from src.utils.config import resolve_project_path
from src.utils.logger import create_logger
from src.utils.seed import set_seed


def _resolve_device(configured_device: str) -> torch.device:
    if configured_device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(configured_device)


def _as_bool(value: Any, default: bool = False) -> bool:
    """Parse config booleans safely, including strings from YAML edits."""

    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _get_gpu_temperature_c() -> int | None:
    """Read the first NVIDIA GPU temperature using nvidia-smi."""

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return None

    first_line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    try:
        return int(first_line.strip())
    except ValueError:
        return None


def _cosine_warmup_lr(epoch: int, num_epochs: int, base_lr: float, min_lr: float, warmup_epochs: int) -> float:
    if warmup_epochs > 0 and epoch <= warmup_epochs:
        warmup_progress = epoch / warmup_epochs
        return min_lr + (base_lr - min_lr) * warmup_progress

    cosine_epochs = max(1, num_epochs - warmup_epochs)
    cosine_progress = min(1.0, max(0.0, (epoch - warmup_epochs) / cosine_epochs))
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * cosine_progress))


def _epoch_learning_rate(
    epoch: int,
    num_epochs: int,
    base_lr: float,
    scheduler: str,
    warmup_epochs: int,
    min_lr: float,
) -> float:
    scheduler_name = scheduler.strip().lower()
    if scheduler_name in {"none", "constant", ""}:
        return base_lr
    if scheduler_name == "cosine":
        return _cosine_warmup_lr(epoch, num_epochs, base_lr, min_lr, warmup_epochs)
    raise ValueError(f"Unsupported scheduler: {scheduler}")


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate


def _apply_run_name(base_path: Path, run_name: str | None) -> Path:
    if not run_name:
        return base_path
    return base_path / run_name


def _infer_run_name_from_resume(root: Path, config: dict[str, Any], resume_path_value: str | Path) -> str | None:
    """Infer run name from a checkpoint path such as checkpoints/simclr/<run>/file.pth."""

    resume_path = resolve_project_path(root, resume_path_value)
    base_checkpoint_dir = resolve_project_path(root, config["checkpoint_dir"])
    parent = resume_path.parent
    if parent == base_checkpoint_dir:
        return None
    return parent.name


def _save_loss_csv(loss_history: list[dict[str, float]], result_dir: Path) -> Path:
    result_dir.mkdir(parents=True, exist_ok=True)
    output_path = result_dir / "simclr_loss.csv"
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", "train_loss"])
        writer.writeheader()
        writer.writerows(loss_history)
    return output_path


def _load_loss_history(result_dir: Path) -> list[dict[str, float]]:
    csv_path = result_dir / "simclr_loss.csv"
    if not csv_path.exists():
        return []

    history: list[dict[str, float]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            history.append({"epoch": int(row["epoch"]), "train_loss": float(row["train_loss"])})
    return history


def _save_loss_curve(loss_history: list[dict[str, float]], figure_dir: Path) -> Path:
    figure_dir.mkdir(parents=True, exist_ok=True)
    output_path = figure_dir / "simclr_loss_curve.png"

    epochs = [row["epoch"] for row in loss_history]
    losses = [row["train_loss"] for row in loss_history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Train loss")
    plt.title("SimCLR training loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def train_simclr(config: dict[str, Any], project_root: str | Path) -> dict[str, Path | float]:
    """Train SimCLR and write checkpoints, loss CSV, loss curve, and logs."""

    root = Path(project_root)
    seed = int(config.get("seed", 42))
    set_seed(seed)

    resume_path_value = config.get("resume_checkpoint")
    run_name = config.get("run_name")
    if not run_name and resume_path_value:
        run_name = _infer_run_name_from_resume(root, config, resume_path_value)
        if run_name:
            config["run_name"] = run_name

    checkpoint_dir = _apply_run_name(resolve_project_path(root, config["checkpoint_dir"]), run_name)
    log_dir = _apply_run_name(resolve_project_path(root, config["log_dir"]), run_name)
    result_dir = _apply_run_name(resolve_project_path(root, config["result_dir"]), run_name)
    figure_dir = _apply_run_name(resolve_project_path(root, config["figure_dir"]), run_name)
    log_path = log_dir / "simclr_train.log"
    logger_mode = "a" if resume_path_value else "w"
    logger = create_logger("simclr_train", log_path, mode=logger_mode)

    data_dir = resolve_project_path(root, config["data_dir"])
    image_size = int(config.get("image_size", 224))
    transform = get_simclr_transform(image_size=image_size)
    dataset = SimCLRDataset(data_dir=data_dir, transform=transform)

    device = _resolve_device(str(config.get("device", "cuda")))
    use_cuda = device.type == "cuda"
    use_amp = _as_bool(config.get("amp"), default=True) and use_cuda
    use_channels_last = _as_bool(config.get("channels_last"), default=True) and use_cuda
    if use_cuda:
        torch.backends.cudnn.benchmark = _as_bool(config.get("cudnn_benchmark"), default=True)
        torch.set_float32_matmul_precision(str(config.get("matmul_precision", "high")))

    batch_size = int(config.get("batch_size", 64))
    num_workers = int(config.get("num_workers", 4))
    drop_last = len(dataset) >= batch_size
    dataloader_kwargs = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": use_cuda,
        "drop_last": drop_last,
    }
    if num_workers > 0:
        dataloader_kwargs["persistent_workers"] = _as_bool(config.get("persistent_workers"), default=True)
        dataloader_kwargs["prefetch_factor"] = int(config.get("prefetch_factor", 2))
    dataloader = DataLoader(dataset, **dataloader_kwargs)

    logger.info("Using device: %s", device)
    logger.info("Run name: %s", run_name if run_name else "default")
    logger.info("Dataset size: %d images", len(dataset))
    logger.info("Batch size: %d", batch_size)
    logger.info("AMP: %s", use_amp)
    logger.info("Channels last: %s", use_channels_last)
    logger.info("Num workers: %d", num_workers)

    encoder, feature_dim = build_backbone(
        name=str(config.get("backbone", "resnet50")),
        pretrained=_as_bool(config.get("pretrained"), default=True),
    )
    projection_head = ProjectionHead(
        input_dim=feature_dim,
        hidden_dim=int(config.get("projection_hidden_dim", config.get("hidden_dim", 512))),
        output_dim=int(config.get("projection_output_dim", config.get("projection_dim", 128))),
    )
    model = SimCLR(encoder=encoder, projection_head=projection_head).to(device)
    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)

    optimizer_name = str(config.get("optimizer", "AdamW"))
    if optimizer_name.strip().lower() != "adamw":
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    learning_rate = float(config.get("learning_rate", 3e-4))
    weight_decay = float(config.get("weight_decay", 1e-4))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    criterion = NTXentLoss(temperature=float(config.get("temperature", 0.5)))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    num_epochs = int(config.get("num_epochs", 100))
    scheduler = str(config.get("scheduler", "none"))
    warmup_epochs = max(0, int(config.get("warmup_epochs", 0)))
    min_lr = float(config.get("min_lr", 0.0))
    early_stopping_enabled = _as_bool(config.get("early_stopping"), default=False)
    early_stopping_patience = int(config.get("early_stopping_patience", 20))
    early_stopping_min_delta = float(config.get("early_stopping_min_delta", 0.0005))
    best_checkpoint_interval = max(0, int(config.get("best_checkpoint_interval", 20)))
    gpu_temp_stop_enabled = _as_bool(config.get("gpu_temp_stop"), default=False) and use_cuda
    gpu_temp_threshold_c = int(config.get("gpu_temp_threshold_c", 83))
    gpu_temp_check_interval = max(1, int(config.get("gpu_temp_check_interval", 10)))
    gpu_temp_consecutive_hits_required = max(1, int(config.get("gpu_temp_consecutive_hits", 5)))
    gpu_temp_consecutive_hits = 0
    gpu_temp_warning_logged = False
    if gpu_temp_stop_enabled:
        logger.info(
            "GPU temperature stop: enabled | threshold=%dC | check_interval=%d batch(es) | consecutive_hits=%d",
            gpu_temp_threshold_c,
            gpu_temp_check_interval,
            gpu_temp_consecutive_hits_required,
        )
    logger.info("Optimizer: %s | lr=%.8f | weight_decay=%.8f", optimizer_name, learning_rate, weight_decay)
    logger.info(
        "Scheduler: %s | warmup_epochs=%d | min_lr=%.8f",
        scheduler,
        warmup_epochs,
        min_lr,
    )
    max_steps_per_epoch = config.get("max_steps_per_epoch")
    if max_steps_per_epoch is not None:
        max_steps_per_epoch = int(max_steps_per_epoch)
    start_epoch = 1
    best_loss = float("inf")
    early_stopping_bad_epochs = 0
    loss_history: list[dict[str, float]] = []

    if resume_path_value:
        resume_path = resolve_project_path(root, resume_path_value)
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        model.projection_head.load_state_dict(checkpoint["projection_head_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_loss = float(checkpoint.get("best_loss", checkpoint.get("loss", float("inf"))))
        early_stopping_bad_epochs = int(checkpoint.get("early_stopping_bad_epochs", 0))
        loss_history = [row for row in _load_loss_history(result_dir) if row["epoch"] < start_epoch]
        logger.info("Resumed from checkpoint: %s", resume_path)
        logger.info("Resume starts at epoch %d/%d", start_epoch, num_epochs)
        if start_epoch > num_epochs:
            raise ValueError(
                f"Resume checkpoint is already at epoch {start_epoch - 1}, "
                f"but num_epochs is {num_epochs}. Increase --epochs to continue."
            )

    for epoch in range(start_epoch, num_epochs + 1):
        epoch_lr = _epoch_learning_rate(
            epoch=epoch,
            num_epochs=num_epochs,
            base_lr=learning_rate,
            scheduler=scheduler,
            warmup_epochs=warmup_epochs,
            min_lr=min_lr,
        )
        _set_optimizer_lr(optimizer, epoch_lr)
        epoch_started_at = time.perf_counter()
        model.train()
        running_loss = 0.0
        num_batches = 0
        gpu_temp_stop_triggered = False

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs}", leave=False)
        for batch_index, (view1, view2) in enumerate(progress_bar, start=1):
            view1 = view1.to(device, non_blocking=True)
            view2 = view2.to(device, non_blocking=True)
            if use_channels_last:
                view1 = view1.contiguous(memory_format=torch.channels_last)
                view2 = view2.contiguous(memory_format=torch.channels_last)

            optimizer.zero_grad(set_to_none=True)
            autocast_context = (
                torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
                if use_amp
                else nullcontext()
            )
            with autocast_context:
                z1, z2 = model(view1, view2)
            loss = criterion(z1.float(), z2.float())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.item())
            num_batches += 1
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

            if gpu_temp_stop_enabled and batch_index % gpu_temp_check_interval == 0:
                gpu_temperature_c = _get_gpu_temperature_c()
                if gpu_temperature_c is None:
                    if not gpu_temp_warning_logged:
                        logger.warning("Could not read GPU temperature from nvidia-smi; skipping this temperature check.")
                        gpu_temp_warning_logged = True
                    gpu_temp_consecutive_hits = 0
                elif gpu_temperature_c is not None:
                    logger.info("GPU temperature: %dC", gpu_temperature_c)
                    if gpu_temperature_c >= gpu_temp_threshold_c:
                        gpu_temp_consecutive_hits += 1
                        if gpu_temp_consecutive_hits >= gpu_temp_consecutive_hits_required:
                            logger.warning(
                                "GPU temperature %dC reached threshold %dC for %d consecutive checks. "
                                "Stopping after this batch.",
                                gpu_temperature_c,
                                gpu_temp_threshold_c,
                                gpu_temp_consecutive_hits,
                            )
                            gpu_temp_stop_triggered = True
                            break
                        logger.warning(
                            "GPU temperature %dC reached threshold %dC (%d/%d consecutive checks).",
                            gpu_temperature_c,
                            gpu_temp_threshold_c,
                            gpu_temp_consecutive_hits,
                            gpu_temp_consecutive_hits_required,
                        )
                    elif gpu_temp_consecutive_hits:
                        logger.info(
                            "GPU temperature back below threshold; reset consecutive high-temperature count from %d to 0.",
                            gpu_temp_consecutive_hits,
                        )
                        gpu_temp_consecutive_hits = 0

            if max_steps_per_epoch is not None and batch_index >= max_steps_per_epoch:
                logger.info("Stopping epoch %d early at max_steps_per_epoch=%d", epoch, max_steps_per_epoch)
                break

        epoch_loss = running_loss / max(num_batches, 1)
        loss_history.append({"epoch": epoch, "train_loss": epoch_loss})
        epoch_seconds = time.perf_counter() - epoch_started_at
        improved = epoch_loss < (best_loss - early_stopping_min_delta)
        logger.info(
            "Epoch %d/%d | lr=%.8f | train_loss=%.6f | time_sec=%.2f | sec_per_batch=%.3f",
            epoch,
            num_epochs,
            epoch_lr,
            epoch_loss,
            epoch_seconds,
            epoch_seconds / max(num_batches, 1),
        )

        best_checkpoint_path = checkpoint_dir / "simclr_encoder_best.pth"
        if improved:
            best_loss = epoch_loss
            early_stopping_bad_epochs = 0
            save_checkpoint(
                encoder_checkpoint(
                    model=model,
                    epoch=epoch,
                    loss=epoch_loss,
                    config=config,
                    best_loss=best_loss,
                    early_stopping_bad_epochs=early_stopping_bad_epochs,
                ),
                best_checkpoint_path,
            )
        else:
            early_stopping_bad_epochs += 1

        if early_stopping_enabled:
            logger.info(
                "Early stopping: bad_epochs=%d/%d | min_delta=%.6f | best_loss=%.6f",
                early_stopping_bad_epochs,
                early_stopping_patience,
                early_stopping_min_delta,
                best_loss,
            )

        # Save resumable state every completed epoch so training can continue
        # after an interruption without starting from the beginning.
        save_checkpoint(
            encoder_checkpoint(
                model=model,
                epoch=epoch,
                loss=epoch_loss,
                config=config,
                best_loss=best_loss,
                early_stopping_bad_epochs=early_stopping_bad_epochs,
            ),
            checkpoint_dir / "simclr_encoder_last.pth",
        )
        save_checkpoint(
            full_model_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=epoch_loss,
                config=config,
                best_loss=best_loss,
                early_stopping_bad_epochs=early_stopping_bad_epochs,
            ),
            checkpoint_dir / "simclr_full_model_last.pth",
        )
        if best_checkpoint_interval and epoch % best_checkpoint_interval == 0:
            interval_best_checkpoint_path = checkpoint_dir / f"simclr_encoder_best_epoch_{epoch:03d}.pth"
            if best_checkpoint_path.exists():
                shutil.copy2(best_checkpoint_path, interval_best_checkpoint_path)
            else:
                save_checkpoint(
                    encoder_checkpoint(
                        model=model,
                        epoch=epoch,
                        loss=epoch_loss,
                        config=config,
                        best_loss=best_loss,
                        early_stopping_bad_epochs=early_stopping_bad_epochs,
                    ),
                    interval_best_checkpoint_path,
                )
            logger.info("Saved interval best encoder checkpoint: %s", interval_best_checkpoint_path)
        _save_loss_csv(loss_history, result_dir)
        _save_loss_curve(loss_history, figure_dir)

        if early_stopping_enabled and early_stopping_bad_epochs >= early_stopping_patience:
            logger.info(
                "Early stopping triggered at epoch %d because loss did not improve for %d epochs.",
                epoch,
                early_stopping_patience,
            )
            break

        if gpu_temp_stop_triggered:
            logger.info(
                "Training stopped by GPU temperature guard at epoch %d. Resume from simclr_full_model_last.pth.",
                epoch,
            )
            break

    final_loss = loss_history[-1]["train_loss"]

    loss_csv_path = _save_loss_csv(loss_history, result_dir)
    loss_curve_path = _save_loss_curve(loss_history, figure_dir)

    logger.info("Saved best encoder checkpoint: %s", checkpoint_dir / "simclr_encoder_best.pth")
    logger.info("Saved last encoder checkpoint: %s", checkpoint_dir / "simclr_encoder_last.pth")
    logger.info("Saved full model checkpoint: %s", checkpoint_dir / "simclr_full_model_last.pth")
    logger.info("Saved loss CSV: %s", loss_csv_path)
    logger.info("Saved loss curve: %s", loss_curve_path)

    return {
        "best_loss": best_loss,
        "last_epoch": loss_history[-1]["epoch"],
        "run_name": run_name if run_name else "default",
        "loss_csv_path": loss_csv_path,
        "loss_curve_path": loss_curve_path,
        "log_path": log_path,
        "checkpoint_dir": checkpoint_dir,
    }
