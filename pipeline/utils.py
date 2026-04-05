"""
pipeline/utils.py
==================
Shared utilities for all pipeline phases.

Covers:
  - Checkpoint saving (best.pt + latest.pt pattern)
  - AMP context manager wrapper
  - Input DataFrame validation
  - Reproducibility seed setting
  - VRAM memory reporting
  - Phase-aware logging
"""

from __future__ import annotations

import logging
import math
import os
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# ─── Logging ────────────────────────────────────────────────────────────────

def get_logger(phase_name: str) -> logging.Logger:
    """Return a logger for the given pipeline phase."""
    logger = logging.getLogger(f"AID.{phase_name}")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "[%(asctime)s] [%(name)s] %(levelname)s — %(message)s",
            datefmt="%H:%M:%S"
        ))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


# ─── Reproducibility ────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # Slower but reproducible
    os.environ["PYTHONHASHSEED"] = str(seed)


# ─── Device Selection ────────────────────────────────────────────────────────

def get_device(compute_tier: str) -> torch.device:
    """Return appropriate torch device for the compute tier."""
    if compute_tier == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        device = torch.device("cuda")
        return device
    return torch.device("cpu")


def log_device_info(device: torch.device, logger: logging.Logger) -> None:
    """Log GPU info for debugging."""
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        logger.info(f"GPU: {props.name} | VRAM: {props.total_memory / 1e9:.1f} GB")
        logger.info(f"CUDA: {torch.version.cuda} | cuDNN: {torch.backends.cudnn.version()}")
    else:
        logger.info("Running on CPU")


def get_vram_used_gb() -> float:
    """Return current GPU VRAM usage in GB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e9
    return 0.0


# ─── AMP Context ─────────────────────────────────────────────────────────────

@contextmanager
def amp_context(device: torch.device, use_amp: bool, amp_dtype: str = "float16"):
    """
    Context manager for AMP autocast.
    Always use this instead of torch.cuda.amp.autocast directly.

    Usage:
        with amp_context(device, USE_AMP) as ctx:
            pred = model(X)
    """
    dtype = torch.float16 if amp_dtype == "float16" else torch.bfloat16
    if use_amp and device.type == "cuda":
        with torch.cuda.amp.autocast(dtype=dtype):
            yield
    else:
        yield


# ─── Checkpoint Saving ───────────────────────────────────────────────────────

def save_checkpoint(
    model: torch.nn.Module,
    path_best: Path,
    path_latest: Path,
    val_loss: float,
    best_val_loss: float,
    extra: Optional[dict] = None,
    logger: Optional[logging.Logger] = None,
) -> float:
    """
    Save latest checkpoint every epoch and best checkpoint on improvement.

    Pattern (Gap #1 fix):
      latest.pt → overwritten EVERY epoch (Colab session recovery)
      best.pt   → overwritten only when val_loss < best_val_loss

    Args:
        model: PyTorch model to save
        path_best: Path for best model checkpoint
        path_latest: Path for latest-epoch checkpoint
        val_loss: Current validation loss
        best_val_loss: Best validation loss so far
        extra: Optional dict with extra state (e.g., epoch, optimizer state)
        logger: Optional logger for printing status

    Returns:
        Updated best_val_loss
    """
    path_best.parent.mkdir(parents=True, exist_ok=True)
    path_latest.parent.mkdir(parents=True, exist_ok=True)

    state = {"model_state_dict": model.state_dict(), "val_loss": val_loss}
    if extra:
        state.update(extra)

    # Always save latest (Colab recovery)
    torch.save(state, path_latest)

    if val_loss < best_val_loss:
        torch.save(state, path_best)
        if logger:
            logger.info(
                f"  ✓ New best: {val_loss:.4f} (prev {best_val_loss:.4f}) → {path_best.name}"
            )
        return val_loss

    return best_val_loss


def load_checkpoint(
    model: torch.nn.Module,
    path: Path,
    device: torch.device,
    logger: Optional[logging.Logger] = None,
) -> torch.nn.Module:
    """Load model weights from checkpoint. Falls back to latest.pt if best.pt missing."""
    if not path.exists():
        # Try sibling latest.pt as fallback
        latest = path.parent / path.name.replace("_best.pt", "_latest.pt")
        if latest.exists():
            if logger:
                logger.warning(f"best.pt not found — loading latest.pt: {latest}")
            path = latest
        else:
            raise FileNotFoundError(f"No checkpoint found at {path} or {latest}")

    state = torch.load(path, map_location=device)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)

    if logger:
        logger.info(f"Loaded checkpoint: {path} (val_loss={state.get('val_loss', 'N/A')})")
    return model


# ─── DataFrame Validation ────────────────────────────────────────────────────

def validate_dataframe(
    df,
    name: str,
    required_cols: Optional[list] = None,
    min_rows: int = 2,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Validate a DataFrame before using it in a pipeline phase.
    Raises ValueError with a descriptive message on failure.

    This implements the "check df.shape[0] > 0 before proceeding" rule.
    """
    import pandas as pd
    log = logger or logging.getLogger("AID.validation")

    if df is None:
        raise ValueError(f"[{name}] DataFrame is None — check data loading step")

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"[{name}] Expected DataFrame, got {type(df)}")

    if df.shape[0] < min_rows:
        raise ValueError(
            f"[{name}] Too few rows: {df.shape[0]} (minimum {min_rows}). "
            f"Check data loading — E-MTAB single-sample bug or empty merge."
        )

    if required_cols:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"[{name}] Missing required columns: {missing}. "
                f"Available: {list(df.columns[:10])}"
            )

    nan_pct = df.isnull().mean().mean() * 100
    if nan_pct > 90:
        log.warning(f"[{name}] {nan_pct:.0f}% NaN values — data may be malformed")

    log.info(f"[{name}] ✓ {df.shape[0]} rows × {df.shape[1]} cols | "
             f"NaN: {nan_pct:.1f}%")


# ─── GAN NaN Guard ───────────────────────────────────────────────────────────

def gan_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss: torch.Tensor,
    scaler: torch.cuda.amp.GradScaler,
    grad_clip: float = 1.0,
    nan_skip: bool = True,
    epoch: int = 0,
    logger: Optional[logging.Logger] = None,
) -> bool:
    """
    Safe GAN training step with NaN gradient detection (Gap #1 fix).

    Pattern (from config.GAN_GRAD_CLIP, GAN_NAN_SKIP):
      1. Unscale gradients before clipping (required for AMP)
      2. Detect NaN gradients — skip step if found
      3. Clip gradients for stability
      4. Apply optimizer step

    Returns:
        True if step was applied, False if skipped due to NaN
    """
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)  # Must unscale before clip

    # NaN gradient detection
    if nan_skip:
        has_nan = any(
            p.grad is not None and torch.isnan(p.grad).any()
            for p in model.parameters()
        )
        if has_nan:
            if logger:
                logger.warning(
                    f"[GAN] NaN gradient at epoch {epoch} — skipping step. "
                    f"VRAM used: {get_vram_used_gb():.2f} GB. "
                    f"If persistent: reduce LR, increase label smoothing, or use bfloat16."
                )
            optimizer.zero_grad()
            scaler.update()
            return False  # Step skipped

    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    return True


def check_gan_collapse(
    disc_losses: list,
    min_loss: float = 0.01,
    patience: int = 10,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Detect GAN mode collapse from discriminator loss history.
    Raises RuntimeError if collapse detected.
    """
    if len(disc_losses) < patience:
        return
    recent = disc_losses[-patience:]
    if all(l < min_loss for l in recent):
        msg = (
            f"GAN mode collapse detected: discriminator loss < {min_loss} "
            f"for {patience} consecutive epochs. "
            f"Try: reducing generator LR, adding noise to inputs, "
            f"increasing label smoothing."
        )
        if logger:
            logger.error(msg)
        raise RuntimeError(msg)


# ─── Progress Timer ──────────────────────────────────────────────────────────

class PhaseTimer:
    """Simple context manager for timing pipeline phases."""

    def __init__(self, phase_name: str, logger: Optional[logging.Logger] = None):
        self.phase_name = phase_name
        self.logger = logger or logging.getLogger(f"AID.{phase_name}")
        self.start_time: float = 0.0

    def __enter__(self):
        self.start_time = time.time()
        self.logger.info(f"{'='*50}")
        self.logger.info(f"Starting Phase: {self.phase_name}")
        self.logger.info(f"{'='*50}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        minutes, seconds = divmod(elapsed, 60)
        if exc_type is None:
            self.logger.info(
                f"Phase {self.phase_name} complete in "
                f"{int(minutes)}m {seconds:.1f}s"
            )
        else:
            self.logger.error(
                f"Phase {self.phase_name} FAILED after "
                f"{int(minutes)}m {seconds:.1f}s: {exc_val}"
            )
        return False  # Don't suppress exceptions
