"""Training-time helper functions for logging, checkpoints and model loading."""

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

import accelerate
import gluonts
import numpy as np
import torch
import torch.distributed as dist
import transformers
from scaleformer.scaleformer.scaleformer import (
    PatchTSTConfig,
    PatchTSTForPrediction,
)


# Utilities for training
def is_main_process() -> bool:
    """
    Check if we're on the main process.
    """
    if not dist.is_torchelastic_launched():
        return True
    return int(os.environ["RANK"]) == 0


def log_on_main(
    msg: str,
    logger: logging.Logger,
    log_level: int = logging.INFO,
) -> None:
    """
    Log the given message using the given logger, if we're on the main process.
    """
    if is_main_process():
        logger.log(log_level, msg)


def get_training_job_info() -> dict:  # not currently used
    """
    Returns info about this training job.
    """
    job_info = {}

    # CUDA info
    job_info["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        job_info["device_count"] = torch.cuda.device_count()

        job_info["device_names"] = {
            idx: torch.cuda.get_device_name(idx)
            for idx in range(torch.cuda.device_count())
        }
        job_info["mem_info"] = {
            idx: torch.cuda.mem_get_info(device=idx)
            for idx in range(torch.cuda.device_count())
        }

    # DDP info
    job_info["torchelastic_launched"] = dist.is_torchelastic_launched()

    if dist.is_torchelastic_launched():
        job_info["world_size"] = dist.get_world_size()

    # Versions
    job_info["python_version"] = sys.version.replace("\n", " ")
    job_info["torch_version"] = torch.__version__
    job_info["numpy_version"] = np.__version__
    job_info["gluonts_version"] = gluonts.__version__
    job_info["transformers_version"] = transformers.__version__
    job_info["accelerate_version"] = accelerate.__version__

    return job_info


def save_training_info(
    ckpt_path: Path,
    model_config: dict,
    train_config: dict,
    all_config: dict,
) -> None:
    """
    Save info about this training job in a json file for documentation.
    """
    assert ckpt_path.is_dir()
    with open(ckpt_path / "training_info.json", "w", encoding="utf-8") as fp:
        json.dump(
            {
                "model_config": model_config,
                "train_config": train_config,
                "all_config": all_config,
                "job_info": get_training_job_info(),
            },
            fp,
            indent=4,
        )


def get_next_path(
    base_fname: str,
    base_dir: Path,
    file_type: str = "yaml",
    separator: str = "-",
    overwrite: bool = False,
) -> Path:
    """
    Gets the next available path in a directory. For example, if `base_fname="results"`
    and `base_dir` has files ["results-0.yaml", "results-1.yaml"], this function returns
    "results-2.yaml".
    """
    if file_type == "":
        # Directory
        items = filter(
            lambda x: x.is_dir() and re.match(f"^{base_fname}{separator}\\d+$", x.stem),
            base_dir.glob("*"),
        )
    else:
        # File
        items = filter(
            lambda x: re.match(f"^{base_fname}{separator}\\d+$", x.stem),
            base_dir.glob(f"*.{file_type}"),
        )
    run_nums = list(
        map(lambda x: int(x.stem.replace(base_fname + separator, "")), items)
    ) + [-1]

    next_num = max(run_nums) + (0 if overwrite else 1)
    fname = f"{base_fname}{separator}{next_num}" + (
        f".{file_type}" if file_type != "" else ""
    )

    return base_dir / fname


def load_patchtst_model(
    model_config: dict[str, Any],
    checkpoint_path: str | None = None,
) -> PatchTSTForPrediction:
    """
    Load a PatchTST prediction model.

    Args:
        model_config: Dictionary containing model configuration parameters
        checkpoint_path: Optional path to a prediction checkpoint

    Returns:
        PatchTSTForPrediction model instance
    """
    config = PatchTSTConfig(**model_config)
    if checkpoint_path is not None:
        pretrained_model = PatchTSTForPrediction.from_pretrained(
            checkpoint_path,
            config=config,
        )
        return pretrained_model  # type: ignore
    return PatchTSTForPrediction(config)


def has_enough_observations(
    entry: dict, min_length: int = 0, max_missing_prop: float = 1.0
) -> bool:
    """
    Check if the given entry has enough observations in the ``"target"`` attribute.

    Parameters
    ----------
    entry
        The data entry (dictionary) to be tested.
    min_length
        The minimum length the ``"target"`` attribute must have.
    max_missing_prop
        The maximum proportion of missing data allowed in the ``"target"``
        attribute.
    """
    if (
        entry["target"].shape[-1] >= min_length
        and np.isnan(entry["target"]).mean() <= max_missing_prop
    ):
        return True
    return False


def ensure_contiguous(model):
    """
    Ensure that all parameters in the model are contiguous.
    If any parameter is not contiguous, make it contiguous.

    Args:
        model: The model whose parameters need to be checked.
    """
    for name, param in model.named_parameters():
        if not param.is_contiguous():
            print(f"Parameter {name} is not contiguous. Making it contiguous.")
            param.data = param.data.contiguous()
