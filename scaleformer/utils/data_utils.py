"""Data IO and normalization helpers used by training/evaluation scripts."""

import os
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Union

import numpy as np
from gluonts.dataset.arrow import ArrowWriter
from gluonts.dataset.common import FileDataset


def get_dim_from_dataset(dataset: FileDataset) -> int:  # type: ignore
    """Return target dimensionality for a GluonTS file dataset."""
    return next(iter(dataset))["target"].shape[0]


def safe_standardize(
    arr: np.ndarray,
    epsilon: float = 1e-10,
    axis: int = -1,
    context: np.ndarray | None = None,
    denormalize: bool = False,
) -> np.ndarray:
    """
    Standardize the trajectories by subtracting the mean and dividing by the standard deviation

    Args:
        arr: The array to standardize
        epsilon: A small value to prevent division by zero
        axis: The axis to standardize along
        context: The context to use for standardization. If provided, use the context to standardize the array.
        denormalize: If True, denormalize the array using the mean and standard deviation of the context.

    Returns:
        The standardized array
    """
    # if no context is provided, use the array itself
    context = arr if context is None else context

    assert (
        arr.ndim == context.ndim
    ), "arr and context must have the same num dims if context is provided"
    assert (
        axis < arr.ndim and axis >= -arr.ndim
    ), "invalid axis specified for standardization"
    mean = np.nanmean(context, axis=axis, keepdims=True)
    std = np.nanstd(context, axis=axis, keepdims=True)
    std = np.where(std < epsilon, epsilon, std)
    if denormalize:
        return arr * std + mean
    return (arr - mean) / std


def convert_to_arrow(
    path: Union[str, Path],
    time_series: Union[List[np.ndarray], np.ndarray],
    compression: Literal["lz4", "zstd"] = "lz4",
    split_coords: bool = False,
):
    """
    Store a given set of series into Arrow format at the specified path.

    Input data can be either a list of 1D numpy arrays, or a single 2D
    numpy array of shape (num_series, time_length).
    """
    assert isinstance(time_series, list) or (
        isinstance(time_series, np.ndarray) and time_series.ndim == 2
    ), "time_series must be a list of 1D numpy arrays or a 2D numpy array"

    # GluonTS requires this datetime format for reading arrow file
    start = datetime.now().strftime("%Y-%m-%d %H:%M")

    if split_coords:
        dataset = [{"start": start, "target": ts} for ts in time_series]
    else:
        dataset = [{"start": start, "target": time_series}]

    ArrowWriter(compression=compression).write_to_file(
        dataset,
        path=Path(path),
    )


def process_trajs(
    base_dir: str,
    timeseries: dict[str, np.ndarray],
    split_coords: bool = False,
    overwrite: bool = False,
    verbose: bool = False,
    base_sample_idx: int = -1,
) -> None:
    """
    Saves each trajectory in timeseries ensemble to a separate directory

    Args:
        base_dir: The base directory to save the trajectories to
        timeseries: A dictionary mapping system names to numpy arrays of shape
            (num_eval_windows * num_datasets, num_channels, T) where T is the prediction
            length or context length.
        split_coords: Whether to split the coordinates by dimension
        overwrite: Whether to overwrite existing trajectories
        verbose: Whether to print verbose output
        base_sample_idx: The base sample index to use for the trajectories
    """
    for sys_name, trajectories in timeseries.items():
        if verbose:
            print(
                f"Processing trajectories of shape {trajectories.shape} for system {sys_name}"
            )

        system_folder = os.path.join(base_dir, sys_name)
        os.makedirs(system_folder, exist_ok=True)

        if not overwrite:
            for filename in os.listdir(system_folder):
                if filename.endswith(".arrow"):
                    sample_idx = int(filename.split("_")[0])
                    base_sample_idx = max(base_sample_idx, sample_idx)

        for i, trajectory in enumerate(trajectories):
            # very hacky, if there is only one trajectory, we can just use the base_sample_idx
            curr_sample_idx = base_sample_idx + i + 1

            if trajectory.ndim == 1:
                trajectory = np.expand_dims(trajectory, axis=0)
            if verbose:
                print(
                    f"Saving {sys_name} trajectory {curr_sample_idx} with shape {trajectory.shape}"
                )

            path = os.path.join(
                system_folder, f"{curr_sample_idx}_T-{trajectory.shape[-1]}.arrow"
            )

            convert_to_arrow(path, trajectory, split_coords=split_coords)
