"""Evaluation utilities for ScaleFormer forecasting experiments."""

from collections import defaultdict
from typing import Callable

import numpy as np
import torch
from gluonts.itertools import batcher
from scaleformer.scaleformer.dataset import TimeSeriesDataset
from scaleformer.scaleformer.pipeline import PatchTSTPipeline
from scaleformer.utils import safe_standardize
from scaleformer.utils.metrics_utils import (
    calculate_gpdim_raw,
    calculate_kld_raw,
    calculate_le_metrics_raw,
    calculate_psd_metrics_raw,
    compute_standard_metrics_per_sample,
    format_ci,
    format_rmse_ci,
    get_system_dt,
)
from tqdm.auto import tqdm
import scaleformer.scaleformer.scaleformer as model_script


def _identity_reduce(samples: np.ndarray) -> np.ndarray:
    """Return forecasts unchanged when no reduction is requested."""
    return samples


def evaluate_forecasting_model(
    pipeline: PatchTSTPipeline,
    systems: dict[str, TimeSeriesDataset],
    batch_size: int,
    prediction_length: int,
    metric_names: list[str] | None = None,
    parallel_sample_reduction_fn: Callable | None = None,
    channel_sampler: Callable | None = None,
    return_predictions: bool = False,
    return_contexts: bool = False,
    return_labels: bool = False,
    redo_normalization: bool = False,
    prediction_kwargs: dict | None = None,
    eval_subintervals: list[tuple[int, int]] | None = None,
) -> tuple[
    dict[str, np.ndarray] | None,
    dict[str, np.ndarray] | None,
    dict[str, np.ndarray] | None,
    dict[int, dict[str, dict[str, str]]],
]:
    """Run batched forecasting and compute aggregate metrics per system.

    Args:
        pipeline: Inference pipeline wrapping a trained ScaleFormer model.
        systems: Mapping from system name to iterable test dataset.
        batch_size: Number of windows per inference batch.
        prediction_length: Forecast horizon.
        metric_names: Metrics to compute, or ``None`` to skip metric computation.
        parallel_sample_reduction_fn: Reduction over sample axis from model output.
        channel_sampler: Optional callable for channel subsampling.
        return_predictions: Whether to return raw predictions.
        return_contexts: Whether to return contexts.
        return_labels: Whether to return labels.
        redo_normalization: Whether to normalize outputs using context statistics.
        prediction_kwargs: Keyword arguments forwarded to ``pipeline.predict``.
        eval_subintervals: Forecast intervals for horizon-specific metrics.

    Returns:
        Tuple of predictions, contexts, labels, and nested metrics dictionary.
    """
    system_predictions = {}
    system_contexts = {}
    system_labels = {}
    system_metrics = defaultdict(dict)
    prediction_kwargs = prediction_kwargs or {}

    if eval_subintervals is None:
        eval_subintervals = [(0, prediction_length)]
    elif (0, prediction_length) not in eval_subintervals:
        eval_subintervals.append((0, prediction_length))

    if parallel_sample_reduction_fn is None:
        parallel_sample_reduction_fn = _identity_reduce

    with torch.cuda.device(pipeline.device):
        for system in tqdm(systems, desc="Forecasting..."):
            dataset = systems[system]
            predictions, labels, contexts = [], [], []

            system_tau = get_system_dt(system)

            for batch in batcher(dataset, batch_size=batch_size):
                past_values, future_values = zip(
                    *[(data["past_values"], data["future_values"]) for data in batch]
                )
                past_batch = torch.stack(past_values, dim=0).to(pipeline.device)

                preds = (
                    pipeline.predict(
                        past_batch,
                        prediction_length=prediction_length,
                        **prediction_kwargs,
                    )
                    .transpose(0, 1)
                    .cpu()
                    .numpy()
                )

                context = past_batch.cpu().numpy()
                future_batch = torch.stack(future_values, dim=0).cpu().numpy()

                if preds.shape[2] > future_batch.shape[1]:
                    preds = preds[..., : future_batch.shape[1], :]

                if channel_sampler is not None:
                    future_batch = channel_sampler(
                        torch.from_numpy(future_batch), resample_inds=False
                    ).numpy()
                    context = channel_sampler(
                        torch.from_numpy(context), resample_inds=False
                    ).numpy()

                if redo_normalization:
                    preds = safe_standardize(preds, context=context[None, :, :], axis=2)
                    future_batch = safe_standardize(
                        future_batch, context=context, axis=1
                    )
                    context = safe_standardize(context, axis=1)

                labels.append(future_batch)
                predictions.append(preds)
                contexts.append(context)

            predictions = np.concatenate(predictions, axis=1)
            predictions = parallel_sample_reduction_fn(predictions)
            labels = np.concatenate(labels, axis=0)
            contexts = np.concatenate(contexts, axis=0)

            if metric_names is not None:
                for start, end in eval_subintervals:
                    horizon_key = end - start
                    if system not in system_metrics[horizon_key]:
                        system_metrics[horizon_key][system] = {}

                    trues_sub = labels[:, start:end, :]
                    preds_sub = predictions[:, start:end, :]

                    std_results = compute_standard_metrics_per_sample(
                        trues_sub, preds_sub, metric_names
                    )

                    for m_name, m_vals in std_results.items():
                        system_metrics[horizon_key][system][m_name] = format_ci(m_vals)

                    if "ME-LRw" in metric_names:
                        raw_lrw = calculate_psd_metrics_raw(trues_sub, preds_sub)
                        system_metrics[horizon_key][system]["ME-LRw"] = format_ci(
                            raw_lrw
                        )

                    if "max_lyap_gt" in metric_names or "max_lyap_pred" in metric_names:
                        traj_len = 32 if (end - start) <= 128 else 128
                        raw_le_true, raw_le_pred = calculate_le_metrics_raw(
                            trues_sub,
                            preds_sub,
                            tau=system_tau,
                            trajectory_len=traj_len,
                        )

                        if "max_lyap_gt" in metric_names:
                            system_metrics[horizon_key][system][
                                "max_lyap_gt"
                            ] = format_ci(raw_le_true)
                        if "max_lyap_pred" in metric_names:
                            system_metrics[horizon_key][system][
                                "max_lyap_pred"
                            ] = format_ci(raw_le_pred)

                    if "gd_rmse" in metric_names:
                        raw_gpdim_diffs = calculate_gpdim_raw(trues_sub, preds_sub)
                        system_metrics[horizon_key][system]["gd_rmse"] = format_rmse_ci(
                            raw_gpdim_diffs
                        )

                    if "kld" in metric_names:
                        raw_kld = calculate_kld_raw(trues_sub, preds_sub)
                        system_metrics[horizon_key][system]["kld"] = format_ci(raw_kld)

            if return_predictions:
                system_predictions[system] = predictions.transpose(0, 2, 1)
            if return_contexts:
                system_contexts[system] = contexts.transpose(0, 2, 1)
            if return_labels:
                system_labels[system] = labels.transpose(0, 2, 1)

    model_script.CURRENT_TEST_SYSTEM = None

    return (
        system_predictions if return_predictions else None,
        system_contexts if return_contexts else None,
        system_labels if return_labels else None,
        system_metrics,
    )
