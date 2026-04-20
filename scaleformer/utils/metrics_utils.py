"""Metric computation helpers for trajectory forecasting evaluation."""

from collections import defaultdict
import math

import dysts.flows as flows
from dysts.analysis import gp_dim
from dysts.metrics import estimate_kl_divergence
import numpy as np
from scipy.signal.windows import hann
from scipy.spatial.distance import cdist
from scipy.stats import sem, spearmanr, t
import torch


def format_ci(values: list | np.ndarray, confidence: float = 0.95) -> str:
    """Format values as mean plus/minus confidence interval."""
    data = np.asarray(values)
    data = data[np.isfinite(data)]
    n = len(data)
    if n < 2:
        return "N/A"

    mean_val = np.mean(data)
    std_err = sem(data)
    half_width = std_err * t.ppf((1 + confidence) / 2.0, n - 1)
    return f"{mean_val:.4f} ± {half_width:.4f}"


def format_rmse_ci(diffs: list | np.ndarray, confidence: float = 0.95) -> str:
    """Format RMSE values as mean plus/minus confidence interval."""
    diffs = np.asarray(diffs)
    diffs = diffs[np.isfinite(diffs)]
    n = len(diffs)
    if n < 2:
        return "N/A"

    squared_errors = diffs**2
    mse_mean = np.mean(squared_errors)
    mse_stderr = sem(squared_errors)
    rmse_mean = np.sqrt(mse_mean)
    rmse_stderr = 0.0 if rmse_mean < 1e-9 else mse_stderr / (2 * rmse_mean)

    half_width = rmse_stderr * t.ppf((1 + confidence) / 2.0, n - 1)
    return f"{rmse_mean:.4f} ± {half_width:.4f}"


def compute_standard_metrics_per_sample(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_list: list[str],
) -> dict[str, list[float]]:
    """Compute requested point-wise metrics for each sample."""
    n_samples = y_true.shape[0]
    results: dict[str, list[float]] = defaultdict(list)

    errors = y_true - y_pred
    abs_errors = np.abs(errors)

    if "mse" in metric_list or "MSE" in metric_list:
        mse_per_sample = np.mean(errors**2, axis=(1, 2))
        results["mse"] = mse_per_sample.tolist()

    if "mae" in metric_list or "MAE" in metric_list:
        mae_per_sample = np.mean(abs_errors, axis=(1, 2))
        results["mae"] = mae_per_sample.tolist()

    if "smape" in metric_list or "sMAPE" in metric_list:
        denominator = np.abs(y_true) + np.abs(y_pred) + 1e-8
        pointwise_smape = 2.0 * abs_errors / denominator
        smape_per_sample = 100.0 * np.mean(pointwise_smape, axis=(1, 2))
        results["smape"] = smape_per_sample.tolist()

    if "spearman" in metric_list or "coefficient_of_variation" in metric_list:
        spearman_vals: list[float] = []
        for i in range(n_samples):
            flat_true = y_true[i].flatten()
            flat_pred = y_pred[i].flatten()

            if np.std(flat_true) < 1e-9 or np.std(flat_pred) < 1e-9:
                spearman_vals.append(np.nan)
            else:
                corr, _ = spearmanr(flat_true, flat_pred)
                spearman_vals.append(corr)
        results["spearman"] = spearman_vals

    return results


def max_lyapunov_exponent_rosenstein_multivariate(
    data: np.ndarray,
    lag: int | None = None,
    min_tsep: int | None = None,
    tau: float = 1,
    trajectory_len: int = 64,
    fit: str = "RANSAC",
    fit_offset: int = 0,
) -> float:
    """Estimate maximal Lyapunov exponent with Rosenstein's method."""
    data = np.asarray(data, dtype="float32")
    n, _ = data.shape
    max_tsep_factor = 0.25

    if lag is None or min_tsep is None:
        f = np.fft.rfft(data[:, 0], n * 2 - 1)

    if min_tsep is None:
        mf = np.fft.rfftfreq(n * 2 - 1) * np.abs(f)
        mf = np.mean(mf[1:]) / np.sum(np.abs(f[1:]))
        min_tsep = int(np.ceil(1.0 / mf))
        if min_tsep > max_tsep_factor * n:
            min_tsep = int(max_tsep_factor * n)

    orbit = data
    n_points = len(orbit)
    dists = cdist(orbit, orbit, metric="euclidean")

    mask = (
        np.abs(np.arange(n_points)[:, None] - np.arange(n_points)[None, :]) < min_tsep
    )
    dists[mask] = float("inf")

    ntraj = n_points - trajectory_len + 1
    min_traj = min_tsep * 2 + 2

    if ntraj <= 0:
        raise ValueError(
            f"Not enough data points. Need {-ntraj + 1} additional data points."
        )
    if ntraj < min_traj:
        raise ValueError(
            "Not enough data points. "
            f"Need {min_traj} trajectories, but only {ntraj} could be created."
        )

    nb_idx = np.argmin(dists[:ntraj, :ntraj], axis=1)

    div_traj = np.zeros(trajectory_len, dtype=float)
    for k in range(trajectory_len):
        indices = (np.arange(ntraj) + k, nb_idx + k)
        div_traj_k = dists[indices]
        nonzero = np.where(div_traj_k != 0)
        div_traj[k] = (
            -np.inf if len(nonzero[0]) == 0 else np.mean(np.log(div_traj_k[nonzero]))
        )

    ks = np.arange(trajectory_len)
    finite = np.where(np.isfinite(div_traj))
    ks = ks[finite]
    div_traj = div_traj[finite]

    if len(ks) < 1:
        return -np.inf

    if fit == "RANSAC":
        try:
            from sklearn.linear_model import RANSACRegressor

            model = RANSACRegressor(random_state=0)
            model.fit(ks[fit_offset:, None], div_traj[fit_offset:, None])
            le = model.estimator_.coef_[0][0] / tau
            return le
        except ImportError:
            print(
                "RANSAC fit failed, falling back to polyfit. "
                "Please install scikit-learn for RANSAC."
            )

    poly = np.polyfit(ks[fit_offset:], div_traj[fit_offset:], 1)
    return poly[0] / tau


def get_system_dt(system_name: str) -> float:
    """Infer a default integration step from the system period."""
    steps_per_period = 128.0

    try:
        system_name_without_pp = system_name.split("_pp")[0]
        is_skew = "_" in system_name_without_pp

        if is_skew:
            driver_name, response_name = system_name_without_pp.split("_")
            driver_system = getattr(flows, driver_name)()
            response_system = getattr(flows, response_name)()
            period = max(driver_system.period, response_system.period)
        else:
            system_obj = getattr(flows, system_name_without_pp)()
            period = system_obj.period

        if steps_per_period <= 0 or period <= 0:
            raise ValueError(f"Invalid period ({period}) or steps ({steps_per_period})")

        return period / steps_per_period
    except Exception as err:
        print(f"Unable to fetch dt for '{system_name}': {err}. Falling back to tau=1.0")
        return 1.0


def _kld_batch_gpu(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_mc: int = 256,
    sigma: float = 1.0,
    device: str | None = None,
    chunk_size: int = 64,
    seed: int = 0,
    n_avg: int = 1,
) -> list[float]:
    """Batch KLD via isotropic-GMM Monte-Carlo on GPU with averaging."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    batch_size, n_steps, n_dims = y_true.shape
    log_coeff = -0.5 * n_dims * math.log(2.0 * math.pi * sigma)
    inv_2sigma = -0.5 / sigma
    log_steps = math.log(n_steps)

    accum = np.zeros(batch_size, dtype=np.float64)
    chunk_starts = list(range(0, batch_size, chunk_size))

    for run_i in range(n_avg):
        run_seed = seed + run_i
        generator = torch.Generator(device="cpu").manual_seed(run_seed)
        run_values: list[float] = []

        for start in chunk_starts:
            end = min(start + chunk_size, batch_size)
            current_batch = end - start

            true_t = torch.as_tensor(
                y_true[start:end], dtype=torch.float32, device=device
            )
            pred_t = torch.as_tensor(
                y_pred[start:end], dtype=torch.float32, device=device
            )

            idx = torch.randint(
                0,
                n_steps,
                (current_batch, n_mc),
                generator=generator,
                device="cpu",
            ).to(device)
            centers = torch.gather(true_t, 1, idx.unsqueeze(-1).expand(-1, -1, n_dims))
            noise = torch.randn(
                current_batch,
                n_mc,
                n_dims,
                generator=generator,
                device="cpu",
            ).to(device=device, dtype=torch.float32)
            samples = centers + noise * math.sqrt(sigma)

            sq_p = torch.cdist(samples, true_t).square()
            sq_q = torch.cdist(samples, pred_t).square()

            log_p = torch.logsumexp(log_coeff + inv_2sigma * sq_p, dim=2) - log_steps
            log_q = torch.logsumexp(log_coeff + inv_2sigma * sq_q, dim=2) - log_steps

            kl = (log_p - log_q).mean(dim=1)
            run_values.extend(kl.cpu().numpy().tolist())

        accum += np.array(run_values, dtype=np.float64)

    accum /= n_avg
    return [value if np.isfinite(value) else np.nan for value in accum]


def calculate_le_metrics_raw(
    trues_batch: np.ndarray,
    preds_batch: np.ndarray,
    tau: float,
    trajectory_len: int,
) -> tuple[list[float], list[float]]:
    """Return Lyapunov exponent estimates for true and predicted trajectories."""
    n_samples, horizon, _ = trues_batch.shape
    le_trues: list[float] = []
    le_preds: list[float] = []

    if (horizon - trajectory_len + 1) <= 0:
        return [], []

    for i in range(n_samples):
        try:
            le_true = max_lyapunov_exponent_rosenstein_multivariate(
                trues_batch[i],
                tau=tau,
                trajectory_len=trajectory_len,
                fit="polyfit",
            )
            le_trues.append(le_true)
        except Exception:
            le_trues.append(np.nan)

        try:
            le_pred = max_lyapunov_exponent_rosenstein_multivariate(
                preds_batch[i],
                tau=tau,
                trajectory_len=trajectory_len,
                fit="polyfit",
            )
            le_preds.append(le_pred)
        except Exception:
            le_preds.append(np.nan)

    return le_trues, le_preds


def calculate_psd_metrics_raw(
    trues_batch: np.ndarray,
    preds_batch: np.ndarray,
) -> list[float]:
    """Return spectral log-ratio weighted discrepancy per sample."""
    n_samples, horizon, n_space = trues_batch.shape
    results: list[float] = []
    epsilon = 1e-8

    if horizon >= 16 and n_space >= 16:
        win_t = hann(horizon)
        win_x = hann(n_space)
        window_2d = np.outer(win_t, win_x)
    else:
        window_2d = np.ones((horizon, n_space))

    scale_factor = 1.0 / (np.sum(window_2d**2) + epsilon)

    for i in range(n_samples):
        data_true = trues_batch[i, :, :]
        data_pred = preds_batch[i, :, :]

        fft_true = np.fft.fft2(data_true * window_2d)
        fft_pred = np.fft.fft2(data_pred * window_2d)

        p_true = np.abs(np.fft.fftshift(fft_true)) ** 2 * scale_factor
        p_pred = np.abs(np.fft.fftshift(fft_pred)) ** 2 * scale_factor

        total_energy = np.sum(p_true) + epsilon
        weights = p_true / total_energy
        log_diff = np.abs(np.log((p_pred + epsilon) / (p_true + epsilon)))
        results.append(np.sum(weights * log_diff))

    return results


def calculate_gpdim_raw(y_true: np.ndarray, y_pred: np.ndarray) -> list[float]:
    """Return per-sample differences of Grassberger-Procaccia dimensions."""
    diffs: list[float] = []
    for i in range(y_true.shape[0]):
        try:
            dim_true = gp_dim(y_true[i])
            dim_pred = gp_dim(y_pred[i])
            diffs.append(dim_true - dim_pred)
        except Exception:
            diffs.append(np.nan)
    return diffs


def calculate_kld_raw(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    kld_samples: int = 256,
    device: str | None = None,
    n_avg: int = 5,
) -> list[float]:
    """Compute KLD with GPU batching and robust CPU fallback."""
    try:
        return _kld_batch_gpu(
            y_true,
            y_pred,
            n_mc=kld_samples,
            sigma=1.0,
            device=device,
            n_avg=n_avg,
        )
    except Exception:
        klds: list[float] = []
        for i in range(y_true.shape[0]):
            try:
                kl = estimate_kl_divergence(
                    y_true[i],
                    y_pred[i],
                    n_samples=kld_samples,
                )
                klds.append(kl if kl is not None and np.isfinite(kl) else np.nan)
            except Exception:
                klds.append(np.nan)
        return klds
