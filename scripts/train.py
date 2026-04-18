"""Training entrypoint for ScaleFormer experiments."""

import logging
import os
from functools import partial
from pathlib import Path

import hydra
import torch
import transformers
from gluonts.dataset.common import FileDataset
from gluonts.itertools import Filter
from omegaconf import OmegaConf
from scaleformer.augmentations import (
    RandomAffineTransform,
    RandomConvexCombinationTransform,
    RandomDimSelectionTransform,
    RandomFourierSeries,
    RandomPhaseSurrogate,
    RandomTakensEmbedding,
    StandardizeTransform,
)
from scaleformer.scaleformer.dataset import TimeSeriesDataset
from scaleformer.utils import (
    ensure_contiguous,
    get_next_path,
    has_enough_observations,
    is_main_process,
    load_patchtst_model,
    log_on_main,
    save_training_info,
)
from transformers import (
    Trainer,
    TrainingArguments,
)

import torch._dynamo

torch._dynamo.config.cache_size_limit = 256
torch._logging.set_logs(dynamo=logging.ERROR)

logger = logging.getLogger(__name__)


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg):
    """Build datasets, configure model/trainer and run training."""
    # set floating point precision
    use_tf32 = cfg.train.tf32
    if use_tf32 and not (
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    ):
        # TF32 floating point format is available only on NVIDIA GPUs
        # with compute capability 8 and above. See link for details.
        # https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability-8-x
        log_on_main(
            "TF32 format is only available on devices with compute capability >= 8. "
            "Setting tf32 to False.",
            logger,
        )
        use_tf32 = False

    # set random seed
    log_on_main(f"Using SEED: {cfg.train.seed}", logger)
    transformers.set_seed(seed=cfg.train.seed)

    # get train data paths
    train_data_dir_lst = cfg.train_data_dirs
    train_data_paths = []
    for train_data_dir in train_data_dir_lst:
        train_data_dir = os.path.expandvars(train_data_dir)
        train_data_paths.extend(
            filter(lambda file: file.is_file(), Path(train_data_dir).rglob("*"))
        )
    # create a new output directory to save results
    output_dir = get_next_path(
        cfg.run_name if cfg.run_name else "run",
        base_dir=Path(cfg.train.output_dir),
        file_type="",
        overwrite=cfg.train.resume_from_checkpoint is not None,
    )

    log_on_main(f"Logging dir: {output_dir}", logger)
    log_on_main(
        f"Loading and filtering {len(train_data_paths)} datasets for training from directories: {train_data_dir_lst}",
        logger,
    )

    train_datasets = [
        Filter(
            partial(
                has_enough_observations,
                min_length=cfg.min_past + cfg.scaleformer.prediction_length,
                max_missing_prop=cfg.max_missing_prop,
            ),
            FileDataset(path=Path(data_path), freq="h", one_dim_target=False),  # type: ignore
        )
        for data_path in train_data_paths
    ]

    # set probabilities (how we weight draws from each data file)
    if isinstance(cfg.probability, float):
        probability = cfg.probability
    elif cfg.probability is None:
        probability = [1.0 / len(train_datasets)] * len(train_datasets)
    assert isinstance(probability, list)
    assert len(train_datasets) == len(probability)

    # adapt number of workers to the number of datasets if there are more workers than datasets
    dataloader_num_workers = cfg.train.dataloader_num_workers
    if dataloader_num_workers > len(train_datasets):
        log_on_main(
            f"Setting the number of data loader workers to {len(train_datasets)}, "
            f"instead of {dataloader_num_workers}.",
            logger,
        )
        dataloader_num_workers = len(train_datasets)

    augmentations = [
        RandomTakensEmbedding(
            lag_range=cfg.augmentations.lag_range,
            random_seed=cfg.train.seed,
        ),
        RandomConvexCombinationTransform(
            alpha=1.0,
            random_seed=cfg.train.seed,
            dim_range=cfg.augmentations.dim_range,
        ),
        RandomAffineTransform(
            dim_range=cfg.augmentations.dim_range,
            scale=1.0,
            random_seed=cfg.train.seed,
        ),
        RandomPhaseSurrogate(
            cutoff=cfg.augmentations.phase_surrogate_cutoff,
            random_seed=cfg.train.seed,
        ),
        RandomFourierSeries(
            max_wavenumber=cfg.augmentations.max_wavenumber,
            max_amp=cfg.augmentations.max_amp,
            mode_range=cfg.augmentations.mode_range,
            random_seed=cfg.train.seed,
        ),
    ]
    if cfg.augmentations.probabilities is None:
        cfg.augmentations.probabilities = [1.0 / len(augmentations)] * len(
            augmentations
        )
    else:  # ensure probabilities sum to 1
        cfg.augmentations.probabilities = [
            prob / sum(cfg.augmentations.probabilities)
            for prob in cfg.augmentations.probabilities
        ]

    log_on_main(
        f"Using augmentations: {[aug for aug, prob in zip(augmentations, cfg.augmentations.probabilities) if prob > 0.0]}",
        logger,
    )

    transforms: list = [
        StandardizeTransform(),
        RandomDimSelectionTransform(num_dims=cfg.fixed_dim),
    ]

    shuffled_train_dataset = TimeSeriesDataset(
        datasets=train_datasets,
        probabilities=probability,
        context_length=cfg.scaleformer.context_length,
        prediction_length=cfg.scaleformer.prediction_length,
        mode="train",
        augmentations=augmentations,
        augmentation_probabilities=cfg.augmentations.probabilities,
        augmentation_rate=cfg.augmentations.augmentation_rate,
        transforms=transforms,
    ).shuffle(shuffle_buffer_length=cfg.shuffle_buffer_length)

    log_on_main("Initializing model", logger)

    checkpoint_path = cfg.scaleformer.get(
        "checkpoint_path", cfg.scaleformer.get("pretrained_pft_path", None)
    )

    model = load_patchtst_model(
        model_config=OmegaConf.to_container(cfg.scaleformer, resolve=True),
        checkpoint_path=checkpoint_path,
    )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_on_main(f"Total trainable parameters: {trainable_params:,}", logger)
    # Define training args
    training_args = TrainingArguments(
        run_name=cfg.run_name,
        output_dir=str(output_dir),
        per_device_train_batch_size=cfg.train.per_device_train_batch_size,
        learning_rate=cfg.train.learning_rate,
        max_grad_norm=cfg.train.max_grad_norm,
        weight_decay=cfg.train.weight_decay,
        optim=cfg.train.optim,
        log_on_each_node=False,
        logging_dir=str(output_dir / "logs"),
        logging_strategy="steps",
        logging_steps=cfg.train.log_steps,
        save_strategy="steps",
        save_steps=cfg.train.save_steps,
        report_to=["tensorboard"],
        max_steps=cfg.train.max_steps,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        dataloader_num_workers=dataloader_num_workers,
        dataloader_prefetch_factor=cfg.train.dataloader_prefetch_factor,
        tf32=use_tf32,  # remove this if not using Ampere GPUs (e.g., A100)
        bf16=cfg.train.bf16,
        torch_compile=cfg.train.torch_compile,
        ddp_find_unused_parameters=cfg.train.ddp_find_unused_parameters,
        ddp_backend=cfg.train.ddp_backend,
        remove_unused_columns=cfg.train.remove_unused_columns,
        seed=cfg.train.seed,
        resume_from_checkpoint=cfg.train.resume_from_checkpoint,
    )

    # check if model weights are contiguous in memory; if not, make them contiguous tensors.
    # This speeds up training and allows checkpoint saving by transformers Trainer
    ensure_contiguous(model)

    trainer = Trainer(
        model=model, args=training_args, train_dataset=shuffled_train_dataset
    )

    log_on_main("Training", logger)
    trainer.train(
        resume_from_checkpoint=cfg.train.resume_from_checkpoint
    )  # Transformers trainer will save model checkpoints automatically

    # save final model checkpoint and training info locally
    if is_main_process():
        model.save_pretrained(output_dir / "checkpoint-final")  # type: ignore
        save_training_info(
            output_dir / "checkpoint-final",
            model_config=OmegaConf.to_container(cfg.scaleformer, resolve=True),  # type: ignore
            train_config=OmegaConf.to_container(cfg.train, resolve=True),  # type: ignore
            all_config=OmegaConf.to_container(cfg, resolve=True),  # type: ignore
        )


if __name__ == "__main__":
    main()
