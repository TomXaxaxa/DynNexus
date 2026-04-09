"""Inference pipeline wrappers for ScaleFormer."""

import warnings
from dataclasses import dataclass
from typing import Callable

import torch
from scaleformer.scaleformer.scaleformer import (
    PatchTSTForPrediction,
)
from scaleformer.utils import left_pad_and_stack_multivariate


class FixedSubsetChannelSampler:
    """
    Generate samples of subsets of channels from a context tensor or a list of context tensors
    """

    def __init__(self, num_channels: int, num_samples: int) -> None:
        self.num_channels = num_channels
        self.num_samples = num_samples
        self._inds = None

    @property
    def inds(self) -> list[torch.Tensor]:
        if self._inds is None:
            raise ValueError("Indices not sampled yet")
        return self._inds

    def _sample_indices(self, C: int) -> torch.Tensor:
        """
        This method is heuristic.

        Sliding window of size C over a random permutation of the channels -
        guarantees that each channel is sampled at least once.

        For a given C and num_channels, the number of samples will be
        equal to: (C - num_channels + 1)
        """
        assert C >= self.num_channels, "cannot sample more channels than available"
        idx_samples = [
            torch.randperm(C).unfold(0, self.num_channels, 1)
            for _ in range(self.num_samples)
        ]
        return torch.cat(idx_samples, dim=0)

    def __call__(
        self, context: torch.Tensor | list[torch.Tensor], resample_inds: bool = True
    ) -> torch.Tensor:
        """
        Generates samples of subsets of channels from a context tensor or a list of context tensors

        The context tensors are expected to have shape:
        (bs, context_length, C) or (context_length, C)

        For each context tensor with channel dim C, the number of samples produced will
        be (C - num_channels + 1), each of shape (bs, context_length, num_channels).
        Fixing the channel dim to num_channels allows for aggregation of the context
        tensor list into a new tensor of shape (bs, S, context_length, num_channels)
        where S is the sum of all samples from all context tensors.

        Args:
            context: A tensor or a list of tensors
            resample_inds: If True, resample the indices for each context tensor

        Returns:
            A tensor with shape (bs, S, context_length, num_channels)
            where S is the sum of all samples from all context tensors
        """
        if not isinstance(context, list):
            context = [context]

        # only subsample context tensors with more than num_channels
        valid_contexts = [c.shape[-1] > self.num_channels for c in context]
        valid_inds = [sum(valid_contexts[:i]) for i in range(len(valid_contexts))]

        # ensure each context tensor has a batch dimension
        for i in range(len(context)):
            if context[i].ndim == 2:
                context[i] = context[i].unsqueeze(0)

        channels = [
            context[i].shape[-1] for i, valid in enumerate(valid_contexts) if valid
        ]

        if (self._inds is None or resample_inds) and sum(valid_contexts) > 0:
            self._inds = [self._sample_indices(c) for c in channels]

        # shape: (batch_size, S, context_length, num_channels)
        selected = torch.cat(
            [
                (
                    c[..., self.inds[i]]  # subsample only if valid
                    if valid
                    else c.unsqueeze(-2)
                )
                for c, valid, i in zip(context, valid_contexts, valid_inds)
            ],
            dim=1,
        ).transpose(2, 1)
        assert selected.ndim == 4

        # shape: (batch_size * S, context_length, num_channels)
        return selected.reshape(-1, *selected.shape[2:])


@dataclass
class PatchTSTPipeline:
    """
    PatchTST pipeline for inference
    """

    model: PatchTSTForPrediction

    @property
    def device(self) -> torch.device:
        return self.model.device

    @classmethod
    def from_pretrained(cls, checkpoint_path: str, **kwargs):
        """
        Load a pretrained model from a path and move it to the specified device.
        """
        model = PatchTSTForPrediction.from_pretrained(checkpoint_path, **kwargs)
        return cls(model=model)

    def _prepare_and_validate_context(
        self, context: torch.Tensor | list[torch.Tensor]
    ) -> torch.Tensor:
        if isinstance(context, list):
            assert len(set(c.shape[-1] for c in context)) == 1, (
                "All contexts must have the same number of channels"
                "Use a channel sampler to subsample a fixed number of channels"
            )
            context = left_pad_and_stack_multivariate(context)
        assert isinstance(context, torch.Tensor)
        if context.ndim == 1:
            context = context.view(1, -1, 1)
        if context.ndim == 2:
            context = context.unsqueeze(0)
        assert context.ndim == 3

        return context.to(self.device)

    @torch.no_grad()
    def predict(
        self,
        context: torch.Tensor | list[torch.Tensor],
        prediction_length: int,
        channel_sampler: (
            Callable[[torch.Tensor | list[torch.Tensor]], torch.Tensor] | None
        ) = None,
        sliding_context: bool = False,
        verbose: bool = True,
    ) -> torch.Tensor:
        """
        Generate an autoregressive forecast for a given context timeseries

        Parameters
        ----------
        context
            Input series. This is either a 1D tensor, or a list
            of 1D tensors, or a 2D tensor whose first dimension
            is sequence length. In the latter case, use left-padding with
            ``torch.nan`` to align series of different lengths.
        prediction_length
            Time steps to predict. Defaults to what specified
            in ``self.model.config``.
        channel_sampler
            A callable that takes a context tensor or a list of context
            tensors and returns a tensor of shape:
                [bs x num_samples x context_length x num_channels]
            where num_samples is specific to the channel_sampler implementation.
            This is intended to be used for subsampling channels e.g. when the
            model was trained with a fixed dim embedding.
        sliding_context
            If True, the context window will be slid over the time series, otherwise
            the context window will be accumulated and grows in memory.

        Returns
        -------
        samples
            Tensor of sample forecasts, of shape
            [bs x num_samples x prediction_length x num_channels]
        """
        # context_tensor: [bs x context_length x num_channels]
        if channel_sampler is not None:
            context_tensor = channel_sampler(context)
        else:
            context_tensor = self._prepare_and_validate_context(context=context)

        if prediction_length > self.model.config.prediction_length and verbose:
            warnings.warn(
                f"We recommend keeping prediction length <= {self.model.config.prediction_length}. "
                "The quality of longer predictions may degrade since the model is not optimized for it. "
            )

        predictions = []
        remaining = prediction_length

        while remaining > 0:
            outputs = self.model.generate(context_tensor)

            # prediction: [bs x num_samples x forecast_len x num_channels]
            prediction = outputs.sequences  # type: ignore

            predictions.append(prediction)
            remaining -= prediction.shape[2]

            if remaining <= 0:
                break

            # need to contract over the num_samples dimension, use median
            context_tensor = torch.cat(
                [context_tensor, prediction.median(dim=1).values], dim=1
            )

            # dont grow the context window, only keep the most recent context_length
            if sliding_context:
                context_tensor = context_tensor[
                    :, -self.model.config.context_length :, :
                ]

        # shape: [bs x num_samples x prediction_length x num_channels]
        predictions = torch.cat(predictions, dim=2)

        return predictions
