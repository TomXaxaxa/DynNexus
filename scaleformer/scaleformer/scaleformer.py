"""ScaleFormer model definition and supporting building blocks."""

try:
    from flash_attn import flash_attn_func

    _flash_attn_available = True
except ImportError:
    _flash_attn_available = False
from typing import Optional, Tuple, Union
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings
import kymatio.torch as kymatio
warnings.filterwarnings("ignore", message="Signal support is too small to avoid border effects")

from transformers import PatchTSTConfig, PatchTSTPreTrainedModel
from transformers.models.patchtst.modeling_patchtst import (
    ACT2CLS,
    PatchTSTForPredictionOutput,
    PatchTSTScaler,
    SamplePatchTSTOutput,
)
from .modules import (
    DyT,
    PatchTSTKernelEmbedding,
    PatchTSTPatchify,
    PatchTSTRMSNorm,
    apply_p_rope_to_qk,
)


class CnnExtractorWithLayerNorm(nn.Module):
    def __init__(self, n_coeffs):
        super().__init__()
        self.conv1 = nn.Conv1d(n_coeffs, 64, kernel_size=7, padding=3)
        self.ln1 = nn.LayerNorm(64)
        self.gelu1 = nn.GELU()
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.ln2 = nn.LayerNorm(128)
        self.gelu2 = nn.GELU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv1(x)
        x = x.permute(0, 2, 1)
        x = self.ln1(x)
        x = x.permute(0, 2, 1)
        x = self.gelu1(x)
        x = self.conv2(x)
        x = x.permute(0, 2, 1)
        x = self.ln2(x)
        x = x.permute(0, 2, 1)
        x = self.gelu2(x)
        x = self.pool(x)
        x = self.flatten(x)
        return x


class WaveletAnalyzer(nn.Module):
    def __init__(self, input_timesteps, feature_dim, J=8, Q=8):
        super().__init__()
        self.scattering = kymatio.Scattering1D(J=J, shape=(input_timesteps,), Q=Q)
        with torch.no_grad():
            dummy_input = torch.randn(1, input_timesteps)
            n_coeffs = self.scattering(dummy_input).shape[1]
        self.cnn_extractor = CnnExtractorWithLayerNorm(n_coeffs)
        self.final_mlp = nn.Linear(128, feature_dim)

    def forward(self, x):
        B, V, T = x.shape
        x_reshaped = x.reshape(B * V, T)
        scattering_coeffs = self.scattering(x_reshaped.contiguous())
        cnn_features = self.cnn_extractor(scattering_coeffs)
        features = self.final_mlp(cnn_features)
        features_reshaped = features.view(B, V, -1)
        final_embedding = features_reshaped.mean(dim=1)
        stabilized_embedding = torch.sign(final_embedding) * torch.log(
            torch.abs(final_embedding) + 1
        )
        return stabilized_embedding


def _compute_squared_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute pairwise squared Euclidean distances."""
    if x.dim() == 3:
        x = x.squeeze(1)
    if y.dim() == 3:
        y = y.squeeze(1)
    x = x.unsqueeze(1)
    y = y.unsqueeze(0)
    return torch.sum((x - y) ** 2, dim=-1)


def rational_quadratic_kernel(
    x: torch.Tensor,
    y: torch.Tensor,
    sigma_list: list[float],
    **kwargs,
) -> torch.Tensor:
    """Evaluate a rational-quadratic kernel mixture."""
    squared_dist = _compute_squared_dist(x, y)
    sigma = torch.tensor(sigma_list, device=x.device).view(-1, 1, 1)
    sigma_squared = sigma**2
    kernel_val = sigma_squared / (sigma_squared + squared_dist.unsqueeze(0))
    return kernel_val.sum(dim=0)


def compute_mmd(
    x: torch.Tensor,
    y: torch.Tensor,
    mean_value: torch.Tensor,
    variance_value: torch.Tensor,
    kernel_params: dict,
) -> torch.Tensor:
    """Compute Maximum Mean Discrepancy between two trajectory batches."""
    if x.dim() == 3:
        x = x.squeeze(1)
    if y.dim() == 3:
        y = y.squeeze(1)
    if mean_value.dim() == 1:
        mean_value = mean_value.unsqueeze(0)
    if variance_value.dim() == 1:
        variance_value = variance_value.unsqueeze(0)
    x = (x - mean_value) / torch.sqrt(variance_value + 1e-6)
    y = (y - mean_value) / torch.sqrt(variance_value + 1e-6)
    xx = rational_quadratic_kernel(x, x, **kernel_params)
    yy = rational_quadratic_kernel(y, y, **kernel_params)
    xy = rational_quadratic_kernel(x, y, **kernel_params)
    B = x.size(0)
    if B > 1:
        term1 = (xx.sum() - xx.diag().sum()) / (B * (B - 1))
        term2 = (yy.sum() - yy.diag().sum()) / (B * (B - 1))
        term3 = xy.sum() / (B * B)
    else:
        term1, term2, term3 = 0, 0, 0
    return (term1 + term2 - 2 * term3).clamp(min=0)


def conditional_mmd_multi_step(
    input_traj,
    true_traj,
    pred_traj,
    mean,
    variance,
    kernel_params: dict,
    steps=None,
) -> torch.Tensor | float:
    """Average MMD across selected prediction steps."""
    H = pred_traj.shape[1]
    if steps is None:
        steps = range(H)
    mmd_sum = 0.0
    for t in steps:
        true_evolved = true_traj[:, t, :]
        model_evolved = pred_traj[:, t, :]
        mmd_sum += compute_mmd(
            true_evolved,
            model_evolved,
            mean,
            variance,
            kernel_params=kernel_params,
        )
    return mmd_sum / len(steps) if len(steps) > 0 else 0.0


class Expert(nn.Module):
    def __init__(self, d_model: int, ffn_dim: int, config: PatchTSTConfig):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(d_model, ffn_dim, bias=config.bias),
            ACT2CLS[config.activation_function](),
            nn.Dropout(config.ff_dropout) if config.ff_dropout > 0 else nn.Identity(),
            nn.Linear(ffn_dim, d_model, bias=config.bias),
        )

    def forward(self, x):
        return self.ff(x)


class NaiveMoE(nn.Module):
    def __init__(
        self,
        d_model: int,
        ffn_dim: int,
        num_experts: int,
        top_k: int,
        config: PatchTSTConfig,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(self.d_model, self.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [Expert(d_model, ffn_dim, config) for _ in range(self.num_experts)]
        )

    def forward(self, x: torch.Tensor):
        original_shape = x.shape
        x_reshaped = x.reshape(-1, self.d_model)
        num_tokens, _ = x_reshaped.shape
        gate_logits = self.gate(x_reshaped)
        router_probs = F.softmax(gate_logits, dim=-1)
        tokens_per_expert_prob = router_probs.mean(dim=0)
        load_balance_loss = self.num_experts * torch.sum(
            tokens_per_expert_prob * tokens_per_expert_prob
        )
        top_k_weights, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_weights = top_k_weights / torch.sum(top_k_weights, dim=-1, keepdim=True)
        flat_top_k_indices = top_k_indices.flatten()
        perm = torch.argsort(flat_top_k_indices)
        perm_flat_top_k_indices = flat_top_k_indices[perm]
        counts = torch.bincount(perm_flat_top_k_indices, minlength=self.num_experts)
        starts = torch.cat((torch.tensor([0], device=x.device), counts.cumsum(0)[:-1]))
        token_indices = torch.arange(num_tokens, device=x.device).repeat_interleave(
            self.top_k
        )
        perm_token_indices = token_indices[perm]
        perm_inputs = x_reshaped[perm_token_indices]
        perm_outputs = torch.zeros_like(perm_inputs)
        for i in range(self.num_experts):
            start, end = starts[i], starts[i] + counts[i]
            if start < end:
                expert_input = perm_inputs[start:end]
                expert_output = self.experts[i](expert_input)
                perm_outputs[start:end] = expert_output
            else:
                dummy_input = torch.zeros((1, self.d_model), dtype=x_reshaped.dtype, device=x.device)
                dummy_output = self.experts[i](dummy_input)
                load_balance_loss = load_balance_loss + dummy_output.sum() * 0.0
        inv_perm = torch.argsort(perm)
        unperm_outputs = perm_outputs[inv_perm]
        unperm_outputs = unperm_outputs * top_k_weights.flatten().unsqueeze(-1)
        final_output_reshaped = torch.zeros_like(x_reshaped).index_add_(
            0, token_indices, unperm_outputs
        )
        return final_output_reshaped.reshape(original_shape), load_balance_loss


class ConvNeXtBlock1D(nn.Module):
    def __init__(
        self, dim, path_dropout=0.0, layer_scale_init_value=1e-6, norm_eps=1e-6
    ):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=norm_eps)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.weight = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = (
            nn.Dropout(path_dropout) if path_dropout > 0.0 else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input = x
        x = x.permute(0, 2, 1)
        x = self.dwconv(x)
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.weight is not None:
            x = self.weight * x
        x = input + self.drop_path(x)
        return x


class PatchMerging(nn.Module):
    def __init__(self, dim: int, norm_eps=1e-6):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(2 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(2 * dim, eps=norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_patches, dim = x.shape
        if num_patches % 2 != 0:
            x = nn.functional.pad(x, (0, 0, 0, 1))
            num_patches += 1
        x = x.reshape(batch_size, num_patches // 2, 2, dim)
        x = x.flatten(2)
        x = self.reduction(x)
        x = self.norm(x)
        return x


class PatchExpansion(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_patches, dim = x.shape
        x = self.expand(x)
        x = x.view(batch_size, num_patches, 2, dim // 2)
        x = x.view(batch_size, -1, dim // 2)
        return x


class PatchTSTEmbedding(nn.Module):
    def __init__(self, config: PatchTSTConfig):
        super().__init__()
        self.input_embedding = nn.Linear(config.patch_length, config.d_model)

    def forward(self, patch_input: torch.Tensor):
        embeddings = self.input_embedding(patch_input)
        return embeddings


class PatchTSTRopeAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        use_rope: bool = True,
        max_wavelength: int = 10000,
        rope_percent: float = 0.5,
        config: Optional[PatchTSTConfig] = None,
        use_flash_attention: bool = True,
    ):
        super().__init__()
        self.embed_dim = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = d_model // num_heads
        self.max_wavelength = max_wavelength
        self.rope_percent = rope_percent
        self.use_rope = use_rope
        self.config = config
        self.use_flash_attention = use_flash_attention and _flash_attn_available
        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def get_seq_pos(self, seq_len, device, dtype, offset=0):
        return torch.arange(seq_len, device=device, dtype=dtype) + offset

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        linear_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        if is_cross_attention and past_key_value is not None:
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
        if self.is_decoder:
            past_key_value = (key_states, value_states)
        query_states = self._shape(query_states, tgt_len, bsz)
        key_states = self._shape(key_states, -1, bsz)
        value_states = self._shape(value_states, -1, bsz)
        src_len = key_states.size(2)
        if self.use_rope:
            q_for_rope = query_states.reshape(-1, tgt_len, self.head_dim)
            k_for_rope = key_states.reshape(-1, src_len, self.head_dim)
            position_ids = self.get_seq_pos(
                src_len, key_states.device, key_states.dtype
            )
            k_for_rope, q_for_rope = apply_p_rope_to_qk(
                k_for_rope,
                q_for_rope,
                position_ids,
                self.head_dim,
                self.max_wavelength,
                self.rope_percent,
            )
            query_states = q_for_rope.view(bsz, self.num_heads, tgt_len, self.head_dim)
            key_states = k_for_rope.view(bsz, self.num_heads, src_len, self.head_dim)
        can_use_flash_attn = (
            self.use_flash_attention
            and hidden_states.is_cuda
            and attention_mask is None
            and hidden_states.dtype in [torch.float16, torch.bfloat16]
        )
        if can_use_flash_attn:
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)
            attn_output = flash_attn_func(
                query_states,
                key_states,
                value_states,
                dropout_p=self.dropout if self.training else 0.0,
                causal=self.is_causal,
            )
            attn_weights_reshaped = None
            attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        else:
            query_states = query_states * self.scaling
            attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1))
            if attn_weights.size() != (bsz, self.num_heads, tgt_len, src_len):
                raise ValueError(
                    f"Attention weights should be of size {(bsz, self.num_heads, tgt_len, src_len)}, but is"
                    f" {attn_weights.size()}"
                )
            if attention_mask is not None:
                if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                    raise ValueError(
                        f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                    )
                attn_weights = attn_weights + attention_mask.to(attn_weights.device)
            if not linear_attn:
                attn_weights = nn.functional.softmax(attn_weights, dim=-1)
            if layer_head_mask is not None:
                if layer_head_mask.size() != (self.num_heads,):
                    raise ValueError(
                        f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                        f" {layer_head_mask.size()}"
                    )
                attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights
            if output_attentions:
                attn_weights_reshaped = attn_weights
            else:
                attn_weights_reshaped = None
            attn_probs = nn.functional.dropout(
                attn_weights, p=self.dropout, training=self.training
            )
            attn_output = torch.matmul(attn_probs, value_states)
            if attn_output.size() != (bsz, self.num_heads, tgt_len, self.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                    f" {attn_output.size()}"
                )
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights_reshaped, past_key_value


class PatchTSTEncoderLayerWithRope(nn.Module):
    def __init__(self, config: PatchTSTConfig, d_model: int, num_heads: int):
        super().__init__()
        self.use_moe = bool(getattr(config, "use_moe", True))
        self.num_experts = int(getattr(config, "moe_num_experts", 8))
        self.top_k = int(getattr(config, "moe_top_k", 2))
        if self.top_k < 1 or self.top_k > self.num_experts:
            raise ValueError(
                f"Invalid moe_top_k={self.top_k}, expected 1 <= moe_top_k <= moe_num_experts({self.num_experts})."
            )
        self.channel_attention = config.channel_attention
        self.temporal_self_attn = PatchTSTRopeAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=config.attention_dropout,
            use_rope=True,
            max_wavelength=config.max_wavelength,
            rope_percent=config.rope_percent,
        )
        if self.channel_attention:
            self.channel_self_attn = PatchTSTRopeAttention(
                d_model=d_model,
                num_heads=num_heads,
                dropout=config.attention_dropout,
                use_rope=config.channel_rope,
                max_wavelength=config.max_wavelength,
                rope_percent=config.rope_percent,
            )
        self.dropout_path1 = (
            nn.Dropout(config.path_dropout)
            if config.path_dropout > 0
            else nn.Identity()
        )
        if config.norm_type == "rmsnorm":
            self.norm_sublayer1 = PatchTSTRMSNorm(d_model, config.norm_eps)
        elif config.norm_type == "layernorm":
            self.norm_sublayer1 = nn.LayerNorm(d_model, eps=config.norm_eps)
        elif config.norm_type == "dyt":
            self.norm_sublayer1 = DyT(d_model)
        else:
            raise ValueError(f"{config.norm_type} is not a supported norm layer type.")
        if self.channel_attention:
            self.dropout_path2 = (
                nn.Dropout(config.path_dropout)
                if config.path_dropout > 0
                else nn.Identity()
            )
            if config.norm_type == "rmsnorm":
                self.norm_sublayer2 = PatchTSTRMSNorm(d_model, config.norm_eps)
            elif config.norm_type == "layernorm":
                self.norm_sublayer2 = nn.LayerNorm(d_model, eps=config.norm_eps)
            elif config.norm_type == "dyt":
                self.norm_sublayer2 = DyT(d_model)
            else:
                raise ValueError(
                    f"{config.norm_type} is not a supported norm layer type."
                )
        ffn_dim = d_model * 4
        if self.use_moe:
            self.ff = NaiveMoE(
                d_model=d_model,
                ffn_dim=ffn_dim,
                num_experts=self.num_experts,
                top_k=self.top_k,
                config=config,
            )
        else:
            self.ff = nn.Sequential(
                nn.Linear(d_model, ffn_dim, bias=config.bias),
                ACT2CLS[config.activation_function](),
                (
                    nn.Dropout(config.ff_dropout)
                    if config.ff_dropout > 0
                    else nn.Identity()
                ),
                nn.Linear(ffn_dim, d_model, bias=config.bias),
            )
        self.dropout_path3 = (
            nn.Dropout(config.path_dropout)
            if config.path_dropout > 0
            else nn.Identity()
        )
        if config.norm_type == "rmsnorm":
            self.norm_sublayer3 = PatchTSTRMSNorm(d_model, config.norm_eps)
        elif config.norm_type == "layernorm":
            self.norm_sublayer3 = nn.LayerNorm(d_model, eps=config.norm_eps)
        elif config.norm_type == "dyt":
            self.norm_sublayer3 = DyT(d_model)
        else:
            raise ValueError(f"{config.norm_type} is not a supported norm layer type.")
        self.pre_norm = config.pre_norm

    def forward(
        self,
        hidden_state: torch.Tensor,
        output_attentions: Optional[bool] = None,
        channel_attention_mask: Optional[torch.Tensor] = None,
        linear_attn: bool = False,
    ):
        batch_size, num_input_channels, sequence_length, d_model = hidden_state.shape
        hidden_state = hidden_state.view(
            batch_size * num_input_channels, sequence_length, d_model
        )
        if self.pre_norm:
            attn_output, attn_weights, _ = self.temporal_self_attn(
                hidden_states=self.norm_sublayer1(hidden_state),
                output_attentions=output_attentions,
            )
            hidden_state = hidden_state + self.dropout_path1(attn_output)
        else:
            attn_output, attn_weights, _ = self.temporal_self_attn(
                hidden_states=hidden_state,
                output_attentions=output_attentions,
                linear_attn=linear_attn,
            )
            hidden_state = self.norm_sublayer1(
                hidden_state + self.dropout_path1(attn_output)
            )
        hidden_state = hidden_state.reshape(
            batch_size, num_input_channels, sequence_length, d_model
        )
        if self.channel_attention:
            hidden_state = hidden_state.transpose(2, 1).contiguous()
            hidden_state = hidden_state.view(
                batch_size * sequence_length, num_input_channels, d_model
            )
            if self.pre_norm:
                attn_output, channel_attn_weights, _ = self.channel_self_attn(
                    hidden_states=self.norm_sublayer2(hidden_state),
                    output_attentions=output_attentions,
                    attention_mask=channel_attention_mask,
                )
                hidden_state = hidden_state + self.dropout_path2(attn_output)
            else:
                attn_output, channel_attn_weights, _ = self.channel_self_attn(
                    hidden_states=hidden_state,
                    output_attentions=output_attentions,
                    attention_mask=channel_attention_mask,
                    linear_attn=linear_attn,
                )
                hidden_state = self.norm_sublayer2(
                    hidden_state + self.dropout_path2(attn_output)
                )
            hidden_state = hidden_state.reshape(
                batch_size, sequence_length, num_input_channels, d_model
            )
            hidden_state = hidden_state.transpose(1, 2).contiguous()
        hidden_state = hidden_state.view(
            batch_size * num_input_channels, sequence_length, d_model
        )
        moe_loss = torch.tensor(0.0, device=hidden_state.device)
        if self.pre_norm:
            normalized_hidden_state = self.norm_sublayer3(hidden_state)
            if self.use_moe:
                ff_output, moe_loss = self.ff(normalized_hidden_state)
            else:
                ff_output = self.ff(normalized_hidden_state)
            hidden_state = hidden_state + self.dropout_path3(ff_output)
        else:
            if self.use_moe:
                ff_output, moe_loss = self.ff(hidden_state)
            else:
                ff_output = self.ff(hidden_state)
            hidden_state = self.norm_sublayer3(
                hidden_state + self.dropout_path3(ff_output)
            )
        hidden_state = hidden_state.reshape(
            batch_size, num_input_channels, sequence_length, d_model
        )
        outputs = (hidden_state,)
        if output_attentions:
            outputs += (
                (attn_weights, channel_attn_weights)
                if self.channel_attention
                else (attn_weights,)
            )
        outputs += (moe_loss,)
        return outputs


class PatchTSTUNetEncoder(nn.Module):
    def __init__(self, config: PatchTSTConfig, depths: list, num_heads_list: list):
        super().__init__()
        self.config = config
        self.stages = nn.ModuleList()
        current_dim = config.d_model
        for i, depth in enumerate(depths):
            stage_layers = nn.ModuleList(
                [
                    PatchTSTEncoderLayerWithRope(
                        config, d_model=current_dim, num_heads=num_heads_list[i]
                    )
                    for _ in range(depth)
                ]
            )
            downsample = (
                PatchMerging(dim=current_dim, norm_eps=config.norm_eps)
                if i < len(depths) - 1
                else None
            )
            self.stages.append(
                nn.ModuleDict({"layers": stage_layers, "downsample": downsample})
            )
            if downsample:
                current_dim *= 2

    def forward(
        self,
        hidden_state,
        output_attentions=None,
        channel_attention_mask=None,
        linear_attn=False,
    ):
        skip_connections = []
        total_moe_loss = 0.0
        for stage in self.stages:
            skip_connections.append(hidden_state)
            for layer in stage["layers"]:
                layer_outputs = layer(
                    hidden_state,
                    output_attentions=output_attentions,
                    channel_attention_mask=channel_attention_mask,
                    linear_attn=linear_attn,
                )
                hidden_state = layer_outputs[0]
                total_moe_loss += layer_outputs[-1]
            if stage["downsample"] is not None:
                batch_size, num_channels, num_patches, d_model = hidden_state.shape
                hidden_state_reshaped = hidden_state.view(
                    batch_size * num_channels, num_patches, d_model
                )
                hidden_state_downsampled = stage["downsample"](hidden_state_reshaped)
                num_patches = hidden_state_downsampled.shape[1]
                d_model = hidden_state_downsampled.shape[2]
                hidden_state = hidden_state_downsampled.view(
                    batch_size, num_channels, num_patches, d_model
                )
        return hidden_state, skip_connections, total_moe_loss


class PatchTSTUNetDecoder(nn.Module):
    def __init__(
        self,
        config: PatchTSTConfig,
        depths: list,
        skip_connections_depths: list,
        num_heads_list: list,
    ):
        super().__init__()
        self.config = config
        self.stages = nn.ModuleList()
        reversed_depths = list(reversed(depths))
        reversed_num_heads = list(reversed(num_heads_list))
        encoder_bottleneck_dim = config.d_model
        current_decoder_dim = encoder_bottleneck_dim
        for i, depth in enumerate(reversed_depths):
            target_dim = encoder_bottleneck_dim // (2**i)
            current_num_heads = reversed_num_heads[i]
            upsample = PatchExpansion(dim=current_decoder_dim) if i > 0 else None
            skip_processor_dim = target_dim
            skip_connection_processor = nn.ModuleList(
                [
                    ConvNeXtBlock1D(skip_processor_dim, norm_eps=config.norm_eps)
                    for _ in range(skip_connections_depths[len(depths) - 1 - i])
                ]
            )
            stage_layers = nn.ModuleList(
                [
                    PatchTSTEncoderLayerWithRope(
                        config, d_model=target_dim, num_heads=current_num_heads
                    )
                    for _ in range(depth)
                ]
            )
            self.stages.append(
                nn.ModuleDict(
                    {
                        "upsample": upsample,
                        "skip_processor": skip_connection_processor,
                        "layers": stage_layers,
                    }
                )
            )
            current_decoder_dim = target_dim

    def forward(
        self,
        hidden_state,
        skip_connections,
        output_attentions=None,
        channel_attention_mask=None,
        linear_attn=False,
    ):
        reversed_skips = list(reversed(skip_connections))
        total_moe_loss = 0.0
        all_stage_outputs = []
        for i, stage in enumerate(self.stages):
            if stage["upsample"] is not None:
                batch_size, num_channels, num_patches, d_model = hidden_state.shape
                hidden_state_reshaped = hidden_state.view(
                    batch_size * num_channels, num_patches, d_model
                )
                hidden_state_upsampled = stage["upsample"](hidden_state_reshaped)
                skip = reversed_skips[i]
                bs_s, nc_s, np_s, nd_s = skip.shape
                skip_reshaped = skip.view(bs_s * nc_s, np_s, nd_s)
                for processor in stage["skip_processor"]:
                    skip_reshaped = processor(skip_reshaped)
                if hidden_state_upsampled.shape[1] != skip_reshaped.shape[1]:
                    diff = hidden_state_upsampled.shape[1] - skip_reshaped.shape[1]
                    skip_reshaped = nn.functional.pad(skip_reshaped, (0, 0, 0, diff))
                hidden_state = hidden_state_upsampled + skip_reshaped
                num_patches, d_model = hidden_state.shape[1], hidden_state.shape[2]
                hidden_state = hidden_state.view(
                    batch_size, num_channels, num_patches, d_model
                )
            for layer in stage["layers"]:
                layer_outputs = layer(
                    hidden_state,
                    output_attentions=output_attentions,
                    channel_attention_mask=channel_attention_mask,
                    linear_attn=linear_attn,
                )
                hidden_state = layer_outputs[0]
                total_moe_loss += layer_outputs[-1]
            all_stage_outputs.append(hidden_state)
        return all_stage_outputs, total_moe_loss


class MultiStagePredictionHead(nn.Module):
    def __init__(
        self,
        config: PatchTSTConfig,
        depths: list,
        wavelet_feature_dim: int = 0,
    ):
        super().__init__()
        self.config = config
        self.wavelet_feature_dim = wavelet_feature_dim
        if self.wavelet_feature_dim > 0:
            self.wavelet_mlp = nn.Sequential(
                nn.Linear(self.wavelet_feature_dim, self.wavelet_feature_dim * 2),
                nn.GELU(),
                nn.Linear(self.wavelet_feature_dim * 2, self.wavelet_feature_dim),
            )
        encoder_bottleneck_dim = config.d_model * (2 ** (len(depths) - 1))
        decoder_dims = [encoder_bottleneck_dim // (2**i) for i in range(len(depths))]
        total_time_domain_dim = sum(decoder_dims)
        head_dim = total_time_domain_dim + self.wavelet_feature_dim
        self.flatten = nn.Flatten(start_dim=2)
        self.projection = nn.Linear(head_dim, config.prediction_length, bias=False)
        self.dropout = (
            nn.Dropout(config.head_dropout)
            if config.head_dropout > 0
            else nn.Identity()
        )

    def forward(
        self,
        decoder_outputs_list: list,
        wavelet_embedding: Optional[torch.Tensor] = None,
    ):
        B, V, _, _ = decoder_outputs_list[0].shape
        pooled_outputs = []
        for embedding in decoder_outputs_list:
            pooled_embedding = embedding.mean(dim=2)
            pooled_outputs.append(pooled_embedding)
        time_domain_embedding = torch.cat(pooled_outputs, dim=-1)
        if wavelet_embedding is not None and self.wavelet_feature_dim > 0:
            processed_wavelet_embedding = self.wavelet_mlp(wavelet_embedding)
            wavelet_embedding_expanded = processed_wavelet_embedding.unsqueeze(
                1
            ).expand(-1, V, -1)
            final_embedding = torch.cat(
                [time_domain_embedding, wavelet_embedding_expanded], dim=-1
            )
        else:
            final_embedding = time_domain_embedding
        flattened_embedding = self.flatten(final_embedding)
        dropped_embedding = self.dropout(flattened_embedding)
        output = self.projection(dropped_embedding)
        output = output.transpose(2, 1)
        return output


class PatchTSTForPrediction(PatchTSTPreTrainedModel):
    """PatchTST-based forecasting model with multi-stage U-Net backbone."""

    def __init__(self, config: PatchTSTConfig):
        super().__init__(config)
        self.config = config
        self.training_truncate_lengths = list(
            getattr(config, "training_truncate_lengths", [128, 256, 384, 512])
        )
        self.inference_truncate_length = int(
            getattr(config, "inference_truncate_length", 512)
        )
        if self.inference_truncate_length not in self.training_truncate_lengths:
            self.training_truncate_lengths.append(self.inference_truncate_length)
        self.training_truncate_lengths = [
            min(length, config.context_length)
            for length in self.training_truncate_lengths
        ]
        self.inference_truncate_length = min(
            self.inference_truncate_length, config.context_length
        )

        self.random = random
        self.depths = list(getattr(config, "depths", [2, 2, 2, 2]))
        self.skip_connections_depths = list(
            getattr(config, "skip_connections_depths", [2, 2, 2, 0])
        )
        self.num_heads_list = list(getattr(config, "num_heads_list", [3, 6, 12, 24]))
        if len(self.depths) != len(self.skip_connections_depths):
            raise ValueError(
                "`depths` and `skip_connections_depths` must have the same length."
            )
        if len(self.depths) != len(self.num_heads_list):
            raise ValueError("`depths` and `num_heads_list` must have the same length.")
        self.load_balance_coeff = float(getattr(config, "load_balance_coeff", 0.1))
        self.mmd_loss_coeff = float(getattr(config, "mmd_loss_coeff", 0.5))
        self.mmd_kernel_params = {
            "sigma_list": list(
                getattr(
                    config, "mmd_rational_quadratic_sigma_list", [0.2, 0.5, 0.9, 1.3]
                )
            )
        }
        self.wavelet_feature_dim = int(getattr(config, "wavelet_feature_dim", 48))
        if self.wavelet_feature_dim < 0:
            raise ValueError("`wavelet_feature_dim` must be >= 0.")
        self.training_target = str(getattr(config, "training_target", "value")).lower()
        if self.training_target not in {"value", "delta"}:
            raise ValueError(
                f"Unsupported training_target: {self.training_target}. "
                "Expected one of ['value', 'delta']."
            )
        if self.wavelet_feature_dim > 0:
            wavelet_scattering_j = int(getattr(config, "wavelet_scattering_j", 8))
            wavelet_scattering_q = int(getattr(config, "wavelet_scattering_q", 8))
            self.freq_analyzer = WaveletAnalyzer(
                input_timesteps=config.context_length,
                feature_dim=self.wavelet_feature_dim,
                J=wavelet_scattering_j,
                Q=wavelet_scattering_q,
            )
        else:
            self.freq_analyzer = None
        self.scaler = PatchTSTScaler(config)
        self.patchifier = PatchTSTPatchify(config)
        if config.use_dynamics_embedding:
            self.encoder_embedding = PatchTSTKernelEmbedding(config)
        else:
            self.encoder_embedding = PatchTSTEmbedding(config)
        original_d_model = config.d_model
        self.encoder = PatchTSTUNetEncoder(
            config, depths=self.depths, num_heads_list=self.num_heads_list
        )
        config.d_model = original_d_model * (2 ** (len(self.depths) - 1))
        self.decoder = PatchTSTUNetDecoder(
            config,
            depths=self.depths,
            skip_connections_depths=self.skip_connections_depths,
            num_heads_list=self.num_heads_list,
        )
        config.d_model = original_d_model
        self.head = MultiStagePredictionHead(
            config,
            depths=self.depths,
            wavelet_feature_dim=self.wavelet_feature_dim,
        )
        if config.loss == "mse":
            self.loss = nn.MSELoss(reduction="mean")
        elif config.loss == "huber":
            self.loss = nn.HuberLoss(reduction="mean", delta=config.huber_delta)
        else:
            raise ValueError(f"Unsupported loss type: {config.loss}")
        self.post_init()

    def forward(
        self,
        past_values: torch.Tensor,
        past_observed_mask: Optional[torch.Tensor] = None,
        future_values: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        channel_attention_mask: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        linear_attn: bool = False,
    ) -> Union[Tuple, PatchTSTForPredictionOutput]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        if past_observed_mask is None:
            past_observed_mask = torch.ones_like(past_values)
        if self.training:
            target_len = self.random.choice(self.training_truncate_lengths)
        else:
            target_len = self.inference_truncate_length
        current_seq_len = past_values.shape[1]
        target_len = min(target_len, current_seq_len)

        past_values_truncated = past_values[:, -target_len:, :]
        past_observed_mask_truncated = past_observed_mask[:, -target_len:, :]

        wavelet_embedding = None
        if self.freq_analyzer is not None:
            wavelet_input_truncated = past_values[:, -target_len:, :].permute(0, 2, 1)
            padding_needed = self.config.context_length - wavelet_input_truncated.shape[2]
            if padding_needed > 0:
                wavelet_input_padded = F.pad(
                    wavelet_input_truncated, (0, padding_needed), "constant", 0
                )
            else:
                wavelet_input_padded = wavelet_input_truncated
            wavelet_embedding = self.freq_analyzer(wavelet_input_padded)
        scaled_past_values, loc, scale = self.scaler(
            past_values_truncated, past_observed_mask_truncated
        )
        patched_values = self.patchifier(scaled_past_values)
        embedded_values = self.encoder_embedding(patched_values)
        encoder_output, skip_connections, encoder_moe_loss = self.encoder(
            embedded_values,
            output_attentions=output_attentions,
            channel_attention_mask=channel_attention_mask,
            linear_attn=linear_attn,
        )
        decoder_outputs_list, decoder_moe_loss = self.decoder(
            encoder_output,
            skip_connections,
            output_attentions=output_attentions,
            channel_attention_mask=channel_attention_mask,
            linear_attn=linear_attn,
        )
        y_hat = self.head(decoder_outputs_list, wavelet_embedding=wavelet_embedding)

        last_past_value = past_values_truncated[:, -1:, :]
        delta_rms = torch.ones_like(last_past_value)

        if self.training_target == "delta":
            if past_values_truncated.shape[1] > 1:
                past_deltas = (
                    past_values_truncated[:, 1:, :] - past_values_truncated[:, :-1, :]
                )
                delta_rms = torch.sqrt(
                    torch.mean(past_deltas**2, dim=1, keepdim=True)
                ).clamp_min(1e-6)
            y_hat_out = last_past_value + y_hat * delta_rms
        else:
            y_hat_out = y_hat * scale + loc

        loss_val = None
        if future_values is not None:
            if self.training_target == "delta":
                prediction_targets = (future_values - last_past_value) / delta_rms
                prediction_outputs = y_hat
            else:
                prediction_targets = future_values
                prediction_outputs = y_hat_out

            prediction_loss = self.loss(prediction_outputs, prediction_targets)
            mmd_loss = torch.tensor(0.0, device=past_values.device)
            if self.mmd_loss_coeff > 0:
                if self.training_target == "delta":
                    batch_mean = torch.zeros(
                        prediction_targets.shape[-1], device=past_values.device
                    )
                    batch_variance = torch.ones(
                        prediction_targets.shape[-1], device=past_values.device
                    )
                else:
                    batch_mean = loc.mean(dim=0)
                    batch_variance = (scale**2).mean(dim=0)
                mmd_loss = conditional_mmd_multi_step(
                    input_traj=None,
                    true_traj=prediction_targets,
                    pred_traj=prediction_outputs,
                    mean=batch_mean,
                    variance=batch_variance,
                    kernel_params=self.mmd_kernel_params,
                )
            total_moe_loss = encoder_moe_loss + decoder_moe_loss
            loss_val = (
                prediction_loss
                + self.mmd_loss_coeff * mmd_loss
                + self.load_balance_coeff * total_moe_loss
            )
        if not return_dict:
            outputs = (y_hat_out, loc, scale)
            return (loss_val,) + outputs if loss_val is not None else outputs
        return PatchTSTForPredictionOutput(
            loss=loss_val,
            prediction_outputs=y_hat_out,
            hidden_states=None,
            attentions=None,
            loc=loc,
            scale=scale,
        )

    def generate(
        self,
        past_values: torch.Tensor,
        past_observed_mask: Optional[torch.Tensor] = None,
        channel_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
    ) -> SamplePatchTSTOutput:
        outputs = self(
            past_values=past_values,
            future_values=None,
            past_observed_mask=past_observed_mask,
            output_hidden_states=False,
            channel_attention_mask=channel_attention_mask,
            output_attentions=output_attentions,
        )
        samples = outputs.prediction_outputs.unsqueeze(1)
        return SamplePatchTSTOutput(sequences=samples)
