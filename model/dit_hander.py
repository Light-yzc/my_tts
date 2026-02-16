from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.func import jvp
from pydantic import BaseModel
from loguru import logger

from dit import AceStepConditionGenerationModel

class CfmConfig(BaseModel):
    sigma_min: float = 1e-6
    solver: str = "euler"
    t_scheduler: str = "log-norm"
    training_cfg_rate: float = 0.1
    inference_cfg_rate: float = 1.0
    reg_loss_type: str = "l1"
    ratio_r_neq_t_range: Tuple[float, float] = (0.25, 0.75)
    noise_cond_prob_range: Tuple[float, float] = (0.0, 0.0)
    noise_cond_scale: float = 0.0


class AceStepConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`AceStepModel`]. It is used to instantiate an
    AceStep model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 64003):
            Vocabulary size of the AceStep model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling the model.
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 22016):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 32):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details, check out [this
            paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to `32`.
        head_dim (`int`, *optional*, defaults to 128):
            The attention head dimension.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 32768):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type
            and you expect the model to work on longer `max_position_embeddings`, we recommend you to update this value
            accordingly.
            Expected contents:
                `rope_type` (`str`):
                    The sub-variant of RoPE to use. Can be one of ['default', 'linear', 'dynamic', 'yarn', 'longrope',
                    'llama3'], with 'default' being the original RoPE implementation.
                `factor` (`float`, *optional*):
                    Used with all rope types except 'default'. The scaling factor to apply to the RoPE embeddings. In
                    most scaling types, a `factor` of x will enable the model to handle sequences of length x *
                    original maximum pre-trained length.
                `original_max_position_embeddings` (`int`, *optional*):
                    Used with 'dynamic', 'longrope' and 'llama3'. The original max position embeddings used during
                    pretraining.
                `attention_factor` (`float`, *optional*):
                    Used with 'yarn' and 'longrope'. The scaling factor to be applied on the attention
                    computation. If unspecified, it defaults to value recommended by the implementation, using the
                    `factor` field to infer the suggested value.
                `beta_fast` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for extrapolation (only) in the linear
                    ramp function. If unspecified, it defaults to 32.
                `beta_slow` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for interpolation (only) in the linear
                    ramp function. If unspecified, it defaults to 1.
                `short_factor` (`list[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to short contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `long_factor` (`list[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to long contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `low_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to low frequency components of the RoPE
                `high_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to high frequency components of the RoPE
        attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        use_sliding_window (`bool`, *optional*, defaults to `False`):
            Whether to use sliding window attention.
        sliding_window (`int`, *optional*, defaults to 4096):
            Sliding window attention (SWA) window size. If not specified, will default to `4096`.
        layer_types (`list`, *optional*):
            Attention pattern for each layer.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.

    ```python
    >>> from acestep.models import AceStepConfig

    >>> # Initializing an AceStep configuration
    >>> configuration = AceStepConfig()

    >>> # Initializing a model from the configuration
    >>> model = AceStepConditionGenerationModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "acestep"
    keys_to_ignore_at_inference = ["past_key_values"]
    
    # Default tensor parallel plan for the base model
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }
    def __init__(
        self,
        vocab_size=64003,
        fsq_dim=2048,
        fsq_input_levels=[8, 8, 8, 5, 5, 5],
        fsq_input_num_quantizers=1,
        # hidden_size=2048,
        # intermediate_size=6144,
        # num_hidden_layers=24,
        # num_attention_heads=16,
        # num_key_value_heads=8,
        # head_dim=128,
        hidden_size=1024,          # 减半
        intermediate_size=4096,    # 4 * hidden
        num_hidden_layers=16,      # 减少层数
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=64,               # 1024/16 = 64
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=True,
        rope_theta=1000000,
        rope_scaling=None,
        attention_bias=False,
        use_sliding_window=True,
        sliding_window=128,
        layer_types=None,
        attention_dropout=0.0,
        num_lyric_encoder_hidden_layers=8,
        audio_acoustic_hidden_dim=64,
        pool_window_size=5,
        text_hidden_dim=1024,
        in_channels=192,
        data_proportion=0.5,
        timestep_mu=-0.4,
        timestep_sigma=1.0,
        timbre_hidden_dim=64,
        num_timbre_encoder_hidden_layers=4,
        timbre_fix_frame=750,
        patch_size=2,
        num_attention_pooler_hidden_layers=2,
        num_audio_decoder_hidden_layers=24,
        model_version="turbo",
        **kwargs,
    ):
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if self.use_sliding_window else None
        
        # Text encoder configuration
        self.text_hidden_dim = text_hidden_dim

        # Lyric encoder configuration
        self.num_lyric_encoder_hidden_layers = num_lyric_encoder_hidden_layers
        self.patch_size = patch_size

        # Audio semantic token generation configuration
        self.audio_acoustic_hidden_dim = audio_acoustic_hidden_dim
        self.pool_window_size = pool_window_size
        self.in_channels = in_channels
        self.data_proportion = data_proportion
        self.timestep_mu = timestep_mu
        self.timestep_sigma = timestep_sigma
        
        # FSQ (Finite Scalar Quantization) configuration
        self.fsq_dim = fsq_dim
        self.fsq_input_levels = fsq_input_levels
        self.fsq_input_num_quantizers = fsq_input_num_quantizers
        
        # Timbre encoder configuration
        self.timbre_hidden_dim = timbre_hidden_dim
        self.num_timbre_encoder_hidden_layers = num_timbre_encoder_hidden_layers
        self.timbre_fix_frame = timbre_fix_frame
        self.num_attention_pooler_hidden_layers = num_attention_pooler_hidden_layers
        self.num_audio_decoder_hidden_layers = num_audio_decoder_hidden_layers
        self.vocab_size = vocab_size

        # Backward compatibility: ensure num_key_value_heads is set
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.model_version = model_version
        
        # Validate rotary position embeddings parameters
        # Backward compatibility: if there is a 'type' field, move it to 'rope_type'
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        self.layer_types = layer_types

        # Set default layer types if not specified
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention" if bool((i + 1) % 2) else "full_attention" for i in range(self.num_hidden_layers)
            ]
        layer_type_validation(self.layer_types)

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class MyDiT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = None
        self.config = None
        self.device = "cpu"
        self.dtype = torch.float32  # Will be set based on device in initialize_service

        # VAE for audio encoding/decoding
        self.vae = None
        self.sample_rate = 48000
        self.batch_size = 2

    def initialize_service(
        self,
        project_root: str,
        config_path: str,
        device: str = "auto",
        use_flash_attention: bool = False,
        compile_model: bool = False,
        offload_dit_to_cpu: bool = False,
        prefer_source: Optional[str] = None,
    ):
        if config_path == "":
            logger.warning(
                "[initialize_service] Empty config_path; pass None to use the default model."
            )
        #vae load
        vae_checkpoint_path = os.path.join(checkpoint_dir, "vae")
        if os.path.exists(vae_checkpoint_path):
            self.vae = AutoencoderOobleck.from_pretrained(vae_checkpoint_path)
            vae_dtype = self._get_vae_dtype(device)
            self.vae = self.vae.to("cpu").to(vae_dtype)
            self.vae.eval()
        
        #dit load
        # 1. Load main model
        # config_path is relative path (e.g., "acestep-v15-turbo"), concatenate to checkpoints directory
        dit_check_base = os.path.join(checkpoint_dir, config_path)
        if os.path.exists(dit_check_base):
            # Force CUDA cleanup before loading DiT to reduce fragmentation on model/mode switch
            if torch.cuda.is_available():
                if getattr(self, "model", None) is not None:
                    del self.model
                    self.model = None
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # Determine attention implementation, then fall back safely.
            if use_flash_attention and self.is_flash_attention_available(device):
                attn_implementation = "flash_attention_2"
            else:
                if use_flash_attention:
                    logger.warning(
                        f"[initialize_service] Flash attention requested but unavailable for device={device}. "
                        "Falling back to SDPA."
                    )
                attn_implementation = "sdpa"

            attn_candidates = [attn_implementation]
            if "sdpa" not in attn_candidates:
                attn_candidates.append("sdpa")
            if "eager" not in attn_candidates:
                attn_candidates.append("eager")

            last_attn_error = None
            self.model = None
            for candidate in attn_candidates:
                try:
                    logger.info(f"[initialize_service] Attempting to load model with attention implementation: {candidate}")
                    self.model = AutoModel.from_pretrained(
                        dit_check_base,
                        trust_remote_code=True,
                        attn_implementation=candidate,
                        torch_dtype=self.dtype,
                    )
                    attn_implementation = candidate
                    break
                except Exception as e:
                    last_attn_error = e
                    logger.warning(f"[initialize_service] Failed to load model with {candidate}: {e}")

            if self.model is None:
                raise RuntimeError(
                    f"Failed to load model with attention implementations {attn_candidates}: {last_attn_error}"
                ) from last_attn_error

            self.model.config._attn_implementation = attn_implementation
            self.config = self.model.config
            # Move model to device and set dtype
            if not self.offload_to_cpu:
                self.model = self.model.to(device).to(self.dtype)
            else:
                # If offload_to_cpu is True, check if we should keep DiT on GPU
                if not self.offload_dit_to_cpu:
                    logger.info(f"[initialize_service] Keeping main model on {device} (persistent)")
                    self.model = self.model.to(device).to(self.dtype)
                else:
                    self.model = self.model.to("cpu").to(self.dtype)
            self.model.eval()
            
            if compile_model:
                # Add __len__ method to model to support torch.compile
                # torch.compile's dynamo requires this method for introspection
                # Note: This modifies the model class, affecting all instances
                if not hasattr(self.model.__class__, '__len__'):
                    def _model_len(model_self):
                        """Return 0 as default length for torch.compile compatibility"""
                        return 0
                    self.model.__class__.__len__ = _model_len
                
                self.model = torch.compile(self.model)
                

                
    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def forward(
        self,
        mu: torch.Tensor,
        n_timesteps: int,
        patch_size: int,
        cond: torch.Tensor,
        temperature: float = 1.0,
        cfg_value: float = 1.0,
        sway_sampling_coef: float = 1.0, 
        use_cfg_zero_star: bool = True,
    ):
        b, _ = mu.shape
        t = patch_size
        z = torch.randn((b, self.in_channels, t), device=mu.device, dtype=mu.dtype) * temperature

        t_span = torch.linspace(1, 0, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        t_span = t_span + sway_sampling_coef * (torch.cos(torch.pi / 2 * t_span) - 1 + t_span)

        return self.solve_euler(
            x=z,
            t_span=t_span,
            mu=mu,
            cond=cond,
            cfg_value=cfg_value,
            use_cfg_zero_star=use_cfg_zero_star,
        )

    def optimized_scale(self, positive_flat: torch.Tensor, negative_flat: torch.Tensor):
        dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)
        squared_norm = torch.sum(negative_flat**2, dim=1, keepdim=True) + 1e-8
        st_star = dot_product / squared_norm
        return st_star

    def solve_euler(
        self,
        x: torch.Tensor,
        t_span: torch.Tensor,
        mu: torch.Tensor,
        cond: torch.Tensor,
        cfg_value: float = 1.0,
        use_cfg_zero_star: bool = True,
    ):
        t, _, dt = t_span[0], t_span[-1], t_span[0] - t_span[1]

        sol = []
        zero_init_steps = max(1, int(len(t_span) * 0.04))
        for step in range(1, len(t_span)):
            if use_cfg_zero_star and step <= zero_init_steps:
                dphi_dt = torch.zeros_like(x)
            else:
                # Classifier-Free Guidance inference introduced in VoiceBox
                b = x.size(0)
                x_in = torch.zeros([2 * b, self.in_channels, x.size(2)], device=x.device, dtype=x.dtype)
                mu_in = torch.zeros([2 * b, mu.size(1)], device=x.device, dtype=x.dtype)
                t_in = torch.zeros([2 * b], device=x.device, dtype=x.dtype)
                dt_in = torch.zeros([2 * b], device=x.device, dtype=x.dtype)
                cond_in = torch.zeros([2 * b, self.in_channels, cond.size(2)], device=x.device, dtype=x.dtype)
                x_in[:b], x_in[b:] = x, x
                mu_in[:b] = mu
                t_in[:b], t_in[b:] = t.unsqueeze(0), t.unsqueeze(0)
                dt_in[:b], dt_in[b:] = dt.unsqueeze(0), dt.unsqueeze(0)
                # not used now
                if not self.mean_mode:
                    dt_in = torch.zeros_like(dt_in)
                cond_in[:b], cond_in[b:] = cond, cond

                dphi_dt = self.estimator(x_in, mu_in, t_in, cond_in, dt_in)
                dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)
                
                if use_cfg_zero_star:
                    positive_flat = dphi_dt.view(b, -1)
                    negative_flat = cfg_dphi_dt.view(b, -1)
                    st_star = self.optimized_scale(positive_flat, negative_flat)
                    st_star = st_star.view(b, *([1] * (len(dphi_dt.shape) - 1)))
                else:
                    st_star = 1.0
                
                dphi_dt = cfg_dphi_dt * st_star + cfg_value * (dphi_dt - cfg_dphi_dt * st_star)

            x = x - dt * dphi_dt
            t = t - dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t - t_span[step + 1]

        return sol[-1]

    # ------------------------------------------------------------------ #
    # Training loss
    # ------------------------------------------------------------------ #
    def adaptive_loss_weighting(self, losses: torch.Tensor, mask: torch.Tensor | None = None, p: float = 0.0, epsilon: float = 1e-3):
        weights = 1.0 / ((losses + epsilon).pow(p))
        if mask is not None:
            weights = weights * mask
        return weights.detach()

    def sample_r_t(self, x: torch.Tensor, mu: float = -0.4, sigma: float = 1.0, ratio_r_neq_t: float = 0.0):
        batch_size = x.shape[0]
        if self.t_scheduler == "log-norm":
            s_r = torch.randn(batch_size, device=x.device, dtype=x.dtype) * sigma + mu
            s_t = torch.randn(batch_size, device=x.device, dtype=x.dtype) * sigma + mu
            r = torch.sigmoid(s_r)
            t = torch.sigmoid(s_t)
        elif self.t_scheduler == "uniform":
            r = torch.rand(batch_size, device=x.device, dtype=x.dtype)
            t = torch.rand(batch_size, device=x.device, dtype=x.dtype)
        else:
            raise ValueError(f"Unsupported t_scheduler: {self.t_scheduler}")

        mask = torch.rand(batch_size, device=x.device, dtype=x.dtype) < ratio_r_neq_t
        r, t = torch.where(
            mask,
            torch.stack([torch.min(r, t), torch.max(r, t)], dim=0),
            torch.stack([t, t], dim=0),
        )

        return r.squeeze(), t.squeeze()

    def compute_loss(
        self,
        x1: torch.Tensor,
        mu: torch.Tensor,
        cond: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
        progress: float = 0.0,
    ):
        b, _, _ = x1.shape

        if self.training_cfg_rate > 0:
            cfg_mask = torch.rand(b, device=x1.device) > self.training_cfg_rate
            mu = mu * cfg_mask.view(-1, 1)

        if cond is None:
            cond = torch.zeros_like(x1)

        noisy_mask = torch.rand(b, device=x1.device) > (
            1.0
            - (
                self.noise_cond_prob_range[0]
                + progress * (self.noise_cond_prob_range[1] - self.noise_cond_prob_range[0])
            )
        )
        cond = cond + noisy_mask.view(-1, 1, 1) * torch.randn_like(cond) * self.noise_cond_scale

        ratio_r_neq_t = (
            self.ratio_r_neq_t_range[0]
            + progress * (self.ratio_r_neq_t_range[1] - self.ratio_r_neq_t_range[0])
            if self.mean_mode
            else 0.0
        )

        r, t = self.sample_r_t(x1, ratio_r_neq_t=ratio_r_neq_t)
        r_ = r.detach().clone()
        t_ = t.detach().clone()
        z = torch.randn_like(x1)
        y = (1 - t_.view(-1, 1, 1)) * x1 + t_.view(-1, 1, 1) * z
        v = z - x1

        def model_fn(z_sample, r_sample, t_sample):
            return self.estimator(z_sample, mu, t_sample, cond, dt=t_sample - r_sample)

        if self.mean_mode:
            v_r = torch.zeros_like(r)
            v_t = torch.ones_like(t)
            from torch.backends.cuda import sdp_kernel

            with sdp_kernel(enable_flash=False, enable_mem_efficient=False):
                u_pred, dudt = jvp(model_fn, (y, r, t), (v, v_r, v_t))
            u_tgt = v - (t_ - r_).view(-1, 1, 1) * dudt
        else:
            u_pred = model_fn(y, r, t)
            u_tgt = v

        losses = F.mse_loss(u_pred, u_tgt.detach(), reduction="none").mean(dim=1)
        if tgt_mask is not None:
            weights = self.adaptive_loss_weighting(losses, tgt_mask.squeeze(1))
            loss = (weights * losses).sum() / torch.sum(tgt_mask)
        else:
            loss = losses.mean()

        return loss
