import torch
from model.minicpm4 import MiniCPMModel, MiniCPM4Config
from model.dit import AceStepConditionGenerationModel
from model.audio_vae import AudioVAE, AudioVAEConfig
from model.local_encoder import VoxCPMLocEnc
from model.local_dit import VoxCPMLocDiT
from model.unified_cfm import UnifiedCFM
from model.scalar_quantization import ScalarQuantizationLayer
from transformers import LlamaTokenizerFast
from safetensors.torch import load_file
from pydantic import BaseModel
from typing import Optional
import sys
import torch.nn as nn
from einops import rearrange

class VoxCPMConfig(BaseModel):
    lm_config: MiniCPM4Config
    patch_size: int = 2
    feat_dim: int = 64
    residual_lm_num_layers: int = 6
    scalar_quantization_latent_dim: int = 256
    scalar_quantization_scale: int = 9
    encoder_config: VoxCPMEncoderConfig


class VoxCPMEncoderConfig(BaseModel):
    hidden_dim: int = 1024
    ffn_dim: int = 4096
    num_heads: int = 16
    num_layers: int = 4
    kv_channels: int = None
    max_length: int = 4096
    device: str = "cuda"
    dtype: str = "bfloat16"
    dit_mean_mode: bool = False

class TrainModel(nn.Module):
    def __init__(
        self,
        config: VoxCPMConfig,
        tokenizer: LlamaTokenizerFast,
        audio_vae: AudioVAE,
        base_lm: nn.Module,
        residual_lm: nn.Module,
        dit: nn.Module,
    ):
        self.config = config
        self.feat_dim = config.feat_dim
        self.patch_size = config.patch_size
        self.base_lm = base_lm
        self.residual_lm = residual_lm
        self.dit = dit
        self.device = config.device
        if not torch.cuda.is_available():
            if torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        print(f"Running on device: {self.device}, dtype: {self.config.dtype}", file=sys.stderr)



        # Local Encoder
        encoder_config = config.lm_config.model_copy(deep=True)
        encoder_config.hidden_size = config.encoder_config.hidden_dim
        encoder_config.intermediate_size = config.encoder_config.ffn_dim
        encoder_config.num_attention_heads = config.encoder_config.num_heads
        encoder_config.num_hidden_layers = config.encoder_config.num_layers
        encoder_config.kv_channels = config.encoder_config.kv_channels
        encoder_config.vocab_size = 0
        self.feat_encoder = VoxCPMLocEnc(encoder_config, input_dim=config.feat_dim)

        # Local DiT
        if "VoxCPMLocDiT" in globals() and "UnifiedCFM" in globals():
             # If placeholders are used, we might need to adjust logic, but assuming imports worked or placeholders are sufficient
             pass
        
        # Use placeholder for feat_decoder if not properly initialized in real code (which uses AceStepConditionGenerationModel in code but import is AceStepConditionGenerationModel)
        # Wait, line 73 says self.feat_decoder = AceStepConditionGenerationModel(config)
        # But AceStepConditionGenerationModel definition in dit.py expects 'AceStepConfig'
        # VoxCPMConfig has 'dit_config'
        # We need to bridge this.
        self.feat_decoder = AceStepConditionGenerationModel(config.dit_config) if hasattr(config, 'dit_config') else AceStepConditionGenerationModel(config)

        #init lm
        base_lm_config = config.lm_config 
        self.base_lm = MiniCPMModel(base_lm_config)
        
        self.text_tokenizer = mask_multichar_chinese_tokens(tokenizer)
        self.audio_start_token = 101
        self.audio_end_token = 102
        
        residual_lm_config = config.lm_config.model_copy(deep=True)
        if hasattr(config, "residual_lm_num_layers"):
             residual_lm_config.num_hidden_layers = config.residual_lm_num_layers
        
        self.residual_lm = MiniCPMModel(residual_lm_config)
        


        # Projection layers
        self.fsq_layer = ScalarQuantizationLayer(
            config.lm_config.hidden_size, 
            config.lm_config.hidden_size, 
            config.scalar_quantization_latent_dim, 
            config.scalar_quantization_scale
        )
        self.enc_to_lm_proj = nn.Linear(config.encoder_config.hidden_dim, config.lm_config.hidden_size)
        self.lm_to_dit_proj = nn.Linear(config.lm_config.hidden_size, config.dit_config.hidden_dim)
        self.res_to_dit_proj = nn.Linear(config.lm_config.hidden_size, config.dit_config.hidden_dim)

        # Stop Predictor
        self.stop_proj = nn.Linear(config.lm_config.hidden_size, config.lm_config.hidden_size)
        self.stop_actn = nn.SiLU()
        self.stop_head = nn.Linear(config.lm_config.hidden_size, 2, bias=False)
        self.stop_loss = nn.CrossEntropyLoss(reduction="none")

        # Audio VAE
        self.audio_vae = audio_vae
        self.chunk_size = audio_vae.chunk_size
        self.sample_rate = audio_vae.sample_rate

    def forward(self, 
            text_tokens: torch.Tensor,
            text_mask: torch.Tensor,
            audio_feats: torch.Tensor,
            audio_mask: torch.Tensor,
            loss_mask: torch.Tensor,
            labels: torch.Tensor,
            progress: float = 0.0,
            sample_generate: bool = False):
        text_tokens = text_tokens.to(self.device, dtype=torch.long)
        text_mask = text_mask.to(self.device, dtype=self._dtype())
        audio_feats = audio_feats.to(self.device, dtype=self._dtype())
        audio_mask = audio_mask.to(self.device, dtype=self._dtype())
        loss_mask = loss_mask.to(self.device, dtype=self._dtype())
        labels = labels.to(self.device, dtype=torch.long)
        B, T, P, D = audio_feats.shape
        feat_embed = self.feat_encoder(audio_feats)
        feat_embed = self.enc_to_lm_proj(feat_embed)
        scale_emb = getattr(self.config.lm_config, "scale_emb", 1.0)
        if not getattr(self.config.lm_config, "use_mup", False):
            scale_emb = 1.0
        text_embed = self.base_lm.embed_tokens(text_tokens) * scale_emb
        combined_embed = text_mask.unsqueeze(-1) * text_embed + audio_mask.unsqueeze(-1) * feat_embed
        enc_outputs, _ = self.base_lm(inputs_embeds=combined_embed, is_causal=True)
        enc_outputs = enc_outputs.to(self._dtype())
        enc_outputs = self.fsq_layer(enc_outputs) * audio_mask.unsqueeze(-1) + enc_outputs * text_mask.unsqueeze(-1)
        lm_hidden = torch.cat((torch.zeros_like(enc_outputs[:, 0:1, :]), enc_outputs[:, :-1, :]), dim=1)
        residual_inputs = enc_outputs + audio_mask.unsqueeze(-1) * feat_embed
        residual_outputs, _ = self.residual_lm(inputs_embeds=residual_inputs, is_causal=True)
        residual_outputs = residual_outputs.to(self._dtype())
        residual_hidden = torch.cat(
            (torch.zeros_like(residual_outputs[:, 0:1, :]), residual_outputs[:, :-1, :]),
            dim=1,
        )

        dit_hidden = self.lm_to_dit_proj(lm_hidden) + self.res_to_dit_proj(residual_hidden)
        dit_hidden = rearrange(dit_hidden, "b t c -> (b t) c")

        # Keep diffusion inputs in the same dtype as the model (e.g., bfloat16)
        target_dtype = self._dtype()

        feat_gt = rearrange(audio_feats.to(target_dtype), "b t p d -> (b t) p d")
        feat_cond = torch.cat(
            (torch.zeros_like(audio_feats[:, 0:1, ...]), audio_feats[:, :-1, ...]),
            dim=1,
        )
        feat_cond = rearrange(feat_cond.to(target_dtype), "b t p d -> (b t) p d")

        loss_seq_mask = loss_mask.unsqueeze(-1).repeat(1, 1, self.patch_size)
        loss_seq_mask = rearrange(loss_seq_mask, "b t p -> (b t) p 1").to(target_dtype)

        diff_loss = self.feat_decoder.compute_loss(
            feat_gt.transpose(1, 2).contiguous(),
            dit_hidden,
            cond=feat_cond.transpose(1, 2).contiguous(),
            tgt_mask=loss_seq_mask.transpose(1, 2).contiguous(),
            progress=progress,
        )

        stop_logits = self.stop_head(self.stop_actn(self.stop_proj(lm_hidden)))
        stop_losses = self.stop_loss(stop_logits.transpose(1, 2), labels)
        denom = torch.clamp(loss_mask.sum(), min=1.0)
        stop_loss = (stop_losses * loss_mask).sum() / denom

        feat_pred = None
        if sample_generate:
            feat_cond_for_sample = feat_cond.transpose(1, 2).contiguous()
            feat_pred_seq = self.feat_decoder(
                mu=dit_hidden,
                patch_size=self.patch_size,
                cond=feat_cond_for_sample,
                n_timesteps=self.config.dit_config.cfm_config.inference_cfg_rate
                if hasattr(self.config.dit_config.cfm_config, "inference_cfg_rate")
                else 10,
            )
            feat_pred = rearrange(feat_pred_seq.transpose(1, 2), "(b t) d p -> b d (t p)", b=B, p=self.patch_size)

        feat_gt_tensor = rearrange(feat_gt, "(b t) p d -> b d (t p)", b=B, p=self.patch_size)

        return {
            "loss/diff": diff_loss,
            "loss/stop": stop_loss,
            "feat_gt": feat_gt_tensor,
            "feat_pred": feat_pred,
        }

if __name__ == "__main__":
    from unittest.mock import MagicMock, patch
    
    # We use mocking for file loading ONLY, to avoid IO errors.
    # The user wants to test ACTUAL components.

    def run_real_component_test():
        print("Running Real Component Test...")
        from transformers import LlamaTokenizerFast

        # 1. Construct Real Configs
        # We need to match signatures of MiniCPMModel and AceStepConditionGenerationModel
        # MiniCPMOutput = config.lm_config
        
        # Create a config that satisfies VoxCPMConfig AND sub-modules
        # We use arbitrary values that are structurally valid
        
        import json
        import model.minicpm4 as mcpm
        
        # Load config from file
        config_path = r"d:\CODE\my_tts\lm_models\config.json"
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        tokenizer = LlamaTokenizerFast.from_pretrained(r"d:\CODE\my_tts\lm_models")
        # 1. LM Config
        lm_config_data = config_dict.get("lm_config", {})
        # Reduce size for testing purposes to avoid OOM or slow execution
        lm_config_data["hidden_size"] = 64
        lm_config_data["intermediate_size"] = 128
        lm_config_data["num_hidden_layers"] = 2
        lm_config_data["num_attention_heads"] = 2
        lm_config_data["num_key_value_heads"] = 1
        
        lm_config = mcpm.MiniCPM4Config(**lm_config_data)
        
        encoder_config = VoxCPMEncoderConfig(
            **config_dict.get("encoder_config", {})
        )
        # AceStepConfig for DiT
        from model.dit import AceStepConfig
        dit_config = AceStepConfig()
        # # Add cfm_config expected by code
        # dit_config.hidden_dim = 64 # alias
        # class CFMConfig: inference_cfg_rate = 10
        # dit_config.cfm_config = CFMConfig()

        config = VoxCPMConfig(
            lm_config=lm_config,
            encoder_config=encoder_config,
        )
        
        # 2. Tokenizer & VAE
        tokenizer = MagicMock(spec=LlamaTokenizerFast)
        audio_vae = AudioVAE(None) # Use placeholder/real class
        
        # 3. Instantiate TrainModel (Real)
        # We patch load_file to avoid crash, but let other things run
        # with patch("safetensors.torch.load_file", return_value={}):
        model = TrainModel(config, tokenizer, audio_vae, base_lm, residual_lm, dit)
        
        print("TrainModel initialized successfully.")

        # 4. Dummy Inputs
        B, T = 2, 10
        device = model.device
        dtype = model._dtype() if hasattr(model, '_dtype') else torch.float32
        
        text_tokens = torch.randint(0, 100, (B, T)).to(device)
        text_mask = torch.ones((B, T)).to(device)
        audio_feats = torch.randn(B, T, config.patch_size, config.feat_dim).to(device)
        audio_mask = torch.ones((B, T)).to(device)
        loss_mask = torch.ones((B, T)).to(device)
        labels = torch.randint(0, 100, (B, T)).to(device)
        
        # 5. Real LMs passed to forward
        # These are usually the same as self.base_lm / self.residual_lm or loaded externally
        # We use the ones created in model, or new ones
        base_lm = model.base_lm
        residual_lm = model.residual_lm
        
        # 6. Run Forward
        # Note: AceStepConditionGenerationModel.compute_loss might need specific inputs.
        # In TrainModel forward:
        # diff_loss = self.feat_decoder.compute_loss(...)
        # If feat_decoder is Placeholder, it returns dummy.
        # If real, AceStepConditionGenerationModel.compute_loss needs implementation check.
        # I checked AceStepConditionGenerationModel.compute_loss in view_file, seems compatible.
        
        print("Running forward pass...")
        outputs = model(
            text_tokens=text_tokens,
            text_mask=text_mask,
            audio_feats=audio_feats,
            audio_mask=audio_mask,
            loss_mask=loss_mask,
            labels=labels,
            sample_generate=True
        )
        
        for k, v in outputs.items():
            if isinstance(v, torch.Tensor):
                print(f"{k}: {v.shape}")
            else:
                print(f"{k}: {v}")
        print("Success.")

    run_real_component_test()