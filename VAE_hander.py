from acestep.training_v2.model_loader import (
    load_vae,
    _resolve_dtype,
)
from acestep.training.dataset_builder_modules.preprocess_audio import load_audio_stereo
import torch
import gc
import logging
import torchaudio  # 需要引入这个库来保存 mp3
from pathlib import Path

# 配置 Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_TARGET_SR = 48000
max_duration = 240.0

def _empty_gpu_cache() -> None:
    """Release cached GPU memory (CUDA / MPS / XPU)."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass

# 设置设备和精度
checkpoint_dir = "你的模型路径"  # 请确保这里有正确路径
audio_files = [] # 你的音频文件列表 Path对象
out_path = Path("output_mp3s")
out_path.mkdir(exist_ok=True, parents=True)

device = 'cuda'
precision = 'auto'
dtype = _resolve_dtype(precision)

logger.info("[Process] Loading VAE...")
vae = load_vae(checkpoint_dir, device, precision)
vae.eval() # 确保进入评估模式

failed = 0
total = len(audio_files)

# 显存监控辅助函数
def get_peak_memory_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0.0

try:
    for i, af in enumerate(audio_files):
        
        mp3_out = out_path / f"{af.stem}_recon.mp3"
        if mp3_out.exists():
            logger.info(f"Skipping (exists): {af.name}")
            continue

        try:
            logger.info(f"Processing {af.name}...")
            
            # 1. Load Audio
            audio, _sr = load_audio_stereo(str(af), _TARGET_SR, max_duration)
            # shape: [C, T] -> [1, C, T]
            audio = audio.unsqueeze(0).to(device).to(vae.dtype)

            # --- 显存监控开始 ---
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            
            # 2. VAE Encode & Decode (无 Tiled，直接整段推理)
            with torch.no_grad():
                # Encode: Audio -> Latent
                # 注意：这里直接取 sample，不再 transpose，因为 decode 需要原始形状 [B, C, L]
                latents = vae.encode(audio).latent_dist.sample()
                
                # Decode: Latent -> Audio
                recon_audio = vae.decode(latents).sample

            # --- 显存监控结束 ---
            peak_mem = get_peak_memory_mb()
            logger.info(f"Peak VRAM used for {af.name}: {peak_mem:.2f} MB")

            # 3. 保存 MP3
            # recon_audio shape is [1, C, T], squeeze to [C, T]
            # 确保数据在 CPU 并且是 float32 (torchaudio save 通常需要 float32)
            save_tensor = recon_audio.squeeze(0).cpu().float()
            
            # 简单的防爆音截断 (Clamp)，虽然 MP3 编码有一定的容忍度，但最好限制在 -1~1
            save_tensor = torch.clamp(save_tensor, -1.0, 1.0)

            torchaudio.save(
                uri=str(mp3_out), 
                src=save_tensor, 
                sample_rate=_TARGET_SR, 
                format="mp3"
            )
            
            # 清理
            del audio, latents, recon_audio, save_tensor
            _empty_gpu_cache()

        except Exception as e:
            logger.error(f"Failed to process {af.name}: {e}")
            failed += 1
            _empty_gpu_cache()

except KeyboardInterrupt:
    logger.info("Interrupted by user.")

logger.info(f"Done. Failed: {failed}/{total}")