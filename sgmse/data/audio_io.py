"""Minimal audio backend with explicit sample-rate probing and float WAV output."""

from pathlib import Path
from typing import Tuple
import wave

import numpy as np
import torch
import torch.nn.functional as functional


def probe_audio(path: str) -> Tuple[int, int, int]:
    """Return ``(sample_rate, frames, channels)`` without resampling."""
    audio_path = Path(path).expanduser()
    if not audio_path.is_file():
        raise FileNotFoundError("Audio file not found: {}".format(audio_path))
    try:
        import torchaudio

        info = torchaudio.info(str(audio_path))
        return int(info.sample_rate), int(info.num_frames), int(info.num_channels)
    except (ImportError, OSError, RuntimeError):
        try:
            import soundfile as sf
        except ImportError:
            with wave.open(str(audio_path), "rb") as stream:
                return (
                    int(stream.getframerate()),
                    int(stream.getnframes()),
                    int(stream.getnchannels()),
                )
        else:
            info = sf.info(str(audio_path))
            return int(info.samplerate), int(info.frames), int(info.channels)


def _resample(waveform: torch.Tensor, source_rate: int, target_rate: int) -> torch.Tensor:
    if source_rate == target_rate:
        return waveform
    try:
        import torchaudio

        return torchaudio.functional.resample(waveform, source_rate, target_rate)
    except (ImportError, OSError, RuntimeError):
        output_length = max(1, int(round(waveform.shape[-1] * target_rate / source_rate)))
        return functional.interpolate(
            waveform.unsqueeze(0), output_length, mode="linear", align_corners=False
        ).squeeze(0)


def load_audio(path: str, sample_rate: int, mono: bool = True) -> torch.Tensor:
    """Load float audio as ``[T]`` for mono or ``[C,T]`` otherwise."""
    audio_path = Path(path).expanduser()
    if not audio_path.is_file():
        raise FileNotFoundError("Audio file not found: {}".format(audio_path))
    try:
        import torchaudio

        waveform, source_rate = torchaudio.load(str(audio_path))
        waveform = waveform.float()
    except (ImportError, OSError, RuntimeError):
        try:
            import soundfile as sf
        except ImportError:
            with wave.open(str(audio_path), "rb") as stream:
                source_rate = stream.getframerate()
                channels = stream.getnchannels()
                width = stream.getsampwidth()
                raw = stream.readframes(stream.getnframes())
            if width == 1:
                samples = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
            elif width == 2:
                samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
            elif width == 4:
                samples = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
            else:
                raise RuntimeError("Unsupported PCM sample width: {}".format(width))
            waveform = torch.from_numpy(samples.reshape(-1, channels).T.copy())
        else:
            samples, source_rate = sf.read(str(audio_path), always_2d=True, dtype="float32")
            waveform = torch.from_numpy(np.asarray(samples).T.copy())
    if waveform.numel() == 0:
        raise ValueError("Empty audio file: {}".format(audio_path))
    if mono:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = _resample(waveform, int(source_rate), int(sample_rate))
    if not torch.isfinite(waveform).all():
        raise ValueError("Non-finite audio samples: {}".format(audio_path))
    return waveform.squeeze(0).contiguous() if mono else waveform.contiguous()


def save_audio(path: str, waveform: torch.Tensor, sample_rate: int) -> None:
    """Save an unclipped mono float waveform.

    Metrics must be computed before this function. Float WAV avoids implicit
    independent peak normalization and preserves out-of-range samples.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    samples = waveform.detach().float().cpu().reshape(-1)
    try:
        import soundfile as sf

        sf.write(str(target), samples.numpy(), sample_rate, subtype="FLOAT")
    except ImportError:
        try:
            import torchaudio
            torchaudio.save(
                str(target),
                samples.unsqueeze(0),
                sample_rate,
                encoding="PCM_F",
                bits_per_sample=32,
            )
        except (ImportError, OSError, RuntimeError):
            # Last-resort stdlib backend. Only the persisted listening copy is
            # quantized/clipped; metrics always use the original float tensor.
            pcm = (
                samples.clamp(-1.0, 1.0).mul(32767.0).round().to(torch.int16).numpy()
            )
            with wave.open(str(target), "wb") as stream:
                stream.setnchannels(1)
                stream.setsampwidth(2)
                stream.setframerate(sample_rate)
                stream.writeframes(pcm.astype("<i2", copy=False).tobytes())
