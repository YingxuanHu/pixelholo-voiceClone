import base64
import io
import os
import sys
import threading
from pathlib import Path

import numpy as np
import torch
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STYLE_TTS2_DIR = PROJECT_ROOT / "lib" / "StyleTTS2"

if STYLE_TTS2_DIR.exists():
    sys.path.insert(0, str(STYLE_TTS2_DIR))

MEAN = -4
STD = 4

app = FastAPI()
_engine_lock = threading.Lock()
_engines: dict[tuple[str, str], "StyleTTS2RepoEngine"] = {}


def _audio_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    audio = np.asarray(audio, dtype=np.float32)
    audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
    audio = np.clip(audio, -1.0, 1.0)
    pcm16 = (audio * 32767.0).astype(np.int16)

    import wave

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())

    return buffer.getvalue()


def _resolve_path(base_dir: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path


def _first_wav(directory: Path | None) -> Path | None:
    if directory is None or not directory.exists():
        return None
    for wav_path in sorted(directory.glob("*.wav")):
        return wav_path
    return None


def _resolve_ref_wav(ref_wav_path: str | None, speaker: str | None) -> Path:
    if ref_wav_path:
        path = Path(ref_wav_path).expanduser()
        if path.exists():
            return path
        raise HTTPException(status_code=400, detail=f"ref_wav_path not found: {ref_wav_path}")

    env_ref = os.getenv("STYLE_TTS2_REF_WAV")
    if env_ref:
        path = Path(env_ref).expanduser()
        if path.exists():
            return path
        raise HTTPException(status_code=400, detail=f"STYLE_TTS2_REF_WAV not found: {env_ref}")

    env_dir = os.getenv("STYLE_TTS2_REF_DIR")
    if env_dir:
        candidate = _first_wav(Path(env_dir).expanduser())
        if candidate:
            return candidate
        raise HTTPException(status_code=400, detail=f"No wav files in STYLE_TTS2_REF_DIR: {env_dir}")

    if speaker:
        candidate = _first_wav(PROJECT_ROOT / "data" / speaker / "processed_wavs")
        if candidate:
            return candidate

    data_root = PROJECT_ROOT / "data"
    processed_dirs = sorted(data_root.glob("*/processed_wavs"))
    if len(processed_dirs) == 1:
        candidate = _first_wav(processed_dirs[0])
        if candidate:
            return candidate

    raise HTTPException(
        status_code=400,
        detail=(
            "Provide ref_wav_path or set STYLE_TTS2_REF_WAV/STYLE_TTS2_REF_DIR. "
            "If using speaker, pass the speaker name."
        ),
    )


def _resolve_config_path(model_path: Path, config_path: str | None) -> Path:
    if config_path:
        path = Path(config_path).expanduser()
        if path.exists():
            return path
        raise HTTPException(status_code=400, detail=f"config_path not found: {config_path}")

    env_config = os.getenv("STYLE_TTS2_CONFIG")
    if env_config:
        path = Path(env_config).expanduser()
        if path.exists():
            return path
        raise HTTPException(status_code=400, detail=f"STYLE_TTS2_CONFIG not found: {env_config}")

    candidate = model_path.parent / "config_ft.yml"
    if candidate.exists():
        return candidate

    raise HTTPException(
        status_code=400,
        detail="config_path is required (or set STYLE_TTS2_CONFIG).",
    )


class GenerateRequest(BaseModel):
    text: str
    model_path: str | None = None
    config_path: str | None = None
    ref_wav_path: str | None = None
    speaker: str | None = None
    alpha: float = 0.1
    beta: float = 0.1
    diffusion_steps: int = 10
    embedding_scale: float = 1.5
    return_base64: bool = False


class StyleTTS2RepoEngine:
    def __init__(self, model_path: Path, config_path: Path, device: str | None = None):
        self.model_path = model_path
        self.config_path = config_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.style_cache: dict[str, torch.Tensor] = {}
        self._load_backend()

    def _load_backend(self) -> None:
        if not STYLE_TTS2_DIR.exists():
            raise RuntimeError("StyleTTS2 repo not found at lib/StyleTTS2.")

        try:
            import librosa
            import phonemizer
            import torchaudio
        except Exception as exc:
            raise RuntimeError(
                "Missing inference deps. Install `librosa`, `phonemizer`, and `torchaudio`."
            ) from exc

        try:
            from nltk.tokenize import word_tokenize
        except Exception:
            word_tokenize = None

        from models import build_model, load_ASR_models, load_F0_models, load_checkpoint
        from utils import length_to_mask, recursive_munch
        from text_utils import TextCleaner
        from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule
        from Utils.PLBERT.util import load_plbert

        self._librosa = librosa
        self._phonemizer = phonemizer.backend.EspeakBackend(
            language="en-us",
            preserve_punctuation=True,
            with_stress=True,
        )
        self._word_tokenize = word_tokenize
        self._length_to_mask = length_to_mask
        self._text_cleaner = TextCleaner()

        with self.config_path.open("r") as handle:
            self.config = yaml.safe_load(handle)

        preprocess_params = self.config.get("preprocess_params", {})
        self.sample_rate = int(preprocess_params.get("sr", 24000))
        spect_params = preprocess_params.get("spect_params", {})
        n_fft = spect_params.get("n_fft", 2048)
        win_length = spect_params.get("win_length", 1200)
        hop_length = spect_params.get("hop_length", 300)
        n_mels = int(self.config.get("model_params", {}).get("n_mels", 80))

        self._to_mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
        )

        model_params = recursive_munch(self.config["model_params"])
        self.model_params = model_params

        asr_config = _resolve_path(STYLE_TTS2_DIR, self.config.get("ASR_config", ""))
        asr_path = _resolve_path(STYLE_TTS2_DIR, self.config.get("ASR_path", ""))
        f0_path = _resolve_path(STYLE_TTS2_DIR, self.config.get("F0_path", ""))
        plbert_dir = _resolve_path(STYLE_TTS2_DIR, self.config.get("PLBERT_dir", ""))

        if not (asr_config and asr_path and f0_path and plbert_dir):
            raise RuntimeError("Missing ASR/F0/PLBERT paths in config.")

        text_aligner = load_ASR_models(str(asr_path), str(asr_config))
        pitch_extractor = load_F0_models(str(f0_path))
        plbert = load_plbert(str(plbert_dir))

        model = build_model(model_params, text_aligner, pitch_extractor, plbert)
        _ = [model[key].to(self.device) for key in model]
        model, _, _, _ = load_checkpoint(model, None, str(self.model_path), load_only_params=True)
        _ = [model[key].eval() for key in model]

        self.model = model
        self.sampler = DiffusionSampler(
            model.diffusion.diffusion,
            sampler=ADPM2Sampler(),
            sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
            clamp=False,
        )

    def _tokenize(self, text: str) -> list[str]:
        if self._word_tokenize is None:
            return text.split()
        try:
            return self._word_tokenize(text)
        except LookupError:
            return text.split()

    def _preprocess(self, wave: np.ndarray) -> torch.Tensor:
        wave_tensor = torch.from_numpy(wave).float()
        mel_tensor = self._to_mel(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - MEAN) / STD
        return mel_tensor

    def _compute_style(self, wav_path: Path) -> torch.Tensor:
        cache_key = str(wav_path)
        if cache_key in self.style_cache:
            return self.style_cache[cache_key]

        wave, sr = self._librosa.load(str(wav_path), sr=self.sample_rate)
        audio, _index = self._librosa.effects.trim(wave, top_db=30)
        if sr != self.sample_rate:
            audio = self._librosa.resample(audio, sr, self.sample_rate)
        mel_tensor = self._preprocess(audio).to(self.device)

        with torch.no_grad():
            ref_s = self.model.style_encoder(mel_tensor.unsqueeze(1))
            ref_p = self.model.predictor_encoder(mel_tensor.unsqueeze(1))

        style = torch.cat([ref_s, ref_p], dim=1)
        self.style_cache[cache_key] = style
        return style

    def generate(
        self,
        text: str,
        ref_wav_path: Path,
        alpha: float,
        beta: float,
        diffusion_steps: int,
        embedding_scale: float,
    ) -> np.ndarray:
        text = text.strip()
        if not text:
            raise ValueError("Text is empty.")

        phonemes = self._phonemizer.phonemize([text])[0]
        tokens = " ".join(self._tokenize(phonemes))
        token_ids = self._text_cleaner(tokens)
        token_ids.insert(0, 0)

        tokens_tensor = torch.LongTensor(token_ids).to(self.device).unsqueeze(0)

        ref_s = self._compute_style(ref_wav_path)
        style_dim = ref_s.shape[-1] // 2

        with torch.no_grad():
            input_lengths = torch.LongTensor([tokens_tensor.shape[-1]]).to(self.device)
            text_mask = self._length_to_mask(input_lengths).to(self.device)

            t_en = self.model.text_encoder(tokens_tensor, input_lengths, text_mask)
            bert_dur = self.model.bert(tokens_tensor, attention_mask=(~text_mask).int())
            d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)

            noise = torch.randn((1, style_dim * 2)).unsqueeze(1).to(self.device)
            s_pred = self.sampler(
                noise=noise,
                embedding=bert_dur,
                embedding_scale=embedding_scale,
                features=ref_s,
                num_steps=diffusion_steps,
            ).squeeze(1)

            s = s_pred[:, style_dim:]
            ref = s_pred[:, :style_dim]

            ref = alpha * ref + (1 - alpha) * ref_s[:, :style_dim]
            s = beta * s + (1 - beta) * ref_s[:, style_dim:]

            d = self.model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
            x, _ = self.model.predictor.lstm(d)
            duration = self.model.predictor.duration_proj(x)

            duration = torch.sigmoid(duration).sum(axis=-1)
            duration = torch.nan_to_num(duration, nan=1.0, posinf=1.0, neginf=1.0)
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)
            if pred_dur.dim() == 0:
                pred_dur = pred_dur.unsqueeze(0)

            total_frames = int(torch.clamp(pred_dur.sum(), min=1).item())
            pred_aln_trg = torch.zeros(
                (input_lengths.item(), total_frames),
                device=self.device,
            )
            c_frame = 0
            for idx, dur in enumerate(pred_dur):
                dur_i = max(1, int(dur.item()))
                pred_aln_trg[idx, c_frame : c_frame + dur_i] = 1
                c_frame += dur_i

            en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0))
            if self.model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(en)
                asr_new[:, :, 0] = en[:, :, 0]
                asr_new[:, :, 1:] = en[:, :, 0:-1]
                en = asr_new

            f0_pred, n_pred = self.model.predictor.F0Ntrain(en, s)

            asr = (t_en @ pred_aln_trg.unsqueeze(0))
            if self.model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(asr)
                asr_new[:, :, 0] = asr[:, :, 0]
                asr_new[:, :, 1:] = asr[:, :, 0:-1]
                asr = asr_new

            out = self.model.decoder(asr, f0_pred, n_pred, ref.squeeze().unsqueeze(0))

        return out.squeeze().cpu().numpy()[..., :-50]


def _get_engine(model_path: Path, config_path: Path) -> StyleTTS2RepoEngine:
    key = (str(model_path), str(config_path))
    with _engine_lock:
        if key not in _engines:
            _engines[key] = StyleTTS2RepoEngine(model_path=model_path, config_path=config_path)
        return _engines[key]


@app.on_event("startup")
def _startup() -> None:
    default_model = os.getenv("STYLE_TTS2_MODEL")
    if default_model:
        model_path = Path(default_model).expanduser()
        config_path = _resolve_config_path(model_path, os.getenv("STYLE_TTS2_CONFIG"))
        _get_engine(model_path, config_path)


@app.post("/generate")
def generate(req: GenerateRequest):
    model_path_value = req.model_path or os.getenv("STYLE_TTS2_MODEL")
    if not model_path_value:
        raise HTTPException(status_code=400, detail="model_path is required")

    model_path = Path(model_path_value).expanduser()
    if not model_path.exists():
        raise HTTPException(status_code=400, detail=f"model_path not found: {model_path}")

    config_path = _resolve_config_path(model_path, req.config_path)
    ref_wav_path = _resolve_ref_wav(req.ref_wav_path, req.speaker)

    try:
        engine = _get_engine(model_path, config_path)
        audio = engine.generate(
            req.text,
            ref_wav_path=ref_wav_path,
            alpha=req.alpha,
            beta=req.beta,
            diffusion_steps=req.diffusion_steps,
            embedding_scale=req.embedding_scale,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    wav_bytes = _audio_to_wav_bytes(audio, engine.sample_rate)

    if req.return_base64:
        payload = base64.b64encode(wav_bytes).decode("ascii")
        return JSONResponse({"audio_base64": payload})

    return Response(content=wav_bytes, media_type="audio/wav")
