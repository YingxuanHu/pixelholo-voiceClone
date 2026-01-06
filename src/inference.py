import base64
import io
import os
import sys
import threading
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from config import DEFAULT_DEVICE, DEFAULT_SAMPLE_RATE  # noqa: E402

app = FastAPI()
_engine_lock = threading.Lock()
_engine = None


def _audio_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    audio = np.asarray(audio, dtype=np.float32)
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


class GenerateRequest(BaseModel):
    text: str
    model_path: str | None = None
    return_base64: bool = False


class StyleTTS2Engine:
    def __init__(self, model_path: str, device: str = DEFAULT_DEVICE):
        self.model_path = model_path
        self.device = device
        self.engine = self._load_backend()

    def _load_backend(self):
        try:
            import importlib

            module = importlib.import_module("styletts2")
        except ImportError as exc:
            raise RuntimeError(
                "styletts2 package not installed. Install it or add a compatible backend."
            ) from exc

        engine_cls = None
        for attr in ("StyleTTS2", "TTS"):
            if hasattr(module, attr):
                engine_cls = getattr(module, attr)
                break

        if engine_cls is None:
            raise RuntimeError("styletts2 module does not expose a supported TTS class.")

        if hasattr(engine_cls, "from_pretrained"):
            engine = engine_cls.from_pretrained(self.model_path, device=self.device)
        else:
            try:
                engine = engine_cls(self.model_path, device=self.device)
            except TypeError:
                engine = engine_cls(self.model_path)

        if hasattr(engine, "to"):
            engine.to(self.device)
        return engine

    def generate(self, text: str):
        if hasattr(self.engine, "inference"):
            return self.engine.inference(text)
        if callable(self.engine):
            return self.engine(text)
        raise RuntimeError("Loaded StyleTTS2 engine does not support inference.")


def _get_engine(model_path: str) -> StyleTTS2Engine:
    global _engine
    with _engine_lock:
        if _engine is None or _engine.model_path != model_path:
            _engine = StyleTTS2Engine(model_path=model_path)
        return _engine


@app.on_event("startup")
def _startup() -> None:
    default_model = os.getenv("STYLE_TTS2_MODEL")
    if default_model:
        _get_engine(default_model)


@app.post("/generate")
def generate(req: GenerateRequest):
    model_path = req.model_path or os.getenv("STYLE_TTS2_MODEL")
    if not model_path:
        raise HTTPException(status_code=400, detail="model_path is required")

    try:
        engine = _get_engine(model_path)
        audio = engine.generate(req.text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    wav_bytes = _audio_to_wav_bytes(audio, DEFAULT_SAMPLE_RATE)

    if req.return_base64:
        payload = base64.b64encode(wav_bytes).decode("ascii")
        return JSONResponse({"audio_base64": payload})

    return Response(content=wav_bytes, media_type="audio/wav")
