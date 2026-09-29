"""Generic raw-IQ input loader."""

from pathlib import Path
import numpy as np
from scipy.io import wavfile


def load_iq(path, dtype=np.float32, iq_order="IQ", max_samples=None):
    """
    Load .bin/.iq/.wav and return complex IQ plus sampling rate.

    .bin/.iq default:
        float32 interleaved I,Q:
        I0,Q0,I1,Q1,...

    .wav:
        stereo -> channel 0 = I, channel 1 = Q
        mono   -> real signal with Q=0

    For .bin/.iq, fs is returned as None because the sampling rate is
    normally metadata supplied by the dataset/application.
    """
    path = Path(path)
    ext = path.suffix.lower()

    if ext in {".bin", ".iq"}:
        raw = np.fromfile(path, dtype=dtype)

        if raw.size < 2:
            raise ValueError("File contains fewer than one complex IQ sample.")

        if max_samples is not None:
            raw = raw[:2 * max_samples]

        raw = raw[:2 * (raw.size // 2)]

        I = raw[0::2]
        Q = raw[1::2]

        if iq_order.upper() == "QI":
            I, Q = Q, I
        elif iq_order.upper() != "IQ":
            raise ValueError("iq_order must be 'IQ' or 'QI'.")

        return (I + 1j * Q).astype(np.complex64), None

    if ext == ".wav":
        fs, data = wavfile.read(path)

        if np.issubdtype(data.dtype, np.integer):
            info = np.iinfo(data.dtype)
            scale = max(abs(info.min), info.max)
            data = data.astype(np.float32) / scale
        else:
            data = data.astype(np.float32)

        if data.ndim == 1:
            iq = data.astype(np.complex64)
        elif data.ndim == 2 and data.shape[1] >= 2:
            iq = (data[:, 0] + 1j * data[:, 1]).astype(np.complex64)
        else:
            raise ValueError("Unsupported WAV channel configuration.")

        if max_samples is not None:
            iq = iq[:max_samples]

        return iq, fs

    raise ValueError("Supported input types are .bin, .iq and .wav.")
