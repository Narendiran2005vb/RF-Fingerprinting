"""Generic IQ preprocessing and time-frequency representation."""

import numpy as np
from scipy import signal


def remove_dc(iq):
    return np.asarray(iq) - np.mean(iq)


def bandpass_filter(iq, fs, lowcut=None, highcut=None, order=5):
    if fs is None:
        raise ValueError("fs is required when filtering is enabled.")

    nyq = fs / 2.0

    if lowcut is None and highcut is None:
        return iq

    if lowcut is not None and lowcut <= 0:
        lowcut = None
    if highcut is not None and highcut >= nyq:
        highcut = None

    if lowcut is not None and highcut is not None:
        if not 0 < lowcut < highcut < nyq:
            raise ValueError("Require 0 < lowcut < highcut < fs/2.")
        wn = [lowcut / nyq, highcut / nyq]
        btype = "bandpass"
    elif lowcut is not None:
        wn = lowcut / nyq
        btype = "highpass"
    else:
        wn = highcut / nyq
        btype = "lowpass"

    sos = signal.butter(order, wn, btype=btype, output="sos")
    return signal.sosfiltfilt(sos, iq)


def preprocess_iq(
    iq,
    fs=None,
    remove_dc_offset=True,
    lowcut=None,
    highcut=None,
    filter_order=5,
    normalize=False,
):
    """
    Generic preprocessing.

    Recommended starting point for unknown IQ:
        remove_dc_offset=True
        lowcut=None
        highcut=None
        normalize=False

    Filtering is optional because a universal cutoff cannot be assumed
    for voice, telemetry, satellite, LoRa, ADS-B, radar, etc.
    """
    iq = np.asarray(iq, dtype=np.complex64)

    if remove_dc_offset:
        iq = remove_dc(iq)

    if lowcut is not None or highcut is not None:
        iq = bandpass_filter(
            iq, fs, lowcut, highcut, order=filter_order
        )

    if normalize:
        rms = np.sqrt(np.mean(np.abs(iq) ** 2))
        if rms > 0:
            iq = iq / rms

    return iq.astype(np.complex64)


def stft_power(
    iq,
    fs,
    nfft=4096,
    nperseg=4096,
    noverlap=3072,
    window="hann",
):
    """
    Return frequency, time, complex STFT and linear power.

    For complex IQ, return_onesided=False is mandatory.
    """
    f, t, Zxx = signal.stft(
        iq,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        return_onesided=False,
        boundary=None,
        padded=False,
    )

    idx = np.fft.fftshift(np.arange(len(f)))
    f = np.fft.fftshift(f)
    Zxx = Zxx[idx, :]

    power = np.abs(Zxx) ** 2
    return f, t, Zxx, power


def iq_to_power_spectrum(iq, fs, nfft=4096, window="hann"):
    """Return frequency bins, FFT and linear power for one IQ block."""
    iq = np.asarray(iq)

    if len(iq) < nfft:
        iq = np.pad(iq, (0, nfft - len(iq)))
    else:
        iq = iq[:nfft]

    if window == "hann":
        w = np.hanning(nfft)
    elif window is None:
        w = np.ones(nfft)
    else:
        raise ValueError("window must be 'hann' or None.")

    X = np.fft.fftshift(np.fft.fft(iq * w, n=nfft))
    f = np.fft.fftshift(np.fft.fftfreq(nfft, 1 / fs))
    power = np.abs(X) ** 2
    return f, X, power
