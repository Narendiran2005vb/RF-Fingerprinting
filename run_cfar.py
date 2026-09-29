"""
COMPLETE LoRadar / generic-IQ detection pipeline.

Input:
    .bin, .iq or .wav containing raw IQ

Output:
    1. IQ waveform information
    2. spectrum
    3. spectrogram
    4. 2-D CA-CFAR threshold/detection
    5. source count
    6. source parameter table
    7. .npz detection results
    8. source list CSV

IMPORTANT:
For .bin/.iq files, set FS explicitly because the file itself normally
does not contain the sampling rate.
"""

from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

from loradar_cfar.iq_loader import load_iq
from loradar_cfar.preprocess import preprocess_iq, stft_power
from loradar_cfar.cfar import ca_cfar_2d
from loradar_cfar.source_detection import extract_sources


# ============================================================
# USER CONFIGURATION
# ============================================================

INPUT_FILE = "./lora_part001/test1.wav"

# For LoRadar screenshot:
FS = 4e6

# During development, limit memory/time. Set None for whole file.
MAX_SAMPLES = 4_000_000

# Generic IQ preprocessing:
REMOVE_DC = True

# Leave these None for a universal first pass.
# Only enable when you know the desired signal band.
LOWCUT = None
HIGHCUT = None

# STFT
NFFT = 4096
NPERSEG = 4096
NOVERLAP = 3072

# ------------------------------------------------------------
# CA-CFAR parameters
# ------------------------------------------------------------
# Training/guard cells are in BOTH directions.
#
# Frequency:
#   20 training cells on each side
#   4 guard cells on each side
#
# Time:
#   8 training frames on each side
#   2 guard frames on each side
#
TRAINING_FREQ = 20
GUARD_FREQ = 4
TRAINING_TIME = 8
GUARD_TIME = 2

PFA = 1e-5

# Remove very small connected regions.
MIN_DETECTION_PIXELS = 10

# Merge small holes/gaps in detected signal regions.
CLOSING_SIZE = (3, 3)

OUTPUT_DIR = Path("cfar_results")


# ============================================================
# PIPELINE
# ============================================================

def main():

    OUTPUT_DIR.mkdir(exist_ok=True)

    # 1. Load
    iq, file_fs = load_iq(
        INPUT_FILE,
        dtype=np.float32,
        iq_order="IQ",
        max_samples=MAX_SAMPLES,
    )

    fs = file_fs if file_fs is not None else FS

    if fs is None:
        raise ValueError(
            "Sampling rate is required for .bin/.iq input. Set FS."
        )

    print("=" * 65)
    print("RAW IQ")
    print("=" * 65)
    print(f"File             : {INPUT_FILE}")
    print(f"Complex samples  : {len(iq):,}")
    print(f"Sampling rate    : {fs/1e6:.6f} MHz")
    print(f"Duration         : {len(iq)/fs:.6f} s")

    # 2. Preprocess
    iq_p = preprocess_iq(
        iq,
        fs=fs,
        remove_dc_offset=REMOVE_DC,
        lowcut=LOWCUT,
        highcut=HIGHCUT,
        normalize=False,
    )

    # 3. STFT
    f, t, Zxx, power = stft_power(
        iq_p,
        fs=fs,
        nfft=NFFT,
        nperseg=NPERSEG,
        noverlap=NOVERLAP,
    )

    # Normalize ONLY for visualization.
    power_db = 10 * np.log10(
        power / (np.max(power) + 1e-30) + 1e-30
    )

    # 4. 2-D CA-CFAR
    detection, threshold, noise, alpha, n_training = ca_cfar_2d(
        power,
        training_freq=TRAINING_FREQ,
        guard_freq=GUARD_FREQ,
        training_time=TRAINING_TIME,
        guard_time=GUARD_TIME,
        pfa=PFA,
    )

    # 5. Source extraction
    sources, labels, source_ids, num_sources = extract_sources(
        detection,
        power,
        f,
        t,
        min_pixels=MIN_DETECTION_PIXELS,
        closing_size=CLOSING_SIZE,
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print("\n" + "=" * 65)
    print("CA-CFAR")
    print("=" * 65)
    print(f"Pfa              : {PFA:g}")
    print(f"Training cells   : {n_training}")
    print(f"Alpha            : {alpha:.6f}")
    print(f"Raw detections   : {np.count_nonzero(detection):,}")
    print(f"Number of sources: {num_sources}")

    print("\nSource ID array:")
    print(source_ids)

    if sources:
        print("\nDetected sources:")
        for s in sources:
            print(
                f"  Source {s['source_id']:2d}: "
                f"f={s['peak_frequency_hz']/1e6:+.6f} MHz, "
                f"BW={s['bandwidth_hz']/1e3:.2f} kHz, "
                f"time={s['time_start_s']:.4f}–{s['time_end_s']:.4f} s"
            )
    else:
        print("No source survived the minimum-pixel criterion.")

    # ========================================================
    # FIGURE 1: SPECTRUM
    # ========================================================

    mid = power.shape[1] // 2
    spectrum_db = 10 * np.log10(
        power[:, mid] / (np.max(power[:, mid]) + 1e-30) + 1e-30
    )

    plt.figure(figsize=(10, 5))
    plt.plot(f / 1e6, spectrum_db)
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Power (dB, normalized)")
    plt.title("LoRadar / IQ Power Spectrum")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_spectrum.png", dpi=160)
    plt.show()

    # ========================================================
    # FIGURE 2: SPECTROGRAM
    # ========================================================

    plt.figure(figsize=(11, 6))
    plt.pcolormesh(
        t,
        f / 1e6,
        power_db,
        shading="auto",
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (MHz)")
    plt.title("Input IQ Power Spectrogram")
    plt.colorbar(label="Power (dB)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_spectrogram.png", dpi=160)
    plt.show()

    # ========================================================
    # FIGURE 3: CFAR DETECTION MAP
    # ========================================================

    plt.figure(figsize=(11, 6))
    plt.pcolormesh(
        t,
        f / 1e6,
        detection.astype(float),
        shading="auto",
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (MHz)")
    plt.title("2-D CA-CFAR Detection Map")
    plt.colorbar(label="Detection (0/1)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_cfar_detection.png", dpi=160)
    plt.show()

    # ========================================================
    # FIGURE 4: SOURCE LABELS
    # ========================================================

    plt.figure(figsize=(11, 6))
    plt.pcolormesh(
        t,
        f / 1e6,
        labels,
        shading="auto",
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (MHz)")
    plt.title(f"Detected Signal Sources: {num_sources}")
    plt.colorbar(label="Source ID")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_sources.png", dpi=160)
    plt.show()

    # ========================================================
    # FIGURE 5: THRESHOLD VS SIGNAL AT A REPRESENTATIVE TIME
    # ========================================================

    # Select the time frame with maximum total received power.
    frame_power = np.sum(power, axis=0)
    k = int(np.argmax(frame_power))

    signal_db = 10 * np.log10(power[:, k] + 1e-30)
    threshold_db = 10 * np.log10(threshold[:, k] + 1e-30)

    plt.figure(figsize=(11, 5))
    plt.plot(f / 1e6, signal_db, label="Signal power")
    plt.plot(f / 1e6, threshold_db, label="CA-CFAR threshold")
    plt.plot(
        f[detection[:, k]] / 1e6,
        signal_db[detection[:, k]],
        "o",
        markersize=4,
        label="Detection",
    )
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Power (dB)")
    plt.title(
        f"CA-CFAR Threshold at t = {t[k]:.6f} s"
    )
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "05_threshold_and_detection.png", dpi=160)
    plt.show()

    # ========================================================
    # SAVE MACHINE-READABLE RESULTS
    # ========================================================

    np.savez_compressed(
        OUTPUT_DIR / "cfar_results.npz",
        frequencies_hz=f,
        times_s=t,
        power=power.astype(np.float32),
        power_db=power_db.astype(np.float32),
        threshold=threshold.astype(np.float32),
        noise_estimate=noise.astype(np.float32),
        detection=detection,
        source_labels=labels,
        source_ids=source_ids,
        num_sources=np.array([num_sources], dtype=np.int32),
        pfa=np.array([PFA]),
        alpha=np.array([alpha]),
        training_cells=np.array([n_training], dtype=np.int32),
    )

    with open(OUTPUT_DIR / "sources.csv", "w", newline="") as fp:
        if sources:
            writer = csv.DictWriter(fp, fieldnames=sources[0].keys())
            writer.writeheader()
            writer.writerows(sources)
        else:
            fp.write("source_id\n")

    print("\nResults saved in:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
