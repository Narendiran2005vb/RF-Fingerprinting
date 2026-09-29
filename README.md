# Generic IQ → STFT → 2-D CA-CFAR → Source Detection

This is Stage 1 of the modular IQ signal-processing project.

It is designed to be usable For Finding Signal Sources from Noise chanels and give it to the next stages:

- LoRa
- voice / audio IQ
- satellite communication
- telemetry
- FSK/PSK/QAM/OFDM-like signals
- unknown RF signals
- other complex baseband recordings

## Architecture

```text
              .bin / .iq / .wav
                       |
                       v
                  load_iq()
                       |
                       v
                  Complex IQ
                       |
                       v
               preprocess_iq()
                       |
                 DC removal
                 optional filter
                       |
                       v
                     STFT
                       |
                       v
                Linear Power
                   |P(f,t)|²
                       |
                       v
                 2-D CA-CFAR
                       |
            +----------+----------+
            |                     |
            v                     v
       Detection mask       Threshold map
            |
            v
    Connected-component
       source extraction
            |
            v
  Number of detected sources
  Frequency / bandwidth
  Start/end time
  Peak frequency
  Peak time
```



## CA-CFAR equation

For 2-D CA-CFAR:

```text
N = number of training cells

noise_estimate = (sum of training-cell power) / N

alpha = N * (Pfa^(-1/N) - 1)

threshold = alpha * noise_estimate

detect if:
    CUT_power > threshold
```

## Current default parameters

```python
TRAINING_FREQ = 20
GUARD_FREQ = 4

TRAINING_TIME = 8
GUARD_TIME = 2

PFA = 1e-5
```

These are starting values, not universal optimum values.

The correct values depend on:

- signal bandwidth
- STFT frequency resolution
- STFT time resolution
- noise statistics
- SNR
- desired probability of false alarm

## Source counting

After CFAR, a 2-D detection mask is produced.

Connected components are then extracted:

```text
CFAR map

       █████
       █████
                    ████
                    ████
          ███

       ↓ connected components

Source 1
Source 2
Source 3
```

The program returns:

```python
num_sources
```

and:

```python
source_ids
```

It also creates:

```python
source_labels
```

which is a 2-D array with:

```text
0 = no detected source
1 = source 1
2 = source 2
3 = source 3
...
```

### Important interpretation

A connected component is an **observed time-frequency source region**.

It does NOT automatically prove there are exactly N physical transmitters.

One physical signal can fragment into multiple components, and multiple overlapping signals can merge into one component. The later source-tracking/classification stage can improve this.


## Running

Install:

```bash
pip install -r requirements.txt
```

Edit:

```python
INPUT_FILE = "./lora_part001/lora_process_energy_1.bin"
FS = 4e6
```

Then:

```bash
python run_cfar.py
```

The program creates:

```text
cfar_results/
├── 01_spectrum.png
├── 02_spectrogram.png
├── 03_cfar_detection.png
├── 04_sources.png
├── 05_threshold_and_detection.png
├── cfar_results.npz
└── sources.csv
```

## Next stage

After validating this detector on LoRadar, we add a separate signal-parameter module:

```text
CFAR detection
      ↓
source regions
      ↓
┌──────────────────────────┐
│ Parameter extraction     │
├──────────────────────────┤
│ Center frequency         │
│ Bandwidth                │
│ Duration                 │
│ SNR                      │
│ Symbol rate              │
│ Carrier frequency        │
│ Modulation               │
│ FEC / interleaving       │
└──────────────────────────┘
```


