# Modular IQ Signal Processing and Modulation Detection

A modular IQ signal-processing pipeline for detecting RF signal sources and subsequently classifying their modulation type.

The system is divided into independent processing stages so that signal detection and modulation classification can be developed, tested, and improved separately.

---

# System Architecture

```text
                    IQ / RF Recording
                           |
                           v
                  ┌─────────────────┐
                  │     Stage 1     │
                  │ Source Detection│
                  └────────┬────────┘
                           |
                    Complex IQ data
                           |
                           v
                  IQ preprocessing
                  DC removal/filter
                           |
                           v
                         STFT
                           |
                           v
                   Power Spectrogram
                           |
                           v
                     2-D CA-CFAR
                           |
                           v
                   Detection Mask
                           |
                           v
              Connected Components
                           |
                           v
              ┌─────────────────────┐
              │ Detected RF Sources │
              ├─────────────────────┤
              │ Frequency           │
              │ Bandwidth           │
              │ Start time          │
              │ End time            │
              │ Peak frequency      │
              │ Source region       │
              └──────────┬──────────┘
                         |
                         v
                  ┌──────────────┐
                  │    Stage 2   │
                  │ CNN Detector │
                  └──────┬───────┘
                         |
                         v
                IQ source segment
                         |
                         v
                 RMS normalization
                         |
                         v
              2 × 1024 IQ representation
                         |
                         v
              Frame extension → 4 × 1024
                         |
                         v
                    CNN model
                         |
                         v
              Modulation classification
                         |
                         v
        ┌────────────────────────────────┐
        │ Predicted Modulation Class     │
        │                                │
        │ e.g. PSK / QAM / FSK / OFDM   │
        │ or other supported classes     │
        └────────────────────────────────┘
```

The intended complete pipeline is therefore:

```text
RF/IQ recording
      ↓
Signal/source detection
      ↓
Source segmentation
      ↓
Signal parameter extraction
      ↓
CNN modulation classification
      ↓
Detected source + modulation information
```

---

# Stage 1 — Signal Source Detection

Stage 1 is a generic time-frequency signal detector designed to identify signal-bearing regions in an IQ recording.

It is intended to work with different types of complex baseband signals, including:

* LoRa
* Voice/audio IQ
* Satellite communication
* Telemetry
* FSK
* PSK
* QAM
* OFDM-like signals
* Unknown RF signals
* Other complex baseband recordings

## Stage 1 Architecture

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
       ├── DC removal
       └── Optional filtering
       |
       v
      STFT
       |
       v
 Linear Power |P(f,t)|²
       |
       v
   2-D CA-CFAR
       |
       ├───────────────┐
       v               v
Detection mask    Threshold map
       |
       v
Connected Components
       |
       v
Detected source regions
```

---

# Stage 1 — 2-D CA-CFAR

The detector uses a two-dimensional Cell-Averaging Constant False Alarm Rate (CA-CFAR) detector on the STFT power spectrogram.

For a CUT (Cell Under Test), the training cells surrounding the CUT are used to estimate the local noise level.

The noise estimate is:

```text
noise_estimate =
    sum(training-cell power) / N
```

The CA-CFAR scaling factor is:

```text
alpha = N * (Pfa^(-1/N) - 1)
```

The detection threshold is:

```text
threshold = alpha × noise_estimate
```

A time-frequency cell is classified as a detection when:

```text
CUT_power > threshold
```

## Current Parameters

```python
TRAINING_FREQ = 20
GUARD_FREQ = 4

TRAINING_TIME = 8
GUARD_TIME = 2

PFA = 1e-5
```

These parameters are configurable.

They are not universal optimum values because their appropriate values depend on:

* Signal bandwidth
* Sampling frequency
* STFT frequency resolution
* STFT time resolution
* Noise characteristics
* SNR
* Desired probability of false alarm

---

# Stage 1 — Source Extraction

After CA-CFAR produces the binary detection mask, connected-component analysis is applied.

For example:

```text
CFAR detection map

       █████
       █████

                    ████
                    ████

          ███
```

The connected components become:

```text
Source 1
Source 2
Source 3
```

The program produces:

```python
num_sources
```

and:

```python
source_ids
```

A 2-D source-label map is also generated:

```python
source_labels
```

where:

```text
0 = no detected source
1 = source 1
2 = source 2
3 = source 3
...
```

### Important interpretation

A connected component represents an **observed time-frequency region**.

It does not necessarily represent exactly one physical transmitter.

For example:

```text
One transmitter
      ↓
Signal fragmentation
      ↓
Multiple components
```

or:

```text
Two overlapping transmitters
      ↓
Merged time-frequency region
      ↓
One component
```

Therefore, source tracking and classification can be used in later stages to improve the physical interpretation.

---

# Stage 2 — CNN Modulation Detector

Stage 2 performs modulation classification on an extracted IQ signal.

The current implementation uses a custom convolutional neural network named:

```text
ProposedCNN
```

The network receives a transformed IQ representation and produces classification logits for:

```text
24 modulation classes
```

The final classification layer is:

```python
nn.Linear(128, 24)
```

---

# Stage 2 Dataset

The CNN is trained using:

## DeepSig RadioML 2018.01A

The dataset is loaded from the Kaggle dataset:

```text
pinxau1000/radioml2018
```

The code uses `kagglehub` to download/locate the dataset automatically:

```python
dataset_dir = kagglehub.dataset_download(
    "pinxau1000/radioml2018"
)
```

The program then searches the downloaded directory for an `.hdf5` file.

The expected RadioML data structure is:

```text
X → IQ samples
Y → modulation labels
Z → SNR values
```

The expected input representation is:

```text
X: (N, 1024, 2)
Y: (N, 24)
Z: (N, 1)
```

where:

```text
N       = number of samples
1024    = number of IQ time samples
2       = I and Q
24      = modulation classes
```

The one-hot encoded labels in `Y` are converted into integer class IDs using:

```python
np.argmax(Y, axis=1)
```

---

# Stage 2 — Input Preprocessing

Each IQ sample initially has the shape:

```text
1024 × 2
```

The two columns represent:

```text
I
Q
```

The data is transposed to:

```text
2 × 1024
```

so that:

```text
Row 0 → I
Row 1 → Q
```

## RMS Normalization

The signal energy is calculated as:

```python
energy = mean(I² + Q²)
```

and:

```python
rms = sqrt(energy)
```

The IQ signal is normalized by:

```text
normalized IQ = IQ / RMS
```

A small epsilon is added to avoid division by zero.

---

# Frame Extension

After normalization, the code creates a horizontally flipped copy:

```python
flipped_signal = norm_signal[:, ::-1]
```

The original and flipped representations are concatenated vertically:

```text
Original IQ
    +
Flipped IQ
    ↓
4 × 1024 representation
```

Therefore:

```text
2 × 1024
      ↓
4 × 1024
```

The final tensor supplied to the CNN has the shape:

```text
1 × 4 × 1024
```

where:

```text
1 = CNN input channel
4 = extended IQ dimension
1024 = temporal dimension
```

---

# Stage 2 — CNN Architecture

The proposed CNN consists of:

```text
Input
  ↓
ABlock
  ↓
BBlock + Residual Connection
  ↓
Average Pooling
  ↓
CBlock 1 + Residual Connection
  ↓
CBlock 2 + Residual Connection
  ↓
Global Average Pooling
  ↓
Fully Connected Layer
  ↓
24 Classes
```

## Complete Architecture

```text
Input
1 × 4 × 1024
      |
      v
┌───────────────────────────┐
│          ABlock           │
│                           │
│ Conv 1×3   1 → 16         │
│ Conv 3×1  16 → 16         │
│ MaxPool 1×2               │
│ Conv 1×3  16 → 16         │
│ Conv 3×1  16 → 16         │
│ MaxPool 1×2               │
└─────────────┬─────────────┘
              |
              v
        16 feature maps
              |
              v
┌───────────────────────────┐
│          BBlock           │
│                           │
│ Conv 1×1  16 → 32         │
│ Conv 1×3  32 → 16         │
│ Conv 3×1  16 → 16         │
│ Conv 1×1  16 → 32         │
│                           │
│ Residual connection       │
└─────────────┬─────────────┘
              |
              v
        Average Pool
        2 × 1
              |
              v
┌───────────────────────────┐
│         CBlock 1          │
│                           │
│ Conv 1×1  32 → 64         │
│ Conv 1×3  64 → 24         │
│ Conv 1×1  24 → 64         │
│                           │
│ Residual connection       │
└─────────────┬─────────────┘
              |
              v
┌───────────────────────────┐
│         CBlock 2          │
│                           │
│ Conv 1×1  64 → 128        │
│ Conv 1×3 128 → 32         │
│ Conv 1×1  32 → 128        │
│                           │
│ Residual connection       │
└─────────────┬─────────────┘
              |
              v
 Global Average Pooling
              |
              v
         128 features
              |
              v
       Fully Connected
          128 → 24
              |
              v
       24 modulation classes
```

Every convolutional block uses:

```text
Conv2D
   ↓
BatchNorm2D
   ↓
ReLU
```

The architecture therefore combines:

* Asymmetric convolutions
* Batch normalization
* ReLU activation
* Max pooling
* Average pooling
* Residual connections
* Global average pooling
* Fully connected classification

---

# Stage 2 — Training Configuration

The current implementation uses:

```python
MAX_EPOCHS = 45
BATCH_SIZE = 64

INITIAL_LR = 0.1

LR_DROP_PERIOD = 20
LR_DROP_FACTOR = 0.1

MOMENTUM = 0.9
```

The optimizer is:

```text
SGD + Momentum
```

with:

```text
learning rate = 0.1
momentum      = 0.9
```

The learning-rate scheduler is:

```text
StepLR
```

with:

```text
step size = 20 epochs
gamma     = 0.1
```

Therefore, the learning rate follows approximately:

```text
Epoch 1–20   → 0.1
Epoch 21–40  → 0.01
Epoch 41–45  → 0.001
```

The loss function is:

```python
CrossEntropyLoss()
```

---

# Dataset Split and Validation

The current code performs an:

```text
80% / 20%
```

random split.

```python
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
```

The split uses a fixed random seed:

```python
torch.Generator().manual_seed(42)
```

which makes the split reproducible.

The resulting datasets are:

```text
80% → training data
20% → evaluation data
```

The 20% subset is currently named:

```python
test_data
```

and is used after every training epoch to calculate:

```text
Test Loss
Test Accuracy
```

### Important terminology

This implementation does **not** currently have a separate validation and test dataset.

The 20% subset functions as an evaluation/validation set during training because it is evaluated after every epoch and is also used to select the best checkpoint.

Therefore, for a formal experimental setup, it would be better to use:

```text
Training set
Validation set
Independent test set
```

For example:

```text
70% → Training
15% → Validation
15% → Final Test
```

The current implementation should therefore be described accurately as:

```text
80% Training
20% Validation/Evaluation
```

rather than claiming that it has an independent test set.

---

# Best Model Selection

After every epoch, the CNN is evaluated on the 20% evaluation subset.

The best model is selected according to:

```python
if test_acc > best_test_acc:
```

The checkpoint is saved whenever a new highest evaluation accuracy is achieved.

Therefore, the final saved model is **not necessarily the model from epoch 45**.

It is the model corresponding to the highest evaluation accuracy observed during training.

---

# CNN Weight Saving

The model weights **are saved** by the current implementation.

The configured path is:

```python
SAVE_WEIGHTS_PATH = "proposed_cnn_weights.pth"
```

When a new best evaluation accuracy is obtained, the following checkpoint is written:

```python
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'accuracy': test_acc
}, SAVE_WEIGHTS_PATH)
```

The resulting file is:

```text
proposed_cnn_weights.pth
```

It contains:

```text
epoch
model_state_dict
optimizer_state_dict
accuracy
```

Therefore, this is a **training checkpoint**, rather than weights only.

---

# Loading the Saved CNN

After training, the best checkpoint is loaded using:

```python
checkpoint = torch.load(
    SAVE_WEIGHTS_PATH,
    map_location=device
)
```

The trained CNN parameters are restored with:

```python
model.load_state_dict(
    checkpoint['model_state_dict']
)
```

The model is then evaluated again.

This provides a final verification of the saved model.

---

# SNR-Based Validation

In addition to overall classification accuracy, Stage 2 calculates classification accuracy for each SNR level.

The evaluation function groups results using:

```python
snr_results[snr]
```

For each SNR, the following are recorded:

```text
Correct predictions
Total samples
Accuracy
```

The final output is similar to:

```text
SNR (dB)   | Accuracy (%) | Samples
------------------------------------------
-20        | xx.xx        | xxxx
-18        | xx.xx        | xxxx
...
18         | xx.xx        | xxxx
20         | xx.xx        | xxxx
```

This provides a more detailed view of CNN performance than overall accuracy alone.

It allows the classifier to be evaluated across different signal-to-noise conditions.

---

# Stage 2 Evaluation Metrics

The current implementation reports:

### Overall Loss

```text
Cross-entropy loss
```

### Overall Accuracy

```text
Correct predictions / Total samples
```

### Per-SNR Accuracy

```text
Accuracy for each SNR level
```

The evaluation process uses:

```python
model.eval()
```

and:

```python
torch.no_grad()
```

so gradients are not calculated during evaluation.

---

# Stage 1 → Stage 2 Integration

The intended integration between the two stages is:

```text
                 Raw IQ Recording
                        |
                        v
                 ┌─────────────┐
                 │   Stage 1   │
                 │    STFT     │
                 │  CA-CFAR    │
                 └──────┬──────┘
                        |
                        v
                Detected Regions
                        |
          ┌─────────────┼─────────────┐
          |             |             |
          v             v             v
       Source 1      Source 2      Source 3
          |             |             |
          v             v             v
       IQ crop        IQ crop        IQ crop
          |             |             |
          └─────────────┼─────────────┘
                        |
                        v
                 ┌─────────────┐
                 │   Stage 2   │
                 │     CNN     │
                 └──────┬──────┘
                        |
                        v
              Modulation Class
```

Each detected source region from Stage 1 can eventually be converted back into an IQ segment and passed to the CNN classifier.

The Stage 2 CNN can then produce a modulation prediction for each detected source.

For example:

```text
Source 1
Frequency: 915.2 MHz
Bandwidth: 125 kHz
Duration: 42 ms
        ↓
CNN
        ↓
LoRa / modulation class
```

and:

```text
Source 2
Frequency: 916.1 MHz
Bandwidth: 200 kHz
Duration: 18 ms
        ↓
CNN
        ↓
Predicted modulation class
```

---

# Important Integration Consideration

The current CNN was trained using **RadioML 2018.01A samples of 1024 IQ samples**.

Stage 1, however, operates on arbitrary IQ recordings and extracts time-frequency regions.

Therefore, the Stage 1 output cannot simply be passed directly to the CNN yet.

An intermediate **source-to-IQ extraction/segmentation module** is required.

The future pipeline should be:

```text
Stage 1 CFAR
      ↓
Detected time-frequency region
      ↓
Determine frequency limits
      ↓
Determine time limits
      ↓
Extract corresponding IQ samples
      ↓
Optional frequency translation
      ↓
Optional resampling
      ↓
Create 1024-sample CNN frame
      ↓
RMS normalization
      ↓
CNN preprocessing
      ↓
Stage 2 classifier
```

This step is important because RadioML and the real-world Stage 1 recordings may have different:

* Sampling rates
* Signal durations
* Bandwidths
* Center frequencies
* SNRs
* Frequency offsets
* Time resolutions

---

# Project Output

Stage 1 produces:

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

Stage 2 produces:

```text
proposed_cnn_weights.pth
```

The checkpoint contains the best model encountered during training.

A future integrated implementation can additionally produce:

```text
classification_results.csv
```

containing information such as:

```text
Source ID
Center Frequency
Bandwidth
Start Time
End Time
SNR
Predicted Modulation
Classification Confidence
```

---

# Complete Modular Architecture

The complete planned system is:

```text
┌──────────────────────────────────────────────┐
│              Raw IQ Recording                │
│             .bin / .iq / .wav                │
└──────────────────────┬───────────────────────┘
                       |
                       v
┌──────────────────────────────────────────────┐
│                  STAGE 1                     │
│           Signal Source Detection            │
│                                              │
│  IQ preprocessing                            │
│       ↓                                      │
│  STFT                                        │
│       ↓                                      │
│  Power Spectrogram                           │
│       ↓                                      │
│  2-D CA-CFAR                                 │
│       ↓                                      │
│  Connected Components                        │
└──────────────────────┬───────────────────────┘
                       |
                       v
             Detected Sources
                       |
                       v
┌──────────────────────────────────────────────┐
│       Signal Parameter Extraction             │
│                                              │
│  Center frequency                            │
│  Bandwidth                                   │
│  Duration                                    │
│  SNR                                         │
│  Time/frequency region                       │
└──────────────────────┬───────────────────────┘
                       |
                       v
                 IQ Extraction
                       |
                       v
┌──────────────────────────────────────────────┐
│                  STAGE 2                     │
│          CNN Modulation Detection            │
│                                              │
│  RMS normalization                           │
│       ↓                                      │
│  IQ frame construction                       │
│       ↓                                      │
│  4 × 1024 representation                     │
│       ↓                                      │
│  Proposed CNN                                │
│       ↓                                      │
│  24-class modulation output                 │
└──────────────────────┬───────────────────────┘
                       |
                       v
              Classified Sources
                       |
                       v
┌──────────────────────────────────────────────┐
│              Final RF Information             │
│                                              │
│  Source ID                                   │
│  Frequency                                   │
│  Bandwidth                                   │
│  Duration                                    │
│  SNR                                         │
│  Modulation                                 │
└──────────────────────────────────────────────┘
```

---

# Current Stage Status

## Stage 1 — Source Detection

```text
Status: Implemented
```

Includes:

* IQ loading
* Preprocessing
* STFT
* Power calculation
* 2-D CA-CFAR
* Detection mask
* Connected-component extraction
* Source labeling
* Source parameter extraction

## Stage 2 — CNN Modulation Detection

```text
Status: CNN training implementation available
```

Includes:

* RadioML 2018.01A dataset
* IQ preprocessing
* RMS normalization
* Frame extension
* Proposed CNN architecture
* Residual blocks
* SGD + momentum
* StepLR learning-rate scheduling
* Model evaluation
* Per-SNR accuracy
* Best-model checkpointing

## Current Integration Status

```text
Stage 1 → Stage 2
```

