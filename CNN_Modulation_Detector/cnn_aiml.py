import os
import time
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import StepLR
import kagglehub  # Added kagglehub import

# ==========================================
# 1. Dataset & Preprocessing Pipeline
# ==========================================

class RadioMLDataset(Dataset):
    """
    Dataset loader for DeepSig RadioML 2018.01A.
    Performs RMS energy normalization and frame extension (2x1024 -> 4x1024).
    """
    def __init__(self, hdf5_path):
        super(RadioMLDataset, self).__init__()
        self.hdf5_path = hdf5_path
        
        print(f"Loading dataset into memory from {hdf5_path}...")
        with h5py.File(hdf5_path, 'r') as f:
            # Shape in RadioML 2018.01A: X is (N, 1024, 2), Y is (N, 24), Z is (N, 1)
            self.X = np.array(f['X'], dtype=np.float32)
            self.Y = np.argmax(np.array(f['Y'], dtype=np.float32), axis=1)
            self.Z = np.array(f['Z'], dtype=np.float32).flatten()

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        # Raw IQ signal shape: (1024, 2)
        raw_signal = self.X[idx] 
        label = self.Y[idx]
        snr = self.Z[idx]

        # Convert to 2 x 1024: Row 0 is I, Row 1 is Q
        iq_signal = raw_signal.T 

        # 1. RMS Normalization
        energy = np.mean(iq_signal[0]**2 + iq_signal[1]**2)
        rms = np.sqrt(energy) + 1e-8
        norm_signal = iq_signal / rms

        # 2. Frame Extension: Horizontal flip of the 2x1024 frame and vertical concat -> 4x1024
        flipped_signal = np.ascontiguousarray(norm_signal[:, ::-1])
        extended_frame = np.concatenate((norm_signal, flipped_signal), axis=0) 

        # Expand channel dimension: shape (1, 4, 1024)
        sample = np.expand_dims(extended_frame, axis=0)

        return torch.tensor(sample, dtype=torch.float32), torch.tensor(label, dtype=torch.long), snr


# ==========================================
# 2. Proposed CNN Model Architecture
# ==========================================

class ConvB(nn.Module):
    """Basic Convolution Block: Conv2d -> BatchNorm2d -> ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvB, self).__init__()
        pad_h = kernel_size[0] // 2
        pad_w = kernel_size[1] // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=(1, 1), padding=(pad_h, pad_w), bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class ProposedCNN(nn.Module):
    def __init__(self, num_classes=24):
        super(ProposedCNN, self).__init__()

        # ABlock
        self.ablock = nn.Sequential(
            ConvB(1, 16, (1, 3)),
            ConvB(16, 16, (3, 1)),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            ConvB(16, 16, (1, 3)),
            ConvB(16, 16, (3, 1)),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        )

        # BBlock
        self.bblock_conv1 = ConvB(16, 32, (1, 1))
        self.bblock_conv2 = ConvB(32, 16, (1, 3))
        self.bblock_conv3 = ConvB(16, 16, (3, 1))
        self.bblock_conv4 = ConvB(16, 32, (1, 1))

        # APool
        self.apool = nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1))

        # CBlock1
        self.cblock1_conv1 = ConvB(32, 64, (1, 1))
        self.cblock1_conv2 = ConvB(64, 24, (1, 3))
        self.cblock1_conv3 = ConvB(24, 64, (1, 1))

        # CBlock2
        self.cblock2_conv1 = ConvB(64, 128, (1, 1))
        self.cblock2_conv2 = ConvB(128, 32, (1, 3))
        self.cblock2_conv3 = ConvB(32, 128, (1, 1))

        # GAP & FC
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        out = self.ablock(x)
        
        res = self.bblock_conv1(out)
        out = self.bblock_conv2(res)
        out = self.bblock_conv3(out)
        out = self.bblock_conv4(out)
        out = out + res

        out = self.apool(out)

        res = self.cblock1_conv1(out)
        out = self.cblock1_conv2(res)
        out = self.cblock1_conv3(out)
        out = out + res

        res = self.cblock2_conv1(out)
        out = self.cblock2_conv2(res)
        out = self.cblock2_conv3(out)
        out = out + res

        out = self.gap(out)
        out = out.flatten(1)
        logits = self.fc(out)
        return logits


# ==========================================
# 3. Training & Validation Pipeline
# ==========================================

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels, _ in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, 100.0 * correct / total

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    snr_results = {}

    with torch.no_grad():
        for inputs, labels, snrs in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            for pred, true, snr in zip(preds.cpu().numpy(), labels.cpu().numpy(), snrs.numpy()):
                snr = int(snr)
                if snr not in snr_results:
                    snr_results[snr] = {'correct': 0, 'total': 0}
                snr_results[snr]['total'] += 1
                if pred == true:
                    snr_results[snr]['correct'] += 1

    return running_loss / total, 100.0 * correct / total, snr_results


# ==========================================
# 4. Main Execution
# ==========================================

if __name__ == "__main__":
    # --- Download and locate dataset via kagglehub ---
    print("Downloading/Locating dataset via kagglehub...")
    dataset_dir = kagglehub.dataset_download("pinxau1000/radioml2018")
    print("Path to dataset files:", dataset_dir)
    
    # Auto-detect the .hdf5 file in the downloaded directory
    HDF5_FILE_PATH = None
    for file in os.listdir(dataset_dir):
        if file.endswith(".hdf5"):
            HDF5_FILE_PATH = os.path.join(dataset_dir, file)
            break
            
    if HDF5_FILE_PATH is None:
        raise FileNotFoundError("Could not find the .hdf5 dataset file in the downloaded Kaggle folder.")
    
    print(f"Found HDF5 dataset at: {HDF5_FILE_PATH}")
    # -------------------------------------------------

    # Hyperparameters from paper Table 2
    SAVE_WEIGHTS_PATH = "proposed_cnn_weights.pth"
    MAX_EPOCHS = 45
    BATCH_SIZE = 64
    INITIAL_LR = 0.1
    LR_DROP_PERIOD = 20
    LR_DROP_FACTOR = 0.1
    MOMENTUM = 0.9

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on hardware device: {device}")

    # Dataset Preparation: 80% train, 20% test split
    full_dataset = RadioMLDataset(hdf5_path=HDF5_FILE_PATH)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size

    train_data, test_data = random_split(
        full_dataset, 
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    # Initialize Architecture, Objective Function, and SGDM Optimizer
    model = ProposedCNN(num_classes=24).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=INITIAL_LR, momentum=MOMENTUM)
    scheduler = StepLR(optimizer, step_size=LR_DROP_PERIOD, gamma=LR_DROP_FACTOR)

    best_test_acc = 0.0

    print("Beginning Training Routine...")
    start_time = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc, _ = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        # Checkpoint Best Model Weights
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': test_acc
            }, SAVE_WEIGHTS_PATH)

        elapsed = time.time() - t0
        print(f"Epoch [{epoch:02d}/{MAX_EPOCHS:02d}] ({elapsed:.1f}s) | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}% | "
              f"LR: {scheduler.get_last_lr()[0]:.4f}")

    print(f"\nTraining completed in {(time.time() - start_time) / 60:.2f} minutes.")
    print(f"Best Test Accuracy: {best_test_acc:.2f}%. Checkpoint saved to: {SAVE_WEIGHTS_PATH}")

    # ==========================================
    # 5. Final Evaluation and SNR Profiling
    # ==========================================
    print("\nLoading best model checkpoint for SNR accuracy profiling...")
    checkpoint = torch.load(SAVE_WEIGHTS_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    _, final_acc, snr_stats = evaluate(model, test_loader, criterion, device)
    print(f"\nVerified Model Overall Accuracy: {final_acc:.2f}%\n")
    print("--- Per-SNR Classification Accuracy ---")
    print(f"{'SNR (dB)':<10} | {'Accuracy (%)':<15} | {'Samples':<10}")
    print("-" * 42)
    for snr in sorted(snr_stats.keys()):
        stats = snr_stats[snr]
        acc = (stats['correct'] / stats['total']) * 100.0 if stats['total'] > 0 else 0.0
        print(f"{snr:<10} | {acc:<15.2f} | {stats['total']:<10}")