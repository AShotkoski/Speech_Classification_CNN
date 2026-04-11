import os
import multiprocessing
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from torchaudio.transforms import MelSpectrogram
import time
from tqdm import tqdm

import CNN
import LibriSpeechDataset

# Config
SAVE_PATH = './training_save.pth'
TOP_K = 20
BATCH_SIZE = 64
NUM_EPOCHS = 2
LOAD_CACHED_PARAMS = True
TRAIN_SPLIT = 0.8


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def load_dataset():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ds = LibriSpeechDataset.LibriSpeechWordDataset(
        root=os.path.join(script_dir, "../LibriSpeech"),
        splits=["dev-clean"],
        top_k=TOP_K,
    )
    print(f"Loaded {len(ds)} dataset entries.")
    return ds


def split_dataset(ds):
    train_size = int(TRAIN_SPLIT * len(ds))
    test_size = len(ds) - train_size
    return random_split(ds, [train_size, test_size])


def make_loader(dataset, num_workers, shuffle=True):
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        collate_fn=LibriSpeechDataset.collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )


def train(net, loader, mel_transform, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    scaler = torch.amp.GradScaler('cuda')

    start_time = time.time()
    for epoch in range(NUM_EPOCHS):
        running_loss = 0
        progress_bar = tqdm(enumerate(loader), total=len(loader),
                            desc=f"Epoch {epoch + 1} of {NUM_EPOCHS}")

        for i, (raw_audio, labels) in progress_bar:
            raw_audio = raw_audio.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                inputs = mel_transform(raw_audio)
                outputs = net(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{running_loss / (i+1):.4f}"})

    elapsed = time.time() - start_time
    print(f'Training done in {elapsed:.2f} seconds')
    torch.save(net.state_dict(), SAVE_PATH)
    print(f"Model saved to {SAVE_PATH}")


def evaluate(net, loader, mel_transform, device):
    net.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for raw_audio, labels in tqdm(loader, desc="Evaluating"):
            raw_audio = raw_audio.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast('cuda'):
                inputs = mel_transform(raw_audio)
                outputs = net(inputs)

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total * 100
    print(f"Test Accuracy: {accuracy:.2f}% ({correct}/{total})")
    return accuracy


def main():
    device = get_device()
    num_workers = max(1, multiprocessing.cpu_count() - 2)

    ds = load_dataset()
    train_ds, test_ds = split_dataset(ds)
    print(f"Train: {len(train_ds)}, Test: {len(test_ds)}")

    net = CNN.net(TOP_K).to(device)
    mel_transform = MelSpectrogram(sample_rate=ds.sample_rate, n_mels=64).to(device)

    # Load cached params or train from scratch
    if LOAD_CACHED_PARAMS and os.path.exists(SAVE_PATH):
        print(f"Loading cached params from {SAVE_PATH}")
        net.load_state_dict(torch.load(SAVE_PATH, map_location=device))
    else:
        if LOAD_CACHED_PARAMS:
            print(f"Cache file {SAVE_PATH} not found. Training from scratch.")
        train_loader = make_loader(train_ds, num_workers, shuffle=True)
        train(net, train_loader, mel_transform, device)

    # Always evaluate on test set
    test_loader = make_loader(test_ds, num_workers, shuffle=False)
    evaluate(net, test_loader, mel_transform, device)


if __name__ == "__main__":
    main()