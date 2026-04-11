import os
import multiprocessing
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn 
from torchaudio.transforms import MelSpectrogram
import time
from tqdm import tqdm

import CNN
import LibriSpeechDataset

# Setup variables
script_dir = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOP_K = 20
BATCH_SIZE = 64
NUM_EPOCHS = 2
NUM_CORES = max(1, multiprocessing.cpu_count() - 2)

print(f"Using device: {device}")

ds = LibriSpeechDataset.LibriSpeechWordDataset(
    root = os.path.join(script_dir, "../LibriSpeech"),
    splits = ["dev-clean"],
    top_k = TOP_K,
)


loader = DataLoader(
    ds, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    collate_fn=LibriSpeechDataset.collate_fn,
    num_workers=NUM_CORES,
    pin_memory=True 
)

print(f"Loaded {len(ds)} dataset entries.")

net = CNN.net(TOP_K).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum= 0.9)

mel_transform = MelSpectrogram(sample_rate=ds.sample_rate, n_mels=64).to(device)

# Initialize the GradScaler for Mixed Precision training
scaler = torch.amp.GradScaler('cuda')


# Train ds
start_time = time.time()
for epoch in range(NUM_EPOCHS):
    running_loss = 0

    progress_bar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch + 1} of {NUM_EPOCHS}")

    for i, data in progress_bar:
        
        # non_blocking=True allows the transfer to overlap with computation
        raw_audio = data[0].to(device, non_blocking=True)
        labels = data[1].to(device, non_blocking=True)

        optimizer.zero_grad()

        # Run the forward pass in Mixed Precision
        with torch.amp.autocast('cuda'):
            # Calculate spectrograms on the GPU
            inputs = mel_transform(raw_audio)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            
        # Scale the gradients and step the optimizer backward
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        
        progress_bar.set_postfix({'loss': f"{running_loss / (i+1):.4f}"})

training_time = time.time() - start_time
print(f'training done in {training_time:.2f} seconds')

PATH = './training_save.pth'
torch.save(net.state_dict(), PATH)