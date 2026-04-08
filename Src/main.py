import os
import multiprocessing
import LibriSpeechDataset
import torch
import CNN
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn 
from torchaudio.transforms import MelSpectrogram

script_dir = os.path.dirname(os.path.abspath(__file__))
TOP_K = 500

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

ds = LibriSpeechDataset.LibriSpeechWordDataset(
    root = os.path.join(script_dir, "../LibriSpeech"),
    splits = ["dev-clean"],
    top_k = TOP_K,
)

BATCH_SIZE = 256
NUM_CORES = max(1, multiprocessing.cpu_count() - 2)

loader = DataLoader(
    ds, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    collate_fn=LibriSpeechDataset.collate_fn,
    num_workers=NUM_CORES,
    pin_memory=True 
)

print(f"{len(ds)}")

net = CNN.net(TOP_K).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum= 0.9)

mel_transform = MelSpectrogram(sample_rate=16000, n_mels=64).to(device)

# Initialize the GradScaler for Mixed Precision training
scaler = torch.amp.GradScaler('cuda')

for epoch in range(20):
    running_loss = 0
    for i, data in enumerate(loader, 0):
        
        # non_blocking=True allows the transfer to overlap with computation
        raw_audio = data[0].to(device, non_blocking=True)
        labels = data[1].to(device, non_blocking=True)

        optimizer.zero_grad()

        # Run the forward pass in Mixed Precision
        with torch.amp.autocast('cuda'):
            # Calculate spectrograms instantly on the GPU!
            inputs = mel_transform(raw_audio)
            
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            
        # Scale the gradients and step the optimizer backward
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        
        if i % 20 == 19:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 50:.3f}')
            running_loss = 0.0
            
print('training done')

PATH = './training_save.pth'
torch.save(net.state_dict(), PATH)