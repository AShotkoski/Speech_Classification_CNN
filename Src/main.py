import os
import LibriSpeechDataset
import torch
import CNN
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn 

script_dir = os.path.dirname(os.path.abspath(__file__))

# Number of words in dataset to keep, sorted by frequency
TOP_K = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

ds = LibriSpeechDataset.LibriSpeechWordDataset(
    root = os.path.join(script_dir, "../LibriSpeech"),
    splits = ["dev-clean"],
    top_k = TOP_K,
)

loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=LibriSpeechDataset.collate_fn)


print(f"Dataset size: {len(ds)}")

net = CNN.net(TOP_K).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum= 0.9)

for epoch in range(2):
    running_loss = 0
    for i, data in enumerate(loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero param gradients
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
print('training done')


# Save training file params
PATH = './training_save.pth'
torch.save(net.state_dict(),PATH)