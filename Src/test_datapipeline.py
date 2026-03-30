import os
import LibriSpeechData
from torch.utils.data import DataLoader

script_dir = os.path.dirname(os.path.abspath(__file__))

ds = LibriSpeechData.LibriSpeechWordDataset(
    root = os.path.join(script_dir, "..\\LibriSpeech"),
    splits = ["dev-clean"],
    top_k = 1000,
)

loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=LibriSpeechData.collate_fn)

features = next(iter(loader))
print("Features shape:", features[0].shape)
print("Labels shape:", features[1].shape)
print(ds[features[1][1]])

print(f"Dataset size: {len(ds)}")
