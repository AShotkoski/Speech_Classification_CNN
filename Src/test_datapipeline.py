import os
import LibriSpeechDataset
from torch.utils.data import DataLoader
from torchaudio.transforms import AmplitudeToDB
import matplotlib.pyplot as plt
from utils import plot_log_mel_spectrogram

script_dir = os.path.dirname(os.path.abspath(__file__))

ds = LibriSpeechDataset.LibriSpeechWordDataset(
    root = os.path.join(script_dir, "../LibriSpeech"),
    splits = ["dev-clean"],
    top_k = 100
)

loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=LibriSpeechDataset.collate_fn)

features, labels = next(iter(loader))
print("Features shape:", features.shape)
print("Labels shape:", labels.shape)


print(f"label {[ds.word_at(label.item()) for label in labels]}")

plot_labels = [ds.word_at(label.item()) for label in labels]
plot_log_mel_spectrogram(features, ds.sample_rate, labels=plot_labels)