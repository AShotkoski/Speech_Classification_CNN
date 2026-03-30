import os
import LibriSpeechData
from torch.utils.data import DataLoader
from torchaudio.transforms import AmplitudeToDB
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))

ds = LibriSpeechData.LibriSpeechWordDataset(
    root = os.path.join(script_dir, "..\\LibriSpeech"),
    splits = ["dev-clean"],
    top_k = 100
)

#loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=LibriSpeechData.collate_fn)

#features, labels, lengths = next(iter(loader))
#print("Features shape:", features.shape)
#print("Labels shape:", labels.shape)
#print("lengths: ", lengths)

print(f"Dataset size: {len(ds)}")

mel_spec1, label1 = ds[-1]
mel_spec2, label2 = ds[-5]

fig, axes = plt.subplots(1, 2, figsize=(16, 4))


im1 = axes[0].imshow(mel_spec1, origin='lower', aspect='auto', cmap='magma')
axes[0].set_title('Mel Spectrogram (ds[-1])')

im2 = axes[1].imshow(mel_spec2, origin='lower', aspect='auto', cmap='magma')
axes[1].set_title('Mel Spectrogram (ds[-5])')

plt.show()

