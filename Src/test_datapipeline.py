import os
import datapipeline

script_dir = os.path.dirname(os.path.abspath(__file__))

ds = datapipeline.LibriSpeechWordDataset(
    root = os.path.join(script_dir, "..\\LibriSpeech"),
    splits = ["train-clean-100"],
    top_k = 1000,
)

print(f"Dataset size: {len(ds)}")