import os
from collections import Counter

import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torchaudio.transforms as T
import soundfile as sf

# All of the librispeech dataset uses 16k hz sample rate
SAMPLE_RATE = 16000
BACKEND = 'soundfile'

# Directory architecture for the dataset is 
# split->speakerID->bookID->list of flac files and speakerID-bookID.alignment.txt if an alignment exists
# each alignment file has the structure of:
# speaker.book.utteranceID "comma deliminated word list" "end times"
# for example
# 19-198-0000 ",NORTHANGER,ABBEY," "0.530,1.270,1.780,1.965" 
# The first and last end time always ends the silences of the file
class LibriSpeechWordDataset(Dataset):

    def __init__(self, root, splits, top_k=None):
        """
        root:      path to the LibriSpeech directory
        splits:    list of subset names, e.g. ["train-clean-100"], or ["all"] for all available splits
        top_k:     if set, only keep the K most frequent words (rest are dropped)
        """
        self.root = root
        # Cache the sample rate
        self.sample_rate = SAMPLE_RATE

        # Handle all option to include all available splits
        if "all" in splits:
            splits = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]

        # collect every word occurrence and count frequencies
        # using the billion nested for loops because of how the dataset is organized
        # plus it lets me base it off the example in the alignment github
        raw_entries = []
        word_counter = Counter()

        for split in splits:
            split_dir = os.path.join(root, split)
            if not os.path.isdir(split_dir):
                raise FileNotFoundError(f"Split directory not found: {split_dir}")

            for speaker_id in sorted(os.listdir(split_dir)):
                speaker_dir = os.path.join(split_dir, speaker_id)
                if not os.path.isdir(speaker_dir):
                    continue

                for book_id in sorted(os.listdir(speaker_dir)):
                    book_dir = os.path.join(speaker_dir, book_id)
                    if not os.path.isdir(book_dir):
                        continue

                    alignment_path = os.path.join(
                        book_dir, f"{speaker_id}-{book_id}.alignment.txt"
                    )
                    if not os.path.exists(alignment_path):
                        continue

                    with open(alignment_path, "r") as f:
                        for line in f:
                            parts = line.strip().split(" ")
                            if len(parts) < 3:
                                continue

                            utterance_id = parts[0]
                            audio_path = os.path.join(book_dir, utterance_id + ".flac")
                            if not os.path.exists(audio_path):
                                continue

                            words = parts[1].replace('"', "").split(",")
                            end_times = [
                                float(t)
                                for t in parts[2].replace('"', "").split(",")
                            ]
                            start_times = [0.0] + end_times[:-1]

                            for word, start, end in zip(words, start_times, end_times):
                                if word == "":
                                    continue
                                raw_entries.append((audio_path, start, end, word))
                                word_counter[word] += 1

        # Build vocabulary 
        if top_k is not None:
            allowed = {w for w, _ in word_counter.most_common(top_k)}
        else:
            allowed = set(word_counter.keys())

        # Set vocabulary for the class
        self.vocab = {
            word: idx
            for idx, (word, _) in enumerate(word_counter.most_common())
            if word in allowed
        }

        # Filter entries to only include words in the vocabulary
        self.entries = [
            (path, start, end, word)
            for path, start, end, word in raw_entries
            if word in allowed
        ]


    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
            """
            Returns a Mel-Spectrogram at a given index 
            """
            audio_path, start_sec, end_sec, word = self.entries[idx]

            start_sample = int(start_sec * SAMPLE_RATE)
            end_sample = int(end_sec * SAMPLE_RATE)
            num_frames = end_sample - start_sample

            data, _ = sf.read(audio_path, start=start_sample, frames=num_frames)
            
            # Convert the soundfile NumPy array directly into a PyTorch tensor
            waveform = torch.from_numpy(data).float()

            mel_spec_transform = MelSpectrogram(SAMPLE_RATE, n_mels=64)
            mel_spec = mel_spec_transform(waveform)

            label = self.vocab[word]
            return mel_spec, label

    def get_vocab(self):
        return dict(self.vocab)
    def word_at(self, idx):
        for word, word_idx in self.vocab.items():
            if word_idx == idx:
                return word
        raise KeyError(f"word index {idx} not in vocabulary.")

def collate_fn(batch):
    """Pad spectrograms but padding the last dimension."""
    spectrograms, labels = zip(*batch)

    max_time_frames = max([spec.shape[-1] for spec in spectrograms])

    padded_spectrograms = []
    for spec in spectrograms:
        pad_amount = max_time_frames - spec.shape[-1]

        padded_spec = torch.nn.functional.pad(spec, (0,pad_amount))
        padded_spectrograms.append(padded_spec)

    batch_spectrograms = torch.stack(padded_spectrograms)
    batch_labels = torch.tensor(labels)

    return batch_spectrograms, batch_labels
