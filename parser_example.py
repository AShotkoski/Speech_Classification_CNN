import sounddevice as sd
import numpy as np
import librosa
import os

# This script extracts individual words from the LibriSpeech dataset using the
# alignment files, then plays each word's audio with its label printed.

librispeech_root = r"C:\Users\Shotk\OneDrive\Documents\Education\Spring2026\Machine Learning\Project\Speech\LibriSpeech"

SAMPLE_RATE = 16000  # Sampling rate of LibriSpeech


def extract_words(audio_fpath, words, end_times):
    """
    Extract individual word audio segments from an utterance using alignment times.
    
    Returns a list of (word, wav_segment) tuples for each non-silence word.
    """
    wav, _ = librosa.load(audio_fpath, sr=SAMPLE_RATE)
    
    start_times = [0.0] + end_times[:-1]
    
    word_segments = []
    for word, start, end in zip(words, start_times, end_times):
        # Skip silence tokens (empty strings)
        if word == '':
            continue
        
        start_sample = int(start * SAMPLE_RATE)
        end_sample = int(end * SAMPLE_RATE)
        segment = wav[start_sample:end_sample]
        
        if len(segment) > 0:
            word_segments.append((word, segment))
    
    return word_segments


def play_word(word, wav_segment):
    """Play a single word's audio and print its label."""
    # Pad with a short silence so sounddevice doesn't cut it off
    padded = np.concatenate((wav_segment, np.zeros(int(SAMPLE_RATE * 0.3))))
    print(f"  Playing: {word}")
    sd.play(padded, SAMPLE_RATE, blocking=True)


if __name__ == '__main__':
    total_words = 0
    
    # Select sets (e.g. dev-clean, train-other-500, ...)
    for set_name in sorted(os.listdir(librispeech_root)):
        set_dir = os.path.join(librispeech_root, set_name)
        if not os.path.isdir(set_dir):
            continue
        
        # Select speakers
        for speaker_id in sorted(os.listdir(set_dir)):
            speaker_dir = os.path.join(set_dir, speaker_id)
            if not os.path.isdir(speaker_dir):
                continue
            
            # Select books
            for book_id in sorted(os.listdir(speaker_dir)):
                book_dir = os.path.join(speaker_dir, book_id)
                if not os.path.isdir(book_dir):
                    continue
                
                # Get the alignment file
                alignment_fpath = os.path.join(
                    book_dir, f"{speaker_id}-{book_id}.alignment.txt"
                )
                if not os.path.exists(alignment_fpath):
                    print(f"Warning: Alignment file not found in {book_dir}, skipping.")
                    continue
                
                # Parse each utterance present in the file
                with open(alignment_fpath, "r") as alignment_file:
                    for line in alignment_file:
                        parts = line.strip().split(' ')
                        if len(parts) < 3:
                            continue
                        
                        utterance_id, words_str, end_times_str = parts[0], parts[1], parts[2]
                        words = words_str.replace('"', '').split(',')
                        end_times = [float(e) for e in end_times_str.replace('"', '').split(',')]
                        audio_fpath = os.path.join(book_dir, utterance_id + '.flac')
                        
                        if not os.path.exists(audio_fpath):
                            continue
                        
                        # Extract and play individual words
                        word_segments = extract_words(audio_fpath, words, end_times)
                        
                        if word_segments:
                            print(f"\nUtterance: {utterance_id}")
                            for word, segment in word_segments:
                                play_word(word, segment)
                                total_words += 1
    
    print(f"\nDone! Played {total_words} individual words.")