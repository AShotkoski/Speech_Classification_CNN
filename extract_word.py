import os
import numpy as np
import librosa
import soundfile as sf

# ---------------------------------------------------------------------------
# Extract specific words from LibriSpeech alignment data and save as audio.
#
# Uses the same directory structure and alignment format as parser_example.py
# and extract_words.py.
# ---------------------------------------------------------------------------

LIBRISPEECH_ROOT = r"C:\Users\Shotk\OneDrive\Documents\Education\Spring2026\Machine Learning\Project\Speech\LibriSpeech"
SAMPLE_RATE = 16000  # LibriSpeech native sample rate


def extract_word(
    words,
    output_dir,
    fmt="wav",
    count=None,
    librispeech_root=LIBRISPEECH_ROOT,
):
    """
    Extract occurrences of specific words from the LibriSpeech corpus and
    save each as an individual audio file.

    Parameters
    ----------
    words : str or list[str]
        A single word or list of words to search for (case-insensitive).
    output_dir : str
        Directory where audio files will be saved (created if it doesn't exist).
    fmt : str, optional
        Output format: ``"wav"`` (default) or ``"mp3"``.
    count : int or None, optional
        Maximum number of occurrences to extract *per word*.
        ``None`` (default) extracts every occurrence found.
    librispeech_root : str, optional
        Path to the LibriSpeech dataset root. Defaults to the module constant.

    Output filenames
    ----------------
    ``{word}_{speaker_id}_{occurrence}.{ext}``

    For example: ``hello_1234_001.wav``, ``hello_5678_002.wav``

    The occurrence number is a per-word running counter across all speakers.
    """

    # Normalize inputs
    if isinstance(words, str):
        words = [words]
    target_words = {w.upper() for w in words}

    os.makedirs(output_dir, exist_ok=True)

    # Per-word running occurrence counter
    occurrence = {w: 0 for w in target_words}

    # Track how many we still need per word (None → unlimited)
    remaining = {w: count for w in target_words}

    def _all_done():
        """Return True when every word has reached its count limit."""
        if count is None:
            return False
        return all(r is not None and r <= 0 for r in remaining.values())

    # ---- traverse LibriSpeech directory tree ----
    for set_name in sorted(os.listdir(librispeech_root)):
        set_dir = os.path.join(librispeech_root, set_name)
        if not os.path.isdir(set_dir):
            continue

        for speaker_id in sorted(os.listdir(set_dir)):
            speaker_dir = os.path.join(set_dir, speaker_id)
            if not os.path.isdir(speaker_dir):
                continue

            for book_id in sorted(os.listdir(speaker_dir)):
                book_dir = os.path.join(speaker_dir, book_id)
                if not os.path.isdir(book_dir):
                    continue

                alignment_fpath = os.path.join(
                    book_dir, f"{speaker_id}-{book_id}.alignment.txt"
                )
                if not os.path.exists(alignment_fpath):
                    continue

                with open(alignment_fpath, "r") as f:
                    for line in f:
                        if _all_done():
                            break

                        parts = line.strip().split(" ")
                        if len(parts) < 3:
                            continue

                        utterance_id = parts[0]
                        words_str = parts[1]
                        end_times_str = parts[2]

                        utt_words = words_str.replace('"', "").split(",")
                        end_times = [
                            float(t) for t in end_times_str.replace('"', "").split(",")
                        ]

                        audio_fpath = os.path.join(book_dir, utterance_id + ".flac")
                        if not os.path.exists(audio_fpath):
                            continue

                        # Check if any target word is in this utterance
                        utt_upper = [w.upper() for w in utt_words]
                        found = target_words.intersection(utt_upper)
                        if not found:
                            continue

                        # Load audio only if needed
                        wav, _ = librosa.load(audio_fpath, sr=SAMPLE_RATE)
                        start_times = [0.0] + end_times[:-1]

                        for word_raw, start, end in zip(
                            utt_words, start_times, end_times
                        ):
                            word_upper = word_raw.upper()
                            if word_upper == "" or word_upper not in target_words:
                                continue

                            # Check per-word limit
                            if remaining[word_upper] is not None and remaining[word_upper] <= 0:
                                continue

                            start_sample = int(start * SAMPLE_RATE)
                            end_sample = int(end * SAMPLE_RATE)
                            segment = wav[start_sample:end_sample]

                            if len(segment) == 0:
                                continue

                            occurrence[word_upper] += 1
                            if remaining[word_upper] is not None:
                                remaining[word_upper] -= 1

                            occ_str = f"{occurrence[word_upper]:03d}"
                            filename = f"{word_upper.lower()}_{speaker_id}_{occ_str}.{fmt}"
                            out_path = os.path.join(output_dir, filename)

                            if fmt == "wav":
                                sf.write(out_path, segment, SAMPLE_RATE)
                            elif fmt == "mp3":
                                try:
                                    from pydub import AudioSegment

                                    # Convert float32 numpy → 16-bit PCM bytes for pydub
                                    pcm16 = (segment * 32767).astype(np.int16)
                                    audio_seg = AudioSegment(
                                        data=pcm16.tobytes(),
                                        sample_width=2,
                                        frame_rate=SAMPLE_RATE,
                                        channels=1,
                                    )
                                    audio_seg.export(out_path, format="mp3")
                                except ImportError:
                                    raise ImportError(
                                        "MP3 export requires the 'pydub' package "
                                        "(pip install pydub) and ffmpeg installed."
                                    )
                            else:
                                raise ValueError(
                                    f"Unsupported format '{fmt}'. Use 'wav' or 'mp3'."
                                )

                            print(f"  Saved: {filename}")

                if _all_done():
                    break
            if _all_done():
                break
        if _all_done():
            break

    # ---- summary ----
    print("\n--- Extraction Summary ---")
    for w in sorted(target_words):
        print(f"  {w.lower()}: {occurrence[w]} file(s) saved")
    total = sum(occurrence.values())
    print(f"  Total: {total} file(s) in '{output_dir}'\n")


# ==========================================================================
# Configure and run here
# ==========================================================================
if __name__ == "__main__":

    extract_word(
        words=["EVERY"],       # single word: "the"  or list: ["the", "hello"]
        output_dir="./extracted",      # output folder (created automatically)
        fmt="wav",                     # "wav" or "mp3"
        count=5,                       # max per word, or None for all occurrences
    )
