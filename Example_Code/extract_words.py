import os
from collections import Counter

# Extract all words from LibriSpeech alignment files and produce two output files:
#   all_words.txt    - every word occurrence, one per line
#   word_counts.txt  - unique words with frequency counts, sorted descending

LIBRISPEECH_ROOT = r""
OUTPUT_DIR = r""


def main():
    all_words = []
    skipped_utterances = 0
    processed_utterances = 0

    for set_name in sorted(os.listdir(LIBRISPEECH_ROOT)):
        set_dir = os.path.join(LIBRISPEECH_ROOT, set_name)
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
                        parts = line.strip().split(' ')
                        if len(parts) < 3:
                            continue

                        utterance_id = parts[0]
                        words_str = parts[1]
                        audio_fpath = os.path.join(book_dir, utterance_id + '.flac')

                        # Only count words if the audio file is downloaded
                        if not os.path.exists(audio_fpath):
                            skipped_utterances += 1
                            continue

                        processed_utterances += 1
                        words = words_str.replace('"', '').split(',')

                        for word in words:
                            if word != '':  # skip silence tokens
                                all_words.append(word)

    # Write all words, one per line
    all_words_path = os.path.join(OUTPUT_DIR, "all_words.txt")
    with open(all_words_path, "w") as f:
        for word in all_words:
            f.write(word + "\n")

    # Write unique words with counts, sorted by frequency (descending)
    word_counts = Counter(all_words)
    word_counts_path = os.path.join(OUTPUT_DIR, "word_counts.txt")
    with open(word_counts_path, "w") as f:
        for word, count in word_counts.most_common():
            f.write(f"{word}\t{count}\n")

    print(f"Processed {processed_utterances} utterances, skipped {skipped_utterances} (missing audio)")
    print(f"Total word occurrences: {len(all_words)}")
    print(f"Unique words: {len(word_counts)}")
    print(f"Wrote: {all_words_path}")
    print(f"Wrote: {word_counts_path}")


if __name__ == '__main__':
    main()
