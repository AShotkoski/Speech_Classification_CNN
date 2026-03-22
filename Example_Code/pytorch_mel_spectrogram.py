import torch
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt

def plot_librispeech_melspectrogram(file_path: str):
    # 1. Load the audio file natively using torchaudio
    waveform, sample_rate = torchaudio.load(file_path)
    
    if sample_rate != 16000:
        print(f"Warning: Expected 16 kHz, got {sample_rate} Hz.")
        
    # 2. Define STFT and Mel parameters
    window_size_ms = 25
    hop_size_ms = 10
    
    n_fft = int(sample_rate * (window_size_ms / 1000))      # 400 samples
    hop_length = int(sample_rate * (hop_size_ms / 1000))    # 160 samples
    n_mels = 80  # Standard number of Mel bins for modern speech models
    
    # 3. Initialize transforms
    mel_spectrogram_transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0 
    )
    
    # Convert power to decibels (dB)
    amplitude_to_db = T.AmplitudeToDB(stype='power', top_db=80)
    
    # 4. Process the waveform
    mel_spec = mel_spectrogram_transform(waveform)
    mel_spec_db = amplitude_to_db(mel_spec)
    
    # Remove the channel dimension (LibriSpeech is mono)
    mel_spec_db = mel_spec_db[0]
    
    # 5. Plot the Mel spectrogram
    time_axis_max = waveform.shape[1] / sample_rate
    
    plt.figure(figsize=(12, 5))
    # Note: Y-axis is now Mel channels (0 to 80), not linear Hz
    plt.imshow(
        mel_spec_db.numpy(), 
        origin='lower', 
        aspect='auto', 
        cmap='magma',
        extent=[0, time_axis_max, 0, n_mels]
    )
    
    plt.title("Mel Spectrogram (LibriSpeech)")
    plt.xlabel("Time (s)")
    plt.ylabel("Mel Filterbank Channel")
    plt.colorbar(format='%+2.0f dB', label='Power')
    plt.tight_layout()
    plt.show()

# Execution:
plot_librispeech_melspectrogram("./extracted/faintheartedness_8842_001.wav")