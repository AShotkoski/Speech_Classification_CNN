import matplotlib.pyplot as plt
import torchaudio.transforms as T
import numpy as np


def plot_log_mel_spectrogram(spectrogram, sample_rate, title=None, ylabel="mel freq", aspect="auto", xmax=None, max_plots=9, labels=None):
    """
    Plot mel spectrograms. Can handle a single spectrogram or a list/array of spectrograms.
    
    Args:
        spectrogram: Single spectrogram tensor or list/array of spectrograms
        sample_rate: Sample rate of the audio
        title: Title for the plot(s)
        ylabel: Label for y-axis
        aspect: Aspect ratio for imshow
        xmax: Maximum x value for xlim
        max_plots: Maximum number of spectrograms to plot (default: 9)
        labels: Optional list/array of labels (strings or values) to display in titles
    """
    amptodb = T.AmplitudeToDB(stype='power', top_db=80)
    
    # Handle single spectrogram or list/array of spectrograms
    if isinstance(spectrogram, (list, tuple)):
        spectrograms = spectrogram
    elif hasattr(spectrogram, 'shape') and len(spectrogram.shape) == 3:
        # Tensor/array with batch dimension - split along first dimension
        spectrograms = [spectrogram[i] for i in range(spectrogram.shape[0])]
    else:
        # Single 2D spectrogram
        spectrograms = [spectrogram]
    
    # Limit number of plots
    num_plots = min(len(spectrograms), max_plots)
    spectrograms = spectrograms[:num_plots]
    
    # Calculate grid dimensions
    cols = int(np.ceil(np.sqrt(num_plots)))
    rows = int(np.ceil(num_plots / cols))
    
    fig, axs = plt.subplots(rows, cols, figsize=(cols*4, rows*3))
    
    # Handle case where there's only one subplot
    if num_plots == 1:
        axs = [axs]
    else:
        axs = axs.flatten()
    
    for idx, spec in enumerate(spectrograms):
        log_mel_spec = amptodb(spec)
        ax = axs[idx]
        
        # Set title with labels if provided
        if labels is not None:
            if hasattr(labels, '__getitem__'):
                label_text = str(labels[idx])
            else:
                label_text = str(labels)
            ax.set_title(f"[{idx}] {label_text}")
        else:
            ax.set_title(f"{title or 'Mel Spectrogram'} ({idx+1})" if num_plots > 1 else title or "Mel Spectrogram (dB)")
        
        ax.set_ylabel(ylabel)
        ax.set_xlabel("frame")
        im = ax.imshow(log_mel_spec, origin="lower", aspect=aspect)
        if xmax:
            ax.set_xlim((0, xmax))
        fig.colorbar(im, ax=ax)
    
    # Hide any unused subplots
    for idx in range(num_plots, len(axs)):
        axs[idx].set_visible(False)
    
    plt.tight_layout()
    plt.show()