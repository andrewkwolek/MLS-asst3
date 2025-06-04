import numpy as np
import librosa
import vggish_params


def waveform_to_examples(data, sample_rate):
    """Converts audio waveform into an array of examples for VGGish.

    Args:
        data: np.array of shape [num_samples] containing the waveform
        sample_rate: the sample rate of the input waveform

    Returns:
        3-D np.array of shape [num_examples, num_frames, num_bands, 1] which represents
        a sequence of examples, each of which contains a patch of log mel
        spectrogram, covering num_frames frames of audio and num_bands mel frequency
        bands.
    """
    # Convert to mono if stereo
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    # Resample to the rate assumed by VGGish
    if sample_rate != vggish_params.SAMPLE_RATE:
        data = librosa.resample(data, orig_sr=sample_rate,
                                target_sr=vggish_params.SAMPLE_RATE)

    # Compute log mel spectrogram features
    log_mel = _waveform_to_log_mel_spectrogram(data, vggish_params.SAMPLE_RATE)

    # Frame the log mel spectrogram into examples
    log_mel_examples = _frame_log_mel_spectrogram(log_mel)

    return log_mel_examples


def _waveform_to_log_mel_spectrogram(data, sample_rate):
    """Converts waveform to a log magnitude mel-frequency spectrogram.

    Args:
        data: 1D np.array of waveform data.
        sample_rate: The sample rate of data.

    Returns:
        2D np.array of (num_frames, num_mel_bins) log magnitude mel-frequency
        spectrogram.
    """

    # Calculate hop length in samples
    hop_length_samples = int(vggish_params.HOP_SIZE * sample_rate)

    # Compute the mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(
        y=data,
        sr=sample_rate,
        hop_length=hop_length_samples,
        n_fft=2048,
        n_mels=vggish_params.NUM_MEL_BINS,
        fmin=vggish_params.MEL_MIN_HZ,
        fmax=vggish_params.MEL_MAX_HZ)

    # Convert to log scale (log1p for numerical stability)
    log_mel_spectrogram = np.log(mel_spectrogram + 1e-6)

    # Transpose to (time, frequency) format expected by VGGish
    log_mel_spectrogram = log_mel_spectrogram.T

    return log_mel_spectrogram


def _frame_log_mel_spectrogram(log_mel_spectrogram):
    """Frame log mel spectrogram into fixed-size examples.

    Args:
        log_mel_spectrogram: 2D array of shape (num_time_frames, num_mel_bins)

    Returns:
        3D array of shape (num_examples, NUM_FRAMES, num_mel_bins, 1)
    """
    num_time_frames, num_mel_bins = log_mel_spectrogram.shape

    # If we don't have enough frames, pad with zeros
    if num_time_frames < vggish_params.NUM_FRAMES:
        padding = vggish_params.NUM_FRAMES - num_time_frames
        log_mel_spectrogram = np.pad(log_mel_spectrogram,
                                     ((0, padding), (0, 0)),
                                     mode='constant',
                                     constant_values=0)
        num_time_frames = vggish_params.NUM_FRAMES

    # Calculate how many examples we can create
    num_examples = 1 + \
        max(0, (num_time_frames - vggish_params.NUM_FRAMES) //
            vggish_params.NUM_FRAMES)

    # If we can't create any full examples, create one padded example
    if num_examples == 0:
        examples = log_mel_spectrogram[:vggish_params.NUM_FRAMES]
        examples = np.expand_dims(examples, axis=0)  # Add batch dimension
        examples = np.expand_dims(examples, axis=-1)  # Add channel dimension
        return examples

    # Create examples by sliding window
    examples = []
    for i in range(num_examples):
        start_frame = i * vggish_params.NUM_FRAMES
        end_frame = start_frame + vggish_params.NUM_FRAMES

        if end_frame <= num_time_frames:
            example = log_mel_spectrogram[start_frame:end_frame]
            examples.append(example)
        else:
            # If we don't have enough frames for the last example, pad it
            example = log_mel_spectrogram[start_frame:]
            padding_needed = vggish_params.NUM_FRAMES - example.shape[0]
            example = np.pad(example, ((0, padding_needed), (0, 0)),
                             mode='constant', constant_values=0)
            examples.append(example)

    # Convert to numpy array and add channel dimension
    examples = np.array(examples)
    examples = np.expand_dims(examples, axis=-1)  # Add channel dimension

    return examples


def wavfile_to_examples(wav_file):
    """Convenience wrapper around waveform_to_examples() for a common WAV format.

    Args:
        wav_file: String path to a wav file.

    Returns:
        See waveform_to_examples.
    """
    data, sr = librosa.load(wav_file, sr=None, mono=True)
    return waveform_to_examples(data, sr)
