import numpy as np
import sounddevice as sd
import librosa
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
import threading
from collections import deque
import sys
import os
from sklearn.preprocessing import StandardScaler

# VGGish imports (for DL model - same as Assignment 2)
try:
    import vggish_params
    from vggish_input import waveform_to_examples
except ImportError:
    print("Warning: VGGish modules not found. DL model may not work correctly.")
    print("Make sure vggish_input.py and vggish_params.py are in your directory.")

# Configuration
SAMPLE_RATE = 16000
WINDOW_LENGTH = 1.0  # 1 second windows
HOP_LENGTH = 0.5     # 0.5 second overlap
BUFFER_SIZE = int(WINDOW_LENGTH * SAMPLE_RATE)
HOP_SIZE = int(HOP_LENGTH * SAMPLE_RATE)

# Activity labels (same as from assignments 1&2)
ACTIVITIES = ['laugh', 'cough', 'clap', 'knock', 'alarm']


class RealTimeAudioProcessor:
    def __init__(self, ml_model_path, dl_model_path):
        """Initialize the real-time audio processor with both models."""

        # Load models
        print("Loading ML model...")
        with open(ml_model_path, 'rb') as f:
            ml_data = pickle.load(f)

        # Handle different ML model save formats
        if isinstance(ml_data, dict):
            self.ml_model = ml_data.get('model', ml_data.get('best_model'))
            self.ml_scaler = ml_data.get('scaler', None)
            print(f"Loaded ML model from dict with keys: {ml_data.keys()}")
        else:
            self.ml_model = ml_data
            self.ml_scaler = None

        print("Loading DL model...")
        self.dl_model = load_model(dl_model_path)

        # Audio buffer for sliding windows
        # Store extra for overlap
        self.audio_buffer = deque(maxlen=BUFFER_SIZE * 2)

        # Results storage
        self.ml_predictions = deque(maxlen=50)
        self.dl_predictions = deque(maxlen=50)
        self.ml_confidences = deque(maxlen=50)
        self.dl_confidences = deque(maxlen=50)
        self.ml_latencies = deque(maxlen=50)
        self.dl_latencies = deque(maxlen=50)
        self.timestamps = deque(maxlen=50)

        # Current window data for visualization
        self.current_window = np.zeros(BUFFER_SIZE)

        # Initialize plot
        self.setup_plot()

        print("Real-time audio processor initialized successfully!")

    def extract_ml_features(self, audio_signal, sr=16000):
        """Extract features for ML model (same as from Assignment 1)."""

        def extract_fft(audio_signal, sr=16000):
            windowed_signal = audio_signal * \
                librosa.filters.get_window('hann', len(audio_signal))
            stft = librosa.stft(windowed_signal)
            magnitude_spectrum = np.abs(stft)
            power_spectrum = magnitude_spectrum ** 2
            freq_bins = librosa.fft_frequencies(
                sr=sr, n_fft=2*(stft.shape[0]-1))

            total_energy = np.sum(power_spectrum)
            spectral_centroid = librosa.feature.spectral_centroid(
                S=power_spectrum, sr=sr)[0, 0]
            spectral_spread = librosa.feature.spectral_bandwidth(
                S=power_spectrum, sr=sr)[0, 0]
            spectral_rolloff = librosa.feature.spectral_rolloff(
                S=power_spectrum, sr=sr)[0, 0]

            bands = [(0, 500), (500, 1000), (1000, 2000),
                     (2000, 4000), (4000, 8000)]
            band_energies = []
            for low, high in bands:
                indices = np.where((freq_bins >= low) & (freq_bins < high))[0]
                if len(indices) > 0 and total_energy > 0:
                    band_energy = np.sum(
                        power_spectrum[indices]) / total_energy
                else:
                    band_energy = 0
                band_energies.append(band_energy)

            features = np.array([total_energy, spectral_centroid,
                                spectral_spread, spectral_rolloff, *band_energies])
            return features

        def extract_mfcc(audio_signal, sr=16000):
            mfccs = librosa.feature.mfcc(y=audio_signal, sr=sr, n_mfcc=13)
            delta_mfccs = librosa.feature.delta(mfccs)
            delta2_mfccs = librosa.feature.delta(mfccs, order=2)

            mfcc_means = np.mean(mfccs, axis=1)
            delta_means = np.mean(delta_mfccs, axis=1)
            delta2_means = np.mean(delta2_mfccs, axis=1)
            mfcc_stds = np.std(mfccs, axis=1)

            selected_features = np.array([
                mfcc_means[0], mfcc_means[1], mfcc_means[2], mfcc_stds[0],
                delta_means[0], delta_means[1], delta2_means[0],
                np.mean(mfcc_means), np.mean(mfcc_stds)
            ])
            return selected_features

        def extract_rms(audio_signal, sr=16000):
            rms_energy = librosa.feature.rms(y=audio_signal)[0]
            rms_mean = np.mean(rms_energy)
            rms_std = np.std(rms_energy)
            rms_max = np.max(rms_energy)
            rms_min = np.min(rms_energy)
            rms_median = np.median(rms_energy)
            rms_range = rms_max - rms_min

            rms_diff = np.diff(rms_energy)
            rms_diff_mean = np.mean(np.abs(rms_diff))
            zcr = librosa.feature.zero_crossing_rate(audio_signal)[0]
            zcr_mean = np.mean(zcr)
            low_energy_ratio = np.mean(rms_energy < rms_mean)

            features = np.array([rms_mean, rms_std, rms_max, rms_min, rms_median,
                                 rms_range, rms_diff_mean, zcr_mean, low_energy_ratio])
            return features

        # Extract all features and combine
        fft_features = extract_fft(audio_signal, sr)
        mfcc_features = extract_mfcc(audio_signal, sr)
        rms_features = extract_rms(audio_signal, sr)

        combined_features = np.concatenate(
            [fft_features, mfcc_features, rms_features])
        return combined_features.reshape(1, -1)  # Reshape for sklearn

    def extract_dl_features(self, audio_signal, sr=16000):
        """Extract features for DL model using VGGish format (same as Assignment 2)."""
        try:
            # Use VGGish preprocessing (same as Assignment 2)
            # Resample to VGGish sample rate if necessary
            if sr != 16000:
                audio_signal = librosa.resample(
                    audio_signal, orig_sr=sr, target_sr=16000)

            # Convert to VGGish examples format
            examples = waveform_to_examples(audio_signal, 16000)

            if examples.shape[0] == 0:
                print("Warning: No VGGish examples generated")
                # Return a dummy input if no examples generated
                return np.zeros((1, 96, 64, 1))

            return examples

        except Exception as e:
            print(f"Error in DL feature extraction: {e}")
            # Return a dummy input with correct shape if feature extraction fails
            return np.zeros((1, 96, 64, 1))

    def predict_ml(self, features):
        """Make prediction using ML model."""
        start_time = time.time()

        try:
            # Apply scaling if scaler is available
            if self.ml_scaler is not None:
                features = self.ml_scaler.transform(features)

            # Get prediction and confidence
            prediction = self.ml_model.predict(features)[0]

            # Get prediction probabilities if available
            if hasattr(self.ml_model, 'predict_proba'):
                confidence = self.ml_model.predict_proba(features)[0]
                max_confidence = np.max(confidence)
            else:
                # For models without predict_proba, use decision function or default confidence
                if hasattr(self.ml_model, 'decision_function'):
                    decision_scores = self.ml_model.decision_function(features)[
                        0]
                    # Convert decision scores to confidence-like values
                    exp_scores = np.exp(
                        decision_scores - np.max(decision_scores))
                    max_confidence = np.max(exp_scores) / np.sum(exp_scores)
                else:
                    max_confidence = 0.5  # Default confidence

            latency = (time.time() - start_time) * \
                1000  # Convert to milliseconds

            return ACTIVITIES[prediction], max_confidence, latency

        except Exception as e:
            print(f"ML prediction error: {e}")
            return "unknown", 0.0, 0.0

    def predict_dl(self, features):
        """Make prediction using DL model."""
        start_time = time.time()

        try:
            # Get prediction - average across all windows if multiple examples
            if features.shape[0] > 1:
                # Multiple windows - average predictions
                prediction_probs = self.dl_model.predict(features, verbose=0)
                avg_pred = np.mean(prediction_probs, axis=0)
            else:
                # Single window
                avg_pred = self.dl_model.predict(features, verbose=0)[0]

            predicted_class = np.argmax(avg_pred)
            confidence = np.max(avg_pred)

            latency = (time.time() - start_time) * \
                1000  # Convert to milliseconds

            return ACTIVITIES[predicted_class], confidence, latency

        except Exception as e:
            print(f"DL prediction error: {e}")
            return "unknown", 0.0, 0.0

    def process_window(self, audio_window):
        """Process a single audio window with both models."""

        # Extract features for both models
        ml_features = self.extract_ml_features(audio_window, SAMPLE_RATE)
        dl_features = self.extract_dl_features(audio_window, SAMPLE_RATE)

        # Make predictions
        ml_pred, ml_conf, ml_lat = self.predict_ml(ml_features)
        dl_pred, dl_conf, dl_lat = self.predict_dl(dl_features)

        # Store results
        current_time = time.time()
        self.timestamps.append(current_time)
        self.ml_predictions.append(ml_pred)
        self.dl_predictions.append(dl_pred)
        self.ml_confidences.append(ml_conf)
        self.dl_confidences.append(dl_conf)
        self.ml_latencies.append(ml_lat)
        self.dl_latencies.append(dl_lat)

        # Print current predictions
        print(f"ML: {ml_pred} (conf: {ml_conf:.3f}, lat: {ml_lat:.1f}ms) | "
              f"DL: {dl_pred} (conf: {dl_conf:.3f}, lat: {dl_lat:.1f}ms)")

    def audio_callback(self, indata, frames, time, status):
        """Callback function for audio input."""
        if status:
            print(f"Audio callback status: {status}")

        # Add new audio data to buffer
        audio_data = indata[:, 0]  # Take first channel if stereo
        self.audio_buffer.extend(audio_data)

        # Check if we have enough data for a window
        if len(self.audio_buffer) >= BUFFER_SIZE:
            # Extract current window
            window_data = np.array(list(self.audio_buffer)[-BUFFER_SIZE:])
            self.current_window = window_data

            # Process window in separate thread to avoid blocking audio callback
            threading.Thread(target=self.process_window, args=(
                window_data,), daemon=True).start()

    def setup_plot(self):
        """Set up the matplotlib plot for real-time visualization."""
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Waveform plot
        self.line1, = self.ax1.plot([], [], 'b-')
        self.ax1.set_xlim(0, BUFFER_SIZE)
        self.ax1.set_ylim(-1, 1)
        self.ax1.set_title('Real-time Audio Waveform (1 second window)')
        self.ax1.set_xlabel('Samples')
        self.ax1.set_ylabel('Amplitude')
        self.ax1.grid(True)

        # Prediction text area
        self.ax2.axis('off')
        self.prediction_text = self.ax2.text(
            0.1, 0.8, '', fontsize=14, family='monospace')

        plt.tight_layout()

    def update_plot(self, frame):
        """Update the plot with current data."""
        # Update waveform
        if len(self.current_window) > 0:
            self.line1.set_data(
                range(len(self.current_window)), self.current_window)

        # Update prediction text
        if len(self.ml_predictions) > 0 and len(self.dl_predictions) > 0:
            ml_pred = self.ml_predictions[-1]
            dl_pred = self.dl_predictions[-1]
            ml_conf = self.ml_confidences[-1]
            dl_conf = self.dl_confidences[-1]
            ml_lat = self.ml_latencies[-1]
            dl_lat = self.dl_latencies[-1]

            prediction_info = f"""
ML Model (Assignment 1):
  Prediction: {ml_pred}
  Confidence: {ml_conf:.3f}
  Latency: {ml_lat:.1f} ms

DL Model (Assignment 2):
  Prediction: {dl_pred}
  Confidence: {dl_conf:.3f}
  Latency: {dl_lat:.1f} ms

Update Rate: ~{1/HOP_LENGTH:.1f} FPS
            """.strip()

            self.prediction_text.set_text(prediction_info)

        return self.line1, self.prediction_text

    def start_realtime_processing(self):
        """Start the real-time audio processing."""
        print("Starting real-time audio processing...")
        print("Perform the 5 activities: laugh, cough, clap, knock, alarm")
        print("Press Ctrl+C to stop")

        try:
            # Start audio stream
            with sd.InputStream(
                callback=self.audio_callback,
                channels=1,
                samplerate=SAMPLE_RATE,
                blocksize=HOP_SIZE,
                dtype=np.float32
            ):
                # Start animation
                self.animation = FuncAnimation(
                    self.fig,
                    self.update_plot,
                    # Update every hop_length seconds
                    interval=int(HOP_LENGTH * 1000),
                    blit=False
                )

                plt.show()

        except KeyboardInterrupt:
            print("\nStopping real-time processing...")
        except Exception as e:
            print(f"Error in real-time processing: {e}")


def main():
    """Main function to run the real-time audio recognition system."""

    # Model paths - adjust these to match your saved models
    ml_model_path = "ml_model.pkl"
    dl_model_path = "dl_model.h5"

    # Check if model files exist
    if not os.path.exists(ml_model_path):
        print(f"Error: ML model file '{ml_model_path}' not found!")
        print("Please ensure your ML model from Assignment 1 is saved as 'ml_model.pkl'")
        return

    if not os.path.exists(dl_model_path):
        print(f"Error: DL model file '{dl_model_path}' not found!")
        print("Please ensure your DL model from Assignment 2 is saved as 'dl_model.h5'")
        return

    # Create and start the real-time processor
    processor = RealTimeAudioProcessor(ml_model_path, dl_model_path)
    processor.start_realtime_processing()


if __name__ == "__main__":
    main()
