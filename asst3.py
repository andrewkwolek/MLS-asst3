import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
import sounddevice as sd
import librosa
import librosa.display
import time
import threading
import queue
from collections import deque
import pickle
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import warnings
import os
import sys
warnings.filterwarnings('ignore')

# Add VGGish to path (make sure you have the vggish files in your directory)
sys.path.append('.')
try:
    from vggish_input import waveform_to_examples
    VGGISH_AVAILABLE = True
except ImportError:
    print("Warning: VGGish not available, using fallback feature extraction")
    VGGISH_AVAILABLE = False


class AudioFeatureExtractor:
    """Handles feature extraction for both ML and DL models"""

    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def extract_fft(self, audio_signal, sr=16000):
        """Extract FFT features for ML model"""
        windowed_signal = audio_signal * \
            librosa.filters.get_window('hann', len(audio_signal))
        stft = librosa.stft(windowed_signal)
        magnitude_spectrum = np.abs(stft)
        power_spectrum = magnitude_spectrum ** 2
        freq_bins = librosa.fft_frequencies(sr=sr, n_fft=2*(stft.shape[0]-1))

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
                band_energy = np.sum(power_spectrum[indices]) / total_energy
            else:
                band_energy = 0
            band_energies.append(band_energy)

        features = np.array([total_energy, spectral_centroid,
                            spectral_spread, spectral_rolloff, *band_energies])
        return features

    def extract_mfcc(self, audio_signal, sr=16000):
        """Extract MFCC features for ML model"""
        mfccs = librosa.feature.mfcc(y=audio_signal, sr=sr, n_mfcc=13)
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)

        mfcc_means = np.mean(mfccs, axis=1)
        mfcc_stds = np.std(mfccs, axis=1)
        delta_means = np.mean(delta_mfccs, axis=1)
        delta2_means = np.mean(delta2_mfccs, axis=1)

        selected_features = np.array([
            mfcc_means[0], mfcc_means[1], mfcc_means[2], mfcc_stds[0],
            delta_means[0], delta_means[1], delta2_means[0],
            np.mean(mfcc_means), np.mean(mfcc_stds)
        ])
        return selected_features

    def extract_rms(self, audio_signal, sr=16000):
        """Extract RMS energy features for ML model"""
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

        features = np.array([
            rms_mean, rms_std, rms_max, rms_min, rms_median,
            rms_range, rms_diff_mean, zcr_mean, low_energy_ratio
        ])
        return features

    def extract_combined_features(self, audio_signal):
        """Extract all features for ML model"""
        fft_features = self.extract_fft(audio_signal, self.sample_rate)
        mfcc_features = self.extract_mfcc(audio_signal, self.sample_rate)
        rms_features = self.extract_rms(audio_signal, self.sample_rate)

        combined_features = np.concatenate(
            [fft_features, mfcc_features, rms_features])
        return combined_features

    def extract_vggish_features(self, audio_signal):
        """Extract VGGish features exactly like Ubicoustics does"""
        if VGGISH_AVAILABLE:
            # Convert to the same format as Ubicoustics
            # Ensure audio is float32 in [-1, 1] range
            if audio_signal.dtype != np.float32:
                audio_signal = audio_signal.astype(np.float32)

            # Normalize to [-1, 1] if needed
            if np.max(np.abs(audio_signal)) > 1.0:
                audio_signal = audio_signal / np.max(np.abs(audio_signal))

            # Use VGGish preprocessing exactly like Ubicoustics
            examples = waveform_to_examples(audio_signal, self.sample_rate)

            if examples.shape[0] == 0:
                # If no examples generated, create dummy data
                examples = np.zeros((1, 96, 64))

            # Reshape to match Ubicoustics format: (n_examples, 96, 64, 1)
            examples = examples.reshape(len(examples), 96, 64, 1)

            return examples
        else:
            # Fallback if VGGish not available
            return self.extract_vggish_features_fallback(audio_signal)

    def extract_vggish_features_fallback(self, audio_signal):
        """Fallback VGGish-style features if vggish_input not available"""
        mel_spec = librosa.feature.melspectrogram(
            y=audio_signal,
            sr=self.sample_rate,
            n_mels=64,
            fmax=8000,
            hop_length=160,
            n_fft=512
        )

        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        target_frames = 96
        if log_mel_spec.shape[1] < target_frames:
            pad_width = target_frames - log_mel_spec.shape[1]
            log_mel_spec = np.pad(
                log_mel_spec, ((0, 0), (0, pad_width)), 'constant', constant_values=-80)
        elif log_mel_spec.shape[1] > target_frames:
            log_mel_spec = log_mel_spec[:, :target_frames]

        features = log_mel_spec.T
        features = np.expand_dims(features, axis=-1)
        features = (features - np.mean(features)) / (np.std(features) + 1e-8)

        # Return in batch format like VGGish
        return np.expand_dims(features, axis=0)


class RealTimeAudioClassifier:
    """Main class for real-time audio classification"""

    def __init__(self, ml_model_path=None, dl_model_path=None):
        # Audio configuration
        self.sample_rate = 16000
        self.window_duration = 1.0  # seconds
        self.hop_duration = 0.5     # seconds
        self.window_samples = int(self.window_duration * self.sample_rate)
        self.hop_samples = int(self.hop_duration * self.sample_rate)

        # Activities - your 5 classes
        self.activities = ['laugh', 'cough', 'clap', 'knock', 'alarm']

        # Ubicoustics labels (30 classes)
        self.ubicoustics_labels = {
            'dog-bark': 0, 'drill': 1, 'hazard-alarm': 2, 'phone-ring': 3, 'speech': 4,
            'vacuum': 5, 'baby-cry': 6, 'chopping': 7, 'cough': 8, 'door': 9,
            'water-running': 10, 'knock': 11, 'microwave': 12, 'shaver': 13, 'toothbrush': 14,
            'blender': 15, 'dishwasher': 16, 'doorbell': 17, 'flush': 18, 'hair-dryer': 19,
            'laugh': 20, 'snore': 21, 'typing': 22, 'hammer': 23, 'car-horn': 24,
            'engine': 25, 'saw': 26, 'cat-meow': 27, 'alarm-clock': 28, 'cooking': 29
        }

        # Human readable labels for Ubicoustics
        self.ubicoustics_human_labels = {
            0: "Dog Barking", 1: "Drill In-Use", 2: "Hazard Alarm", 3: "Phone Ringing",
            4: "Person Talking", 5: "Vacuum In-Use", 6: "Baby Crying", 7: "Chopping",
            8: "Coughing", 9: "Door In-Use", 10: "Water Running", 11: "Knocking",
            12: "Microwave In-Use", 13: "Shaver In-Use", 14: "Toothbrushing", 15: "Blender In-Use",
            16: "Dishwasher In-Use", 17: "Doorbell In-Use", 18: "Toilet Flushing", 19: "Hair Dryer In-Use",
            20: "Laughing", 21: "Snoring", 22: "Typing", 23: "Hammering", 24: "Car Honking",
            25: "Vehicle Running", 26: "Saw In-Use", 27: "Cat Meowing", 28: "Alarm Clock", 29: "Utensils and Cutlery"
        }

        # Mapping from Ubicoustics classes to your activities
        self.ubicoustics_to_activity = {
            20: 0,  # laugh -> laugh
            8: 1,   # cough -> cough
            11: 2,  # knock -> clap (closest match)
            11: 3,  # knock -> knock
            2: 4,   # hazard-alarm -> alarm
            28: 4   # alarm-clock -> alarm
        }

        # Feature extractor
        self.feature_extractor = AudioFeatureExtractor(self.sample_rate)

        # Audio buffer
        self.audio_buffer = deque(maxlen=self.window_samples * 3)
        self.audio_queue = queue.Queue()

        # Results storage
        self.ml_results = deque(maxlen=50)
        self.dl_results = deque(maxlen=50)
        self.ml_times = deque(maxlen=50)
        self.dl_times = deque(maxlen=50)
        self.timestamps = deque(maxlen=50)

        # Load models
        self.ml_model = self.load_ml_model(ml_model_path)
        self.dl_model = self.load_dl_model(dl_model_path)

        # Control flags
        self.is_recording = False
        self.is_processing = False

        # Prediction thresholds (like in Ubicoustics)
        self.prediction_threshold = 0.5
        self.db_threshold = -40

        # Initialize GUI
        self.setup_gui()

    def load_ml_model(self, model_path):
        """Load or create ML model"""
        if model_path and os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    print(f"Loaded ML model from {model_path}")
                    return model_data['model'], model_data['scaler']
            except Exception as e:
                print(f"Could not load ML model: {e}")

        # Create dummy ML model for demonstration
        print("Creating dummy ML model...")
        model = MLPClassifier(
            hidden_layer_sizes=(100,),
            activation='relu',
            alpha=0.0001,
            max_iter=1000,
            random_state=42
        )

        # Train on dummy data
        dummy_X = np.random.randn(100, 27)  # 9+9+9 features
        dummy_y = np.random.randint(0, len(self.activities), 100)
        model.fit(dummy_X, dummy_y)

        scaler = StandardScaler()
        scaler.fit(dummy_X)

        return model, scaler

    def load_dl_model(self, model_path):
        """Load DL model (Ubicoustics)"""
        if model_path and os.path.exists(model_path):
            try:
                model = keras.models.load_model(model_path)
                print(f"Loaded DL model from {model_path}")
                print(f"Model input shape: {model.input_shape}")
                print(f"Model output shape: {model.output_shape}")
                return model
            except Exception as e:
                print(f"Could not load DL model: {e}")

        print("Warning: No valid DL model found. Please provide the path to your Ubicoustics model.")
        return None

    def map_ubicoustics_to_activities(self, ubicoustics_probs):
        """Map 30-class Ubicoustics predictions to 5-class activity predictions"""
        activity_probs = np.zeros(len(self.activities))

        # Direct mapping for classes that match
        for ubicoustics_idx, activity_idx in self.ubicoustics_to_activity.items():
            if ubicoustics_idx < len(ubicoustics_probs):
                activity_probs[activity_idx] += ubicoustics_probs[ubicoustics_idx]

        # For clap, we don't have a direct match, so we use knock as proxy
        # clap gets knock probability
        activity_probs[2] = ubicoustics_probs[11]

        # Normalize probabilities
        if np.sum(activity_probs) > 0:
            activity_probs = activity_probs / np.sum(activity_probs)

        return activity_probs

    def compute_audio_level_db(self, audio_data):
        """Compute audio level in dB like Ubicoustics"""
        rms = np.sqrt(np.mean(audio_data**2))
        if rms > 0:
            db = 20 * np.log10(rms)
        else:
            db = -80  # Very quiet
        return db

    def setup_gui(self):
        """Setup the matplotlib GUI"""
        self.fig, ((self.ax_waveform, self.ax_ml), (self.ax_dl,
                   self.ax_confidence)) = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle(
            'Real-Time Acoustic Activity Recognition', fontsize=16)

        # Waveform plot
        self.ax_waveform.set_title('Audio Waveform (Last 1 second)')
        self.ax_waveform.set_xlabel('Time (s)')
        self.ax_waveform.set_ylabel('Amplitude')
        self.ax_waveform.set_ylim(-1, 1)
        self.line_waveform, = self.ax_waveform.plot([], [], 'b-')

        # ML predictions plot
        self.ax_ml.set_title('ML Model Predictions')
        self.ax_ml.set_xlabel('Time')
        self.ax_ml.set_ylabel('Activity')
        self.ax_ml.set_yticks(range(len(self.activities)))
        self.ax_ml.set_yticklabels(self.activities)

        # DL predictions plot
        self.ax_dl.set_title('DL Model Predictions')
        self.ax_dl.set_xlabel('Time')
        self.ax_dl.set_ylabel('Activity')
        self.ax_dl.set_yticks(range(len(self.activities)))
        self.ax_dl.set_yticklabels(self.activities)

        # Confidence comparison
        self.ax_confidence.set_title('Model Confidence Comparison')
        self.ax_confidence.set_xlabel('Activity')
        self.ax_confidence.set_ylabel('Confidence')
        self.ax_confidence.set_xticks(range(len(self.activities)))
        self.ax_confidence.set_xticklabels(self.activities, rotation=45)

        # Control buttons
        ax_start = plt.axes([0.02, 0.02, 0.1, 0.04])
        ax_stop = plt.axes([0.13, 0.02, 0.1, 0.04])

        self.btn_start = Button(ax_start, 'Start')
        self.btn_stop = Button(ax_stop, 'Stop')

        self.btn_start.on_clicked(self.start_recording)
        self.btn_stop.on_clicked(self.stop_recording)

        # Status text
        self.status_text = self.fig.text(
            0.02, 0.95, 'Status: Ready', fontsize=12)
        self.ml_info_text = self.fig.text(
            0.5, 0.95, 'ML: --- (--- ms)', fontsize=10)
        self.dl_info_text = self.fig.text(
            0.5, 0.92, 'DL: --- (--- ms)', fontsize=10)

    def audio_callback(self, indata, frames, time, status):
        """Callback function for audio input"""
        if status:
            print(f'Audio callback status: {status}')

        if self.is_recording:
            # Add audio data to buffer
            audio_data = indata[:, 0]  # Take first channel
            self.audio_buffer.extend(audio_data)

            # Add to processing queue if we have enough data
            if len(self.audio_buffer) >= self.window_samples:
                window_data = np.array(
                    list(self.audio_buffer)[-self.window_samples:])
                self.audio_queue.put(window_data)

    def process_audio(self):
        """Process audio in a separate thread"""
        while self.is_processing:
            try:
                # Get audio data from queue
                audio_data = self.audio_queue.get(timeout=0.1)

                # Compute audio level
                audio_db = self.compute_audio_level_db(audio_data)

                # ML Model Processing
                start_time = time.time()
                try:
                    ml_features = self.feature_extractor.extract_combined_features(
                        audio_data)
                    ml_features_scaled = self.ml_model[1].transform(
                        [ml_features])
                    ml_probs = self.ml_model[0].predict_proba(
                        ml_features_scaled)[0]
                    ml_prediction = np.argmax(ml_probs)
                    ml_confidence = ml_probs[ml_prediction]
                    ml_time = (time.time() - start_time) * 1000  # ms
                except Exception as e:
                    print(f"ML processing error: {e}")
                    ml_prediction = 0
                    ml_confidence = 0.0
                    ml_probs = np.zeros(len(self.activities))
                    ml_time = 0

                # DL Model Processing (Ubicoustics style)
                start_time = time.time()
                dl_prediction = 0
                dl_confidence = 0.0
                dl_probs = np.zeros(len(self.activities))
                dl_time = 0

                if self.dl_model is not None:
                    try:
                        # Extract VGGish features exactly like Ubicoustics
                        vggish_examples = self.feature_extractor.extract_vggish_features(
                            audio_data)

                        if vggish_examples.shape[0] > 0:
                            # Make prediction with Ubicoustics model
                            ubicoustics_predictions = self.dl_model.predict(
                                vggish_examples, verbose=0)

                            # Average predictions across all examples
                            avg_ubicoustics_pred = np.mean(
                                ubicoustics_predictions, axis=0)

                            # Map to your 5 activities
                            dl_probs = self.map_ubicoustics_to_activities(
                                avg_ubicoustics_pred)
                            dl_prediction = np.argmax(dl_probs)
                            dl_confidence = dl_probs[dl_prediction]

                            # Apply thresholds like Ubicoustics
                            if dl_confidence < self.prediction_threshold or audio_db < self.db_threshold:
                                dl_prediction = -1  # No prediction
                                dl_confidence = 0.0

                        dl_time = (time.time() - start_time) * 1000  # ms

                    except Exception as e:
                        print(f"DL processing error: {e}")
                        import traceback
                        traceback.print_exc()

                # Store results
                current_time = time.time()
                self.ml_results.append(
                    (ml_prediction, ml_confidence, ml_probs))
                self.dl_results.append(
                    (dl_prediction, dl_confidence, dl_probs))
                self.ml_times.append(ml_time)
                self.dl_times.append(dl_time)
                self.timestamps.append(current_time)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing error: {e}")

    def update_plots(self, frame):
        """Update all plots"""
        if not self.is_recording:
            return []

        # Update waveform
        if len(self.audio_buffer) >= self.window_samples:
            audio_data = np.array(list(self.audio_buffer)
                                  [-self.window_samples:])
            time_axis = np.linspace(0, self.window_duration, len(audio_data))
            self.line_waveform.set_data(time_axis, audio_data)
            self.ax_waveform.set_xlim(0, self.window_duration)

            # Auto-scale y-axis
            if np.max(np.abs(audio_data)) > 0:
                self.ax_waveform.set_ylim(
                    np.min(audio_data) * 1.1, np.max(audio_data) * 1.1)

        # Update ML predictions
        self.ax_ml.clear()
        self.ax_ml.set_title('ML Model Predictions')
        self.ax_ml.set_ylabel('Activity')
        self.ax_ml.set_yticks(range(len(self.activities)))
        self.ax_ml.set_yticklabels(self.activities)

        if self.ml_results:
            recent_ml = list(self.ml_results)[-20:]
            ml_preds = [r[0] for r in recent_ml]
            ml_confs = [r[1] for r in recent_ml]
            x_pos = range(len(ml_preds))

            colors = plt.cm.viridis(ml_confs)
            self.ax_ml.scatter(x_pos, ml_preds, c=colors, s=50)

            if ml_preds:
                latest_ml = recent_ml[-1]
                self.ax_ml.set_title(
                    f'ML: {self.activities[latest_ml[0]]} ({latest_ml[1]:.2f})')

        # Update DL predictions
        self.ax_dl.clear()
        self.ax_dl.set_title('DL Model Predictions')
        self.ax_dl.set_ylabel('Activity')
        self.ax_dl.set_yticks(range(-1, len(self.activities)))
        self.ax_dl.set_yticklabels(['No Pred'] + self.activities)

        if self.dl_results:
            recent_dl = list(self.dl_results)[-20:]
            dl_preds = [r[0] for r in recent_dl]
            dl_confs = [r[1] for r in recent_dl]
            x_pos = range(len(dl_preds))

            colors = plt.cm.plasma(dl_confs)
            self.ax_dl.scatter(x_pos, dl_preds, c=colors, s=50)

            if dl_preds:
                latest_dl = recent_dl[-1]
                if latest_dl[0] >= 0:
                    self.ax_dl.set_title(
                        f'DL: {self.activities[latest_dl[0]]} ({latest_dl[1]:.2f})')
                else:
                    self.ax_dl.set_title(
                        f'DL: No Prediction ({latest_dl[1]:.2f})')

        # Update confidence comparison
        self.ax_confidence.clear()
        self.ax_confidence.set_title('Latest Predictions Confidence')
        self.ax_confidence.set_xlabel('Activity')
        self.ax_confidence.set_ylabel('Confidence')
        self.ax_confidence.set_xticks(range(len(self.activities)))
        self.ax_confidence.set_xticklabels(self.activities, rotation=45)

        if self.ml_results and self.dl_results:
            latest_ml_probs = self.ml_results[-1][2]
            latest_dl_probs = self.dl_results[-1][2]

            if len(latest_ml_probs) == len(self.activities) and len(latest_dl_probs) == len(self.activities):
                x_pos = np.arange(len(self.activities))
                width = 0.35

                self.ax_confidence.bar(x_pos - width/2, latest_ml_probs, width,
                                       label='ML Model', alpha=0.8, color='blue')
                self.ax_confidence.bar(x_pos + width/2, latest_dl_probs, width,
                                       label='DL Model', alpha=0.8, color='red')
                self.ax_confidence.legend()
                self.ax_confidence.set_ylim(0, 1)

        # Update status
        if self.ml_times and self.dl_times:
            avg_ml_time = np.mean(list(self.ml_times)[-10:])
            avg_dl_time = np.mean(list(self.dl_times)[-10:])

            if self.ml_results:
                self.ml_info_text.set_text(
                    f'ML: {self.activities[self.ml_results[-1][0]]} ({avg_ml_time:.1f} ms)')

            if self.dl_results and self.dl_results[-1][0] >= 0:
                self.dl_info_text.set_text(
                    f'DL: {self.activities[self.dl_results[-1][0]]} ({avg_dl_time:.1f} ms)')
            else:
                self.dl_info_text.set_text(
                    f'DL: No Prediction ({avg_dl_time:.1f} ms)')

        return []

    def start_recording(self, event):
        """Start audio recording and processing"""
        if not self.is_recording:
            print("Starting recording...")
            self.is_recording = True
            self.is_processing = True

            # Start audio stream
            self.stream = sd.InputStream(
                callback=self.audio_callback,
                channels=1,
                samplerate=self.sample_rate,
                blocksize=self.hop_samples
            )
            self.stream.start()

            # Start processing thread
            self.processing_thread = threading.Thread(
                target=self.process_audio)
            self.processing_thread.daemon = True
            self.processing_thread.start()

            self.status_text.set_text('Status: Recording and Processing')

    def stop_recording(self, event):
        """Stop audio recording and processing"""
        if self.is_recording:
            print("Stopping recording...")
            self.is_recording = False
            self.is_processing = False

            # Stop audio stream
            if hasattr(self, 'stream'):
                self.stream.stop()
                self.stream.close()

            self.status_text.set_text('Status: Stopped')

    def run(self):
        """Run the real-time classifier"""
        print("Real-Time Audio Classifier")
        print("Activities:", self.activities)
        if self.dl_model is None:
            print("WARNING: No DL model loaded. Only ML predictions will be available.")
        print("Click 'Start' to begin recording and classification")
        print("Click 'Stop' to end recording")
        print("Close the window to exit")

        # Start animation
        self.ani = animation.FuncAnimation(
            self.fig, self.update_plots, interval=100, blit=False
        )

        plt.tight_layout()
        plt.show()


def main():
    """Main function"""
    import os

    # Paths to your trained models
    ml_model_path = "ml_model.pkl"  # Path to your ML model from Assignment 1
    # Path to your Ubicoustics model from Assignment 2
    dl_model_path = "models/ubicoustics_model.h5"

    # Check if models exist
    if not os.path.exists(ml_model_path):
        print(f"ML model not found at {ml_model_path}, will use dummy model")
        ml_model_path = None

    if not os.path.exists(dl_model_path):
        print(f"DL model not found at {dl_model_path}")
        print("Please provide the path to your Ubicoustics model from Assignment 2")
        dl_model_path = None

    # Create and run classifier
    classifier = RealTimeAudioClassifier(ml_model_path, dl_model_path)
    classifier.run()


if __name__ == "__main__":
    main()
