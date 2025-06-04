import vggish_params
from vggish_input import waveform_to_examples
import pickle
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
import threading
import time
import queue
import librosa
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import load_model
import sys
import os

# Add the vggish modules to path (assuming they're in the same directory)
sys.path.append('.')


class RealtimeAudioClassifier:
    def __init__(self):
        # Audio parameters
        self.sample_rate = 16000
        self.window_length = 1.0  # 1 second windows
        self.hop_length = 0.5     # 0.5 second overlap
        self.buffer_size = int(self.window_length * self.sample_rate)
        self.hop_size = int(self.hop_length * self.sample_rate)

        # Data buffers
        # Keep extra for overlap
        self.audio_buffer = deque(maxlen=self.buffer_size * 2)
        self.audio_queue = queue.Queue()

        # Activities from assignments
        self.activities = ['laugh', 'cough', 'clap', 'knock', 'alarm']

        # Ubicoustics mapping for DL model
        self.ubicoustics_mapping = {
            'laugh': 'laugh',
            'cough': 'cough',
            'knock': 'knock',
            'hazard-alarm': 'alarm',
            'knock': 'clap'  # Map clap to knock as closest match
        }

        # Load models
        self.load_models()

        # GUI setup
        self.setup_gui()

        # Prediction history for display
        self.ml_predictions = []
        self.dl_predictions = []
        self.ml_confidences = []
        self.dl_confidences = []
        self.ml_latencies = []
        self.dl_latencies = []

        # Audio stream
        self.stream = None
        self.is_running = False

    def load_models(self):
        """Load both ML and DL models"""
        try:
            # Load ML model (from Assignment 1)
            print("Loading ML model...")
            with open('ml_model.pkl', 'rb') as f:
                ml_data = pickle.load(f)
            self.ml_model = ml_data['model']
            self.ml_scaler = ml_data['scaler']
            print("ML model loaded successfully")

            # Load DL model (from Assignment 2)
            print("Loading DL model...")
            self.dl_model = load_model('dl_model.h5', compile=False)
            print("DL model loaded successfully")

        except Exception as e:
            print(f"Error loading models: {e}")
            print(
                "Please ensure ml_model.pkl and ubicoustics_model.h5 are in the current directory")

    def extract_ml_features(self, audio_signal):
        """Extract features for ML model (same as in Assignment 1)"""
        sr = self.sample_rate

        # FFT features
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

        # Band energies
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

        fft_features = np.array(
            [total_energy, spectral_centroid, spectral_spread, spectral_rolloff, *band_energies])

        # MFCC features
        mfccs = librosa.feature.mfcc(y=audio_signal, sr=sr, n_mfcc=13)
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)

        mfcc_means = np.mean(mfccs, axis=1)
        delta_means = np.mean(delta_mfccs, axis=1)
        delta2_means = np.mean(delta2_mfccs, axis=1)
        mfcc_stds = np.std(mfccs, axis=1)

        mfcc_features = np.array([
            mfcc_means[0], mfcc_means[1], mfcc_means[2], mfcc_stds[0],
            delta_means[0], delta_means[1], delta2_means[0],
            np.mean(mfcc_means), np.mean(mfcc_stds)
        ])

        # RMS features
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

        rms_features = np.array([rms_mean, rms_std, rms_max, rms_min, rms_median,
                                rms_range, rms_diff_mean, zcr_mean, low_energy_ratio])

        # Combine all features
        features = np.concatenate([fft_features, mfcc_features, rms_features])
        return features

    def predict_ml(self, audio_signal):
        """Make prediction using ML model"""
        start_time = time.time()

        # Extract features
        features = self.extract_ml_features(audio_signal)
        features = features.reshape(1, -1)

        # Scale features
        features_scaled = self.ml_scaler.transform(features)

        # Predict
        prediction = self.ml_model.predict(features_scaled)[0]
        probabilities = self.ml_model.predict_proba(features_scaled)[0]
        confidence = np.max(probabilities)

        latency = (time.time() - start_time) * 1000  # Convert to ms

        return self.activities[prediction], confidence, latency

    def predict_dl(self, audio_signal):
        """Make prediction using DL model"""
        start_time = time.time()

        # Convert to VGGish input format
        examples = waveform_to_examples(audio_signal, self.sample_rate)

        if examples.shape[0] == 0:
            return "silence", 0.0, 0.0

        # Run prediction
        predictions = self.dl_model.predict(examples, verbose=0)
        avg_pred = np.mean(predictions, axis=0)

        # Get top prediction
        top_class_idx = np.argmax(avg_pred)
        confidence = avg_pred[top_class_idx]

        # Map to our activities (simplified mapping)
        # In a real implementation, you'd have the full Ubicoustics label mapping
        activity_mapping = {0: 'laugh', 1: 'cough',
                            2: 'alarm', 3: 'knock', 4: 'clap'}
        predicted_activity = activity_mapping.get(top_class_idx % 5, 'unknown')

        latency = (time.time() - start_time) * 1000

        return predicted_activity, confidence, latency

    def audio_callback(self, indata, frames, time, status):
        """Callback function for audio stream"""
        if status:
            print(f"Audio callback status: {status}")

        # Add audio data to buffer
        audio_data = indata[:, 0]  # Get mono channel
        self.audio_queue.put(audio_data.copy())

    def process_audio(self):
        """Process audio in separate thread"""
        while self.is_running:
            try:
                # Get audio data from queue
                if not self.audio_queue.empty():
                    audio_chunk = self.audio_queue.get(timeout=0.1)
                    self.audio_buffer.extend(audio_chunk)

                    # Check if we have enough data for a window
                    if len(self.audio_buffer) >= self.buffer_size:
                        # Extract window
                        window_data = np.array(
                            list(self.audio_buffer)[-self.buffer_size:])

                        # Make predictions
                        ml_pred, ml_conf, ml_lat = self.predict_ml(window_data)
                        dl_pred, dl_conf, dl_lat = self.predict_dl(window_data)

                        # Store results (keep last 10 predictions)
                        self.ml_predictions.append(ml_pred)
                        self.ml_confidences.append(ml_conf)
                        self.ml_latencies.append(ml_lat)
                        self.dl_predictions.append(dl_pred)
                        self.dl_confidences.append(dl_conf)
                        self.dl_latencies.append(dl_lat)

                        if len(self.ml_predictions) > 10:
                            self.ml_predictions.pop(0)
                            self.ml_confidences.pop(0)
                            self.ml_latencies.pop(0)
                            self.dl_predictions.pop(0)
                            self.dl_confidences.pop(0)
                            self.dl_latencies.pop(0)

                        # Update display
                        self.update_display(
                            window_data, ml_pred, ml_conf, ml_lat, dl_pred, dl_conf, dl_lat)

                        # Wait for hop time
                        time.sleep(self.hop_length)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in audio processing: {e}")

    def setup_gui(self):
        """Setup GUI for real-time display"""
        self.root = tk.Tk()
        self.root.title("Real-time Acoustic Activity Recognition")
        self.root.geometry("1200x800")

        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Audio waveform plot
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, main_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Prediction display frame
        pred_frame = ttk.LabelFrame(
            main_frame, text="Real-time Predictions", padding=10)
        pred_frame.pack(fill=tk.X, pady=(10, 0))

        # ML predictions
        ml_frame = ttk.LabelFrame(
            pred_frame, text="ML Classifier (Assignment 1)", padding=5)
        ml_frame.pack(fill=tk.X, pady=2)

        self.ml_label_var = tk.StringVar(value="Activity: -")
        self.ml_conf_var = tk.StringVar(value="Confidence: -")
        self.ml_lat_var = tk.StringVar(value="Latency: -")

        ttk.Label(ml_frame, textvariable=self.ml_label_var,
                  font=("Arial", 12, "bold")).pack(anchor=tk.W)
        ttk.Label(ml_frame, textvariable=self.ml_conf_var).pack(anchor=tk.W)
        ttk.Label(ml_frame, textvariable=self.ml_lat_var).pack(anchor=tk.W)

        # DL predictions
        dl_frame = ttk.LabelFrame(
            pred_frame, text="DL Classifier (Assignment 2)", padding=5)
        dl_frame.pack(fill=tk.X, pady=2)

        self.dl_label_var = tk.StringVar(value="Activity: -")
        self.dl_conf_var = tk.StringVar(value="Confidence: -")
        self.dl_lat_var = tk.StringVar(value="Latency: -")

        ttk.Label(dl_frame, textvariable=self.dl_label_var,
                  font=("Arial", 12, "bold")).pack(anchor=tk.W)
        ttk.Label(dl_frame, textvariable=self.dl_conf_var).pack(anchor=tk.W)
        ttk.Label(dl_frame, textvariable=self.dl_lat_var).pack(anchor=tk.W)

        # Control buttons
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(10, 0))

        self.start_button = ttk.Button(
            control_frame, text="Start Recording", command=self.start_recording)
        self.start_button.pack(side=tk.LEFT, padx=(0, 10))

        self.stop_button = ttk.Button(
            control_frame, text="Stop Recording", command=self.stop_recording, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT)

        # Status label
        self.status_var = tk.StringVar(value="Ready to start")
        ttk.Label(control_frame, textvariable=self.status_var).pack(
            side=tk.RIGHT)

    def update_display(self, audio_data, ml_pred, ml_conf, ml_lat, dl_pred, dl_conf, dl_lat):
        """Update GUI display with new predictions"""
        # Update prediction labels
        self.ml_label_var.set(f"Activity: {ml_pred.upper()}")
        self.ml_conf_var.set(f"Confidence: {ml_conf:.3f}")
        self.ml_lat_var.set(f"Latency: {ml_lat:.1f} ms")

        self.dl_label_var.set(f"Activity: {dl_pred.upper()}")
        self.dl_conf_var.set(f"Confidence: {dl_conf:.3f}")
        self.dl_lat_var.set(f"Latency: {dl_lat:.1f} ms")

        # Update waveform plot
        self.ax1.clear()
        time_axis = np.linspace(0, self.window_length, len(audio_data))
        self.ax1.plot(time_axis, audio_data, 'b-', linewidth=0.5)
        self.ax1.set_title("Audio Waveform (Current Window)")
        self.ax1.set_xlabel("Time (s)")
        self.ax1.set_ylabel("Amplitude")
        self.ax1.grid(True, alpha=0.3)

        # Update confidence plot
        self.ax2.clear()
        if len(self.ml_confidences) > 1:
            self.ax2.plot(range(len(self.ml_confidences)),
                          self.ml_confidences, 'b-o', label='ML Model', markersize=4)
            self.ax2.plot(range(len(self.dl_confidences)),
                          self.dl_confidences, 'r-s', label='DL Model', markersize=4)
            self.ax2.set_title("Prediction Confidence Over Time")
            self.ax2.set_xlabel("Window Number")
            self.ax2.set_ylabel("Confidence")
            self.ax2.legend()
            self.ax2.grid(True, alpha=0.3)
            self.ax2.set_ylim(0, 1)

        self.canvas.draw()

    def start_recording(self):
        """Start audio recording and processing"""
        try:
            self.is_running = True

            # Start audio stream
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                callback=self.audio_callback,
                blocksize=self.hop_size
            )
            self.stream.start()

            # Start processing thread
            self.processing_thread = threading.Thread(
                target=self.process_audio)
            self.processing_thread.daemon = True
            self.processing_thread.start()

            # Update GUI
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.status_var.set("Recording and classifying...")

        except Exception as e:
            print(f"Error starting recording: {e}")
            self.status_var.set(f"Error: {e}")

    def stop_recording(self):
        """Stop audio recording and processing"""
        self.is_running = False

        if self.stream:
            self.stream.stop()
            self.stream.close()

        # Update GUI
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_var.set("Stopped")

    def run(self):
        """Run the application"""
        print("Real-time Acoustic Activity Recognition System")
        print("Activities: laugh, cough, clap, knock, alarm")
        print("Window size: 1 second with 0.5 second overlap")
        print("Click 'Start Recording' to begin...")

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

    def on_closing(self):
        """Handle application closing"""
        self.stop_recording()
        self.root.destroy()


if __name__ == "__main__":
    # Check if required files exist
    if not os.path.exists('ml_model.pkl'):
        print("Error: ml_model.pkl not found. Please ensure you have the ML model from Assignment 1.")
        exit(1)

    if not os.path.exists('dl_model.h5'):
        print("Error: ubicoustics_model.h5 not found. Please ensure you have the DL model from Assignment 2.")
        exit(1)

    # Create and run the application
    app = RealtimeAudioClassifier()
    app.run()
