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

sys.path.append('.')


class RealtimeAudioClassifier:
    def __init__(self):
        self.sample_rate = 16000
        self.window_length = 1.0
        self.hop_length = 0.5
        self.buffer_size = int(self.window_length * self.sample_rate)
        self.hop_size = int(self.hop_length * self.sample_rate)

        self.audio_buffer = deque(maxlen=self.buffer_size * 2)
        self.audio_queue = queue.Queue()

        self.activities = ['laugh', 'cough', 'clap', 'knock', 'alarm']

        self.ubicoustics_mapping = {
            'laugh': 'laugh',
            'cough': 'cough',
            'knock': 'knock',
            'hazard-alarm': 'alarm',
            'knock': 'clap'
        }

        self.load_models()
        self.setup_gui()

        self.ml_predictions = []
        self.dl_predictions = []
        self.ml_confidences = []
        self.dl_confidences = []
        self.ml_latencies = []
        self.dl_latencies = []

        self.stream = None
        self.is_running = False

    def load_models(self):
        try:
            print("Loading ML model...")
            with open('ml_model.pkl', 'rb') as f:
                ml_data = pickle.load(f)
            self.ml_model = ml_data['model']
            self.ml_scaler = ml_data['scaler']
            print("ML model loaded successfully")

            print("Loading DL model...")
            self.dl_model = load_model('dl_model.h5', compile=False)
            print("DL model loaded successfully")

        except Exception as e:
            print(f"Error loading models: {e}")
            print(
                "Please ensure ml_model.pkl and ubicoustics_model.h5 are in the current directory")

    def extract_ml_features(self, audio_signal):
        sr = self.sample_rate

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

        fft_features = np.array(
            [total_energy, spectral_centroid, spectral_spread, spectral_rolloff, *band_energies])

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

        rms_features = np.array([rms_mean, rms_std, rms_max, rms_min,
                                rms_median, rms_range, rms_diff_mean, zcr_mean, low_energy_ratio])

        features = np.concatenate([fft_features, mfcc_features, rms_features])
        return features

    def predict_ml(self, audio_signal):
        start_time = time.time()

        features = self.extract_ml_features(audio_signal)
        features = features.reshape(1, -1)

        features_scaled = self.ml_scaler.transform(features)

        prediction = self.ml_model.predict(features_scaled)[0]
        probabilities = self.ml_model.predict_proba(features_scaled)[0]
        confidence = np.max(probabilities)

        latency = (time.time() - start_time) * 1000

        return self.activities[prediction], confidence, latency

    def predict_dl(self, audio_signal):
        start_time = time.time()

        examples = waveform_to_examples(audio_signal, self.sample_rate)

        if examples.shape[0] == 0:
            return "silence", 0.0, 0.0

        predictions = self.dl_model.predict(examples, verbose=0)
        avg_pred = np.mean(predictions, axis=0)

        top_class_idx = np.argmax(avg_pred)
        confidence = avg_pred[top_class_idx]

        activity_mapping = {0: 'laugh', 1: 'cough',
                            2: 'alarm', 3: 'knock', 4: 'clap'}
        predicted_activity = activity_mapping.get(top_class_idx % 5, 'unknown')

        latency = (time.time() - start_time) * 1000

        return predicted_activity, confidence, latency

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(f"Audio callback status: {status}")

        audio_data = indata[:, 0]
        self.audio_queue.put(audio_data.copy())

    def process_audio(self):
        while self.is_running:
            try:
                if not self.audio_queue.empty():
                    audio_chunk = self.audio_queue.get(timeout=0.1)
                    self.audio_buffer.extend(audio_chunk)

                    if len(self.audio_buffer) >= self.buffer_size:
                        window_data = np.array(
                            list(self.audio_buffer)[-self.buffer_size:])

                        ml_pred, ml_conf, ml_lat = self.predict_ml(window_data)
                        dl_pred, dl_conf, dl_lat = self.predict_dl(window_data)

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

                        self.update_display(
                            window_data, ml_pred, ml_conf, ml_lat, dl_pred, dl_conf, dl_lat)

                        time.sleep(self.hop_length)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in audio processing: {e}")

    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("Real-time Acoustic Activity Recognition")
        self.root.geometry("1200x800")

        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, main_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        pred_frame = ttk.LabelFrame(
            main_frame, text="Real-time Predictions", padding=10)
        pred_frame.pack(fill=tk.X, pady=(10, 0))

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

        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(10, 0))

        self.start_button = ttk.Button(
            control_frame, text="Start Recording", command=self.start_recording)
        self.start_button.pack(side=tk.LEFT, padx=(0, 10))

        self.stop_button = ttk.Button(
            control_frame, text="Stop Recording", command=self.stop_recording, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT)

        self.status_var = tk.StringVar(value="Ready to start")
        ttk.Label(control_frame, textvariable=self.status_var).pack(
            side=tk.RIGHT)

    def update_display(self, audio_data, ml_pred, ml_conf, ml_lat, dl_pred, dl_conf, dl_lat):
        self.ml_label_var.set(f"Activity: {ml_pred.upper()}")
        self.ml_conf_var.set(f"Confidence: {ml_conf:.3f}")
        self.ml_lat_var.set(f"Latency: {ml_lat:.1f} ms")

        self.dl_label_var.set(f"Activity: {dl_pred.upper()}")
        self.dl_conf_var.set(f"Confidence: {dl_conf:.3f}")
        self.dl_lat_var.set(f"Latency: {dl_lat:.1f} ms")

        self.ax1.clear()
        time_axis = np.linspace(0, self.window_length, len(audio_data))
        self.ax1.plot(time_axis, audio_data, 'b-', linewidth=0.5)
        self.ax1.set_title("Audio Waveform (Current Window)")
        self.ax1.set_xlabel("Time (s)")
        self.ax1.set_ylabel("Amplitude")
        self.ax1.grid(True, alpha=0.3)

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
        try:
            self.is_running = True

            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                callback=self.audio_callback,
                blocksize=self.hop_size
            )
            self.stream.start()

            self.processing_thread = threading.Thread(
                target=self.process_audio)
            self.processing_thread.daemon = True
            self.processing_thread.start()

            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.status_var.set("Recording and classifying...")

        except Exception as e:
            print(f"Error starting recording: {e}")
            self.status_var.set(f"Error: {e}")

    def stop_recording(self):
        self.is_running = False

        if self.stream:
            self.stream.stop()
            self.stream.close()

        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_var.set("Stopped")

    def run(self):
        print("Real-time Acoustic Activity Recognition System")
        print("Activities: laugh, cough, clap, knock, alarm")
        print("Window size: 1 second with 0.5 second overlap")
        print("Click 'Start Recording' to begin...")

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

    def on_closing(self):
        self.stop_recording()
        self.root.destroy()


if __name__ == "__main__":
    if not os.path.exists('ml_model.pkl'):
        print("Error: ml_model.pkl not found. Please ensure you have the ML model from Assignment 1.")
        exit(1)

    if not os.path.exists('dl_model.h5'):
        print("Error: ubicoustics_model.h5 not found. Please ensure you have the DL model from Assignment 2.")
        exit(1)

    app = RealtimeAudioClassifier()
    app.run()
