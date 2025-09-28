import os

import numpy as np
import librosa
import random
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.signal import butter, filtfilt
import cv2

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, mean_squared_error, mean_absolute_error, r2_score

def classify_emotion(valence, arousal):
    if valence >= 5 and arousal >= 5:
        return 'happy'
    if valence >= 5 >= arousal:
        return 'relaxed'
    if valence <= 5 and arousal <= 5:
        return 'sad'
    if valence <= 5 <= arousal:
        return 'anger'


def classify_emotion_regression(valence, arousal):
    if valence >= 0 and arousal >= 0:
        return 'happy'
    if valence >= 0 >= arousal:
        return 'relaxed'
    if valence <= 0 and arousal <= 0:
        return 'sad'
    if valence <= 0 <= arousal:
        return 'angry'


def augment_audio(segment, sr, number):
    # Time Stretch
    # rate = random.uniform(0.8, 1.2)
    # stretched = librosa.effects.time_stretch(y=segment, rate=rate)
    # if len(stretched) >= len(segment):
    #     stretched = stretched[:len(segment)]
    # else:
    #     stretched = np.pad(stretched, (0, len(segment) - len(stretched)))
    # augmented.append(stretched)

    if number % 3 == 0:
        # Pitch Shift
        return librosa.effects.pitch_shift(segment, sr=sr, n_steps=random.choice([-2, 2]))

    elif number % 3 == 1:
        # Add noise
        return segment + 0.005 * np.random.randn(len(segment))

    else:
        # Time Shift
        shift = int(0.1 * sr)
        return np.roll(segment, shift)


def extract_spectral_entropy(y):
    stft = np.abs(librosa.stft(y)) ** 2  # moc widma
    ps = stft / (np.sum(stft, axis=0, keepdims=True) + 1e-10) # Normalizacja w każdym oknie (kolumnie)
    return -np.sum(ps * np.log2(ps + 1e-10), axis=0) # Entropia Shannona dla każdego okna


def extract_attack_time(y, sr):
    rms = librosa.feature.rms(y=y)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr)

    rms_norm = rms / np.max(rms)
    # czas od pierwszego wzrostu >10% max do osiągnięcia 90% max
    threshold_low, threshold_high = 0.1, 0.9
    try:
        start_idx = np.where(rms_norm >= threshold_low)[0][0]
        peak_idx = np.where(rms_norm >= threshold_high)[0][0]
        result = times[peak_idx] - times[start_idx]
    except IndexError:
        result = None

    return result


def extract_features(path):
    features = dict()
    y, sr = librosa.load(path, sr=44100, mono=True)
    y = butter_filter(y, sr, cutoff_low=100, order=2, btype='high')
    y = butter_filter(y, sr, cutoff_high=5000, order=2, btype='low')

    rmse = librosa.feature.rms(y=y)
    low_energy = rmse < np.mean(rmse)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_entropy = extract_spectral_entropy(y)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    attack_time = extract_attack_time(y, sr)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)

    features['rmse_mean'] = np.mean(rmse)
    features['low_energy'] = np.mean(low_energy)
    features['tempo_mean'] = tempo[0]
    features['spectral_centroid_mean'] = np.mean(spectral_centroid)
    features['spectral_entropy'] = np.mean(spectral_entropy)
    features['zcr_mean'] = np.mean(zcr)
    for i in range(13):
        features[f'mfcc_mean_{i+1}'] = np.mean(mfcc[i]) if mfcc[i] is not None else None

    features['attack_time_mean'] = np.mean(attack_time)
    # features['attack_time_std'] = np.std(attack_time)
    for i in range(12):
        features[f'chroma_stft_mean_{i+1}'] = np.mean(chroma_stft[i]) if chroma_stft[i] is not None else None

    # S = np.abs(librosa.stft(y))
    # chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr)
    # chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
    # melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    # spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    # contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
    # rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    # poly_features = librosa.feature.poly_features(S=S, sr=sr)
    # tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    # harmonic = librosa.effects.harmonic(y)
    # percussive = librosa.effects.percussive(y)
    # mfcc_delta = librosa.feature.delta(mfcc)
    # onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    # frames_to_time = librosa.frames_to_time(onset_frames[:20], sr=sr)

    return features


def extract_features_segmented(path, song_id, start_time=15.0, frame_duration=0.5, sr=44100):
    y, sr = librosa.load(path, sr=sr, mono=True)
    y = butter_filter(y, sr, cutoff_low=100, order=2, btype='high')
    y = butter_filter(y, sr, cutoff_high=5000, order=2, btype='low')

    start_sample = int(start_time * sr)
    y = y[start_sample:]

    frame_length = int(frame_duration * sr)
    num_frames = len(y) // frame_length

    features_list = []

    for i in range(num_frames):
        segment = y[i * frame_length: (i + 1) * frame_length]
        if len(segment) < frame_length:
            continue

        rmse = librosa.feature.rms(y=segment)
        low_energy = rmse < np.mean(rmse)
        tempo, beats = librosa.beat.beat_track(y=segment, sr=sr)
        spectral_centroid = librosa.feature.spectral_centroid(y=segment, sr=sr)
        spectral_entropy = extract_spectral_entropy(segment)
        zcr = librosa.feature.zero_crossing_rate(segment)
        attack_time = extract_attack_time(y, sr)

        chroma_mean = librosa.feature.chroma_stft(y=segment, sr=sr, n_chroma=12).mean(axis=1)
        mfcc_mean = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13).mean(axis=1)

        feature_dict = {
            "song_id": song_id,
            "second": int((i * frame_duration) + start_time),
            "rmse_mean": np.mean(rmse),
            "low_energy": np.mean(low_energy),
            "tempo_mean": tempo[0] if isinstance(tempo, list) else tempo,
            "spectral_centroid_mean": np.mean(spectral_centroid),
            "spectral_entropy": np.mean(spectral_entropy),
            "zcr_mean": np.mean(zcr),
            "attack_time_mean": np.mean(attack_time),
        }
        for j, coeff in enumerate(mfcc_mean):
            feature_dict[f"mfcc_mean_{j + 1}"] = coeff

        for j, coeff in enumerate(chroma_mean):
            feature_dict[f"chroma_mean_{j + 1}"] = coeff

        features_list.append(feature_dict)

    return pd.DataFrame(features_list)


def butter_filter(y, sr, cutoff_low=None, cutoff_high=None, order=2, btype='low'):
    """
    Filtr Butterwortha do sygnału audio.

    y : sygnał audio
    sr : częstotliwość próbkowania
    cutoff_low : dolna częstotliwość graniczna (Hz) dla highpass/bandpass
    cutoff_high : górna częstotliwość graniczna (Hz) dla lowpass/bandpass
    order : rząd filtra
    btype : 'low', 'high', 'bandpass', 'bandstop'
    """
    nyq = 0.5 * sr  # częstotliwość Nyquista

    if btype == 'low':
        normal_cutoff = cutoff_high / nyq
        b, a = butter(order, normal_cutoff, btype='low')
    elif btype == 'high':
        normal_cutoff = cutoff_low / nyq
        b, a = butter(order, normal_cutoff, btype='high')
    elif btype in ['bandpass', 'bandstop']:
        low = cutoff_low / nyq
        high = cutoff_high / nyq
        b, a = butter(order, [low, high], btype=btype)
    else:
        raise ValueError("btype musi być 'low', 'high', 'bandpass' lub 'bandstop'")

    y_filtered = filtfilt(b, a, y)
    return y_filtered


def show_class_proportions(y):
    y.value_counts().plot(kind='bar', color='skyblue')
    plt.xlabel('Emotion')
    plt.ylabel('Count')
    plt.title('Counts of Emotions')
    plt.show()


def show_classification_results(y_pred, y_test, label_encoder):
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    cm = confusion_matrix(y_test, y_pred)
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, cmap='Blues')
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(label_encoder.classes_)
    ax.yaxis.set_ticklabels(label_encoder.classes_)


def plot_trainig_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'][1:], label='Train Loss')
    plt.plot(history.history['val_loss'][1:], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'][1:], label='Train accuracy')
    plt.plot(history.history['val_accuracy'][1:], label='Val accuracy')
    plt.title('Model accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


def extract_melspectrogram_segments(path, sr=22050, segment_duration=5, n_mels=128, n_fft=2048, hop_length=512):
    y, sr = librosa.load(path, sr=sr)
    song_duration = librosa.get_duration(y=y, sr=sr)

    # ile próbek odpowiada segmentowi
    samples_per_segment = segment_duration * sr
    num_segments = int(song_duration // segment_duration)

    segments_specs = []

    for i in range(num_segments):
        start = i * samples_per_segment
        end = start + samples_per_segment
        segment = y[start:end]

        if len(segment) < samples_per_segment:
            continue

        mel_spec = librosa.feature.melspectrogram(y=segment,
                                                  sr=sr,
                                                  n_mels=n_mels,
                                                  n_fft=n_fft,
                                                  hop_length=hop_length)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # normalizacja (0–1)
        mel_spec_db -= mel_spec_db.min()
        mel_spec_db /= mel_spec_db.max()

        # dodanie wymiaru kanału (dla CNN: (n_mels, czas, 1))
        mel_spec_db = np.expand_dims(mel_spec_db, axis=-1)

        segments_specs.append(mel_spec_db)

    return segments_specs


def melspectrogram_to_rgb(X):
    X_rgb = []
    for mel in X:
        mel_norm = cv2.normalize(mel, None, 0, 255, cv2.NORM_MINMAX)
        mel_uint8 = mel_norm.astype(np.uint8)

        mel_resized = cv2.resize(mel_uint8, (224, 224))
        mel_rgb = cv2.cvtColor(mel_resized, cv2.COLOR_GRAY2RGB)
        X_rgb.append(mel_rgb)

    return np.array(X_rgb)


def evaluate_regression_model(y_test, emotions_test, y_pred):
    mse_valence = mean_squared_error(y_test['valence'], y_pred[:, 1])
    mse_arousal = mean_squared_error(y_test['arousal'], y_pred[:, 0])
    print("REGRESSION ERRORS")
    print("-------------------------------------")
    print(f"MSE Valence: {mse_valence:.4f}")
    print(f"MSE Arousal: {mse_arousal:.4f}")
    print("-------------------------------------")

    mae_valence = mean_absolute_error(y_test['valence'], y_pred[:, 1])
    mae_arousal = mean_absolute_error(y_test['arousal'], y_pred[:, 0])
    print(f"MAE Valence: {mae_valence:.4f}")
    print(f"MAE Arousal: {mae_arousal:.4f}")
    print("-------------------------------------")

    r2_valence = r2_score(y_test['valence'], y_pred[:, 1])
    r2_arousal = r2_score(y_test['arousal'], y_pred[:, 0])
    print(f"R² Valence: {r2_valence:.4f}")
    print(f"R² Arousal: {r2_arousal:.4f}")
    print("-------------------------------------")

    print("CLASSIFICATION ACCURACY\n")

    classification_pred = []
    classification_true = []
    for i in range(len(y_test)):
        pred_vals = y_pred[i]
        valence_mean = np.mean(pred_vals[1])
        arousal_mean = np.mean(pred_vals[0])

        classification_pred.append(classify_emotion_regression(valence_mean, arousal_mean))
        classification_true.append(emotions_test.iloc[i]["emotion"])

    print(classification_report(classification_true, classification_pred))
    conusion_matrix = confusion_matrix(classification_true, classification_pred)

    ax = plt.subplot()
    sns.heatmap(conusion_matrix, annot=True, cmap='Blues')
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(np.unique(classification_true))
    ax.yaxis.set_ticklabels(np.unique(classification_true))
