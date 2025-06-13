import numpy as np
import librosa
from scipy.spatial.distance import cdist
from collections import Counter

def extract_string_segment(path, min_db=-35):
    y, sr = librosa.load(path, sr=None, mono=True)
    y_harmonic, _ = librosa.effects.hpss(y)
    rms = librosa.feature.rms(y=y_harmonic)[0]
    frames = np.nonzero(rms > librosa.db_to_amplitude(min_db))[0]
    if frames.size == 0:
        return None, sr
    times = librosa.frames_to_samples(frames)
    start = max(0, times[0])
    end = min(len(y_harmonic), times[-1] + 2048)
    return y_harmonic[start:end], sr


def extract_string_features(path):
    try:
        y, sr = extract_string_segment(path)
        if y is None or len(y) == 0 or np.max(np.abs(y)) < 1e-4:
            return None

        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beats, sr=sr)
        chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
        if chroma_cqt.shape[1] == 0:
            return None
        beat_chroma = librosa.util.sync(chroma_cqt, beats, aggregate=np.median)

        active_notes = np.argmax(beat_chroma, axis=0)
        note_counts = Counter(active_notes)

        return {
            "tempo": tempo,
            "beat_times": beat_times,
            "chroma": beat_chroma,
            "notes": set(note_counts.keys())
        }
    except Exception:
        return None


def compare_chroma(chroma1, chroma2):
    if chroma1.shape[1] == 0 or chroma2.shape[1] == 0:
        return 0.0
    min_len = min(chroma1.shape[1], chroma2.shape[1])
    chroma1 = chroma1[:, :min_len]
    chroma2 = chroma2[:, :min_len]
    dist = cdist(chroma1.T, chroma2.T, metric="cosine")
    dist = np.nan_to_num(dist, nan=1.0, posinf=1.0, neginf=1.0)
    alignment = np.mean(np.min(dist, axis=1))
    return max(0.0, 100 * (1 - alignment))


def compare_beat_structure(beat1, beat2):
    if len(beat1) == 0 or len(beat2) == 0:
        return 0.0
    len_diff = abs(len(beat1) - len(beat2)) / max(len(beat1), len(beat2))
    return max(0.0, 100 * (1 - len_diff))


def compare_notes(notes1, notes2):
    if not notes1 or not notes2:
        return 0.0
    overlap = notes1.intersection(notes2)
    mastered = len(overlap) / len(notes2) if notes2 else 0.0
    return round(100 * mastered, 2)


def compare_string_tracks(file1, file2):
    f1 = extract_string_features(file1)
    f2 = extract_string_features(file2)
    if f1 is None or f2 is None:
        return {"error": "Invalid or silent string instrument audio"}

    chroma_score = compare_chroma(f1['chroma'], f2['chroma'])
    beat_score = compare_beat_structure(f1['beat_times'], f2['beat_times'])
    tempo_diff = abs(f1['tempo'] - f2['tempo']) / max(f1['tempo'], f2['tempo']) if max(f1['tempo'], f2['tempo']) > 0 else 1.0
    tempo_score = max(0.0, 100 * (1 - tempo_diff))
    notes_mastered = compare_notes(f1['notes'], f2['notes'])

    final_score = round((chroma_score * 0.4 + beat_score * 0.2 + tempo_score * 0.2 + notes_mastered * 0.2), 2)

    return {
        "tempo1": round(f1['tempo'], 2),
        "tempo2": round(f2['tempo'], 2),
        "chroma_similarity": round(chroma_score, 2),
        "beat_similarity": round(beat_score, 2),
        "tempo_similarity": round(tempo_score, 2),
        "notes_mastered": round(notes_mastered, 2),
        "final_similarity": final_score
    }



print(result = compare_string_tracks("string1.wav", "string2.wav"))