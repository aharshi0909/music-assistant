import numpy as np
import librosa

def extract_drum_segment(path, min_db=-35):
    y, sr = librosa.load(path, sr=None, mono=True)
    _, y_percussive = librosa.effects.hpss(y)
    rms = librosa.feature.rms(y=y_percussive)[0]
    frames = np.nonzero(rms > librosa.db_to_amplitude(min_db))[0]
    if frames.size == 0:
        return None, sr
    times = librosa.frames_to_samples(frames)
    start = max(0, times[0])
    end = min(len(y_percussive), times[-1] + 2048)
    return y_percussive[start:end], sr

def extract_drum_features(path):
    try:
        y, sr = extract_drum_segment(path)
        if y is None or len(y) == 0 or np.max(np.abs(y)) < 1e-4:
            return None

        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beats, sr=sr)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        beat_energy = librosa.util.sync(onset_env, beats)
        return {
            "tempo": tempo,
            "beat_times": beat_times,
            "energy_pattern": beat_energy
        }
    except Exception:
        return None

def compare_beat_structure(beat1, beat2):
    if len(beat1) == 0 or len(beat2) == 0:
        return 0.0
    len_diff = abs(len(beat1) - len(beat2)) / max(len(beat1), len(beat2))
    return max(0.0, 100 * (1 - len_diff))


def compare_energy_pattern(e1, e2):
    min_len = min(e1.shape[-1], e2.shape[-1])
    if min_len == 0:
        return 0.0
    e1 = e1[..., :min_len]
    e2 = e2[..., :min_len]
    dist = np.mean(np.abs(e1 - e2))
    return max(0.0, 100 * (1 - dist))

def compare_drum_tracks(file1, file2):
    f1 = extract_drum_features(file1)
    f2 = extract_drum_features(file2)
    if f1 is None or f2 is None:
        return {"error": "Invalid or silent drum audio"}

    beat_score = compare_beat_structure(f1['beat_times'], f2['beat_times'])
    energy_score = compare_energy_pattern(f1['energy_pattern'], f2['energy_pattern'])
    tempo_diff = abs(f1['tempo'] - f2['tempo']) / max(f1['tempo'], f2['tempo']) if max(f1['tempo'], f2['tempo']) > 0 else 1.0
    tempo_score = max(0.0, 100 * (1 - tempo_diff))

    final_score = round((energy_score * 0.5 + beat_score * 0.3 + tempo_score * 0.2), 2)

    return {
        "tempo1": round(f1['tempo'], 2),
        "tempo2": round(f2['tempo'], 2),
        "energy_similarity": round(energy_score, 2),
        "beat_similarity": round(beat_score, 2),
        "tempo_similarity": round(tempo_score, 2),
        "final_similarity": final_score
    }

result = compare_drum_tracks("drum1.wav", "drum2.wav")
print(result)
