import parselmouth
import numpy as np

def safe_shimmer_call(snd, point_process, command):
    try:
        return parselmouth.praat.call([snd, point_process], command, 0, 0, 0.0001, 0.02, 1.3, 1.6)
    except Exception:
        return 0.0

def extract_features(file_path):
    snd = parselmouth.Sound(file_path)
    pitch = snd.to_pitch()
    point_process = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 500)

    pitch_values = pitch.selected_array['frequency']
    pitch_values = pitch_values[pitch_values > 0]  # Remove unvoiced frames

    mdvp_fo = pitch_values.mean() if len(pitch_values) > 0 else 0.0
    mdvp_fhi = pitch_values.max() if len(pitch_values) > 0 else 0.0
    mdvp_flo = pitch_values.min() if len(pitch_values) > 0 else 0.0

    jitter_local = parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    jitter_local_abs = parselmouth.praat.call(point_process, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    jitter_rap = parselmouth.praat.call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    jitter_ppq5 = parselmouth.praat.call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    jitter_ddp = parselmouth.praat.call(point_process, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)

    shimmer_local = safe_shimmer_call(snd, point_process, "Get shimmer (local)")
    shimmer_local_db = safe_shimmer_call(snd, point_process, "Get shimmer (local, dB)")
    shimmer_apq3 = safe_shimmer_call(snd, point_process, "Get shimmer (apq3)")
    shimmer_apq5 = safe_shimmer_call(snd, point_process, "Get shimmer (apq5)")
    shimmer_apq = safe_shimmer_call(snd, point_process, "Get shimmer (apq)")
    shimmer_dda = safe_shimmer_call(snd, point_process, "Get shimmer (dda)")

    harmonicity = parselmouth.praat.call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr = parselmouth.praat.call(harmonicity, "Get mean", 0, 0)

    # The following features are placeholders (not computed here)
    rpde = 0.0
    dfa = 0.0
    spread1 = 0.0
    spread2 = 0.0
    d2 = 0.0
    ppe = 0.0

    # Additional placeholder features (Feature22 to Feature28)
    feature22 = 0.0
    feature23 = 0.0
    feature24 = 0.0
    feature25 = 0.0
    feature26 = 0.0
    feature27 = 0.0
    feature28 = 0.0

    features = np.array([
        mdvp_fo,
        mdvp_fhi,
        mdvp_flo,
        jitter_local,
        jitter_local_abs,
        jitter_rap,
        jitter_ppq5,
        jitter_ddp,
        shimmer_local,
        shimmer_local_db,
        shimmer_apq3,
        shimmer_apq5,
        shimmer_apq,
        shimmer_dda,
        hnr,
        rpde,
        dfa,
        spread1,
        spread2,
        d2,
        ppe,
        feature22,
        feature23,
        feature24,
        feature25,
        feature26,
        feature27,
        feature28
    ])

    return features
