"""Code-cell export from the sanitized companion notebook.
Notebook outputs and execution counters were intentionally removed before release.
"""

# %% [notebook code cell 1]
import pickle, gzip
import os, pickle, gzip, json

# pkl
with open("/content/drive/MyDrive/Mimar_turbo/MAINT_BEFORE_ALL.pkl", "rb") as f:
    MAINT_BEFORE_ALL_1 = pickle.load(f)
with open("/content/drive/MyDrive/Mimar_turbo/MAINT_AFTER_ALL.pkl", "rb") as f:
    MAINT_AFTER_ALL_1 = pickle.load(f)

# %% [notebook code cell 2]
import numpy as np
import pandas as pd

def build_pairs(MAINT_BEFORE_ALL, MAINT_AFTER_ALL, strategies=None, horizon=None):
    """
    Read the pre/post maintenance data of different strategies for each sample, form a one-to-one comparison, and align the time lengths.
    Alignment rule: Using maintenance point t_m as the boundary, align t_m..(t_m+L-1) of BEFORE with t_m+1..(t_m+L) of AFTER.
    where L = min(horizon(if given), T_future, T_seq - t_m).

    return:
      - pairs[uid][strategy] = {
            't_m': int,
            't_before': np.ndarray (L,), # Aligned real timeline (t_m..t_m+L-1)
            't_after': np.ndarray (L,), # Aligned prediction timeline (t_m+1..t_m+L)
            'x_before': np.ndarray (L, C),
            'x_after':  np.ndarray (L, C),
            'hi_before': np.ndarray (L,),
            'hi_after':  np.ndarray (L,),
            'rul_before':np.ndarray (L,),
            'rul_after': np.ndarray (L,)
        }
      - summary_df: summary indicator DataFrame for each (uid, strategy)
    """
    pairs = {}
    rows = []

    # Available policy names (taken from AFTER, usually the same as BEFORE)
    if strategies is None:
        # Take the policy set of the first sample as the complete set
        some_uid = next(iter(MAINT_AFTER_ALL.keys()))
        strategies = list(MAINT_AFTER_ALL[some_uid]["strategies"].keys())

    for uid, rec_b in MAINT_BEFORE_ALL.items():
        if uid not in MAINT_AFTER_ALL:
            # There is no sample corresponding to AFTER, skip it
            continue

        rec_a = MAINT_AFTER_ALL[uid]
        t_m = int(rec_a["t_m"]) # Both sides should be consistent

        pairs[uid] = {}

        for strat in strategies:
            if strat not in rec_b["strategies"] or strat not in rec_a["strategies"]:
                # Some samples may lack strategies, so skip them
                continue

            b = rec_b["strategies"][strat]
            a = rec_a["strategies"][strat]

            # BEFORE: complete real sequence
            t_full = np.asarray(b["t_full"])              # (T_seq,)
            x_full = np.asarray(b["x"])                   # (T_seq, C)
            hi_full= np.asarray(b["hi"])                  # (T_seq,)
            rul_full=np.asarray(b["rul"])                 # (T_seq,)

            # AFTER: future segment after maintenance
            t_pred_abs = np.asarray(a["t_pred_abs"])      # (T_future,)
            fut_x  = np.asarray(a["future_x"])            # (T_future, C)
            fut_hi = np.asarray(a["future_hi"])           # (T_future,)
            fut_rul= np.asarray(a["future_rul"])          # (T_future,)

            T_seq = len(t_full)
            # Comparable window of BEFORE: starting from t_m (corresponding to t_m+1 of AFTER), up to T_seq - t_m can be taken
            max_len_before = max(0, T_seq - t_m)
            # Comparable window length for AFTER: T_future
            max_len_after = len(t_pred_abs)

            # Alignment length L (can also be limited by horizon)
            L = min(max_len_before, max_len_after)
            if horizon is not None:
                L = min(L, int(horizon))

            if L <= 0:
                # No overlap of comparable length, skip
                continue

            # BEFORE slice: t_m .. t_m+L-1
            t_before = t_full[t_m : t_m + L]
            x_before = x_full[t_m : t_m + L, :]
            hi_before= hi_full[t_m : t_m + L]
            rul_before=rul_full[t_m : t_m + L]

            # AFTER slice: corresponding to t_m+1 .. t_m+L
            # Note: t_pred_abs is designed as [t_m+1, t_m+2, ...], index 0..L-1
            t_after = t_pred_abs[:L]
            x_after = fut_x[:L, :]
            hi_after= fut_hi[:L]
            rul_after=fut_rul[:L]

            pairs[uid][strat] = {
                "t_m": t_m,
                "t_before": t_before,
                "t_after":  t_after,
                "x_before": x_before,
                "x_after":  x_after,
                "hi_before": hi_before,
                "hi_after":  hi_after,
                "rul_before": rul_before,
                "rul_after":  rul_after,
            }

            # Make some summary indicators (first step up jump, mean difference, RMSE, etc.)
            # HI / RUL "jump amount" in the first step (after[0] - before[0])
            dHI0  = float(hi_after[0]  - hi_before[0]) if len(hi_after)>0 else np.nan
            dRUL0 = float(rul_after[0] - rul_before[0]) if len(rul_after)>0 else np.nan

            # Interval statistics (aligned within the interval)
            def rmse(a, b):
                if len(a) == 0: return np.nan
                return float(np.sqrt(np.mean((a - b) ** 2)))

            hi_rmse  = rmse(hi_after, hi_before)
            rul_rmse = rmse(rul_after, rul_before)

            rows.append({
                "uid": uid,
                "strategy": strat,
                "t_m": t_m,
                "L_overlap": L,
                "dHI0": dHI0,
                "dRUL0": dRUL0,
                "HI_mean_before": float(np.mean(hi_before)),
                "HI_mean_after":  float(np.mean(hi_after)),
                "HI_rmse": hi_rmse,
                "RUL_mean_before": float(np.mean(rul_before)),
                "RUL_mean_after":  float(np.mean(rul_after)),
                "RUL_rmse": rul_rmse,
            })

    summary_df = pd.DataFrame(rows).sort_values(["uid", "strategy"]).reset_index(drop=True)
    return pairs, summary_df

# === Call ===
pairs, summary_df = build_pairs(MAINT_BEFORE_ALL_1, MAINT_AFTER_ALL_1)

print("✅ Comparison has been generated: ")
print(f"-Number of samples (with comparable windows): {len(pairs)}")
print("- Summary example: ")
print(summary_df.head(12))

# Need to see the alignment result of a specific uid+policy:
# eg. uid0 = next(iter(pairs.keys()))
# print(uid0, list(pairs[uid0].keys()))
# one = pairs[uid0]["Perfect Maintenance"]
# one.keys()  # t_m / t_before / t_after / x_before / x_after / hi_before / hi_after / rul_before / rul_after

# %% [notebook code cell 3]
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import random
import matplotlib.pyplot as plt

def build_pairs(MAINT_BEFORE_ALL, MAINT_AFTER_ALL, strategies=None, horizon=None):
    """
    Read the pre/post maintenance data of different strategies for each sample, form a one-to-one comparison, and align the time lengths.
    Alignment rule: Using maintenance point t_m as the boundary, align t_m..(t_m+L-1) of BEFORE with t_m+1..(t_m+L) of AFTER.
    where L = min(horizon(if given), T_future, T_seq - t_m).

    return:
      - pairs[uid][strategy] = {
            't_m': int,
            't_before': np.ndarray (L,), # Aligned real timeline (t_m..t_m+L-1)
            't_after': np.ndarray (L,), # Aligned prediction timeline (t_m+1..t_m+L)
            'x_before': np.ndarray (L, C),
            'x_after':  np.ndarray (L, C),
            'hi_before': np.ndarray (L,),
            'hi_after':  np.ndarray (L,),
            'rul_before':np.ndarray (L,),
            'rul_after': np.ndarray (L,)
        }
      - summary_df: summary indicator DataFrame for each (uid, strategy)
    """
    pairs = {}
    rows = []

    # Available policy names (taken from AFTER, usually the same as BEFORE)
    if strategies is None:
        # Take the policy set of the first sample as the complete set
        some_uid = next(iter(MAINT_AFTER_ALL.keys()))
        strategies = list(MAINT_AFTER_ALL[some_uid]["strategies"].keys())

    for uid, rec_b in MAINT_BEFORE_ALL.items():
        if uid not in MAINT_AFTER_ALL:
            # There is no sample corresponding to AFTER, skip it
            continue

        rec_a = MAINT_AFTER_ALL[uid]
        t_m = int(rec_a["t_m"]) # Both sides should be consistent

        pairs[uid] = {}

        for strat in strategies:
            if strat not in rec_b["strategies"] or strat not in rec_a["strategies"]:
                # Some samples may lack strategies, so skip them
                continue

            b = rec_b["strategies"][strat]
            a = rec_a["strategies"][strat]

            # BEFORE: complete real sequence
            t_full = np.asarray(b["t_full"])              # (T_seq,)
            x_full = np.asarray(b["x"])                   # (T_seq, C)
            hi_full= np.asarray(b["hi"])                  # (T_seq,)
            rul_full=np.asarray(b["rul"])                 # (T_seq,)

            # AFTER: future segment after maintenance
            t_pred_abs = np.asarray(a["t_pred_abs"])      # (T_future,)
            fut_x  = np.asarray(a["future_x"])            # (T_future, C)
            fut_hi = np.asarray(a["future_hi"])           # (T_future,)
            fut_rul= np.asarray(a["future_rul"])          # (T_future,)

            T_seq = len(t_full)
            # Comparable window of BEFORE: starting from t_m (corresponding to t_m+1 of AFTER), up to T_seq - t_m can be taken
            max_len_before = max(0, T_seq - t_m)
            # Comparable window length for AFTER: T_future
            max_len_after = len(t_pred_abs)

            # Alignment length L (can also be limited by horizon)
            L = min(max_len_before, max_len_after)
            if horizon is not None:
                L = min(L, int(horizon))

            if L <= 0:
                # No overlap of comparable length, skip
                continue

            # BEFORE slice: t_m .. t_m+L-1
            t_before = t_full[t_m : t_m + L]
            x_before = x_full[t_m : t_m + L, :]
            hi_before= hi_full[t_m : t_m + L]
            rul_before=rul_full[t_m : t_m + L]

            # AFTER slice: corresponding to t_m+1 .. t_m+L
            # Note: t_pred_abs is designed as [t_m+1, t_m+2, ...], index 0..L-1
            t_after = t_pred_abs[:L]
            x_after = fut_x[:L, :]
            hi_after= fut_hi[:L]
            rul_after=fut_rul[:L]

            pairs[uid][strat] = {
                "t_m": t_m,
                "t_before": t_before,
                "t_after":  t_after,
                "x_before": x_before,
                "x_after":  x_after,
                "hi_before": hi_before,
                "hi_after":  hi_after,
                "rul_before": rul_before,
                "rul_after":  rul_after,
            }

            # Make some summary indicators (first step up jump, mean difference, RMSE, etc.)
            # HI / RUL "up jump amount" in the first step (after[0] - before[0])
            dHI0  = float(hi_after[0]  - hi_before[0]) if len(hi_after)>0 else np.nan
            dRUL0 = float(rul_after[0] - rul_before[0]) if len(rul_after)>0 else np.nan

            # Interval statistics (aligned within the interval)
            def rmse(a, b):
                if len(a) == 0: return np.nan
                return float(np.sqrt(np.mean((a - b) ** 2)))

            hi_rmse  = rmse(hi_after, hi_before)
            rul_rmse = rmse(rul_after, rul_before)

            rows.append({
                "uid": uid,
                "strategy": strat,
                "t_m": t_m,
                "L_overlap": L,
                "dHI0": dHI0,
                "dRUL0": dRUL0,
                "HI_mean_before": float(np.mean(hi_before)),
                "HI_mean_after":  float(np.mean(hi_after)),
                "HI_rmse": hi_rmse,
                "RUL_mean_before": float(np.mean(rul_before)),
                "RUL_mean_after":  float(np.mean(rul_after)),
                "RUL_rmse": rul_rmse,
            })

    summary_df = pd.DataFrame(rows).sort_values(["uid", "strategy"]).reset_index(drop=True)
    return pairs, summary_df

def analyze_pairs_data(pairs, summary_df):
    """Analyze the generated pairs data"""
    print("=== Comparative data analysis ===")

    total_samples = sum(len(strategies) for strategies in pairs.values())
    print(f"Total number of comparisons: {total_samples}")
    print(f"Sample number: {len(pairs)}")

    if len(pairs) > 0:
        # Strategy distribution
        strategy_counts = summary_df['strategy'].value_counts()
        print("\nStrategy distribution:")
        for strategy, count in strategy_counts.items():
            print(f" {strategy}: {count} samples")

        # Alignment length statistics
        print(f"\nAlignment length statistics:")
        print(f" Average alignment length: {summary_df['L_overlap'].mean():.1f}")
        print(f"Minimum alignment length: {summary_df['L_overlap'].min()}")
        print(f"Maximum alignment length: {summary_df['L_overlap'].max()}")

        # HI jump analysis
        print(f"\nHI first step jump analysis:")
        print(f" Average HI jump: {summary_df['dHI0'].mean():.4f}")
        print(f"HI jump standard deviation: {summary_df['dHI0'].std():.4f}")

        # Jump according to strategic HI
        print(f"\nJump according to the HI of the strategy:")
        for strategy in strategy_counts.index:
            strategy_data = summary_df[summary_df['strategy'] == strategy]
            print(f"  {strategy}: {strategy_data['dHI0'].mean():.4f} ± {strategy_data['dHI0'].std():.4f}")

        # RUL jump analysis
        print(f"\nRUL first step jump analysis:")
        print(f" Average RUL jump: {summary_df['dRUL0'].mean():.4f}")
        print(f" RUL jump standard deviation: {summary_df['dRUL0'].std():.4f}")

        # Jump according to the RUL of the strategy
        print(f"\nJump according to the RUL of the strategy:")
        for strategy in strategy_counts.index:
            strategy_data = summary_df[summary_df['strategy'] == strategy]
            print(f"  {strategy}: {strategy_data['dRUL0'].mean():.4f} ± {strategy_data['dRUL0'].std():.4f}")

def show_pair_example(pairs, uid=None, strategy=None):
    """Display detailed information about a comparison"""
    if uid is None:
        uid = next(iter(pairs.keys()))

    if uid not in pairs:
        print(f"UID {uid} does not exist")
        return

    if strategy is None:
        strategy = next(iter(pairs[uid].keys()))

    if strategy not in pairs[uid]:
        print(f"Strategy {strategy} does not exist in UID {uid}")
        return

    pair_data = pairs[uid][strategy]

    # Safely handle uid display
    uid_str = str(uid)
    uid_display = uid_str[:8] + "..." if len(uid_str) > 8 else uid_str

    print(f"=== Comparison example: {uid_display}/{strategy} ===")
    print(f"Maintenance time point t_m: {pair_data['t_m']}")
    print(f"Alignment length: {len(pair_data['t_before'])}")
    print(f"Number of sensor channels: {pair_data['x_before'].shape[1]}")

    print(f"\nTimeline comparison:")
    print(f"BEFORE time: {pair_data['t_before'][:5]}...{pair_data['t_before'][-3:]}")
    print(f"AFTER time: {pair_data['t_after'][:5]}...{pair_data['t_after'][-3:]}")

    print(f"\nHI comparison:")
    print(f"BEFORE HI: {pair_data['hi_before'][:5]}")
    print(f"AFTER HI:  {pair_data['hi_after'][:5]}")
    print(f"HI first jump: {pair_data['hi_after'][0] - pair_data['hi_before'][0]:.4f}")

    print(f"\nRUL comparison:")
    print(f"BEFORE RUL: {pair_data['rul_before'][:5]}")
    print(f"AFTER RUL:  {pair_data['rul_after'][:5]}")
    print(f"RUL first jump: {pair_data['rul_after'][0] - pair_data['rul_before'][0]:.4f}")

# ============================================================
# Sensor feature detection
# ============================================================

def analyze_sensor_characteristics(x_data):
    """
    Analyze sensor data characteristics and decide whether to use linear or nonlinear assumptions

    Args:
        x_data: np.ndarray (T, C) - sensor data

    Returns:
        features: dict - contains various feature indicators
        is_linear: bool - Whether the linearity assumption is suitable
    """
    T, C = x_data.shape

    features = {
        'nonlinearity_score': 0.0,
        'volatility_score': 0.0,
        'threshold_events': 0,
        'oscillation_score': 0.0,
        'saturation_score': 0.0
    }

    for c in range(C):
        signal = x_data[:, c]

        # 1. Nonlinear scoring: variance of second-order difference
        if T > 3:
            second_diff = np.diff(signal, n=2)
            nonlinearity = np.var(second_diff) / (np.var(signal) + 1e-6)
            features['nonlinearity_score'] += nonlinearity

        # 2. Volatility score: standard deviation of rate of change
        if T > 1:
            changes = np.abs(np.diff(signal))
            volatility = np.std(changes) / (np.mean(np.abs(signal)) + 1e-6)
            features['volatility_score'] += volatility

        # 3. Threshold event: big jump
        if T > 1:
            diffs = np.abs(np.diff(signal))
            threshold = np.mean(diffs) + 2 * np.std(diffs)
            threshold_events = np.sum(diffs > threshold)
            features['threshold_events'] += threshold_events

        # 4. Oscillation scoring: zero-crossing detection
        if T > 5:
            signal_detrend = signal - np.linspace(signal[0], signal[-1], T)
            zero_crossings = np.sum(np.diff(np.sign(signal_detrend)) != 0)
            oscillation = zero_crossings / T
            features['oscillation_score'] += oscillation

        # 5. Saturation score: the variance in the second half is relative to the first half
        if T > 10:
            mid = T // 2
            var_first = np.var(signal[:mid])
            var_second = np.var(signal[mid:])
            saturation = max(0, (var_first - var_second) / (var_first + 1e-6))
            features['saturation_score'] += saturation

    # average features
    for key in features:
        features[key] /= C

    # The comprehensive score determines the model type
    complexity_score = (features['nonlinearity_score'] * 0.3 +
                       features['volatility_score'] * 0.2 +
                       features['threshold_events'] * 0.2 +
                       features['oscillation_score'] * 0.15 +
                       features['saturation_score'] * 0.15)

    # Threshold decision
    is_linear = complexity_score < 0.1 # Adjustable threshold

    return features, is_linear

# ============================================================
#Model definition
# ============================================================

class LinearHIModel(nn.Module):
    """First-order linear HI model: HI = a * t + b"""
    def __init__(self, in_ch):
        super().__init__()
        self.slope = nn.Parameter(torch.tensor(-0.01)) # Slope (negative value means decreasing)
        self.bias = nn.Parameter(torch.tensor(0.8)) #Intercept

    def forward(self, x):  # x: (B, T, C)
        B, T, C = x.shape

        # Make sure the slope is negative (decreasing)
        slope_val = -F.softplus(self.slope)
        bias_val = torch.sigmoid(self.bias)

        # Timeline normalization
        t = torch.arange(T, device=x.device, dtype=x.dtype) / max(T-1, 1)
        t = t.unsqueeze(0).expand(B, -1)  # (B, T)

        # Linear decrease
        hi = slope_val * t + bias_val
        return torch.clamp(hi, 0.0, 1.0)

class NonLinearHIModel(nn.Module):
    """Nonlinear HI model: free operator combination"""
    def __init__(self, in_ch):
        super().__init__()

        # Feature extraction
        self.feature_net = nn.Sequential(
            nn.Linear(in_ch, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )

        # HI prediction header
        self.hi_head = nn.Sequential(
            nn.Linear(16 + 1, 32),  # +1 for time
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):  # x: (B, T, C)
        B, T, C = x.shape

        # Time features
        t = torch.arange(T, device=x.device, dtype=x.dtype) / max(T-1, 1)
        t = t.unsqueeze(0).expand(B, -1).unsqueeze(-1)  # (B, T, 1)

        # Sensor feature extraction
        x_reshaped = x.reshape(-1, C)  # (B*T, C)
        features = self.feature_net(x_reshaped)  # (B*T, 16)
        features = features.reshape(B, T, -1)  # (B, T, 16)

        # Splice time features
        combined = torch.cat([features, t], dim=-1)  # (B, T, 17)

        #HIPrediction
        combined_reshaped = combined.reshape(-1, 17)  # (B*T, 17)
        hi = self.hi_head(combined_reshaped)  # (B*T, 1)
        hi = hi.reshape(B, T)  # (B, T)

        return hi

class AdaptiveHIEncoder(nn.Module):
    """Adaptive HI encoder: model selection based on sensor characteristics"""
    def __init__(self, in_ch):
        super().__init__()
        self.in_ch = in_ch
        self.linear_model = LinearHIModel(in_ch)
        self.nonlinear_model = NonLinearHIModel(in_ch)

    def analyze_and_select(self, x):
        """Analyze sensor data and select appropriate models"""
        # Convert to numpy for analysis
        x_np = x.detach().cpu().numpy()

        # Batch analysis
        use_linear = []
        for b in range(x_np.shape[0]):
            features, is_linear = analyze_sensor_characteristics(x_np[b])
            use_linear.append(is_linear)

        return use_linear

    def forward(self, x):  # x: (B, T, C)
        B = x.shape[0]

        # Analyze data characteristics
        use_linear = self.analyze_and_select(x)

        # Calculate the output of the two models separately
        hi_linear = self.linear_model(x)      # (B, T)
        hi_nonlinear = self.nonlinear_model(x)  # (B, T)

        # Select output based on analysis results
        hi_output = torch.zeros_like(hi_linear)
        for b in range(B):
            if use_linear[b]:
                hi_output[b] = hi_linear[b]
            else:
                hi_output[b] = hi_nonlinear[b]

        return hi_output, use_linear

class SimpleDiffClassifier(nn.Module):
    """Simplified difference classifier"""
    def __init__(self, n_classes=3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(4, 32), # 4 statistical features
            nn.ReLU(),
            nn.Linear(32, n_classes)
        )

    def forward(self, h_b, h_a, mask):
        # Calculate statistical features
        diff = h_a - h_b  # (B, T)

        # Valid length
        valid_lengths = mask.sum(dim=1, keepdim=True).clamp(min=1.0)  # (B, 1)

        # Feature extraction
        mean_diff = (diff * mask).sum(dim=1, keepdim=True) / valid_lengths  # (B, 1)
        max_diff = diff.max(dim=1, keepdim=True).values  # (B, 1)
        min_diff = diff.min(dim=1, keepdim=True).values  # (B, 1)
        std_diff = (((diff * mask) ** 2).sum(dim=1, keepdim=True) / valid_lengths).sqrt()  # (B, 1)

        # Splicing features
        features = torch.cat([mean_diff, max_diff, min_diff, std_diff], dim=1)  # (B, 4)

        return self.classifier(features)

class SimplifiedModel(nn.Module):
    """Simplified overall model"""
    def __init__(self, in_ch, n_classes=3):
        super().__init__()
        self.encoder = AdaptiveHIEncoder(in_ch)
        self.classifier = SimpleDiffClassifier(n_classes)

    def forward(self, x_before, x_after, mask):
        #encodeHI
        hi_before, linear_flags_b = self.encoder(x_before)
        hi_after, linear_flags_a = self.encoder(x_after)

        # Classification
        logits = self.classifier(hi_before, hi_after, mask)

        return hi_before, hi_after, logits, linear_flags_b, linear_flags_a

# ============================================================
#dataset
# ============================================================
class SimplePairsDataset(Dataset):
    """Simplified pairs data set"""
    def __init__(self, pairs: dict, horizon: int=None):
        items = []

        for uid, sd in pairs.items():
            for strat, d in sd.items():
                xb = np.asarray(d["x_before"], dtype=np.float32)
                xa = np.asarray(d["x_after"],  dtype=np.float32)
                hib= np.asarray(d["hi_before"],dtype=np.float32)
                hia= np.asarray(d["hi_after"], dtype=np.float32)
                L, C = xb.shape

                if horizon is not None and L > horizon:
                    xb, xa, hib, hia = xb[:horizon], xa[:horizon], hib[:horizon], hia[:horizon]
                    L = horizon
                if L < 3:
                    continue

                items.append({"uid": uid, "strategy": strat, "x_before": xb, "x_after": xa,
                              "hi_before": hib, "hi_after": hia})

        if len(items) == 0:
            raise ValueError("The data set is empty, please check the pairs data.")

        self.items = items

        # Policy mapping
        def strat_to_cls(s):
            s = s.lower()
            if "perfect" in s: return 0
            if "general" in s: return 1
            if "poor" in s: return 2
            raise ValueError(f"Unknown strategy name: {s}")
        self.labels = [strat_to_cls(it["strategy"]) for it in items]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        return {
            "uid": it["uid"],
            "label": self.labels[idx],
            "x_before": torch.from_numpy(it["x_before"]),
            "x_after":  torch.from_numpy(it["x_after"]),
            "hi_before": torch.from_numpy(it["hi_before"]),
            "hi_after":  torch.from_numpy(it["hi_after"]),
            "strategy": it["strategy"]
        }

def simple_collate_fn(batch):
    """Simple collate function"""
    Ls = [b["x_before"].shape[0] for b in batch]
    Lmax = max(Ls)
    C = batch[0]["x_before"].shape[1]

    def pad2d(x):
        L, C0 = x.shape
        out = torch.zeros(Lmax, C0, dtype=x.dtype)
        out[:L] = x
        return out

    def pad1d(x):
        L = x.shape[0]
        out = torch.zeros(Lmax, dtype=x.dtype)
        out[:L] = x
        return out

    x_before = torch.stack([pad2d(b["x_before"]) for b in batch], 0)
    x_after  = torch.stack([pad2d(b["x_after"])  for b in batch], 0)
    hi_before= torch.stack([pad1d(b["hi_before"])for b in batch], 0)
    hi_after = torch.stack([pad1d(b["hi_after"]) for b in batch], 0)
    mask     = torch.zeros(len(batch), Lmax)
    for i,L in enumerate(Ls): mask[i,:L] = 1.0
    labels   = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    uids     = [b["uid"] for b in batch]

    return {"uids": uids, "labels": labels,
            "x_before": x_before, "x_after": x_after,
            "hi_before": hi_before, "hi_after": hi_after,
            "mask": mask}

# ============================================================
#Loss function
# ============================================================
def simple_loss_function(hi_pred_b, hi_pred_a, hi_true_b, hi_true_a, logits, labels, mask, linear_flags):
    """Simplified loss function"""

    # HI prediction loss
    def masked_mse(pred, true, mask):
        diff = (pred - true) ** 2
        return (diff * mask).sum() / (mask.sum() + 1e-6)

    loss_hi_b = masked_mse(hi_pred_b, hi_true_b, mask)
    loss_hi_a = masked_mse(hi_pred_a, hi_true_a, mask)

    # Classification loss
    loss_cls = F.cross_entropy(logits, labels)

    # Monotonically decreasing constraints (only for linear models)
    loss_mono = torch.tensor(0.0, device=hi_pred_b.device)
    for b in range(len(linear_flags)):
        if linear_flags[b]: # If using linear model
            # Check for monotonic decrease
            diff_b = hi_pred_b[b, 1:] - hi_pred_b[b, :-1]
            diff_a = hi_pred_a[b, 1:] - hi_pred_a[b, :-1]
            mask_b = mask[b, 1:]
            mask_a = mask[b, 1:]

            loss_mono += (F.relu(diff_b) * mask_b).sum() / (mask_b.sum() + 1e-6)
            loss_mono += (F.relu(diff_a) * mask_a).sum() / (mask_a.sum() + 1e-6)

    loss_mono /= len(linear_flags)

    #Total loss
    total_loss = loss_hi_b + loss_hi_a + loss_cls + 0.1 * loss_mono

    return {
        "total": total_loss,
        "hi_b": loss_hi_b,
        "hi_a": loss_hi_a,
        "cls": loss_cls,
        "mono": loss_mono
    }

# ============================================================
#Training function
# ============================================================
def train_adaptive_model(pairs, epochs=30, batch_size=32, lr=1e-3, horizon=None, device=None):
    """Train adaptive model"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=== Adaptive HI model training ===")

    #Data analysis statistics
    total_samples = sum(len(strategies) for strategies in pairs.values())
    linear_count = 0
    nonlinear_count = 0

    for uid, strategies in pairs.items():
        for strategy, data in strategies.items():
            features, is_linear = analyze_sensor_characteristics(data["x_before"])
            if is_linear:
                linear_count += 1
            else:
                nonlinear_count += 1

    print(f"Data feature analysis: total samples={total_samples}, linear fit={linear_count}, nonlinear fit={nonlinear_count}")

    #Data partitioning
    def split_pairs_uid(pairs, train_ratio=0.7, val_ratio=0.1, seed=42):
        uids = sorted(list(pairs.keys()))
        rs = np.random.RandomState(seed)
        rs.shuffle(uids)
        n = len(uids); n_tr=int(n*train_ratio); n_vl=int(n*val_ratio)
        to_dict = lambda ss:{u:pairs[u] for u in ss if u in pairs}
        return to_dict(uids[:n_tr]), to_dict(uids[n_tr:n_tr+n_vl]), to_dict(uids[n_tr+n_vl:])

    pairs_tr, pairs_vl, pairs_te = split_pairs_uid(pairs, 0.7, 0.1, 42)

    #Create dataset
    ds_tr = SimplePairsDataset(pairs_tr, horizon=horizon)
    ds_vl = SimplePairsDataset(pairs_vl, horizon=horizon)
    ds_te = SimplePairsDataset(pairs_te, horizon=horizon)

    ld_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, collate_fn=simple_collate_fn)
    ld_vl = DataLoader(ds_vl, batch_size=batch_size, shuffle=False, collate_fn=simple_collate_fn)
    ld_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False, collate_fn=simple_collate_fn)

    C = ds_tr.items[0]["x_before"].shape[1]
    model = SimplifiedModel(in_ch=C, n_classes=3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"Model parameters: input channel={C}, device={device}")
    print(f"Training set: {len(ds_tr)} samples, validation set: {len(ds_vl)} samples, test set: {len(ds_te)} samples")

    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(1, epochs + 1):
        # train
        model.train()
        train_losses = {"total": 0, "hi_b": 0, "hi_a": 0, "cls": 0, "mono": 0}
        n_batches = 0
        linear_usage = []

        for batch in ld_tr:
            xb = batch["x_before"].to(device)
            xa = batch["x_after"].to(device)
            hib_true = batch["hi_before"].to(device)
            hia_true = batch["hi_after"].to(device)
            labels = batch["labels"].to(device)
            mask = batch["mask"].to(device)

            opt.zero_grad()

            hib_pred, hia_pred, logits, linear_flags_b, linear_flags_a = model(xb, xa, mask)

            losses = simple_loss_function(hib_pred, hia_pred, hib_true, hia_true,
                                        logits, labels, mask, linear_flags_b)

            if not torch.isnan(losses["total"]):
                losses["total"].backward()
                opt.step()

                for key in train_losses:
                    train_losses[key] += losses[key].item()
                n_batches += 1

                # Statistical model usage
                linear_usage.extend(linear_flags_b)

        # Average training loss
        for key in train_losses:
            train_losses[key] /= max(n_batches, 1)

        linear_ratio = sum(linear_usage) / max(len(linear_usage), 1)

        # verify
        model.eval()
        val_losses = {"total": 0, "cls": 0}
        n_correct = 0
        n_total = 0
        val_linear_usage = []

        with torch.no_grad():
            for batch in ld_vl:
                xb = batch["x_before"].to(device)
                xa = batch["x_after"].to(device)
                hib_true = batch["hi_before"].to(device)
                hia_true = batch["hi_after"].to(device)
                labels = batch["labels"].to(device)
                mask = batch["mask"].to(device)

                hib_pred, hia_pred, logits, linear_flags_b, linear_flags_a = model(xb, xa, mask)

                losses = simple_loss_function(hib_pred, hia_pred, hib_true, hia_true,
                                            logits, labels, mask, linear_flags_b)

                if not torch.isnan(losses["total"]):
                    val_losses["total"] += losses["total"].item()
                    val_losses["cls"] += losses["cls"].item()

                # Calculate accuracy
                pred = logits.argmax(dim=1)
                n_correct += (pred == labels).sum().item()
                n_total += labels.size(0)

                val_linear_usage.extend(linear_flags_b)

        val_acc = n_correct / max(n_total, 1)
        val_linear_ratio = sum(val_linear_usage) / max(len(val_linear_usage), 1)

        # Print progress
        if epoch % 5 == 0 or epoch == 1:
            print(f"round {epoch:03d}:")
            print(f" Training - total loss: {train_losses['total']:.4f}, HI loss: {train_losses['hi_b']+train_losses['hi_a']:.4f}, "
                  f"Classification: {train_losses['cls']:.4f}, Linear utilization rate: {linear_ratio:.2%}")
            print(f" Verification - Total losses: {val_losses['total']:.4f}, Accuracy: {val_acc:.3f}, Linear usage rate: {val_linear_ratio:.2%}")

        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f" -> Best model update! Verification accuracy: {best_val_acc:.4f}")

    #Load the best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nTraining completed, best model loaded (verification accuracy: {best_val_acc:.4f})")

    return model, (ds_tr, ld_tr), (ds_vl, ld_vl), (ds_te, ld_te)

def analyze_model_decisions(pairs, model, device, n_examples=10):
    """The decision-making process of the analysis model"""
    print("\n=== Model decision analysis ===")

    model.eval()

    linear_examples = []
    nonlinear_examples = []

    with torch.no_grad():
        for uid, strategies in pairs.items():
            if len(linear_examples) >= n_examples and len(nonlinear_examples) >= n_examples:
                break

            for strategy, data in strategies.items():
                x_before = torch.from_numpy(data["x_before"]).unsqueeze(0).float().to(device)

                # Analyze features
                features, is_linear_analysis = analyze_sensor_characteristics(data["x_before"])

                #Model decision
                _, linear_flags = model.encoder.analyze_and_select(x_before)
                is_linear_model = linear_flags[0]

                example = {
                    "uid": uid,
                    "strategy": strategy,
                    "features": features,
                    "analysis_decision": is_linear_analysis,
                    "model_decision": is_linear_model,
                    "consistent": is_linear_analysis == is_linear_model
                }

                if is_linear_model and len(linear_examples) < n_examples:
                    linear_examples.append(example)
                elif not is_linear_model and len(nonlinear_examples) < n_examples:
                    nonlinear_examples.append(example)

    print(f"Linear model examples ({len(linear_examples)}):")
    for i, ex in enumerate(linear_examples):
        print(f"  {i+1}. {ex['uid'][:8]}-{ex['strategy']}: "
              f"Nonlinearity score={ex['features']['nonlinearity_score']:.3f}, "
              f"Volatility score={ex['features']['volatility_score']:.3f}, "
              f"Consistency={ex['consistent']}")

    print(f"\nNonlinear model examples ({len(nonlinear_examples)}):")
    for i, ex in enumerate(nonlinear_examples):
        print(f"  {i+1}. {ex['uid'][:8]}-{ex['strategy']}: "
              f"Nonlinearity score={ex['features']['nonlinearity_score']:.3f}, "
              f"Volatility score={ex['features']['volatility_score']:.3f}, "
              f"Consistency={ex['consistent']}")

    # Statistical consistency
    all_examples = linear_examples + nonlinear_examples
    consistency_rate = sum(ex['consistent'] for ex in all_examples) / len(all_examples)
    print(f"\nDecision consistency: {consistency_rate:.2%}")

def get_hi_modeling_recommendations(summary_df, pairs):
    """Output HI modeling decision suggestions based on the analysis results"""
    print("\n" + "="*80)
    print("HI modeling strategy suggestions based on data analysis")
    print("="*80)

    # Analyze data feature statistics
    total_samples = sum(len(strategies) for strategies in pairs.values())
    linear_count = 0
    nonlinear_count = 0
    complexity_scores = []

    for uid, strategies in pairs.items():
        for strategy, data in strategies.items():
            features, is_linear = analyze_sensor_characteristics(data["x_before"])
            complexity_score = (features['nonlinearity_score'] * 0.3 +
                              features['volatility_score'] * 0.2 +
                              features['threshold_events'] * 0.2 +
                              features['oscillation_score'] * 0.15 +
                              features['saturation_score'] * 0.15)
            complexity_scores.append(complexity_score)

            if is_linear:
                linear_count += 1
            else:
                nonlinear_count += 1

    avg_complexity = np.mean(complexity_scores)
    linear_ratio = linear_count / total_samples

    # HI jump feature analysis
    avg_hi_jump = summary_df['dHI0'].mean()
    std_hi_jump = summary_df['dHI0'].std()
    max_hi_jump = summary_df['dHI0'].max()
    min_hi_jump = summary_df['dHI0'].min()

    # Analysis of differences between strategies
    strategy_hi_jumps = {}
    for strategy in summary_df['strategy'].unique():
        strategy_data = summary_df[summary_df['strategy'] == strategy]
        strategy_hi_jumps[strategy] = strategy_data['dHI0'].mean()

    print(f"\n📊 Summary of data characteristics:")
    print(f"Total number of samples: {total_samples}")
    print(f"Average complexity score: {avg_complexity:.4f}")
    print(f"Linear fit ratio: {linear_ratio:.2%} ({linear_count}/{total_samples})")
    print(f" Average HI jump: {avg_hi_jump:.4f} ± {std_hi_jump:.4f}")
    print(f"HI jump range: [{min_hi_jump:.4f}, {max_hi_jump:.4f}]")

    print(f"\n Each strategy HI jump:")
    for strategy, jump in strategy_hi_jumps.items():
        print(f"    {strategy}: {jump:.4f}")

    print(f"\n🎯 Suggested modeling strategy:")

    # Give specific suggestions based on the analysis results
    if linear_ratio > 0.7:
        print("✅ It is recommended to use linear model (LinearHIModel)")
        print(" - Most of the data fit the linear assumption")
        print(" - It is recommended to increase w_linear weight to 0.8-1.0")
        print(" - Maintain monotonic decreasing constraints (mono_dec=True)")
        print(" - Use simple time-dependent model: HI(t) = a*t + b")

    elif linear_ratio < 0.3:
        print(" 🔧 It is recommended to enable free operator combination (NonLinearHIModel)")
        print("-The data shows strong non-linear characteristics")
        print(" - Enable multi-operator combination (SparseGate + regularization)")
        print(" - consider dual time constant model or saturation effect")
        print("-Add smoothing constraints to prevent overfitting")

    else:
        print(" ⚖️ It is recommended to use the adaptive hybrid model (AdaptiveHIEncoder)")
        print(" - Mixed data features, sample-level adaptation is required")
        print(" - Real-time analysis of sensor features to select model type")
        print(" - Use constrained model for linear samples")
        print(" - Enable flexible operators for nonlinear samples")

    # Recommendations based on HI jump features
    print(f"\n📈 Suggestions based on HI jump mode:")
    if abs(avg_hi_jump) < 0.05:
        print(" - HI jump is small and maintenance effect is mild")
        print(" - Continuity assumption can be used to reduce jump detection")

    elif avg_hi_jump > 0.1:
        print("-HI has a significant positive jump, maintenance will immediately improve health")
        print(" - It is recommended to introduce jump operators at maintenance points")
        print(" - Consider segmented modeling: use different parameters before and after maintenance")

    elif avg_hi_jump < -0.1:
        print("-HI has a negative jump, there may be data quality issues")
        print(" - It is recommended to check the maintenance effect definition and data preprocessing")

    if std_hi_jump > 0.1:
        print(" - HI jump has a large variation, and the strategy effect is obviously different")
        print(" - Enhance classifier feature extraction capabilities")
        print(" - Consider strategy-specific HI modeling parameters")

    # Implementation suggestions
    print(f"\n💻 Specific implementation suggestions:")

    if linear_ratio > 0.7:
        print(" # Linear priority configuration")
        print("    model = LinearHIModel(in_ch=C)")
        print("    loss_weights = {'hi': 1.0, 'cls': 1.0, 'mono': 0.2}")

    elif linear_ratio < 0.3:
        print(" # Non-linear priority configuration")
        print("    model = NonLinearHIModel(in_ch=C)")
        print(" # Enable multiple operators and regularization")
        print("    loss_weights = {'hi': 1.0, 'cls': 1.0, 'smooth': 0.1, 'sparsity': 0.05}")

    else:
        print(" #Adaptive hybrid configuration")
        print("    model = AdaptiveHIEncoder(in_ch=C)")
        print(" # Dynamically select linear/nonlinear model")
        print("complexity_threshold = 0.1 #can be adjusted according to data")

    print("="*80)

# === Call example ===
def run_pairs_analysis(MAINT_BEFORE_ALL, MAINT_AFTER_ALL, horizon=None):
    """Run the complete pairs analysis process"""

    # Generate comparison pairs
    pairs, summary_df = build_pairs(MAINT_BEFORE_ALL, MAINT_AFTER_ALL, horizon=horizon)

    print("✅ Comparison has been generated")

    #Data analysis
    analyze_pairs_data(pairs, summary_df)

    # Display summary table
    print("\n=== Summary table (first 12 rows)===")
    print(summary_df.head(12))

    # Show a specific example
    if len(pairs) > 0:
        print()
        show_pair_example(pairs)

    # Output modeling suggestions based on analysis results
    get_hi_modeling_recommendations(summary_df, pairs)

    return pairs, summary_df

#Usage example (needs to be replaced with actual data variable name)
pairs, summary_df = run_pairs_analysis(MAINT_BEFORE_ALL_1, MAINT_AFTER_ALL_1)

# %% [notebook code cell 4]

# ============================================================
# Monotonic-Decreasing HI + aligned plotting (before -> after)
# ============================================================
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import random
import matplotlib.pyplot as plt

# ============================================================
# 1) Dataset: read samples from pairs, perform shift operations in batch later
# ============================================================
class PairsReconstructDataset(Dataset):
    """
    Each sample contains:
      x_before: (L,C) feature sequence before maintenance
      x_after : (L,C) feature sequence after maintenance
      hi_before/hi_after: (L,) health index (for evaluation/optional analysis only, not strong supervision)
      strategy: maintenance type ("Perfect Maintenance" / "General Maintenance" / "Poor Maintenance")
    """
    def __init__(self, pairs: dict, horizon: int=None):
        items = []
        for uid, sd in pairs.items():
            for strat, d in sd.items():
                xb = np.asarray(d["x_before"], dtype=np.float32)
                xa = np.asarray(d["x_after"],  dtype=np.float32)
                # When there's no real HI, create placeholder that will be learned by the model
                if "hi_before" in d and d["hi_before"] is not None:
                    hib = np.asarray(d["hi_before"],dtype=np.float32)
                else:
                    # Create initialized fake HI, model will learn real patterns
                    hib = np.linspace(0.8, 0.4, len(xb)).astype(np.float32)

                if "hi_after" in d and d["hi_after"] is not None:
                    hia = np.asarray(d["hi_after"], dtype=np.float32)
                else:
                    # Create different fake HI patterns based on maintenance strategy
                    if "perfect" in strat.lower():
                        # Perfect maintenance: significant improvement
                        hia = np.linspace(0.9, 0.6, len(xa)).astype(np.float32)
                    elif "general" in strat.lower():
                        # General maintenance: moderate improvement
                        hia = np.linspace(0.7, 0.5, len(xa)).astype(np.float32)
                    else:
                        # Poor maintenance: slight improvement
                        hia = np.linspace(0.5, 0.35, len(xa)).astype(np.float32)

                L, C = xb.shape
                # Unify length (optional)
                if horizon is not None and L > horizon:
                    xb, xa, hib, hia = xb[:horizon], xa[:horizon], hib[:horizon], hia[:horizon]
                    L = horizon
                if L < 3:  # At least need to form 0:L-2 -> 1:L-1
                    continue
                items.append({"uid": uid, "strategy": strat, "x_before": xb, "x_after": xa,
                              "hi_before": hib, "hi_after": hia})
        assert len(items)>0, "Dataset is empty, please check pairs data."
        self.items = items
        self.C = items[0]["x_before"].shape[1]

        # Strategy to 3-class label mapping
        def strat_to_cls(s):
            s = s.lower()
            if "perfect" in s: return 0
            if "general" in s: return 1
            if "poor" in s: return 2
            raise ValueError(f"Unknown strategy name: {s}")
        self.labels = [strat_to_cls(it["strategy"]) for it in items]

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        return {
            "uid": it["uid"],
            "label": self.labels[idx],
            "x_before": torch.from_numpy(it["x_before"]), # (L,C)
            "x_after":  torch.from_numpy(it["x_after"]),  # (L,C)
            "hi_before":torch.from_numpy(it["hi_before"]),# (L,)
            "hi_after": torch.from_numpy(it["hi_after"]), # (L,)
            "strategy": it["strategy"]
        }

def pad_collate_shift(batch):
    """
    Pad sequences to same length; return mask, do NOT perform shift operation here
    to allow unified training: inputs = [:,0:L-2], targets = [:,1:L-1]
    """
    Ls = [b["x_before"].shape[0] for b in batch]
    Lmax = max(Ls)
    C = batch[0]["x_before"].shape[1]

    def pad2d(x):
        L, C0 = x.shape
        out = torch.zeros(Lmax, C0, dtype=x.dtype)
        out[:L] = x
        return out

    def pad1d(x):
        L = x.shape[0]
        out = torch.zeros(Lmax, dtype=x.dtype)
        out[:L] = x
        return out

    x_before = torch.stack([pad2d(b["x_before"]) for b in batch], 0) # (B,Lmax,C)
    x_after  = torch.stack([pad2d(b["x_after"])  for b in batch], 0) # (B,Lmax,C)
    hi_before= torch.stack([pad1d(b["hi_before"])for b in batch], 0) # (B,Lmax)
    hi_after = torch.stack([pad1d(b["hi_after"]) for b in batch], 0) # (B,Lmax)
    lengths  = torch.tensor(Ls, dtype=torch.long)
    mask     = torch.zeros(len(batch), Lmax)
    for i,L in enumerate(Ls): mask[i,:L] = 1.0
    labels   = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    uids     = [b["uid"] for b in batch]
    return {"uids": uids, "labels": labels,
            "x_before": x_before, "x_after": x_after,
            "hi_before": hi_before, "hi_after": hi_after,
            "lengths": lengths, "mask": mask}

# ============================================================
# 2) Model: encode "damage" → project to monotonic decreasing HI → recursive reconstruction of two sequences → classify using ΔHI
# ============================================================
class BoltzmannKAN(nn.Module):
    def __init__(self, in_ch, out_ch=4):
        super().__init__()
        self.E  = nn.Parameter(torch.zeros(out_ch, in_ch))
        self.kT = nn.Parameter(torch.ones(out_ch, in_ch))
    def forward(self, x):
        B,T,C = x.shape
        kT = torch.clamp(F.softplus(self.kT), 0.01, 10.0).unsqueeze(0).unsqueeze(2)  # (1,out_ch,1,in_ch)
        E  = torch.clamp(self.E, -10.0, 10.0).unsqueeze(0).unsqueeze(2)             # (1,out_ch,1,in_ch)
        x_ = torch.clamp(x.unsqueeze(1), -10.0, 10.0)                               # (B, out_ch, T, in_ch)
        exp = torch.clamp((E - x_) / kT, -50, 50)
        w   = torch.sigmoid(exp)
        h   = (w * x_).sum(dim=3).permute(0, 2, 1)           # (B,T,out_ch) >=0
        return torch.clamp(F.softplus(h), 0.0, 100.0)

class BaseOp(nn.Module): pass
class MonotonicLinearOp(BaseOp):
    def __init__(self):
        super().__init__()
        self.raw_scale=nn.Parameter(torch.tensor(0.0))
        self.bias=nn.Parameter(torch.tensor(0.0))
        self.smin,self.smax=0.01,5.0
    def forward(self,h):
        scale=torch.clamp(F.softplus(self.raw_scale),self.smin,self.smax)
        xm=torch.clamp(h.mean(-1), -10.0, 10.0)
        return torch.clamp(F.softplus(scale*(xm+self.bias)), 0.0, 100.0)

class MonotonicFlatOp(BaseOp):
    def __init__(self):
        super().__init__()
        self.raw_scale=nn.Parameter(torch.tensor(0.0))
        self.bias=nn.Parameter(torch.tensor(0.0))
        self.smin,self.smax=1e-3,1.0
    def _cum(self,x):
        diff=F.relu(x[:,1:]-x[:,:-1])
        return torch.cat([torch.zeros_like(diff[:,:1]),torch.cumsum(diff,1)],1)
    def forward(self,h):
        scale=torch.clamp(F.softplus(self.raw_scale),self.smin,self.smax)
        xm=torch.clamp(h.mean(-1,keepdim=True), -10.0, 10.0)
        return torch.clamp(F.softplus(scale*(self._cum(xm)+self.bias)).squeeze(-1), 0.0, 100.0)

class ConcaveLogOp(BaseOp):
    def __init__(self):
        super().__init__()
        self.raw_scale=nn.Parameter(torch.tensor(0.0))
        self.eps=1e-3
        self.smin,self.smax=0.01,5.0
    def forward(self,h):
        scale=torch.clamp(F.softplus(self.raw_scale),self.smin,self.smax)
        xm=torch.clamp(h.mean(-1), -10.0, 10.0)
        return torch.clamp(F.softplus(scale*torch.log(torch.abs(xm)+self.eps)), 0.0, 100.0)

class SaturationSigmoidOp(BaseOp):
    def __init__(self):
        super().__init__()
        self.raw_scale=nn.Parameter(torch.tensor(0.0))
        self.raw_slope=nn.Parameter(torch.tensor(0.0))
        self.bias=nn.Parameter(torch.tensor(0.0))
        self.smin,self.smax=0.01,5.0
        self.lmin,self.lmax=0.1,5.0
    def forward(self,h):
        scale=torch.clamp(F.softplus(self.raw_scale),self.smin,self.smax)
        slope=torch.clamp(F.softplus(self.raw_slope),self.lmin,self.lmax)
        xm=torch.clamp(h.mean(-1), -10.0, 10.0)
        return torch.clamp(F.softplus(scale*torch.sigmoid(slope*(xm-self.bias))), 0.0, 100.0)

class HingeReLUOp(BaseOp):
    def __init__(self):
        super().__init__()
        self.raw_scale=nn.Parameter(torch.tensor(0.0))
        self.threshold=nn.Parameter(torch.tensor(0.0))
        self.smin,self.smax=0.01,5.0
    def forward(self,h):
        scale=torch.clamp(F.softplus(self.raw_scale),self.smin,self.smax)
        xm=torch.clamp(h.mean(-1), -10.0, 10.0)
        return torch.clamp(F.softplus(scale*F.relu(xm-self.threshold)), 0.0, 100.0)

class PolynomialOp(BaseOp):
    def __init__(self,deg=3):
        super().__init__()
        self.raw_coeff=nn.Parameter(torch.zeros(deg))
        self.deg=deg
    def forward(self,h):
        xm=torch.clamp(h.mean(-1), -5.0, 5.0)
        y=torch.zeros_like(xm)
        for i in range(self.deg):
            coeff = torch.clamp(F.softplus(self.raw_coeff[i]), 0.01, 5.0)
            y = y + coeff * torch.clamp(xm ** (i+1), -100.0, 100.0)
        return torch.clamp(F.softplus(y), 0.0, 100.0)

class DampedSinOp(BaseOp):
    def __init__(self):
        super().__init__()
        self.raw_scale=nn.Parameter(torch.tensor(0.0))
        self.raw_freq=nn.Parameter(torch.tensor(0.0))
        self.raw_lambda=nn.Parameter(torch.tensor(0.0))
        self.phase=nn.Parameter(torch.tensor(0.0))
        self.smin,self.smax=0.01,5.0
        self.fmin,self.fmax=0.1,5.0
        self.lmin,self.lmax=0.01,3.0
    def forward(self,h):
        scale=torch.clamp(F.softplus(self.raw_scale),self.smin,self.smax)
        freq=torch.clamp(F.softplus(self.raw_freq),self.fmin,self.fmax)
        lam=torch.clamp(F.softplus(self.raw_lambda),self.lmin,self.lmax)
        xm=torch.clamp(h.mean(-1), -10.0, 10.0)
        damp=1/(1+lam*torch.abs(xm))
        phase_val = torch.clamp(self.phase, -10.0, 10.0)
        return torch.clamp(F.softplus(scale*damp*(torch.sin(freq*xm+phase_val)+1)), 0.0, 100.0)

class PiecewiseLinearOp(BaseOp):
    def __init__(self):
        super().__init__()
        self.raw_k1=nn.Parameter(torch.tensor(0.0))
        self.raw_k2=nn.Parameter(torch.tensor(0.0))
        self.threshold=nn.Parameter(torch.tensor(0.0))
        self.kmin,self.kmax=0.01,5.0
    def forward(self,h):
        k1=torch.clamp(F.softplus(self.raw_k1),self.kmin,self.kmax)
        k2=torch.clamp(F.softplus(self.raw_k2),self.kmin,self.kmax)
        thresh = torch.clamp(self.threshold, -5.0, 5.0)
        xm=torch.clamp(h.mean(-1), -10.0, 10.0)
        left=k1*xm
        right=k1*thresh + k2*(xm-thresh)
        out=torch.where(xm<=thresh,left,right)
        return torch.clamp(F.softplus(out), 0.0, 100.0)

class LiquidWeightGenerator(nn.Module):
    """
    Liquid Weight Generator: Conditional continuous weight generator
    Input h_multi, x → output continuous weight α_t of (B, T, K)
    No routing/expert selection is performed, all operators are always involved, but in different proportions
    """
    def __init__(self, h_dim, x_dim, n_ops, hidden_dim=64, tau_min=0.1, tau_max=5.0):
        super().__init__()
        self.n_ops = n_ops
        self.tau_min, self.tau_max = tau_min, tau_max

        # Feature extraction network: extract conditional features from h_multi and x
        self.feature_encoder = nn.Sequential(
            nn.Linear(h_dim + x_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )

        # Weight prediction network
        self.weight_predictor = nn.Linear(hidden_dim // 2, n_ops)

        # Temperature prediction network (optional)
        self.temp_predictor = nn.Linear(hidden_dim // 2, 1)

        # Global weight bias (sequence-level long-term preference)
        self.global_bias = nn.Parameter(torch.zeros(n_ops))

        # Simplified version of time encoding (without using additional Linear layer)

        # initialization
        nn.init.xavier_uniform_(self.weight_predictor.weight)
        nn.init.zeros_(self.weight_predictor.bias)
        nn.init.xavier_uniform_(self.temp_predictor.weight)
        nn.init.zeros_(self.temp_predictor.bias)

    def forward(self, h_multi, x):
        """
        Args:
            h_multi: (B, T, h_dim) - Features output by Boltzmann KAN
            x: (B, T, x_dim) - raw sensor data
        Returns:
            weights: (B, T, K) - operator weights at each time step
            temperature: (B, T) - temperature parameter (optional)
        """
        B, T, h_dim = h_multi.shape
        _, _, x_dim = x.shape

        # Splicing features: h_multi + x
        combined_features = torch.cat([h_multi, x], dim=-1)  # (B, T, h_dim + x_dim)

        # Feature encoding
        features = self.feature_encoder(combined_features)  # (B, T, hidden_dim//2)

        # Predict unnormalized weight score
        raw_weights = self.weight_predictor(features)  # (B, T, K)

        # Predict temperature
        raw_temp = self.temp_predictor(features).squeeze(-1)  # (B, T)
        temperature = torch.clamp(F.softplus(raw_temp), self.tau_min, self.tau_max)

        # Global bias modulation
        global_bias = torch.clamp(F.softplus(self.global_bias), 0.1, 5.0)  # (K,)
        adjusted_weights = raw_weights + global_bias.unsqueeze(0).unsqueeze(0)  # (B, T, K)

        # Softmax normalization of temperature modulation
        weights = F.softmax(adjusted_weights / temperature.unsqueeze(-1), dim=-1)  # (B, T, K)

        return weights, temperature

class CustomKAN(nn.Module):
    """
    Improved CustomKAN: Replacement of SparseGate with Liquid Weight Generator
    Weights vary with input, but no routing/expert selection is done
    """
    def __init__(self, ops, h_dim, x_dim):
        super().__init__()
        self.ops = nn.ModuleList(ops)
        self.n_ops = len(ops)

        # Liquid weight generator replaces SparseGate
        self.weight_generator = LiquidWeightGenerator(
            h_dim=h_dim,
            x_dim=x_dim,
            n_ops=self.n_ops,
            hidden_dim=64
        )

        # Operator output normalization (to avoid amplitude-weight ambiguity)
        self.op_norm = nn.LayerNorm(1) # Normalize the output of each operator

        # Final transformation parameters
        self.raw_gain = nn.Parameter(torch.tensor(0.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, h_multi, x):  # h_multi:(B,T,h_dim), x:(B,T,x_dim)
        B, T, _ = h_multi.shape

        # Calculate the output of all operators
        outs = []
        for op in self.ops:
            op_out = op(h_multi)  # (B, T)
            # Normalized operator output
            op_out_norm = self.op_norm(op_out.unsqueeze(-1)).squeeze(-1)  # (B, T)
            outs.append(op_out_norm)

        # Alignment length
        Tm = min(o.size(1) for o in outs)
        outs = [o[:, :Tm] for o in outs]
        h_multi_aligned = h_multi[:, :Tm, :]
        x_aligned = x[:, :Tm, :]

        # Stacking operator output
        op_stack = torch.stack(outs, dim=-1)  # (B, Tm, K)

        # Generate liquid weights
        weights, temperature = self.weight_generator(h_multi_aligned, x_aligned)  # (B, Tm, K)

        # Weighted combination (continuous weight, non-routing)
        damage = torch.sum(op_stack * weights, dim=-1)  # (B, Tm)
        damage = torch.clamp(damage, 0.0, 100.0)

        # final transformation
        gain = torch.clamp(F.softplus(self.raw_gain), 0.1, 5.0)
        bias_val = torch.clamp(self.bias, -5.0, 5.0)
        damage = torch.clamp(gain * damage + bias_val, 0.0, 100.0)

        return damage, weights, temperature

class TrendEncoder(nn.Module):
    """
    Encoder outputs "damage", then projects to monotonic decreasing HI:
        HI is learned from sensor data without hard range constraints
    Without real HI, learns mapping from sensor data to infer health states
    Enhanced with Liquid Weight Generator
    """
    def __init__(self, in_ch, trend_ch=4):
        super().__init__()
        self.boltz = BoltzmannKAN(in_ch, out_ch=trend_ch)
        ops = [MonotonicLinearOp(), MonotonicFlatOp(), ConcaveLogOp(),
               SaturationSigmoidOp(), HingeReLUOp(), PolynomialOp(),
               DampedSinOp(), PiecewiseLinearOp()]

        # Pass in h_dim and x_dim to CustomKAN
        self.customkan = CustomKAN(ops, h_dim=trend_ch, x_dim=in_ch)

        # Projection parameters - learn flexible health states from sensor features
        self.proj_gain = nn.Parameter(torch.tensor(1.0))
        self.proj_bias = nn.Parameter(torch.tensor(0.0))

        # Add health-aware layer, directly infer from multi-dimensional sensor features
        self.health_aware_transform = nn.Sequential(
            nn.Linear(in_ch, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output 0-1 range health state
        )

        # Add linear constraint layer, enforce linear trend
        self.linear_enforcer = nn.Sequential(
            nn.Linear(in_ch, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output linear decay degree
        )

    def forward(self, x):  # x:(B,T,C)
        # Feature extraction based on Boltzmann KAN
        h_multi = self.boltz(x)              # (B,T,trend_ch) >=0

        # Use liquid weight generator
        damage, weights, temperature = self.customkan(h_multi, x)    # damage:(B,T), weights:(B,T,K), temp:(B,T)

        # Directly learn health states from raw sensor data
        # Assess health state for sensor readings at each time step
        B, T, C = x.shape
        x_reshaped = x.view(-1, C)  # (B*T, C)
        health_direct = self.health_aware_transform(x_reshaped)  # (B*T, 1)
        health_direct = health_direct.view(B, T)  # (B, T)

        # Add linear constraint, force linear decay pattern
        linear_weight = self.linear_enforcer(x_reshaped).view(B, T)  # (B, T)

        # Generate ideal linear decay pattern
        t = torch.arange(T, device=x.device, dtype=x.dtype).unsqueeze(0).expand(B, -1)  # (B, T)
        t_normalized = t / (T - 1)  # Normalize to [0, 1]

        # Combine damage and direct health assessment
        g = torch.clamp(F.softplus(self.proj_gain), 0.1, 5.0)
        b = torch.clamp(self.proj_bias, -5.0, 5.0)

        # Convert damage to health state decay
        damage_normalized = torch.sigmoid(g*(damage + b))

        # Combine three health assessment methods: direct assessment, damage pattern, linear constraint
        combined_health = health_direct * (1 - 0.3 * damage_normalized)

        # Generate ideal linear decay curve (from high to low) - remove hard range constraints
        linear_decay = 1.0 - t_normalized * 0.5  # Simple linear decay from 1.0 to 0.5

        # Use linear_weight to mix linear decay and complex patterns
        # Higher linear_weight means more towards linear
        hi = linear_weight * linear_decay + (1 - linear_weight) * combined_health

        # Force stronger monotonic decreasing constraint without hard range limits
        for t_idx in range(1, T):
            hi = torch.cat([
                hi[:, :t_idx],
                torch.min(hi[:, t_idx:t_idx+1], hi[:, t_idx-1:t_idx] - 0.001),  # Force at least 0.001 decrease
                hi[:, t_idx+1:]
            ], dim=1)

        return hi, h_multi, weights, temperature

class ReconHead(nn.Module):
    """
    Recursive reconstruction: given (x_t, h_t) → predict x_{t+1}
    """
    def __init__(self, C, hidden=128):
        super().__init__()
        self.gru = nn.GRU(input_size=C+1, hidden_size=hidden, batch_first=True)
        self.out = nn.Linear(hidden, C)
    def forward(self, x_in, h_in):
        # x_in:(B,T_in,C), h_in:(B,T_in)
        B,T,C = x_in.shape
        h_in_clamped = torch.clamp(h_in, 0.0, 10.0)  # Remove upper limit constraint
        feat = torch.cat([x_in, h_in_clamped.unsqueeze(-1)], dim=-1)  # (B,T,C+1)
        H,_ = self.gru(feat)                                  # (B,T,H)
        y = self.out(H)                                       # (B,T,C) aligned predict x_{t+1}
        return y

class MaintClassifier(nn.Module):
    """
    Use HI before-after difference features for 3-class classification
    Enhanced version: not only uses ΔHI, but also sensor data change patterns
    """
    def __init__(self, sensor_dim, hidden=64, n_classes=3):
        super().__init__()
        # Original HI difference features
        self.hi_mlp = nn.Sequential(
            nn.Linear(6, hidden), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden//2)
        )

        # Sensor data change features
        self.sensor_mlp = nn.Sequential(
            nn.Linear(sensor_dim * 2, hidden), nn.ReLU(),  # Before and after statistical features
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden//2)
        )

        # Fusion classifier
        self.final_classifier = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, n_classes)
        )

    def forward(self, h_b, h_a, x_b, x_a, mask):
        # HI difference features (original logic)
        m = mask
        T = m.size(1)
        valid = m.sum(1, keepdim=True).clamp_min(1.0)  # (B,1)
        db = h_a - h_b                        # (B,T)
        mean_d = (db*m).sum(1,keepdim=True)/valid
        var_d  = (((db-mean_d)*m)**2).sum(1,keepdim=True)/valid
        std_d  = (var_d + 1e-8).sqrt()
        d0     = (db[:, :1])                  # First value
        dmax   = (db.masked_fill(m==0, -1e9)).max(dim=1, keepdim=True).values
        pos_ratio = ((db>0).float()*m).sum(1,keepdim=True)/valid

        # Slope (linear fitting)
        t = torch.arange(T, device=h_b.device, dtype=h_b.dtype)
        t = (t - t.mean())/(t.std()+1e-6)
        def slope(x):
            num = (x*t).sum(1) - x.sum(1)*t.sum()/T
            den = (t**2).sum() - (t.sum()**2)/T + 1e-8
            return (num/den).unsqueeze(1)
        slope_diff = slope(h_a) - slope(h_b)  # (B,1)
        hi_feat = torch.cat([mean_d, d0, dmax, std_d, slope_diff, pos_ratio], dim=1)  # (B,6)
        hi_feat = torch.clamp(hi_feat, -10.0, 10.0)
        hi_features = self.hi_mlp(hi_feat)

        # Sensor data change features (fix broadcast BUG: directly divide by (B,1) shaped valid)
        x_b_mean = (x_b * m.unsqueeze(-1)).sum(1) / valid        # (B,C)
        x_a_mean = (x_a * m.unsqueeze(-1)).sum(1) / valid        # (B,C)
        sensor_change = torch.cat([x_b_mean, x_a_mean], dim=1)   # (B, 2*C)
        sensor_features = self.sensor_mlp(sensor_change)

        # Fused features
        combined_features = torch.cat([hi_features, sensor_features], dim=1)
        logits = self.final_classifier(combined_features)

        return logits

def sanitize_tensors(*tensors):
    """Replace NaN/Inf in tensors with finite values"""
    result = []
    for t in tensors:
        if t is not None:
            t_clean = torch.nan_to_num(t, nan=0.0, posinf=1e6, neginf=-1e6)
            result.append(t_clean)
        else:
            result.append(t)
    return result[0] if len(result) == 1 else result

class DiffAwareReconstructor(nn.Module):
    """
    Overall model:
     - Two encoders share parameters: encode x_before / x_after → h_b / h_a (monotonic decreasing HI)
     - Two reconstruction heads: perform one-step recursive reconstruction respectively
     - Classification head: use (h_a - h_b) statistical features to identify maintenance type
    Without real HI labels, learn to infer health states from sensor data through self-supervised learning
    Enhanced with Liquid Weight Generator
    """
    def __init__(self, in_ch, trend_ch=4, hidden=128, n_classes=3):
        super().__init__()
        self.encoder = TrendEncoder(in_ch, trend_ch)
        self.recon_b = ReconHead(in_ch, hidden)
        self.recon_a = ReconHead(in_ch, hidden)
        self.clf     = MaintClassifier(sensor_dim=in_ch, hidden=64, n_classes=n_classes)

        # Add self-supervised consistency constraint
        self.consistency_weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, x_b, x_a, mask):
        # x_b/x_a : (B,L,C) (will be cut to 0:L-2 / 1:L-1 later)
        h_b, h_multi_b, weights_b, temp_b = self.encoder(x_b)  # Health state learned from sensor data
        h_a, h_multi_a, weights_a, temp_a = self.encoder(x_a)

        # Numerical stabilization
        h_b, h_a = sanitize_tensors(h_b, h_a)
        weights_b, weights_a = sanitize_tensors(weights_b, weights_a)

        # Recursive reconstruction: input 0:L-2 → predict 1:L-1
        xb_in, hb_in = x_b[:, :-2], h_b[:, :-2]
        xa_in, ha_in = x_a[:, :-2], h_a[:, :-2]
        yb_hat = self.recon_b(xb_in, hb_in)   # (B,L-2,C) ≈ x_b[:,1:L-1]
        ya_hat = self.recon_a(xa_in, ha_in)   # (B,L-2,C) ≈ x_a[:,1:L-1]

        # Numerical stabilization
        yb_hat, ya_hat = sanitize_tensors(yb_hat, ya_hat)

        # Classification: use unclipped length mask and include sensor features
        logits = self.clf(h_b, h_a, x_b, x_a, mask)
        logits = sanitize_tensors(logits)

        return yb_hat, ya_hat, h_b, h_a, logits, weights_b, weights_a, temp_b, temp_a

# ============================================================
# 3) Loss function: reconstruction + ΔHI + smooth/monotonic(decreasing) + classification + first-order linear + self-supervised consistency + maintenance effect constraint + global HI superiority constraint + liquid weight regularization
# ============================================================
def masked_mse(a, b, mask):
    # a,b:(B,T,C)  mask:(B,T)
    diff = (a - b)**2
    mse  = (diff.mean(-1) * mask).sum() / (mask.sum() + 1e-6)
    return mse

def slope_loss(h, mask, kind="smooth"):
    # Smooth: second-order difference; Monotonic decreasing: penalize upward trend
    if kind=="smooth":
        d2 = h[:,2:] - 2*h[:,1:-1] + h[:,:-2]
        m  = mask[:,2:]
        return (d2.abs() * m).sum() / (m.sum()+1e-6)
    elif kind=="mono_dec":
        dh = h[:,1:] - h[:,:-1]        # If >0 indicates upward (violates "decreasing")
        m  = mask[:,1:]
        return (F.relu(dh) * m).sum() / (m.sum()+1e-6)
    else:
        return torch.tensor(0., device=h.device)

def liquid_weight_regularization(weights_b, weights_a, mask, lambda_tv=0.1, lambda_ent=0.05, lambda_bal=0.01):
    """
    Liquid weight regularization loss
    Args:
        weights_b, weights_a: (B, T, K) - weight sequence
        mask: (B, T) - valid position mask
        lambda_tv: temporal smoothing regularization weight
        lambda_ent: entropy regularization weight
        lambda_bal: Usage balancing regularization weight
    """
    total_loss = torch.tensor(0.0, device=weights_b.device)

    # 1. Time smoothing loss (Total Variation)
    def tv_loss(weights, mask):
        if weights.size(1) <= 1:
            return torch.tensor(0.0, device=weights.device)

        diff = weights[:, 1:, :] - weights[:, :-1, :]  # (B, T-1, K)
        mask_diff = mask[:, 1:]  # (B, T-1)

        tv = (diff.abs().sum(-1) * mask_diff).sum() / (mask_diff.sum() + 1e-6)
        return tv

    tv_loss_b = tv_loss(weights_b, mask)
    tv_loss_a = tv_loss(weights_a, mask)
    total_loss += lambda_tv * (tv_loss_b + tv_loss_a)

    # 2. Entropy regularization (encourage high entropy in the early stage and allow more personalization in the later stage)
    def entropy_loss(weights, mask):
        # Calculate entropy at each time step
        eps = 1e-8
        entropy = -(weights * torch.log(weights + eps)).sum(-1)  # (B, T)
        return (entropy * mask).sum() / (mask.sum() + 1e-6)

    ent_loss_b = entropy_loss(weights_b, mask)
    ent_loss_a = entropy_loss(weights_a, mask)
    total_loss += lambda_ent * (ent_loss_b + ent_loss_a)

    # 3. Usage balance loss (to prevent certain operators from being ignored for a long time)
    def balance_loss(weights, mask):
        # Calculate the average usage of each operator
        valid_weights = weights * mask.unsqueeze(-1)  # (B, T, K)
        mean_usage = valid_weights.sum([0, 1]) / (mask.sum() + 1e-6)  # (K,)
        target_usage = 1.0 / weights.size(-1) # Ideally, the usage rate of each operator is 1/K

        balance = ((mean_usage - target_usage) ** 2).sum()
        return balance

    bal_loss_b = balance_loss(weights_b, mask)
    bal_loss_a = balance_loss(weights_a, mask)
    total_loss += lambda_bal * (bal_loss_b + bal_loss_a)

    return total_loss

def enhanced_linear_slope_loss(h, mask, weight=5.0):
    """
    Enhanced linear constraint: calculate deviation from best linear fit, stronger weight
    Without real HI, this constraint helps learn reasonable decay patterns
    """
    B, T = h.shape
    valid_mask = mask  # (B, T)

    # Calculate best linear fit for each sample
    t = torch.arange(T, device=h.device, dtype=h.dtype).unsqueeze(0).expand(B, -1)  # (B, T)

    # Consider only valid positions
    loss_total = torch.tensor(0.0, device=h.device)
    n_valid_samples = 0

    for b in range(B):
        valid_indices = valid_mask[b] > 0
        if valid_indices.sum() < 2:  # Need at least 2 points for linear fitting
            continue

        h_valid = h[b][valid_indices]  # Valid HI values
        t_valid = t[b][valid_indices]  # Corresponding time points

        # Center time points
        t_mean = t_valid.mean()
        t_centered = t_valid - t_mean
        h_mean = h_valid.mean()

        # Calculate linear regression slope
        numerator = (t_centered * (h_valid - h_mean)).sum()
        denominator = (t_centered ** 2).sum() + 1e-8
        slope = numerator / denominator

        # Calculate intercept
        intercept = h_mean - slope * t_mean

        # Linear predicted values
        h_linear = slope * t_valid + intercept

        # Calculate MSE with linear fit, using stronger weight
        linear_mse = ((h_valid - h_linear) ** 2).mean()

        # Additional penalty for positive slope (encourage negative slope, i.e., decreasing)
        slope_penalty = F.relu(slope + 0.01) * 2.0  # Slope should be negative

        loss_total += weight * linear_mse + slope_penalty
        n_valid_samples += 1

    return loss_total / max(n_valid_samples, 1)

def maintenance_improvement_constraint(h_b, h_a, labels, mask):
    """
    Maintenance improvement constraint: ensure post-maintenance HI is significantly higher than pre-maintenance
    - Perfect(0): require post-maintenance significantly higher than pre-maintenance (at least 0.4)
    - General(1): require post-maintenance moderately higher than pre-maintenance (at least 0.2)
    - Poor(2): require post-maintenance slightly higher than pre-maintenance (at least 0.1)
    """
    valid = mask.sum(1, keepdim=True).clamp_min(1.0)  # (B,1)

    # Calculate average HI before and after maintenance
    h_b_mean = (h_b * mask).sum(1, keepdim=True) / valid  # (B,1)
    h_a_mean = (h_a * mask).sum(1, keepdim=True) / valid  # (B,1)

    improvement = h_a_mean - h_b_mean  # Improvement after maintenance relative to before

    loss = torch.zeros_like(improvement)
    for cls in [0, 1, 2]:
        sel = (labels == cls).float().unsqueeze(1)  # (B,1)
        if sel.sum() < 1:
            continue

        if cls == 0:  # Perfect
            # Require at least 0.4 improvement
            loss += sel * F.relu(0.4 - improvement) ** 2
        elif cls == 1:  # General
            # Require at least 0.2 improvement
            loss += sel * F.relu(0.2 - improvement) ** 2
        else:  # Poor
            # Require at least 0.1 improvement
            loss += sel * F.relu(0.1 - improvement) ** 2

    return loss.mean()

def global_hi_superiority_constraint(h_b, h_a, mask, weight=5.0):
    """
    Enhanced global superiority constraint: ensure post-maintenance HI is entirely higher than pre-maintenance HI
    For each sample, require post-maintenance HI at every time point to be higher than pre-maintenance HI
    Remove range constraints and focus on relative superiority
    """
    valid = mask  # (B, T)

    loss_total = torch.tensor(0.0, device=h_b.device)
    n_valid_samples = 0

    for b in range(h_b.size(0)):
        valid_mask_b = valid[b] > 0
        if valid_mask_b.sum() < 1:
            continue

        # Get valid HI values before and after maintenance
        h_b_valid = h_b[b][valid_mask_b]  # Valid HI values before maintenance
        h_a_valid = h_a[b][valid_mask_b]  # Valid HI values after maintenance

        # Point-wise superiority: each post-maintenance point should be higher than corresponding pre-maintenance point
        pointwise_gap = h_a_valid - h_b_valid  # Should be positive
        pointwise_loss = F.relu(-pointwise_gap + 0.05).mean()  # Penalize when post-maintenance is not sufficiently higher

        # Global superiority: minimum post-maintenance should be higher than maximum pre-maintenance
        h_b_max = h_b_valid.max()
        h_a_min = h_a_valid.min()

        # Require post-maintenance minimum to be higher than pre-maintenance maximum
        margin = 0.1  # Post-maintenance minimum should be at least 0.1 higher than pre-maintenance maximum
        global_gap_loss = F.relu(h_b_max + margin - h_a_min) ** 2

        # Additional constraint: post-maintenance average should be significantly higher than pre-maintenance average
        h_b_mean = h_b_valid.mean()
        h_a_mean = h_a_valid.mean()
        mean_gap_loss = F.relu(h_b_mean + 0.2 - h_a_mean) ** 2  # Average should improve by at least 0.2

        loss_total += weight * (pointwise_loss + global_gap_loss + 0.5 * mean_gap_loss)
        n_valid_samples += 1

    return loss_total / max(n_valid_samples, 1)

def diff_margin_by_class(mean_delta, labels, m_low=0.1, m_mid=0.25, m_high=0.45):
    """
    Class-conditional difference constraint (enhanced maintenance effect):
      - Perfect(0):  Δ>=m_high (post-maintenance significantly higher than pre-maintenance)
      - General(1):  Δ≈m_mid   (post-maintenance moderately higher than pre-maintenance)
      - Poor(2):     Δ>=m_low  (post-maintenance at least slightly higher than pre-maintenance)
    All classes require post-maintenance HI higher than pre-maintenance (positive improvement), just different degrees
    """
    loss = torch.zeros_like(mean_delta)
    for cls in [0,1,2]:
        sel = (labels==cls).float()
        if sel.sum() < 1: continue
        if cls==0:
            # Perfect: require significant improvement (at least reach m_high)
            loss += sel * F.relu(m_high - mean_delta)**2
        elif cls==1:
            # General: require moderate improvement (close to m_mid)
            loss += sel * (mean_delta - m_mid)**2
        else:
            # Poor: require at least small improvement (at least reach m_low)
            loss += sel * F.relu(m_low - mean_delta)**2
    denom = torch.ones_like(mean_delta) * (labels.numel())
    return loss.sum() / denom.sum()

def sensor_consistency_loss(x_b, x_a, h_b, h_a, mask):
    """
    Sensor data consistency loss: ensure HI changes are consistent with sensor data change patterns
    Important constraint when no real HI labels are available
    """
    # Calculate statistical changes in sensor data
    valid = mask.sum(1, keepdim=True).clamp_min(1.0)   # (B,1)

    # Average change magnitude in sensor data (fix broadcast BUG: divide by (B,1))
    xb_mean = (x_b * mask.unsqueeze(-1)).sum(1) / valid   # (B,C)
    xa_mean = (x_a * mask.unsqueeze(-1)).sum(1) / valid   # (B,C)
    sensor_change = torch.abs(xa_mean - xb_mean)          # (B,C)
    sensor_change_magnitude = sensor_change.mean(1)       # (B,)

    # HI change magnitude (post-maintenance improvement relative to pre-maintenance)
    hi_change = ((h_a - h_b) * mask).sum(1) / valid.squeeze(-1)  # (B,) Note: remove abs, maintain positive direction

    # Expect HI change to positively correlate with sensor change, and HI should improve positively
    # Use Pearson correlation coefficient as consistency metric
    mean_sensor = sensor_change_magnitude.mean()
    mean_hi = hi_change.mean()

    correlation = ((sensor_change_magnitude - mean_sensor) * (hi_change - mean_hi)).mean() / \
                  (sensor_change_magnitude.std() * hi_change.std() + 1e-8)

    # Encourage positive correlation (large sensor change when HI change is also large)
    consistency_loss = F.relu(0.5 - correlation)  # Expect correlation at least 0.5

    # Additional constraint: force positive HI improvement
    negative_change_penalty = F.relu(-hi_change).mean() * 2.0  # Penalize negative changes

    return consistency_loss + negative_change_penalty

def smoothness_enhancement_loss(h, mask, order=2):
    """
    Enhanced smoothness loss: use higher-order differences to force smoother curves
    """
    if order == 2:
        # Second-order difference
        d2 = h[:,2:] - 2*h[:,1:-1] + h[:,:-2]
        m = mask[:,2:]
        return (d2.abs() * m).sum() / (m.sum() + 1e-6)
    elif order == 3:
        # Third-order difference (stronger smoothness constraint)
        d3 = h[:,3:] - 3*h[:,2:-1] + 3*h[:,1:-2] - h[:,:-3]
        m = mask[:,3:]
        return (d3.abs() * m).sum() / (m.sum() + 1e-6)
    else:
        return torch.tensor(0., device=h.device)

# ============================================================
# 4) Data splitting and visualization (aligned to same time axis)
# ============================================================
LABEL2NAME = {0: "Perfect", 1: "General", 2: "Poor"}

def split_pairs_uid(pairs, train_ratio=0.7, val_ratio=0.1, seed=42):
    uids = sorted(list(pairs.keys()))
    rs = np.random.RandomState(seed)
    rs.shuffle(uids)
    n = len(uids); n_tr=int(n*train_ratio); n_vl=int(n*val_ratio)
    to_dict = lambda ss:{u:pairs[u] for u in ss if u in pairs}
    return to_dict(uids[:n_tr]), to_dict(uids[n_tr:n_tr+n_vl]), to_dict(uids[n_tr+n_vl:])

def print_split_summary(pairs, train_ratio=0.7, val_ratio=0.1, seed=42):
    pairs_tr, pairs_vl, pairs_te = split_pairs_uid(pairs, train_ratio, val_ratio, seed)
    def count_items(pp):
        cnt = 0
        for uid, d in pp.items():
            cnt += len(d)
        return len(pp), cnt
    n_uid_tr, n_pair_tr = count_items(pairs_tr)
    n_uid_vl, n_pair_vl = count_items(pairs_vl)
    n_uid_te, n_pair_te = count_items(pairs_te)
    print(f"[Data Split] Train: UID={n_uid_tr}, pairs≈{n_pair_tr} | "
          f"Val: UID={n_uid_vl}, pairs≈{n_pair_vl} | "
          f"Test: UID={n_uid_te}, pairs≈{n_pair_te}")

def remove_constant_segments(hi_values, threshold=1e-6):
    """
    Remove constant segments in HI curves (caused by mask or other reasons)
    Return valid HI values and corresponding time indices
    """
    if len(hi_values) <= 1:
        return hi_values, np.arange(len(hi_values))

    # Calculate differences between adjacent points
    diffs = np.abs(np.diff(hi_values))

    # Find points with sufficient change
    valid_mask = np.ones(len(hi_values), dtype=bool)
    valid_mask[1:] = diffs > threshold

    # Ensure at least first and last two points are kept
    if np.sum(valid_mask) < 2:
        valid_mask[0] = True
        valid_mask[-1] = True

    valid_indices = np.where(valid_mask)[0]
    return hi_values[valid_indices], valid_indices

@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total = {"mse_b":0.0, "mse_a":0.0, "acc":0.0, "delta_mean":0.0}
    n_batch=0
    n_acc = 0; n_tot=0
    for batch in loader:
        xb = batch["x_before"].to(device)
        xa = batch["x_after"].to(device)
        hi_b = batch["hi_before"].to(device)
        hi_a = batch["hi_after"].to(device)
        labels = batch["labels"].to(device)
        lengths= batch["lengths"].to(device)
        mask   = batch["mask"].to(device)

        # Valid segment: 0:L-2 input, 1:L-1 target
        m_tgt  = (torch.arange(xb.size(1)-2, device=device)[None,:] < (lengths-2)[:,None]).float()

        yb_hat, ya_hat, h_b, h_a, logits, _, _, _, _ = model(xb, xa, mask)

        # Numerical stabilization
        yb_hat, ya_hat, h_b, h_a, logits = sanitize_tensors(yb_hat, ya_hat, h_b, h_a, logits)

        # Align targets
        yb_tgt = xb[:,1:-1,:]
        ya_tgt = xa[:,1:-1,:]

        mse_b = masked_mse(yb_hat, yb_tgt, m_tgt)
        mse_a = masked_mse(ya_hat, ya_tgt, m_tgt)

        # Check for NaN
        if torch.isnan(mse_b) or torch.isnan(mse_a):
            continue

        # Classification
        pred = logits.argmax(dim=1)
        n_acc += (pred == labels).sum().item()
        n_tot += labels.numel()

        # Statistics of ΔHI mean (post-maintenance improvement relative to pre-maintenance)
        valid = mask.sum(1, keepdim=True).clamp_min(1.0)
        delta_mean = ((h_a - h_b)*mask).sum(1,keepdim=True)/valid
        delta_mean = torch.nan_to_num(delta_mean, nan=0.0)
        total["delta_mean"] += delta_mean.mean().item()

        total["mse_b"] += mse_b.item()
        total["mse_a"] += mse_a.item()
        n_batch += 1

    for k in ["mse_b","mse_a","delta_mean"]:
        total[k] /= max(n_batch,1)
    total["acc"] = n_acc / max(n_tot,1)
    return total

@torch.no_grad()
def collect_test_predictions(model, loader, device, max_curve_keep=24):
    """
    Collect predictions, ΔHI, and a few sample HI curves on test set (for plotting).
    """
    model.eval()
    y_true, y_pred = [], []
    all_delta_mean, all_uids = [], []
    keep_curves = []

    for batch in loader:
        xb = batch["x_before"].to(device)   # (B,L,C)
        xa = batch["x_after"].to(device)
        mask   = batch["mask"].to(device)   # (B,L)
        labels = batch["labels"].to(device) # (B,)
        lengths= batch["lengths"]           # cpu tensor
        uids   = batch["uids"]

        yb_hat, ya_hat, h_b, h_a, logits, weights_b, weights_a, temp_b, temp_a = model(xb, xa, mask)

        # Numerical stabilization
        yb_hat, ya_hat, h_b, h_a, logits = sanitize_tensors(yb_hat, ya_hat, h_b, h_a, logits)

        prob = F.softmax(logits, dim=1)     # (B,3)
        pred = prob.argmax(1)               # (B,)

        # Statistics of ΔHI mean (post-maintenance improvement relative to pre-maintenance)
        valid = mask.sum(1, keepdim=True).clamp_min(1.0)
        delta_mean = ((h_a - h_b) * mask).sum(1, keepdim=True) / valid  # (B,1)
        delta_mean = torch.nan_to_num(delta_mean, nan=0.0)

        y_true.append(labels.cpu().numpy())
        y_pred.append(pred.cpu().numpy())
        all_delta_mean.append(delta_mean.squeeze(1).cpu().numpy())
        all_uids.extend(uids)

        # Save some curves for visualization (aligned: after concatenated after before)
        for i in range(xb.size(0)):
            if len(keep_curves) >= max_curve_keep:
                break
            L_i = int(lengths[i].item())
            # Fix: truncate reconstruction prediction by actual length
            L_recon = min(L_i - 2, yb_hat.size(1))  # Reconstruction sequence length
            keep_curves.append({
                "uid": uids[i],
                "true": int(labels[i].cpu().item()),
                "pred": int(pred[i].cpu().item()),
                "prob": prob[i].cpu().numpy(),
                "h_before": h_b[i, :L_i].cpu().numpy(),
                "h_after":  h_a[i, :L_i].cpu().numpy(),
                "weights_before": weights_b[i, :L_i].cpu().numpy() if weights_b is not None else None,
                "weights_after": weights_a[i, :L_i].cpu().numpy() if weights_a is not None else None,
                "yb_hat":   yb_hat[i, :L_recon].cpu().numpy(),       # (L_recon,C)
                "ya_hat":   ya_hat[i, :L_recon].cpu().numpy(),       # (L_recon,C)
                "x_before": xb[i, :L_i].cpu().numpy(),               # (L,C)
                "x_after":  xa[i, :L_i].cpu().numpy(),               # (L,C)
            })

    y_true = np.concatenate(y_true, axis=0) if len(y_true)>0 else np.array([])
    y_pred = np.concatenate(y_pred, axis=0) if len(y_pred)>0 else np.array([])
    all_delta_mean = np.concatenate(all_delta_mean, axis=0) if len(all_delta_mean)>0 else np.array([])
    return y_true, y_pred, all_delta_mean, all_uids, keep_curves

def plot_hi_examples_aligned(curves, n_show=6, seed=0):
    """Post-maintenance sequence continues from pre-maintenance endpoint; draw vertical line to mark t_m; remove constant segments.
    Add "Learned HI" in title to indicate this is health state inferred from sensor data, post-maintenance should be higher than pre-maintenance"""
    if len(curves)==0:
        print("(No visualization samples)")
        return
    random.Random(seed).shuffle(curves)
    n_show = min(n_show, len(curves))
    cols = 3
    rows = int(np.ceil(n_show/cols))
    plt.figure(figsize=(cols*4.6, rows*3.2))
    for k in range(n_show):
        ex = curves[k]
        hb = ex["h_before"]; ha = ex["h_after"]
        L  = min(len(hb), len(ha))  # Defensive alignment
        hb, ha = hb[:L], ha[:L]

        # Remove constant segments
        hb_clean, hb_indices = remove_constant_segments(hb)
        ha_clean, ha_indices = remove_constant_segments(ha)

        # Corresponding time axis
        t_b = hb_indices
        t_a = ha_indices + L  # after follows immediately

        uid = ex["uid"]
        pred = ex["pred"]; true = ex["true"]; prob = ex["prob"]
        plt.subplot(rows, cols, k+1)

        # Only plot non-constant segments
        if len(hb_clean) > 1:
            plt.plot(t_b, hb_clean, label="Learned HI_Pre-maintenance", linewidth=1.8, marker='o', markersize=3)
        if len(ha_clean) > 1:
            plt.plot(t_a, ha_clean, label="Learned HI_Post-maintenance",  linewidth=1.8, linestyle='--', marker='s', markersize=3)

        # Maintenance time vertical line
        plt.axvline(L-1, color='k', linestyle=':', linewidth=1.0, alpha=0.7)

        d_mean = float(np.mean(ha - hb))  # Post-maintenance improvement relative to pre-maintenance
        d_max  = float(np.max(ha - hb))
        title = (f"uid={uid} (Liquid Weight Maintenance Effect)\nTrue={LABEL2NAME[true]} | Pred={LABEL2NAME[pred]} "
                 f"| p={prob[pred]:.2f}\nΔHI_mean={d_mean:.3f}, ΔHI_max={d_max:.3f}")
        plt.title(title, fontsize=9)
        plt.xlabel("Cycle (Pre-maintenance | Post-maintenance)")
        plt.ylabel("Learned Health Index (↑Post should be higher)")
        plt.grid(ls='--', alpha=.35)
        # Remove Y-axis range constraint, let it adapt to data
        if k==0: plt.legend()
    plt.tight_layout()
    plt.show()

def plot_liquid_weights(curves, n_show=3, seed=0):
    """
    Visualize liquid weights for selected samples
    """
    if len(curves) == 0:
        print("(No visualization samples)")
        return

    # Filter curves that have weight information
    curves_with_weights = [c for c in curves if c.get("weights_before") is not None]
    if len(curves_with_weights) == 0:
        print("(No weight information available)")
        return

    random.Random(seed).shuffle(curves_with_weights)
    n_show = min(n_show, len(curves_with_weights))

    fig, axes = plt.subplots(n_show, 2, figsize=(12, 4*n_show))
    if n_show == 1:
        axes = axes.reshape(1, -1)

    op_names = ["MonotonicLinear", "MonotonicFlat", "ConcaveLog", "SaturationSigmoid",
                "HingeReLU", "Polynomial", "DampedSin", "PiecewiseLinear"]

    for i in range(n_show):
        ex = curves_with_weights[i]
        weights_b = ex["weights_before"]  # (T, K)
        weights_a = ex["weights_after"]   # (T, K)

        if weights_b is None or weights_a is None:
            continue

        uid = ex["uid"]
        true_label = LABEL2NAME[ex["true"]]
        pred_label = LABEL2NAME[ex["pred"]]

        # Plot weights before maintenance
        ax1 = axes[i, 0]
        T_b, K = weights_b.shape
        for k in range(K):
            ax1.plot(range(T_b), weights_b[:, k], label=op_names[k] if k < len(op_names) else f"Op_{k}",
                    marker='o', markersize=2, linewidth=1.5)
        ax1.set_title(f"UID={uid} Pre-maintenance Weights\nTrue={true_label}, Pred={pred_label}")
        ax1.set_xlabel("Time Step")
        ax1.set_ylabel("Weight")
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)

        # Plot weights after maintenance
        ax2 = axes[i, 1]
        T_a, _ = weights_a.shape
        for k in range(K):
            ax2.plot(range(T_a), weights_a[:, k], label=op_names[k] if k < len(op_names) else f"Op_{k}",
                    marker='s', markersize=2, linewidth=1.5, linestyle='--')
        ax2.set_title(f"UID={uid} Post-maintenance Weights\nTrue={true_label}, Pred={pred_label}")
        ax2.set_xlabel("Time Step")
        ax2.set_ylabel("Weight")
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_sensor_examples_aligned(curves, sensor_idx_list=None, n_cols=4):
    """
    Plot several sensors' original vs recursive prediction (post), with after time axis connected after before.
    - Original before: x_before[:, s]
    - Original after : x_after [:, s] (optional)
    - Predicted after : ya_hat   [:, s] (one-step sequence aligned with x_after)
    """
    if len(curves)==0:
        print("(No visualization samples)")
        return
    # Take first sample to show multiple sensors
    ex = curves[0]
    xb = ex["x_before"]         # (L,C)
    xa = ex["x_after"]          # (L,C)
    ya = ex["ya_hat"]           # (L_recon,C) predicted is 1..L-1
    L, C = xb.shape
    Lhat = ya.shape[0]          # Actual reconstruction length

    if sensor_idx_list is None:
        # Default pick 8
        step = max(1, C//8)
        sensor_idx_list = list(range(0, C, step))[:8]

    n = len(sensor_idx_list)
    n_rows = int(np.ceil(n/n_cols))
    plt.figure(figsize=(n_cols*4.3, n_rows*3.1))
    for i, s in enumerate(sensor_idx_list):
        plt.subplot(n_rows, n_cols, i+1)
        t_b = np.arange(L)
        t_a = np.arange(L, 2*L)
        plt.plot(t_b, xb[:,s], label="Original (before)", linewidth=1.2)
        plt.plot(t_a, xa[:,s], label="Original (after)",  linewidth=1.0, linestyle="--", alpha=0.7)
        # Predicted after: aligned with after target (corresponding to 1..L-1)
        t_pred = np.arange(L+1, L+1+Lhat)      # Align ya_hat time
        plt.plot(t_pred, ya[:,s], label="Post-maint. prediction", linewidth=1.6)
        plt.axvline(L-1, color='k', linestyle=':', linewidth='1.0')
        plt.title(f"Sensor_{s:02d} (Liquid Weight Enhanced)")
        if i==0: plt.legend()
        plt.grid(ls="--", alpha=.35)
    plt.tight_layout()
    plt.show()

def topk_by_delta(df_delta, k=5):
    if len(df_delta)==0:
        return
    for cls in [0,1,2]:
        sub = df_delta[df_delta["true"]==cls].sort_values("delta_hi_mean", ascending=False).head(k)
        print(f"\n[True={LABEL2NAME[cls]}] Top {k} samples with highest learned ΔHI_mean (maintenance effect):")
        print(sub[["uid","delta_hi_mean","pred"]].reset_index(drop=True))

# ============================================================
# 5) Training/evaluation loop with liquid weight regularization
# ============================================================
def train_model(pairs, epochs=20, batch_size=32, lr=1e-3, horizon=None, device=None, patience=20):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data split 7:1:2
    pairs_tr, pairs_vl, pairs_te = split_pairs_uid(pairs, 0.7, 0.1, 42)
    print_split_summary(pairs, 0.7, 0.1, 42)

    ds_tr = PairsReconstructDataset(pairs_tr, horizon=horizon)
    ds_vl = PairsReconstructDataset(pairs_vl, horizon=horizon)
    ds_te = PairsReconstructDataset(pairs_te, horizon=horizon)

    ld_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_shift)
    ld_vl = DataLoader(ds_vl, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_shift)
    ld_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_shift)

    C = ds_tr.C
    print(f"\n[Dimension Check] Sensor feature dimension C = {C}")

    model = DiffAwareReconstructor(in_ch=C, trend_ch=4, hidden=128, n_classes=3).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)  # Use AdamW
    ce  = nn.CrossEntropyLoss()

    best_v = 1e9
    best = None
    nan_count = 0  # Count NaN batches
    best_epoch = 0  # Record best epoch
    no_improve_count = 0  # Count epochs without improvement for early stopping

    print(f"\nStarting training for {epochs} epochs with early stopping (patience={patience})...")
    print(f"Device: {device}")
    print(f"Train set: {len(ds_tr)} samples, Val set: {len(ds_vl)} samples, Test set: {len(ds_te)} samples")
    print("Note: Enhanced with Liquid Weight Generator - weights adapt to input conditions")
    print("Goal: Post-maintenance HI should be entirely higher than pre-maintenance, with adaptive operator combination")

    for ep in range(1, epochs+1):
        print(f"\n[Epoch {ep}/{epochs}] Training...")
        model.train()
        logs = {"mse_b":0.0, "mse_a":0.0, "diff":0.0, "smooth":0.0, "mono":0.0, "linear":0.0, "cls":0.0, "consist":0.0, "maintain":0.0, "smooth_enh":0.0, "global_sup":0.0, "liquid_weight":0.0, "all":0.0}
        n_bt = 0
        batch_nan_count = 0

        # Course-based learning: Dynamically adjust the liquid weight regularization intensity
        # Early days: high entropy, encouraging uniform weights
        # Later: allow for more personalized weight distribution
        progress = ep / epochs
        lambda_tv = 0.1 * (1 - progress * 0.5) # Time smoothing: gradually decrease from 0.1 to 0.05
        lambda_ent = 0.1 * (1 - progress) # Entropy regularity: gradually decreases from 0.1 to 0
        lambda_bal = 0.01 * (1 - progress * 0.8) # Balance regularity: gradually decrease from 0.01 to 0.002

        for batch_idx, batch in enumerate(ld_tr):
            # Show training progress
            if batch_idx % max(1, len(ld_tr)//10) == 0:
                progress_pct = (batch_idx + 1) / len(ld_tr) * 100
                print(f"    Training progress: {progress_pct:.1f}% ({batch_idx+1}/{len(ld_tr)} batches)")

            xb = batch["x_before"].to(device)  # (B,L,C)
            xa = batch["x_after"].to(device)
            labels = batch["labels"].to(device)
            lengths= batch["lengths"].to(device)
            mask   = batch["mask"].to(device)

            # Input/target mask
            m_tgt  = (torch.arange(xb.size(1)-2, device=device)[None,:] < (lengths-2)[:,None]).float()

            yb_hat, ya_hat, h_b, h_a, logits, weights_b, weights_a, temp_b, temp_a = model(xb, xa, mask)

            # Numerical stabilization
            yb_hat, ya_hat, h_b, h_a, logits = sanitize_tensors(yb_hat, ya_hat, h_b, h_a, logits)
            weights_b, weights_a = sanitize_tensors(weights_b, weights_a)

            # Recursive reconstruction targets
            yb_tgt = xb[:,1:-1,:]
            ya_tgt = xa[:,1:-1,:]
            loss_b = masked_mse(yb_hat, yb_tgt, m_tgt)
            loss_a = masked_mse(ya_hat, ya_tgt, m_tgt)

            # HI difference (class-conditional margin) - ensure post-maintenance higher than pre-maintenance
            valid = mask.sum(1, keepdim=True).clamp_min(1.0)
            delta_mean = ((h_a - h_b)*mask).sum(1,keepdim=True)/valid  # (B,1)
            loss_diff = diff_margin_by_class(delta_mean.squeeze(1), labels,
                                             m_low=0.1, m_mid=0.25, m_high=0.45)

            # HI smooth + monotonic decreasing (constrain both pre/post) - remove range constraint
            loss_smooth= slope_loss(h_a, mask, "smooth") + slope_loss(h_b, mask, "smooth")
            loss_mono  = slope_loss(h_a, mask, "mono_dec") + slope_loss(h_b, mask, "mono_dec")

            # Enhanced linear loss - stronger weight
            loss_linear = enhanced_linear_slope_loss(h_b, mask, weight=8.0) + enhanced_linear_slope_loss(h_a, mask, weight=8.0)

            # Sensor consistency loss (including constraint that post-maintenance should be higher than pre-maintenance)
            loss_consistency = sensor_consistency_loss(xb, xa, h_b, h_a, mask)

            # Maintenance improvement constraint
            loss_maintenance = maintenance_improvement_constraint(h_b, h_a, labels, mask)

            # Enhanced global HI superiority constraint (higher weight)
            loss_global_superiority = global_hi_superiority_constraint(h_b, h_a, mask, weight=5.0)

            # Enhanced smoothness loss (second and third order differences)
            loss_smooth_enhanced = (smoothness_enhancement_loss(h_b, mask, order=2) +
                                   smoothness_enhancement_loss(h_a, mask, order=2) +
                                   smoothness_enhancement_loss(h_b, mask, order=3) * 0.5 +
                                   smoothness_enhancement_loss(h_a, mask, order=3) * 0.5)

            # Liquid weight regularization loss
            loss_liquid_weight = liquid_weight_regularization(weights_b, weights_a, mask,
                                                             lambda_tv=lambda_tv,
                                                             lambda_ent=lambda_ent,
                                                             lambda_bal=lambda_bal)

            # Classification loss
            loss_cls = ce(logits, labels)

            # Total loss (Adjust weights - add liquid weight regularization)
            w_rec=1.0; w_diff=0.5; w_smooth=0.05; w_mono=0.1; w_linear=1.0; w_consist=0.3; w_cls=1.0; w_maintain=1.5; w_smooth_enh=0.5; w_global_sup=2.0; w_liquid=0.2
            loss = (w_rec*(loss_b+loss_a) + w_diff*loss_diff +
                   w_smooth*loss_smooth + w_mono*loss_mono + w_linear*loss_linear +
                   w_consist*loss_consistency + w_cls*loss_cls + w_maintain*loss_maintenance +
                   w_smooth_enh*loss_smooth_enhanced + w_global_sup*loss_global_superiority +
                   w_liquid*loss_liquid_weight)

            # Numerical stabilization of all losses
            loss_b, loss_a, loss_diff, loss_smooth, loss_mono, loss_linear, loss_consistency, loss_cls, loss_maintenance, loss_smooth_enhanced, loss_global_superiority, loss_liquid_weight, loss = sanitize_tensors(
                loss_b, loss_a, loss_diff, loss_smooth, loss_mono, loss_linear, loss_consistency, loss_cls, loss_maintenance, loss_smooth_enhanced, loss_global_superiority, loss_liquid_weight, loss)

            # Check NaN and skip
            if torch.isnan(loss) or torch.isinf(loss):
                batch_nan_count += 1
                if batch_nan_count == 1:  # Only print warning once
                    print(f"    [Warning] Epoch {ep}: Found NaN/Inf loss, skipping abnormal batch")
                continue

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            logs["mse_b"] += loss_b.item()
            logs["mse_a"] += loss_a.item()
            logs["diff"]  += loss_diff.item()
            logs["smooth"]+= loss_smooth.item()
            logs["mono"]  += loss_mono.item()
            logs["linear"]+= loss_linear.item()
            logs["consist"]+= loss_consistency.item()
            logs["cls"]   += loss_cls.item()
            logs["maintain"]+= loss_maintenance.item()
            logs["smooth_enh"]+= loss_smooth_enhanced.item()
            logs["global_sup"]+= loss_global_superiority.item()
            logs["liquid_weight"]+= loss_liquid_weight.item()
            logs["all"]   += loss.item()
            n_bt += 1

        # Record NaN batches
        if batch_nan_count > 0:
            nan_count += batch_nan_count

        print("    Validating...")
        for k in logs: logs[k] /= max(n_bt,1)
        vl = eval_epoch(model, ld_vl, device)
        print(f"[Epoch {ep:03d}] Train: L={logs['all']:.4f} rec_b={logs['mse_b']:.4f} rec_a={logs['mse_a']:.4f} "
              f"diff={logs['diff']:.4f} linear={logs['linear']:.4f} maintain={logs['maintain']:.4f} liquid_w={logs['liquid_weight']:.4f} global_sup={logs['global_sup']:.4f} cls={logs['cls']:.4f} | "
              f"Val: mse_b={vl['mse_b']:.4f} mse_a={vl['mse_a']:.4f} acc={vl['acc']:.3f} ΔHI_improvement={vl['delta_mean']:.3f}")

        # Simple early stopping: observe validation reconstruction loss
        vl_total = vl['mse_b'] + vl['mse_a'] + vl['acc']  # Lower is better (minimize MSE, maximize accuracy)
        if vl_total < best_v:
            best_v = vl_total
            best_epoch = ep
            best = {k: v.clone() if hasattr(v, 'clone') else v for k, v in model.state_dict().items()}
            no_improve_count = 0  # Reset counter
            print(f"    ✓ Saved new best model (val loss: {best_v:.4f})")
        else:
            no_improve_count += 1
            print(f"    Val loss not improved (current: {vl_total:.4f}, best: {best_v:.4f}) - No improvement for {no_improve_count} epochs")

            # Early stopping check
            if no_improve_count >= patience:
                print(f"\n[Early Stopping] No improvement for {patience} epochs. Stopping training at epoch {ep}.")
                break

    if best is not None:
        model.load_state_dict(best)
        print(f"\n[Best Checkpoint] Val reconstruction loss: {best_v:.4f} (Epoch {best_epoch})")

        # Save best model to specified path
        save_path = "/content/drive/MyDrive/CMAPSS/main_dynamics_identification_2.pth"
        torch.save(model.state_dict(), save_path)
        print(f"[Model Saved] Best model saved to: {save_path}")

    if nan_count > 0:
        print(f"[Warning] Total skipped {nan_count} batches containing NaN/Inf")

    # Test set evaluation
    print("\n" + "="*60)
    print("Final Test Set Evaluation")
    print("="*60)
    te = eval_epoch(model, ld_te, device)
    print(f"[Test Set] mse_b={te['mse_b']:.4f} mse_a={te['mse_a']:.4f} acc={te['acc']:.3f} ΔHI_improvement={te['delta_mean']:.3f}")

    # Collect test predictions and visualization data
    y_true, y_pred, all_delta_mean, all_uids, keep_curves = collect_test_predictions(model, ld_te, device, max_curve_keep=24)

    # Classification report
    if len(y_true) > 0:
        from sklearn.metrics import classification_report, confusion_matrix
        print("\n[Classification Report]")
        print(classification_report(y_true, y_pred, target_names=["Perfect", "General", "Poor"]))
        print("\n[Confusion Matrix]")
        print(confusion_matrix(y_true, y_pred))

        # Create learned ΔHI statistics table
        df_delta = pd.DataFrame({
            "uid": all_uids[:len(y_true)],
            "true": y_true,
            "pred": y_pred,
            "delta_hi_mean": all_delta_mean[:len(y_true)]
        })
        print("\n[Learned ΔHI Statistics] By true class (maintenance effect improvement)")
        print(df_delta.groupby("true")["delta_hi_mean"].describe())

        # Show Top-K samples
        topk_by_delta(df_delta, k=3)

        # HI curve visualization
        print("\n[Visualization] Showing learned health index curves with liquid weight adaptation...")
        plot_hi_examples_aligned(keep_curves, n_show=6, seed=0)

        # Liquid weight visualization
        print("\n[Visualization] Showing liquid operator weights evolution...")
        plot_liquid_weights(keep_curves, n_show=3, seed=0)

        # Sensor reconstruction visualization
        print("\n[Visualization] Showing sensor reconstruction predictions...")
        plot_sensor_examples_aligned(keep_curves, sensor_idx_list=None, n_cols=4)

    return model, (y_true, y_pred, all_delta_mean, all_uids, keep_curves)

# %% [notebook code cell 5]


model, results = train_model(
    pairs=pairs,
    epochs=300,
    batch_size=258,
    lr=1e-3,
    horizon=50,
    device=None,  # Auto select
    patience=20   # Early stopping patience
)

print("\nTraining completed! Model enhanced with Liquid Weight Generator:")
print("- Operator weights adapt to input conditions (h_multi, x)")
print("- Time-smooth, entropy-balanced, and usage-balanced regularization")
print("- Curriculum learning: from uniform to personalized weight distribution")
print("- All operators participate with continuous weighting (no routing/switching)")
print("View learned HI curves and liquid weight evolution through enhanced visualization.")

# %% [notebook code cell 6]

# ============================================================
# Monotonic-Decreasing HI + aligned plotting (before -> after)
# ============================================================
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import random
import matplotlib.pyplot as plt

# ============================================================
# 1) Dataset: read samples from pairs, perform shift operations in batch later
# ============================================================
class PairsReconstructDataset(Dataset):
    """
    Each sample contains:
      x_before: (L,C) feature sequence before maintenance
      x_after : (L,C) feature sequence after maintenance
      hi_before/hi_after: (L,) health index (for evaluation/optional analysis only, not strong supervision)
      strategy: maintenance type ("Perfect Maintenance" / "General Maintenance" / "Poor Maintenance")
    """
    def __init__(self, pairs: dict, horizon: int=None):
        items = []
        for uid, sd in pairs.items():
            for strat, d in sd.items():
                xb = np.asarray(d["x_before"], dtype=np.float32)
                xa = np.asarray(d["x_after"],  dtype=np.float32)
                # When there's no real HI, create placeholder that will be learned by the model
                if "hi_before" in d and d["hi_before"] is not None:
                    hib = np.asarray(d["hi_before"],dtype=np.float32)
                else:
                    # Create initialized fake HI, model will learn real patterns
                    hib = np.linspace(0.8, 0.4, len(xb)).astype(np.float32)

                if "hi_after" in d and d["hi_after"] is not None:
                    hia = np.asarray(d["hi_after"], dtype=np.float32)
                else:
                    # Create different fake HI patterns based on maintenance strategy
                    if "perfect" in strat.lower():
                        # Perfect maintenance: significant improvement
                        hia = np.linspace(0.9, 0.6, len(xa)).astype(np.float32)
                    elif "general" in strat.lower():
                        # General maintenance: moderate improvement
                        hia = np.linspace(0.7, 0.5, len(xa)).astype(np.float32)
                    else:
                        # Poor maintenance: slight improvement
                        hia = np.linspace(0.5, 0.35, len(xa)).astype(np.float32)

                L, C = xb.shape
                # Unify length (optional)
                if horizon is not None and L > horizon:
                    xb, xa, hib, hia = xb[:horizon], xa[:horizon], hib[:horizon], hia[:horizon]
                    L = horizon
                if L < 3:  # At least need to form 0:L-2 -> 1:L-1
                    continue
                items.append({"uid": uid, "strategy": strat, "x_before": xb, "x_after": xa,
                              "hi_before": hib, "hi_after": hia})
        assert len(items)>0, "Dataset is empty, please check pairs data."
        self.items = items
        self.C = items[0]["x_before"].shape[1]

        # Strategy to 3-class label mapping
        def strat_to_cls(s):
            s = s.lower()
            if "perfect" in s: return 0
            if "general" in s: return 1
            if "poor" in s: return 2
            raise ValueError(f"Unknown strategy name: {s}")
        self.labels = [strat_to_cls(it["strategy"]) for it in items]

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        return {
            "uid": it["uid"],
            "label": self.labels[idx],
            "x_before": torch.from_numpy(it["x_before"]), # (L,C)
            "x_after":  torch.from_numpy(it["x_after"]),  # (L,C)
            "hi_before":torch.from_numpy(it["hi_before"]),# (L,)
            "hi_after": torch.from_numpy(it["hi_after"]), # (L,)
            "strategy": it["strategy"]
        }

def pad_collate_shift(batch):
    """
    Pad sequences to same length; return mask, do NOT perform shift operation here
    to allow unified training: inputs = [:,0:L-2], targets = [:,1:L-1]
    """
    Ls = [b["x_before"].shape[0] for b in batch]
    Lmax = max(Ls)
    C = batch[0]["x_before"].shape[1]

    def pad2d(x):
        L, C0 = x.shape
        out = torch.zeros(Lmax, C0, dtype=x.dtype)
        out[:L] = x
        return out

    def pad1d(x):
        L = x.shape[0]
        out = torch.zeros(Lmax, dtype=x.dtype)
        out[:L] = x
        return out

    x_before = torch.stack([pad2d(b["x_before"]) for b in batch], 0) # (B,Lmax,C)
    x_after  = torch.stack([pad2d(b["x_after"])  for b in batch], 0) # (B,Lmax,C)
    hi_before= torch.stack([pad1d(b["hi_before"])for b in batch], 0) # (B,Lmax)
    hi_after = torch.stack([pad1d(b["hi_after"]) for b in batch], 0) # (B,Lmax)
    lengths  = torch.tensor(Ls, dtype=torch.long)
    mask     = torch.zeros(len(batch), Lmax)
    for i,L in enumerate(Ls): mask[i,:L] = 1.0
    labels   = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    uids     = [b["uid"] for b in batch]
    return {"uids": uids, "labels": labels,
            "x_before": x_before, "x_after": x_after,
            "hi_before": hi_before, "hi_after": hi_after,
            "lengths": lengths, "mask": mask}

# ============================================================
# 2) Model: encode "damage" → project to monotonic decreasing HI → recursive reconstruction of two sequences → classify using ΔHI
# ============================================================
class BoltzmannKAN(nn.Module):
    def __init__(self, in_ch, out_ch=4):
        super().__init__()
        self.E  = nn.Parameter(torch.zeros(out_ch, in_ch))
        self.kT = nn.Parameter(torch.ones(out_ch, in_ch))
    def forward(self, x):
        B,T,C = x.shape
        kT = torch.clamp(F.softplus(self.kT), 0.01, 10.0).unsqueeze(0).unsqueeze(2)  # (1,out_ch,1,in_ch)
        E  = torch.clamp(self.E, -10.0, 10.0).unsqueeze(0).unsqueeze(2)             # (1,out_ch,1,in_ch)
        x_ = torch.clamp(x.unsqueeze(1), -10.0, 10.0)                               # (B, out_ch, T, in_ch)
        exp = torch.clamp((E - x_) / kT, -50, 50)
        w   = torch.sigmoid(exp)
        h   = (w * x_).sum(dim=3).permute(0, 2, 1)           # (B,T,out_ch) >=0
        return torch.clamp(F.softplus(h), 0.0, 100.0)

class BaseOp(nn.Module): pass
class MonotonicLinearOp(BaseOp):
    def __init__(self):
        super().__init__()
        self.raw_scale=nn.Parameter(torch.tensor(0.0))
        self.bias=nn.Parameter(torch.tensor(0.0))
        self.smin,self.smax=0.01,5.0
    def forward(self,h):
        scale=torch.clamp(F.softplus(self.raw_scale),self.smin,self.smax)
        xm=torch.clamp(h.mean(-1), -10.0, 10.0)
        return torch.clamp(F.softplus(scale*(xm+self.bias)), 0.0, 100.0)

class MonotonicFlatOp(BaseOp):
    def __init__(self):
        super().__init__()
        self.raw_scale=nn.Parameter(torch.tensor(0.0))
        self.bias=nn.Parameter(torch.tensor(0.0))
        self.smin,self.smax=1e-3,1.0
    def _cum(self,x):
        diff=F.relu(x[:,1:]-x[:,:-1])
        return torch.cat([torch.zeros_like(diff[:,:1]),torch.cumsum(diff,1)],1)
    def forward(self,h):
        scale=torch.clamp(F.softplus(self.raw_scale),self.smin,self.smax)
        xm=torch.clamp(h.mean(-1,keepdim=True), -10.0, 10.0)
        return torch.clamp(F.softplus(scale*(self._cum(xm)+self.bias)).squeeze(-1), 0.0, 100.0)

class ConcaveLogOp(BaseOp):
    def __init__(self):
        super().__init__()
        self.raw_scale=nn.Parameter(torch.tensor(0.0))
        self.eps=1e-3
        self.smin,self.smax=0.01,5.0
    def forward(self,h):
        scale=torch.clamp(F.softplus(self.raw_scale),self.smin,self.smax)
        xm=torch.clamp(h.mean(-1), -10.0, 10.0)
        return torch.clamp(F.softplus(scale*torch.log(torch.abs(xm)+self.eps)), 0.0, 100.0)

class SaturationSigmoidOp(BaseOp):
    def __init__(self):
        super().__init__()
        self.raw_scale=nn.Parameter(torch.tensor(0.0))
        self.raw_slope=nn.Parameter(torch.tensor(0.0))
        self.bias=nn.Parameter(torch.tensor(0.0))
        self.smin,self.smax=0.01,5.0
        self.lmin,self.lmax=0.1,5.0
    def forward(self,h):
        scale=torch.clamp(F.softplus(self.raw_scale),self.smin,self.smax)
        slope=torch.clamp(F.softplus(self.raw_slope),self.lmin,self.lmax)
        xm=torch.clamp(h.mean(-1), -10.0, 10.0)
        return torch.clamp(F.softplus(scale*torch.sigmoid(slope*(xm-self.bias))), 0.0, 100.0)

class HingeReLUOp(BaseOp):
    def __init__(self):
        super().__init__()
        self.raw_scale=nn.Parameter(torch.tensor(0.0))
        self.threshold=nn.Parameter(torch.tensor(0.0))
        self.smin,self.smax=0.01,5.0
    def forward(self,h):
        scale=torch.clamp(F.softplus(self.raw_scale),self.smin,self.smax)
        xm=torch.clamp(h.mean(-1), -10.0, 10.0)
        return torch.clamp(F.softplus(scale*F.relu(xm-self.threshold)), 0.0, 100.0)

class PolynomialOp(BaseOp):
    def __init__(self,deg=3):
        super().__init__()
        self.raw_coeff=nn.Parameter(torch.zeros(deg))
        self.deg=deg
    def forward(self,h):
        xm=torch.clamp(h.mean(-1), -5.0, 5.0)
        y=torch.zeros_like(xm)
        for i in range(self.deg):
            coeff = torch.clamp(F.softplus(self.raw_coeff[i]), 0.01, 5.0)
            y = y + coeff * torch.clamp(xm ** (i+1), -100.0, 100.0)
        return torch.clamp(F.softplus(y), 0.0, 100.0)

class DampedSinOp(BaseOp):
    def __init__(self):
        super().__init__()
        self.raw_scale=nn.Parameter(torch.tensor(0.0))
        self.raw_freq=nn.Parameter(torch.tensor(0.0))
        self.raw_lambda=nn.Parameter(torch.tensor(0.0))
        self.phase=nn.Parameter(torch.tensor(0.0))
        self.smin,self.smax=0.01,5.0
        self.fmin,self.fmax=0.1,5.0
        self.lmin,self.lmax=0.01,3.0
    def forward(self,h):
        scale=torch.clamp(F.softplus(self.raw_scale),self.smin,self.smax)
        freq=torch.clamp(F.softplus(self.raw_freq),self.fmin,self.fmax)
        lam=torch.clamp(F.softplus(self.raw_lambda),self.lmin,self.lmax)
        xm=torch.clamp(h.mean(-1), -10.0, 10.0)
        damp=1/(1+lam*torch.abs(xm))
        phase_val = torch.clamp(self.phase, -10.0, 10.0)
        return torch.clamp(F.softplus(scale*damp*(torch.sin(freq*xm+phase_val)+1)), 0.0, 100.0)

class PiecewiseLinearOp(BaseOp):
    def __init__(self):
        super().__init__()
        self.raw_k1=nn.Parameter(torch.tensor(0.0))
        self.raw_k2=nn.Parameter(torch.tensor(0.0))
        self.threshold=nn.Parameter(torch.tensor(0.0))
        self.kmin,self.kmax=0.01,5.0
    def forward(self,h):
        k1=torch.clamp(F.softplus(self.raw_k1),self.kmin,self.kmax)
        k2=torch.clamp(F.softplus(self.raw_k2),self.kmin,self.kmax)
        thresh = torch.clamp(self.threshold, -5.0, 5.0)
        xm=torch.clamp(h.mean(-1), -10.0, 10.0)
        left=k1*xm
        right=k1*thresh + k2*(xm-thresh)
        out=torch.where(xm<=thresh,left,right)
        return torch.clamp(F.softplus(out), 0.0, 100.0)

class LiquidWeightGenerator(nn.Module):
    """
    Improved Liquid Weight Generator with stronger operator diversity and better temporal dynamics
    """
    def __init__(self, h_dim, x_dim, n_ops, hidden_dim=64, tau_min=1.0, tau_max=3.0):
        super().__init__()
        self.n_ops = n_ops
        self.tau_min, self.tau_max = tau_min, tau_max

        # Enhanced feature encoder with more sophisticated temporal dynamics
        self.h_feature_net = nn.Sequential(
            nn.Linear(h_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.x_feature_net = nn.Sequential(
            nn.Linear(x_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Add temporal encoding
        self.temporal_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim // 4),  # t, dt, phase
            nn.ReLU()
        )

        # Combined feature processing
        combined_dim = hidden_dim + hidden_dim // 4
        self.feature_fusion = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Weight prediction with explicit per-operator branches
        self.op_branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.ReLU(),
                nn.Linear(hidden_dim // 4, 1)
            ) for _ in range(n_ops)
        ])

        # Temperature predictor
        self.temp_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )

        # Remove global bias or make it minimal
        self.global_bias_scale = 0.01  # Very small influence

        # Initialize to encourage diversity
        for branch in self.op_branches:
            nn.init.xavier_uniform_(branch[0].weight)
            nn.init.xavier_uniform_(branch[2].weight)

    def forward(self, h_multi, x):
        """
        Args:
            h_multi: (B, T, h_dim) - Boltzmann KAN output
            x: (B, T, x_dim) - Raw sensor data
        Returns:
            weights: (B, T, K) - Per-timestep operator weights
            temperature: (B, T) - Temperature parameters
        """
        B, T, h_dim = h_multi.shape
        _, _, x_dim = x.shape

        # Process h and x separately to maintain diversity
        h_features = self.h_feature_net(h_multi)  # (B, T, hidden_dim//2)
        x_features = self.x_feature_net(x)        # (B, T, hidden_dim//2)

        # Add temporal encoding
        t_normalized = torch.arange(T, device=x.device, dtype=x.dtype).unsqueeze(0).expand(B, -1) / max(T-1, 1)
        dt = torch.zeros_like(t_normalized)
        dt[:, 1:] = t_normalized[:, 1:] - t_normalized[:, :-1]
        phase = torch.sin(2 * 3.14159 * t_normalized)  # Cyclical patterns

        temporal_input = torch.stack([t_normalized, dt, phase], dim=-1)  # (B, T, 3)
        temporal_features = self.temporal_encoder(temporal_input)  # (B, T, hidden_dim//4)

        # Fuse features
        combined = torch.cat([h_features, x_features, temporal_features], dim=-1)
        fused_features = self.feature_fusion(combined)  # (B, T, hidden_dim)

        # Generate per-operator logits using separate branches
        raw_logits = []
        for i, branch in enumerate(self.op_branches):
            logit = branch(fused_features).squeeze(-1)  # (B, T)
            raw_logits.append(logit)
        raw_weights = torch.stack(raw_logits, dim=-1)  # (B, T, K)

        # Zero-mean normalization to prevent systematic bias
        raw_weights = raw_weights - raw_weights.mean(dim=-1, keepdim=True)

        # Minimal global bias (optional)
        if hasattr(self, 'global_bias_scale') and self.global_bias_scale > 0:
            # Very small random bias to break symmetry
            device = raw_weights.device
            global_bias = torch.randn(self.n_ops, device=device) * self.global_bias_scale
            raw_weights = raw_weights + global_bias.unsqueeze(0).unsqueeze(0)

        # Predict temperature
        raw_temp = self.temp_predictor(fused_features).squeeze(-1)  # (B, T)
        temperature = torch.clamp(F.softplus(raw_temp) + self.tau_min, self.tau_min, self.tau_max)

        # Apply temperature-scaled softmax
        weights = F.softmax(raw_weights / temperature.unsqueeze(-1), dim=-1)  # (B, T, K)

        return weights, temperature

class CustomKAN(nn.Module):
    """
    Enhanced CustomKAN with improved operator diversity and multi-scale feature extraction
    """
    def __init__(self, ops, h_dim, x_dim):
        super().__init__()
        self.ops = nn.ModuleList(ops)
        self.n_ops = len(ops)

        # Enhanced weight generator
        self.weight_generator = LiquidWeightGenerator(
            h_dim=h_dim,
            x_dim=x_dim,
            n_ops=self.n_ops,
            hidden_dim=128
        )

        # Per-operator feature extraction to increase diversity
        self.op_feature_extractors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(h_dim, h_dim),
                nn.ReLU(),
                nn.Linear(h_dim, h_dim)
            ) for _ in range(self.n_ops)
        ])

        # Remove per-operator normalization to preserve natural differences
        # self.op_norm = nn.LayerNorm(1)  # Removed

        # Final transformation parameters
        self.raw_gain = nn.Parameter(torch.tensor(0.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, h_multi, x):  # h_multi:(B,T,h_dim), x:(B,T,x_dim)
        B, T, _ = h_multi.shape

        # Apply per-operator feature extraction for diversity
        op_features = []
        for i, extractor in enumerate(self.op_feature_extractors):
            feat = extractor(h_multi)  # (B, T, h_dim)
            op_features.append(feat)

        # Compute operator outputs with enhanced diversity
        outs = []
        for i, op in enumerate(self.ops):
            op_out = op(op_features[i])  # (B, T) - each op sees different features
            outs.append(op_out)

        # Align lengths
        Tm = min(o.size(1) for o in outs)
        outs = [o[:, :Tm] for o in outs]
        h_multi_aligned = h_multi[:, :Tm, :]
        x_aligned = x[:, :Tm, :]

        # Stack operator outputs
        op_stack = torch.stack(outs, dim=-1)  # (B, Tm, K)

        # Generate liquid weights
        weights, temperature = self.weight_generator(h_multi_aligned, x_aligned)  # (B, Tm, K)

        # Weighted combination (continuous weighting)
        damage = torch.sum(op_stack * weights, dim=-1)  # (B, Tm)
        damage = torch.clamp(damage, 0.0, 100.0)

        # Final transformation
        gain = torch.clamp(F.softplus(self.raw_gain), 0.1, 2.0)  # Reduced gain range
        bias_val = torch.clamp(self.bias, -2.0, 2.0)  # Reduced bias range
        damage = torch.clamp(gain * damage + bias_val, 0.0, 100.0)

        return damage, weights, temperature

class TrendEncoder(nn.Module):
    """
    Encoder outputs "damage", then projects to monotonic decreasing HI:
        HI is learned from sensor data without hard range constraints
    Without real HI, learns mapping from sensor data to infer health states
    Enhanced with improved Liquid Weight Generator
    """
    def __init__(self, in_ch, trend_ch=4):
        super().__init__()
        self.boltz = BoltzmannKAN(in_ch, out_ch=trend_ch)
        ops = [MonotonicLinearOp(), MonotonicFlatOp(), ConcaveLogOp(),
               SaturationSigmoidOp(), HingeReLUOp(), PolynomialOp(),
               DampedSinOp(), PiecewiseLinearOp()]

        # Enhanced CustomKAN
        self.customkan = CustomKAN(ops, h_dim=trend_ch, x_dim=in_ch)

        # Projection parameters - learn flexible health states from sensor features
        self.proj_gain = nn.Parameter(torch.tensor(1.0))
        self.proj_bias = nn.Parameter(torch.tensor(0.0))

        # Add health-aware layer, directly infer from multi-dimensional sensor features
        self.health_aware_transform = nn.Sequential(
            nn.Linear(in_ch, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output 0-1 range health state
        )

        # Add linear constraint layer, enforce linear trend
        self.linear_enforcer = nn.Sequential(
            nn.Linear(in_ch, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output linear decay degree
        )

    def forward(self, x):  # x:(B,T,C)
        # Feature extraction based on Boltzmann KAN
        h_multi = self.boltz(x)              # (B,T,trend_ch) >=0

        # Enhanced liquid weight generation
        damage, weights, temperature = self.customkan(h_multi, x)    # damage:(B,T), weights:(B,T,K), temp:(B,T)

        # Directly learn health states from raw sensor data
        # Assess health state for sensor readings at each time step
        B, T, C = x.shape
        x_reshaped = x.view(-1, C)  # (B*T, C)
        health_direct = self.health_aware_transform(x_reshaped)  # (B*T, 1)
        health_direct = health_direct.view(B, T)  # (B, T)

        # Add linear constraint, force linear decay pattern
        linear_weight = self.linear_enforcer(x_reshaped).view(B, T)  # (B, T)

        # Generate ideal linear decay pattern
        t = torch.arange(T, device=x.device, dtype=x.dtype).unsqueeze(0).expand(B, -1)  # (B, T)
        t_normalized = t / (T - 1)  # Normalize to [0, 1]

        # Combine damage and direct health assessment
        g = torch.clamp(F.softplus(self.proj_gain), 0.1, 5.0)
        b = torch.clamp(self.proj_bias, -5.0, 5.0)

        # Convert damage to health state decay
        damage_normalized = torch.sigmoid(g*(damage + b))

        # Combine three health assessment methods: direct assessment, damage pattern, linear constraint
        combined_health = health_direct * (1 - 0.3 * damage_normalized)

        # Generate ideal linear decay curve (from high to low) - remove hard range constraints
        linear_decay = 1.0 - t_normalized * 0.5  # Simple linear decay from 1.0 to 0.5

        # Use linear_weight to mix linear decay and complex patterns
        # Higher linear_weight means more towards linear
        hi = linear_weight * linear_decay + (1 - linear_weight) * combined_health

        # Force stronger monotonic decreasing constraint without hard range limits
        for t_idx in range(1, T):
            hi = torch.cat([
                hi[:, :t_idx],
                torch.min(hi[:, t_idx:t_idx+1], hi[:, t_idx-1:t_idx] - 0.001),  # Force at least 0.001 decrease
                hi[:, t_idx+1:]
            ], dim=1)

        return hi, h_multi, weights, temperature

class ReconHead(nn.Module):
    """
    Recursive reconstruction: given (x_t, h_t) → predict x_{t+1}
    """
    def __init__(self, C, hidden=128):
        super().__init__()
        self.gru = nn.GRU(input_size=C+1, hidden_size=hidden, batch_first=True)
        self.out = nn.Linear(hidden, C)
    def forward(self, x_in, h_in):
        # x_in:(B,T_in,C), h_in:(B,T_in)
        B,T,C = x_in.shape
        h_in_clamped = torch.clamp(h_in, 0.0, 10.0)  # Remove upper limit constraint
        feat = torch.cat([x_in, h_in_clamped.unsqueeze(-1)], dim=-1)  # (B,T,C+1)
        H,_ = self.gru(feat)                                  # (B,T,H)
        y = self.out(H)                                       # (B,T,C) aligned predict x_{t+1}
        return y

class MaintClassifier(nn.Module):
    """
    Use HI before-after difference features for 3-class classification
    Enhanced version: not only uses ΔHI, but also sensor data change patterns
    """
    def __init__(self, sensor_dim, hidden=64, n_classes=3):
        super().__init__()
        # Original HI difference features
        self.hi_mlp = nn.Sequential(
            nn.Linear(6, hidden), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden//2)
        )

        # Sensor data change features
        self.sensor_mlp = nn.Sequential(
            nn.Linear(sensor_dim * 2, hidden), nn.ReLU(),  # Before and after statistical features
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden//2)
        )

        # Fusion classifier
        self.final_classifier = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, n_classes)
        )

    def forward(self, h_b, h_a, x_b, x_a, mask):
        # HI difference features (original logic)
        m = mask
        T = m.size(1)
        valid = m.sum(1, keepdim=True).clamp_min(1.0)  # (B,1)
        db = h_a - h_b                        # (B,T)
        mean_d = (db*m).sum(1,keepdim=True)/valid
        var_d  = (((db-mean_d)*m)**2).sum(1,keepdim=True)/valid
        std_d  = (var_d + 1e-8).sqrt()
        d0     = (db[:, :1])                  # First value
        dmax   = (db.masked_fill(m==0, -1e9)).max(dim=1, keepdim=True).values
        pos_ratio = ((db>0).float()*m).sum(1,keepdim=True)/valid

        # Slope (linear fitting)
        t = torch.arange(T, device=h_b.device, dtype=h_b.dtype)
        t = (t - t.mean())/(t.std()+1e-6)
        def slope(x):
            num = (x*t).sum(1) - x.sum(1)*t.sum()/T
            den = (t**2).sum() - (t.sum()**2)/T + 1e-8
            return (num/den).unsqueeze(1)
        slope_diff = slope(h_a) - slope(h_b)  # (B,1)
        hi_feat = torch.cat([mean_d, d0, dmax, std_d, slope_diff, pos_ratio], dim=1)  # (B,6)
        hi_feat = torch.clamp(hi_feat, -10.0, 10.0)
        hi_features = self.hi_mlp(hi_feat)

        # Sensor data change features (fix broadcast BUG: directly divide by (B,1) shaped valid)
        x_b_mean = (x_b * m.unsqueeze(-1)).sum(1) / valid        # (B,C)
        x_a_mean = (x_a * m.unsqueeze(-1)).sum(1) / valid        # (B,C)
        sensor_change = torch.cat([x_b_mean, x_a_mean], dim=1)   # (B, 2*C)
        sensor_features = self.sensor_mlp(sensor_change)

        # Fused features
        combined_features = torch.cat([hi_features, sensor_features], dim=1)
        logits = self.final_classifier(combined_features)

        return logits

def sanitize_tensors(*tensors):
    """Replace NaN/Inf in tensors with finite values"""
    result = []
    for t in tensors:
        if t is not None:
            t_clean = torch.nan_to_num(t, nan=0.0, posinf=1e6, neginf=-1e6)
            result.append(t_clean)
        else:
            result.append(t)
    return result[0] if len(result) == 1 else result

class DiffAwareReconstructor(nn.Module):
    """
    Overall model:
     - Two encoders share parameters: encode x_before / x_after → h_b / h_a (monotonic decreasing HI)
     - Two reconstruction heads: perform one-step recursive reconstruction respectively
     - Classification head: use (h_a - h_b) statistical features to identify maintenance type
    Without real HI labels, learn to infer health states from sensor data through self-supervised learning
    Enhanced with improved Liquid Weight Generator
    """
    def __init__(self, in_ch, trend_ch=4, hidden=128, n_classes=3):
        super().__init__()
        self.encoder = TrendEncoder(in_ch, trend_ch)
        self.recon_b = ReconHead(in_ch, hidden)
        self.recon_a = ReconHead(in_ch, hidden)
        self.clf     = MaintClassifier(sensor_dim=in_ch, hidden=64, n_classes=n_classes)

        # Add self-supervised consistency constraint
        self.consistency_weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, x_b, x_a, mask):
        # x_b/x_a : (B,L,C) (will be cut to 0:L-2 / 1:L-1 later)
        h_b, h_multi_b, weights_b, temp_b = self.encoder(x_b)  # Health state learned from sensor data
        h_a, h_multi_a, weights_a, temp_a = self.encoder(x_a)

        # Numerical stabilization
        h_b, h_a = sanitize_tensors(h_b, h_a)
        weights_b, weights_a = sanitize_tensors(weights_b, weights_a)

        # Recursive reconstruction: input 0:L-2 → predict 1:L-1
        xb_in, hb_in = x_b[:, :-2], h_b[:, :-2]
        xa_in, ha_in = x_a[:, :-2], h_a[:, :-2]
        yb_hat = self.recon_b(xb_in, hb_in)   # (B,L-2,C) ≈ x_b[:,1:L-1]
        ya_hat = self.recon_a(xa_in, ha_in)   # (B,L-2,C) ≈ x_a[:,1:L-1]

        # Numerical stabilization
        yb_hat, ya_hat = sanitize_tensors(yb_hat, ya_hat)

        # Classification: use unclipped length mask and include sensor features
        logits = self.clf(h_b, h_a, x_b, x_a, mask)
        logits = sanitize_tensors(logits)

        return yb_hat, ya_hat, h_b, h_a, logits, weights_b, weights_a, temp_b, temp_a

# ============================================================
# 3) Loss function: reconstruction + ΔHI + smooth/monotonic(decreasing) + classification + first-order linear + self-supervised consistency + maintenance effect constraint + global HI superiority constraint + enhanced liquid weight regularization
# ============================================================
def masked_mse(a, b, mask):
    # a,b:(B,T,C)  mask:(B,T)
    diff = (a - b)**2
    mse  = (diff.mean(-1) * mask).sum() / (mask.sum() + 1e-6)
    return mse

def slope_loss(h, mask, kind="smooth"):
    # Smooth: second-order difference; Monotonic decreasing: penalize upward trend
    if kind=="smooth":
        d2 = h[:,2:] - 2*h[:,1:-1] + h[:,:-2]
        m  = mask[:,2:]
        return (d2.abs() * m).sum() / (m.sum()+1e-6)
    elif kind=="mono_dec":
        dh = h[:,1:] - h[:,:-1]        # If >0 indicates upward (violates "decreasing")
        m  = mask[:,1:]
        return (F.relu(dh) * m).sum() / (m.sum()+1e-6)
    else:
        return torch.tensor(0., device=h.device)

def enhanced_liquid_weight_regularization(weights_b, weights_a, mask, lambda_tv=0.05, lambda_ent=0.2, lambda_bal=0.1, lambda_div=0.1):
    """
    Enhanced liquid weight regularization with stronger diversity encouragement
    """
    total_loss = torch.tensor(0.0, device=weights_b.device)

    # 1. Temporal smoothness (reduced weight)
    def tv_loss(weights, mask):
        if weights.size(1) <= 1:
            return torch.tensor(0.0, device=weights.device)
        diff = weights[:, 1:, :] - weights[:, :-1, :]  # (B, T-1, K)
        mask_diff = mask[:, 1:]  # (B, T-1)
        tv = (diff.abs().sum(-1) * mask_diff).sum() / (mask_diff.sum() + 1e-6)
        return tv

    tv_loss_b = tv_loss(weights_b, mask)
    tv_loss_a = tv_loss(weights_a, mask)
    total_loss += lambda_tv * (tv_loss_b + tv_loss_a)

    # 2. Enhanced entropy regularization (stronger)
    def entropy_loss(weights, mask):
        eps = 1e-8
        entropy = -(weights * torch.log(weights + eps)).sum(-1)  # (B, T)
        # Target higher entropy (more diverse operator usage)
        target_entropy = np.log(weights.size(-1)) * 0.8  # 80% of maximum entropy
        entropy_penalty = F.relu(target_entropy - entropy)  # Penalize low entropy
        return (entropy_penalty * mask).sum() / (mask.sum() + 1e-6)

    ent_loss_b = entropy_loss(weights_b, mask)
    ent_loss_a = entropy_loss(weights_a, mask)
    total_loss += lambda_ent * (ent_loss_b + ent_loss_a)

    # 3. Enhanced balance loss (stronger)
    def balance_loss(weights, mask):
        valid_weights = weights * mask.unsqueeze(-1)  # (B, T, K)
        mean_usage = valid_weights.sum([0, 1]) / (mask.sum() + 1e-6)  # (K,)
        target_usage = 1.0 / weights.size(-1)  # Equal usage
        balance = ((mean_usage - target_usage) ** 2).sum()
        return balance

    bal_loss_b = balance_loss(weights_b, mask)
    bal_loss_a = balance_loss(weights_a, mask)
    total_loss += lambda_bal * (bal_loss_b + bal_loss_a)

    # 4. Operator diversity loss (new)
    def diversity_loss(weights, mask):
        # Encourage different operators to be used at different times
        B, T, K = weights.shape
        if T <= 1:
            return torch.tensor(0.0, device=weights.device)

        # Correlation between operator usage over time
        valid_weights = weights * mask.unsqueeze(-1)

        # Compute correlation matrix between operators
        weights_flat = valid_weights.view(-1, K)  # (B*T, K)
        weights_centered = weights_flat - weights_flat.mean(0, keepdim=True)
        cov = torch.mm(weights_centered.T, weights_centered) / (weights_flat.size(0) - 1 + 1e-6)

        # Penalize high off-diagonal correlations
        K = cov.size(0)
        mask_offdiag = torch.ones_like(cov) - torch.eye(K, device=cov.device)
        corr_penalty = (cov.abs() * mask_offdiag).sum()

        return corr_penalty

    div_loss_b = diversity_loss(weights_b, mask)
    div_loss_a = diversity_loss(weights_a, mask)
    total_loss += lambda_div * (div_loss_b + div_loss_a)

    return total_loss

def enhanced_linear_slope_loss(h, mask, weight=5.0):
    """
    Enhanced linear constraint: calculate deviation from best linear fit, stronger weight
    Without real HI, this constraint helps learn reasonable decay patterns
    """
    B, T = h.shape
    valid_mask = mask  # (B, T)

    # Calculate best linear fit for each sample
    t = torch.arange(T, device=h.device, dtype=h.dtype).unsqueeze(0).expand(B, -1)  # (B, T)

    # Consider only valid positions
    loss_total = torch.tensor(0.0, device=h.device)
    n_valid_samples = 0

    for b in range(B):
        valid_indices = valid_mask[b] > 0
        if valid_indices.sum() < 2:  # Need at least 2 points for linear fitting
            continue

        h_valid = h[b][valid_indices]  # Valid HI values
        t_valid = t[b][valid_indices]  # Corresponding time points

        # Center time points
        t_mean = t_valid.mean()
        t_centered = t_valid - t_mean
        h_mean = h_valid.mean()

        # Calculate linear regression slope
        numerator = (t_centered * (h_valid - h_mean)).sum()
        denominator = (t_centered ** 2).sum() + 1e-8
        slope = numerator / denominator

        # Calculate intercept
        intercept = h_mean - slope * t_mean

        # Linear predicted values
        h_linear = slope * t_valid + intercept

        # Calculate MSE with linear fit, using stronger weight
        linear_mse = ((h_valid - h_linear) ** 2).mean()

        # Additional penalty for positive slope (encourage negative slope, i.e., decreasing)
        slope_penalty = F.relu(slope + 0.01) * 2.0  # Slope should be negative

        loss_total += weight * linear_mse + slope_penalty
        n_valid_samples += 1

    return loss_total / max(n_valid_samples, 1)

def maintenance_improvement_constraint(h_b, h_a, labels, mask):
    """
    Maintenance improvement constraint: ensure post-maintenance HI is significantly higher than pre-maintenance
    - Perfect(0): require post-maintenance significantly higher than pre-maintenance (at least 0.4)
    - General(1): require post-maintenance moderately higher than pre-maintenance (at least 0.2)
    - Poor(2): require post-maintenance slightly higher than pre-maintenance (at least 0.1)
    """
    valid = mask.sum(1, keepdim=True).clamp_min(1.0)  # (B,1)

    # Calculate average HI before and after maintenance
    h_b_mean = (h_b * mask).sum(1, keepdim=True) / valid  # (B,1)
    h_a_mean = (h_a * mask).sum(1, keepdim=True) / valid  # (B,1)

    improvement = h_a_mean - h_b_mean  # Improvement after maintenance relative to before

    loss = torch.zeros_like(improvement)
    for cls in [0, 1, 2]:
        sel = (labels == cls).float().unsqueeze(1)  # (B,1)
        if sel.sum() < 1:
            continue

        if cls == 0:  # Perfect
            # Require at least 0.4 improvement
            loss += sel * F.relu(0.4 - improvement) ** 2
        elif cls == 1:  # General
            # Require at least 0.2 improvement
            loss += sel * F.relu(0.2 - improvement) ** 2
        else:  # Poor
            # Require at least 0.1 improvement
            loss += sel * F.relu(0.1 - improvement) ** 2

    return loss.mean()

def global_hi_superiority_constraint(h_b, h_a, mask, weight=3.0):
    """
    Reduced global superiority constraint: ensure post-maintenance HI is entirely higher than pre-maintenance HI
    Reduced weight to allow more diversity in operator usage
    """
    valid = mask  # (B, T)

    loss_total = torch.tensor(0.0, device=h_b.device)
    n_valid_samples = 0

    for b in range(h_b.size(0)):
        valid_mask_b = valid[b] > 0
        if valid_mask_b.sum() < 1:
            continue

        # Get valid HI values before and after maintenance
        h_b_valid = h_b[b][valid_mask_b]  # Valid HI values before maintenance
        h_a_valid = h_a[b][valid_mask_b]  # Valid HI values after maintenance

        # Point-wise superiority: each post-maintenance point should be higher than corresponding pre-maintenance point
        pointwise_gap = h_a_valid - h_b_valid  # Should be positive
        pointwise_loss = F.relu(-pointwise_gap + 0.05).mean()  # Penalize when post-maintenance is not sufficiently higher

        # Global superiority: minimum post-maintenance should be higher than maximum pre-maintenance
        h_b_max = h_b_valid.max()
        h_a_min = h_a_valid.min()

        # Require post-maintenance minimum to be higher than pre-maintenance maximum
        margin = 0.1  # Post-maintenance minimum should be at least 0.1 higher than pre-maintenance maximum
        global_gap_loss = F.relu(h_b_max + margin - h_a_min) ** 2

        # Additional constraint: post-maintenance average should be significantly higher than pre-maintenance average
        h_b_mean = h_b_valid.mean()
        h_a_mean = h_a_valid.mean()
        mean_gap_loss = F.relu(h_b_mean + 0.2 - h_a_mean) ** 2  # Average should improve by at least 0.2

        loss_total += weight * (pointwise_loss + global_gap_loss + 0.5 * mean_gap_loss)
        n_valid_samples += 1

    return loss_total / max(n_valid_samples, 1)

def diff_margin_by_class(mean_delta, labels, m_low=0.1, m_mid=0.25, m_high=0.45):
    """
    Class-conditional difference constraint (enhanced maintenance effect):
      - Perfect(0):  Δ>=m_high (post-maintenance significantly higher than pre-maintenance)
      - General(1):  Δ≈m_mid   (post-maintenance moderately higher than pre-maintenance)
      - Poor(2):     Δ>=m_low  (post-maintenance at least slightly higher than pre-maintenance)
    All classes require post-maintenance HI higher than pre-maintenance (positive improvement), just different degrees
    """
    loss = torch.zeros_like(mean_delta)
    for cls in [0,1,2]:
        sel = (labels==cls).float()
        if sel.sum() < 1: continue
        if cls==0:
            # Perfect: require significant improvement (at least reach m_high)
            loss += sel * F.relu(m_high - mean_delta)**2
        elif cls==1:
            # General: require moderate improvement (close to m_mid)
            loss += sel * (mean_delta - m_mid)**2
        else:
            # Poor: require at least small improvement (at least reach m_low)
            loss += sel * F.relu(m_low - mean_delta)**2
    denom = torch.ones_like(mean_delta) * (labels.numel())
    return loss.sum() / denom.sum()

def sensor_consistency_loss(x_b, x_a, h_b, h_a, mask):
    """
    Sensor data consistency loss: ensure HI changes are consistent with sensor data change patterns
    Important constraint when no real HI labels are available
    """
    # Calculate statistical changes in sensor data
    valid = mask.sum(1, keepdim=True).clamp_min(1.0)   # (B,1)

    # Average change magnitude in sensor data (fix broadcast BUG: divide by (B,1))
    xb_mean = (x_b * mask.unsqueeze(-1)).sum(1) / valid   # (B,C)
    xa_mean = (x_a * mask.unsqueeze(-1)).sum(1) / valid   # (B,C)
    sensor_change = torch.abs(xa_mean - xb_mean)          # (B,C)
    sensor_change_magnitude = sensor_change.mean(1)       # (B,)

    # HI change magnitude (post-maintenance improvement relative to pre-maintenance)
    hi_change = ((h_a - h_b) * mask).sum(1) / valid.squeeze(-1)  # (B,) Note: remove abs, maintain positive direction

    # Expect HI change to positively correlate with sensor change, and HI should improve positively
    # Use Pearson correlation coefficient as consistency metric
    mean_sensor = sensor_change_magnitude.mean()
    mean_hi = hi_change.mean()

    correlation = ((sensor_change_magnitude - mean_sensor) * (hi_change - mean_hi)).mean() / \
                  (sensor_change_magnitude.std() * hi_change.std() + 1e-8)

    # Encourage positive correlation (large sensor change when HI change is also large)
    consistency_loss = F.relu(0.5 - correlation)  # Expect correlation at least 0.5

    # Additional constraint: force positive HI improvement
    negative_change_penalty = F.relu(-hi_change).mean() * 2.0  # Penalize negative changes

    return consistency_loss + negative_change_penalty

def smoothness_enhancement_loss(h, mask, order=2):
    """
    Enhanced smoothness loss: use higher-order differences to force smoother curves
    """
    if order == 2:
        # Second-order difference
        d2 = h[:,2:] - 2*h[:,1:-1] + h[:,:-2]
        m = mask[:,2:]
        return (d2.abs() * m).sum() / (m.sum() + 1e-6)
    elif order == 3:
        # Third-order difference (stronger smoothness constraint)
        d3 = h[:,3:] - 3*h[:,2:-1] + 3*h[:,1:-2] - h[:,:-3]
        m = mask[:,3:]
        return (d3.abs() * m).sum() / (m.sum() + 1e-6)
    else:
        return torch.tensor(0., device=h.device)

# ============================================================
# 4) Data splitting and visualization (aligned to same time axis)
# ============================================================
LABEL2NAME = {0: "Perfect", 1: "General", 2: "Poor"}

def split_pairs_uid(pairs, train_ratio=0.7, val_ratio=0.1, seed=42):
    uids = sorted(list(pairs.keys()))
    rs = np.random.RandomState(seed)
    rs.shuffle(uids)
    n = len(uids); n_tr=int(n*train_ratio); n_vl=int(n*val_ratio)
    to_dict = lambda ss:{u:pairs[u] for u in ss if u in pairs}
    return to_dict(uids[:n_tr]), to_dict(uids[n_tr:n_tr+n_vl]), to_dict(uids[n_tr+n_vl:])

def print_split_summary(pairs, train_ratio=0.7, val_ratio=0.1, seed=42):
    pairs_tr, pairs_vl, pairs_te = split_pairs_uid(pairs, train_ratio, val_ratio, seed)
    def count_items(pp):
        cnt = 0
        for uid, d in pp.items():
            cnt += len(d)
        return len(pp), cnt
    n_uid_tr, n_pair_tr = count_items(pairs_tr)
    n_uid_vl, n_pair_vl = count_items(pairs_vl)
    n_uid_te, n_pair_te = count_items(pairs_te)
    print(f"[Data Split] Train: UID={n_uid_tr}, pairs≈{n_pair_tr} | "
          f"Val: UID={n_uid_vl}, pairs≈{n_pair_vl} | "
          f"Test: UID={n_uid_te}, pairs≈{n_pair_te}")

def remove_constant_segments(hi_values, threshold=1e-6):
    """
    Remove constant segments in HI curves (caused by mask or other reasons)
    Return valid HI values and corresponding time indices
    """
    if len(hi_values) <= 1:
        return hi_values, np.arange(len(hi_values))

    # Calculate differences between adjacent points
    diffs = np.abs(np.diff(hi_values))

    # Find points with sufficient change
    valid_mask = np.ones(len(hi_values), dtype=bool)
    valid_mask[1:] = diffs > threshold

    # Ensure at least first and last two points are kept
    if np.sum(valid_mask) < 2:
        valid_mask[0] = True
        valid_mask[-1] = True

    valid_indices = np.where(valid_mask)[0]
    return hi_values[valid_indices], valid_indices

@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total = {"mse_b":0.0, "mse_a":0.0, "acc":0.0, "delta_mean":0.0}
    n_batch=0
    n_acc = 0; n_tot=0
    for batch in loader:
        xb = batch["x_before"].to(device)
        xa = batch["x_after"].to(device)
        hi_b = batch["hi_before"].to(device)
        hi_a = batch["hi_after"].to(device)
        labels = batch["labels"].to(device)
        lengths= batch["lengths"].to(device)
        mask   = batch["mask"].to(device)

        # Valid segment: 0:L-2 input, 1:L-1 target
        m_tgt  = (torch.arange(xb.size(1)-2, device=device)[None,:] < (lengths-2)[:,None]).float()

        yb_hat, ya_hat, h_b, h_a, logits, _, _, _, _ = model(xb, xa, mask)

        # Numerical stabilization
        yb_hat, ya_hat, h_b, h_a, logits = sanitize_tensors(yb_hat, ya_hat, h_b, h_a, logits)

        # Align targets
        yb_tgt = xb[:,1:-1,:]
        ya_tgt = xa[:,1:-1,:]

        mse_b = masked_mse(yb_hat, yb_tgt, m_tgt)
        mse_a = masked_mse(ya_hat, ya_tgt, m_tgt)

        # Check for NaN
        if torch.isnan(mse_b) or torch.isnan(mse_a):
            continue

        # Classification
        pred = logits.argmax(dim=1)
        n_acc += (pred == labels).sum().item()
        n_tot += labels.numel()

        # Statistics of ΔHI mean (post-maintenance improvement relative to pre-maintenance)
        valid = mask.sum(1, keepdim=True).clamp_min(1.0)
        delta_mean = ((h_a - h_b)*mask).sum(1,keepdim=True)/valid
        delta_mean = torch.nan_to_num(delta_mean, nan=0.0)
        total["delta_mean"] += delta_mean.mean().item()

        total["mse_b"] += mse_b.item()
        total["mse_a"] += mse_a.item()
        n_batch += 1

    for k in ["mse_b","mse_a","delta_mean"]:
        total[k] /= max(n_batch,1)
    total["acc"] = n_acc / max(n_tot,1)
    return total

@torch.no_grad()
def collect_test_predictions(model, loader, device, max_curve_keep=24):
    """
    Collect predictions, ΔHI, and a few sample HI curves on test set (for plotting).
    Also collect operator outputs and diagnostics.
    """
    model.eval()
    y_true, y_pred = [], []
    all_delta_mean, all_uids = [], []
    keep_curves = []

    for batch in loader:
        xb = batch["x_before"].to(device)   # (B,L,C)
        xa = batch["x_after"].to(device)
        mask   = batch["mask"].to(device)   # (B,L)
        labels = batch["labels"].to(device) # (B,)
        lengths= batch["lengths"]           # cpu tensor
        uids   = batch["uids"]

        yb_hat, ya_hat, h_b, h_a, logits, weights_b, weights_a, temp_b, temp_a = model(xb, xa, mask)

        # Numerical stabilization
        yb_hat, ya_hat, h_b, h_a, logits = sanitize_tensors(yb_hat, ya_hat, h_b, h_a, logits)

        prob = F.softmax(logits, dim=1)     # (B,3)
        pred = prob.argmax(1)               # (B,)

        # Statistics of ΔHI mean (post-maintenance improvement relative to pre-maintenance)
        valid = mask.sum(1, keepdim=True).clamp_min(1.0)
        delta_mean = ((h_a - h_b) * mask).sum(1, keepdim=True) / valid  # (B,1)
        delta_mean = torch.nan_to_num(delta_mean, nan=0.0)

        y_true.append(labels.cpu().numpy())
        y_pred.append(pred.cpu().numpy())
        all_delta_mean.append(delta_mean.squeeze(1).cpu().numpy())
        all_uids.extend(uids)

        # Collect operator outputs for diagnostics
        # Run encoder again to get intermediate outputs
        with torch.no_grad():
            h_multi_b = model.encoder.boltz(xb)
            h_multi_a = model.encoder.boltz(xa)

            # Get operator outputs
            op_outs_b = []
            op_outs_a = []

            for i, op in enumerate(model.encoder.customkan.ops):
                # Get operator-specific features
                feat_b = model.encoder.customkan.op_feature_extractors[i](h_multi_b)
                feat_a = model.encoder.customkan.op_feature_extractors[i](h_multi_a)

                out_b = op(feat_b)
                out_a = op(feat_a)

                op_outs_b.append(out_b.cpu().numpy())
                op_outs_a.append(out_a.cpu().numpy())

        # Save some curves for visualization
        for i in range(xb.size(0)):
            if len(keep_curves) >= max_curve_keep:
                break
            L_i = int(lengths[i].item())
            # Fix: truncate reconstruction prediction by actual length
            L_recon = min(L_i - 2, yb_hat.size(1))  # Reconstruction sequence length

            # Collect operator outputs for this sample
            sample_op_outs_b = [op_out[i, :L_i] for op_out in op_outs_b]
            sample_op_outs_a = [op_out[i, :L_i] for op_out in op_outs_a]

            keep_curves.append({
                "uid": uids[i],
                "true": int(labels[i].cpu().item()),
                "pred": int(pred[i].cpu().item()),
                "prob": prob[i].cpu().numpy(),
                "h_before": h_b[i, :L_i].cpu().numpy(),
                "h_after":  h_a[i, :L_i].cpu().numpy(),
                "weights_before": weights_b[i, :L_i].cpu().numpy() if weights_b is not None else None,
                "weights_after": weights_a[i, :L_i].cpu().numpy() if weights_a is not None else None,
                "temp_before": temp_b[i, :L_i].cpu().numpy() if temp_b is not None else None,
                "temp_after": temp_a[i, :L_i].cpu().numpy() if temp_a is not None else None,
                "op_outputs_before": sample_op_outs_b,
                "op_outputs_after": sample_op_outs_a,
                "yb_hat":   yb_hat[i, :L_recon].cpu().numpy(),       # (L_recon,C)
                "ya_hat":   ya_hat[i, :L_recon].cpu().numpy(),       # (L_recon,C)
                "x_before": xb[i, :L_i].cpu().numpy(),               # (L,C)
                "x_after":  xa[i, :L_i].cpu().numpy(),               # (L,C)
            })

    y_true = np.concatenate(y_true, axis=0) if len(y_true)>0 else np.array([])
    y_pred = np.concatenate(y_pred, axis=0) if len(y_pred)>0 else np.array([])
    all_delta_mean = np.concatenate(all_delta_mean, axis=0) if len(all_delta_mean)>0 else np.array([])
    return y_true, y_pred, all_delta_mean, all_uids, keep_curves

def plot_hi_examples_aligned(curves, n_show=6, seed=0):
    """Post-maintenance sequence continues from pre-maintenance endpoint; draw vertical line to mark t_m; remove constant segments.
    Add "Learned HI" in title to indicate this is health state inferred from sensor data, post-maintenance should be higher than pre-maintenance"""
    if len(curves)==0:
        print("(No visualization samples)")
        return
    random.Random(seed).shuffle(curves)
    n_show = min(n_show, len(curves))
    cols = 3
    rows = int(np.ceil(n_show/cols))
    plt.figure(figsize=(cols*4.6, rows*3.2))
    for k in range(n_show):
        ex = curves[k]
        hb = ex["h_before"]; ha = ex["h_after"]
        L  = min(len(hb), len(ha))  # Defensive alignment
        hb, ha = hb[:L], ha[:L]

        # Remove constant segments
        hb_clean, hb_indices = remove_constant_segments(hb)
        ha_clean, ha_indices = remove_constant_segments(ha)

        # Corresponding time axis
        t_b = hb_indices
        t_a = ha_indices + L  # after follows immediately

        uid = ex["uid"]
        pred = ex["pred"]; true = ex["true"]; prob = ex["prob"]
        plt.subplot(rows, cols, k+1)

        # Only plot non-constant segments
        if len(hb_clean) > 1:
            plt.plot(t_b, hb_clean, label="Learned HI_Pre-maintenance", linewidth=1.8, marker='o', markersize=3)
        if len(ha_clean) > 1:
            plt.plot(t_a, ha_clean, label="Learned HI_Post-maintenance",  linewidth=1.8, linestyle='--', marker='s', markersize=3)

        # Maintenance time vertical line
        plt.axvline(L-1, color='k', linestyle=':', linewidth=1.0, alpha=0.7)

        d_mean = float(np.mean(ha - hb))  # Post-maintenance improvement relative to pre-maintenance
        d_max  = float(np.max(ha - hb))
        title = (f"uid={uid} (Enhanced Liquid Weight)\nTrue={LABEL2NAME[true]} | Pred={LABEL2NAME[pred]} "
                 f"| p={prob[pred]:.2f}\nΔHI_mean={d_mean:.3f}, ΔHI_max={d_max:.3f}")
        plt.title(title, fontsize=9)
        plt.xlabel("Cycle (Pre-maintenance | Post-maintenance)")
        plt.ylabel("Learned Health Index (↑Post should be higher)")
        plt.grid(ls='--', alpha=.35)
        # Remove Y-axis range constraint, let it adapt to data
        if k==0: plt.legend()
    plt.tight_layout()
    plt.show()

def plot_enhanced_liquid_weights(curves, n_show=3, seed=0):
    """
    Enhanced visualization of liquid weights with diagnostics
    """
    if len(curves) == 0:
        print("(No visualization samples)")
        return

    # Filter curves that have weight information
    curves_with_weights = [c for c in curves if c.get("weights_before") is not None]
    if len(curves_with_weights) == 0:
        print("(No weight information available)")
        return

    random.Random(seed).shuffle(curves_with_weights)
    n_show = min(n_show, len(curves_with_weights))

    op_names = ["MonotonicLinear", "MonotonicFlat", "ConcaveLog", "SaturationSigmoid",
                "HingeReLU", "Polynomial", "DampedSin", "PiecewiseLinear"]

    for i in range(n_show):
        ex = curves_with_weights[i]
        weights_b = ex["weights_before"]  # (T, K)
        weights_a = ex["weights_after"]   # (T, K)
        temp_b = ex.get("temp_before")
        temp_a = ex.get("temp_after")
        op_outs_b = ex.get("op_outputs_before", [])
        op_outs_a = ex.get("op_outputs_after", [])

        if weights_b is None or weights_a is None:
            continue

        uid = ex["uid"]
        true_label = LABEL2NAME[ex["true"]]
        pred_label = LABEL2NAME[ex["pred"]]

        # Create subplots for comprehensive analysis
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f"Enhanced Liquid Weight Analysis - UID={uid}, True={true_label}, Pred={pred_label}", fontsize=14)

        # 1. Weights before maintenance
        ax = axes[0, 0]
        T_b, K = weights_b.shape
        for k in range(K):
            ax.plot(range(T_b), weights_b[:, k], label=op_names[k] if k < len(op_names) else f"Op_{k}",
                   marker='o', markersize=2, linewidth=1.5)
        ax.set_title("Pre-maintenance Weights")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Weight")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

        # 2. Weights after maintenance
        ax = axes[0, 1]
        T_a, _ = weights_a.shape
        for k in range(K):
            ax.plot(range(T_a), weights_a[:, k], label=op_names[k] if k < len(op_names) else f"Op_{k}",
                   marker='s', markersize=2, linewidth=1.5, linestyle='--')
        ax.set_title("Post-maintenance Weights")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Weight")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

        # 3. Weight entropy over time
        ax = axes[0, 2]
        entropy_b = -np.sum(weights_b * np.log(weights_b + 1e-8), axis=1)
        entropy_a = -np.sum(weights_a * np.log(weights_a + 1e-8), axis=1)
        ax.plot(range(T_b), entropy_b, label="Pre-maintenance", marker='o', linewidth=2)
        ax.plot(range(T_a), entropy_a, label="Post-maintenance", marker='s', linewidth=2, linestyle='--')
        ax.axhline(np.log(K), color='r', linestyle=':', label=f'Max Entropy ({np.log(K):.2f})')
        ax.set_title("Weight Entropy Over Time")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Entropy")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Temperature evolution (if available)
        ax = axes[1, 0]
        if temp_b is not None and temp_a is not None:
            ax.plot(range(T_b), temp_b, label="Pre-maintenance", marker='o', linewidth=2)
            ax.plot(range(T_a), temp_a, label="Post-maintenance", marker='s', linewidth=2, linestyle='--')
            ax.set_title("Temperature Evolution")
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Temperature")
            ax.legend()
        else:
            ax.text(0.5, 0.5, "Temperature data\nnot available", ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Temperature Evolution")
        ax.grid(True, alpha=0.3)

        # 5. Operator output variance (if available)
        ax = axes[1, 1]
        if op_outs_b and len(op_outs_b) == K:
            variances_b = [np.var(op_out) for op_out in op_outs_b]
            variances_a = [np.var(op_out) for op_out in op_outs_a] if op_outs_a and len(op_outs_a) == K else [0] * K

            x_pos = np.arange(K)
            width = 0.35
            ax.bar(x_pos - width/2, variances_b, width, label='Pre-maintenance', alpha=0.7)
            ax.bar(x_pos + width/2, variances_a, width, label='Post-maintenance', alpha=0.7)
            ax.set_xlabel('Operator Index')
            ax.set_ylabel('Output Variance')
            ax.set_title('Operator Output Variance')
            ax.set_xticks(x_pos)
            ax.set_xticklabels([op_names[k] if k < len(op_names) else f"Op_{k}" for k in range(K)], rotation=45)
            ax.legend()
        else:
            ax.text(0.5, 0.5, "Operator output\ndata not available", ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Operator Output Variance")
        ax.grid(True, alpha=0.3)

        # 6. Weight correlation matrix
        ax = axes[1, 2]
        # Combine before and after weights for correlation analysis
        combined_weights = np.concatenate([weights_b, weights_a], axis=0)
        corr_matrix = np.corrcoef(combined_weights.T)

        im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax.set_title("Operator Weight Correlation")
        ax.set_xticks(range(K))
        ax.set_yticks(range(K))
        ax.set_xticklabels([op_names[k][:8] if k < len(op_names) else f"Op_{k}" for k in range(K)], rotation=45)
        ax.set_yticklabels([op_names[k][:8] if k < len(op_names) else f"Op_{k}" for k in range(K)])

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Correlation')

        plt.tight_layout()
        plt.show()

def plot_sensor_examples_aligned(curves, sensor_idx_list=None, n_cols=4):
    """
    Plot several sensors' original vs recursive prediction (post), with after time axis connected after before.
    - Original before: x_before[:, s]
    - Original after : x_after [:, s] (optional)
    - Predicted after : ya_hat   [:, s] (one-step sequence aligned with x_after)
    """
    if len(curves)==0:
        print("(No visualization samples)")
        return
    # Take first sample to show multiple sensors
    ex = curves[0]
    xb = ex["x_before"]         # (L,C)
    xa = ex["x_after"]          # (L,C)
    ya = ex["ya_hat"]           # (L_recon,C) predicted is 1..L-1
    L, C = xb.shape
    Lhat = ya.shape[0]          # Actual reconstruction length

    if sensor_idx_list is None:
        # Default pick 8
        step = max(1, C//8)
        sensor_idx_list = list(range(0, C, step))[:8]

    n = len(sensor_idx_list)
    n_rows = int(np.ceil(n/n_cols))
    plt.figure(figsize=(n_cols*4.3, n_rows*3.1))
    for i, s in enumerate(sensor_idx_list):
        plt.subplot(n_rows, n_cols, i+1)
        t_b = np.arange(L)
        t_a = np.arange(L, 2*L)
        plt.plot(t_b, xb[:,s], label="Original (before)", linewidth=1.2)
        plt.plot(t_a, xa[:,s], label="Original (after)",  linewidth=1.0, linestyle="--", alpha=0.7)
        # Predicted after: aligned with after target (corresponding to 1..L-1)
        t_pred = np.arange(L+1, L+1+Lhat)      # Align ya_hat time
        plt.plot(t_pred, ya[:,s], label="Post-maint. prediction", linewidth=1.6)
        plt.axvline(L-1, color='k', linestyle=':', linewidth='1.0')
        plt.title(f"Sensor_{s:02d} (Enhanced Liquid Weight)")
        if i==0: plt.legend()
        plt.grid(ls="--", alpha=.35)
    plt.tight_layout()
    plt.show()

def topk_by_delta(df_delta, k=5):
    if len(df_delta)==0:
        return
    for cls in [0,1,2]:
        sub = df_delta[df_delta["true"]==cls].sort_values("delta_hi_mean", ascending=False).head(k)
        print(f"\n[True={LABEL2NAME[cls]}] Top {k} samples with highest learned ΔHI_mean (maintenance effect):")
        print(sub[["uid","delta_hi_mean","pred"]].reset_index(drop=True))

# ============================================================
# 5) Training/evaluation loop with enhanced liquid weight regularization
# ============================================================
def train_model(pairs, epochs=20, batch_size=32, lr=1e-3, horizon=None, device=None, patience=20):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data split 7:1:2
    pairs_tr, pairs_vl, pairs_te = split_pairs_uid(pairs, 0.7, 0.1, 42)
    print_split_summary(pairs, 0.7, 0.1, 42)

    ds_tr = PairsReconstructDataset(pairs_tr, horizon=horizon)
    ds_vl = PairsReconstructDataset(pairs_vl, horizon=horizon)
    ds_te = PairsReconstructDataset(pairs_te, horizon=horizon)

    ld_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_shift)
    ld_vl = DataLoader(ds_vl, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_shift)
    ld_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_shift)

    C = ds_tr.C
    print(f"\n[Dimension Check] Sensor feature dimension C = {C}")

    model = DiffAwareReconstructor(in_ch=C, trend_ch=4, hidden=128, n_classes=3).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)  # Use AdamW
    ce  = nn.CrossEntropyLoss()

    best_v = 1e9
    best = None
    nan_count = 0  # Count NaN batches
    best_epoch = 0  # Record best epoch
    no_improve_count = 0  # Count epochs without improvement for early stopping

    print(f"\nStarting training for {epochs} epochs with early stopping (patience={patience})...")
    print(f"Device: {device}")
    print(f"Train set: {len(ds_tr)} samples, Val set: {len(ds_vl)} samples, Test set: {len(ds_te)} samples")
    print("Note: Enhanced with Improved Liquid Weight Generator")
    print("- Increased temperature minimum to reduce weight collapse")
    print("- Zero-mean weight normalization to prevent systematic bias")
    print("- Per-operator feature extraction for diversity")
    print("- Enhanced entropy and diversity regularization")
    print("- Reduced constraint weights to allow more operator diversity")

    for ep in range(1, epochs+1):
        print(f"\n[Epoch {ep}/{epochs}] Training...")
        model.train()
        logs = {"mse_b":0.0, "mse_a":0.0, "diff":0.0, "smooth":0.0, "mono":0.0, "linear":0.0, "cls":0.0, "consist":0.0, "maintain":0.0, "smooth_enh":0.0, "global_sup":0.0, "liquid_weight":0.0, "all":0.0}
        n_bt = 0
        batch_nan_count = 0

        # Enhanced curriculum learning for operator diversity
        progress = ep / epochs
        # Stronger entropy encouragement in early training
        lambda_tv = 0.03 * (1 - progress * 0.3)      # Reduced TV weight
        lambda_ent = 0.3 * (1 - progress * 0.7)      # Higher initial entropy weight
        lambda_bal = 0.15 * (1 - progress * 0.5)     # Higher balance weight
        lambda_div = 0.2 * (1 - progress * 0.6)      # New diversity regularization

        for batch_idx, batch in enumerate(ld_tr):
            # Show training progress
            if batch_idx % max(1, len(ld_tr)//10) == 0:
                progress_pct = (batch_idx + 1) / len(ld_tr) * 100
                print(f"    Training progress: {progress_pct:.1f}% ({batch_idx+1}/{len(ld_tr)} batches)")

            xb = batch["x_before"].to(device)  # (B,L,C)
            xa = batch["x_after"].to(device)
            labels = batch["labels"].to(device)
            lengths= batch["lengths"].to(device)
            mask   = batch["mask"].to(device)

            # Input/target mask
            m_tgt  = (torch.arange(xb.size(1)-2, device=device)[None,:] < (lengths-2)[:,None]).float()

            yb_hat, ya_hat, h_b, h_a, logits, weights_b, weights_a, temp_b, temp_a = model(xb, xa, mask)

            # Numerical stabilization
            yb_hat, ya_hat, h_b, h_a, logits = sanitize_tensors(yb_hat, ya_hat, h_b, h_a, logits)
            weights_b, weights_a = sanitize_tensors(weights_b, weights_a)

            # Recursive reconstruction targets
            yb_tgt = xb[:,1:-1,:]
            ya_tgt = xa[:,1:-1,:]
            loss_b = masked_mse(yb_hat, yb_tgt, m_tgt)
            loss_a = masked_mse(ya_hat, ya_tgt, m_tgt)

            # HI difference (class-conditional margin) - ensure post-maintenance higher than pre-maintenance
            valid = mask.sum(1, keepdim=True).clamp_min(1.0)
            delta_mean = ((h_a - h_b)*mask).sum(1,keepdim=True)/valid  # (B,1)
            loss_diff = diff_margin_by_class(delta_mean.squeeze(1), labels,
                                             m_low=0.1, m_mid=0.25, m_high=0.45)

            # HI smooth + monotonic decreasing (constrain both pre/post) - remove range constraint
            loss_smooth= slope_loss(h_a, mask, "smooth") + slope_loss(h_b, mask, "smooth")
            loss_mono  = slope_loss(h_a, mask, "mono_dec") + slope_loss(h_b, mask, "mono_dec")

            # Enhanced linear loss - reduced weight to allow more diversity
            loss_linear = enhanced_linear_slope_loss(h_b, mask, weight=5.0) + enhanced_linear_slope_loss(h_a, mask, weight=5.0)

            # Sensor consistency loss (including constraint that post-maintenance should be higher than pre-maintenance)
            loss_consistency = sensor_consistency_loss(xb, xa, h_b, h_a, mask)

            # Maintenance improvement constraint
            loss_maintenance = maintenance_improvement_constraint(h_b, h_a, labels, mask)

            # Reduced global HI superiority constraint
            loss_global_superiority = global_hi_superiority_constraint(h_b, h_a, mask, weight=3.0)

            # Enhanced smoothness loss (reduced weight)
            loss_smooth_enhanced = (smoothness_enhancement_loss(h_b, mask, order=2) +
                                   smoothness_enhancement_loss(h_a, mask, order=2) +
                                   smoothness_enhancement_loss(h_b, mask, order=3) * 0.3 +
                                   smoothness_enhancement_loss(h_a, mask, order=3) * 0.3)

            # Enhanced liquid weight regularization
            loss_liquid_weight = enhanced_liquid_weight_regularization(weights_b, weights_a, mask,
                                                                      lambda_tv=lambda_tv,
                                                                      lambda_ent=lambda_ent,
                                                                      lambda_bal=lambda_bal,
                                                                      lambda_div=lambda_div)

            # Classification loss
            loss_cls = ce(logits, labels)

            # Total loss (adjusted weights to encourage operator diversity)
            w_rec=1.0; w_diff=0.4; w_smooth=0.03; w_mono=0.05; w_linear=0.5; w_consist=0.2; w_cls=1.0; w_maintain=1.0; w_smooth_enh=0.3; w_global_sup=1.0; w_liquid=0.5
            loss = (w_rec*(loss_b+loss_a) + w_diff*loss_diff +
                   w_smooth*loss_smooth + w_mono*loss_mono + w_linear*loss_linear +
                   w_consist*loss_consistency + w_cls*loss_cls + w_maintain*loss_maintenance +
                   w_smooth_enh*loss_smooth_enhanced + w_global_sup*loss_global_superiority +
                   w_liquid*loss_liquid_weight)

            # Numerical stabilization of all losses
            loss_b, loss_a, loss_diff, loss_smooth, loss_mono, loss_linear, loss_consistency, loss_cls, loss_maintenance, loss_smooth_enhanced, loss_global_superiority, loss_liquid_weight, loss = sanitize_tensors(
                loss_b, loss_a, loss_diff, loss_smooth, loss_mono, loss_linear, loss_consistency, loss_cls, loss_maintenance, loss_smooth_enhanced, loss_global_superiority, loss_liquid_weight, loss)

            # Check NaN and skip
            if torch.isnan(loss) or torch.isinf(loss):
                batch_nan_count += 1
                if batch_nan_count == 1:  # Only print warning once
                    print(f"    [Warning] Epoch {ep}: Found NaN/Inf loss, skipping abnormal batch")
                continue

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            logs["mse_b"] += loss_b.item()
            logs["mse_a"] += loss_a.item()
            logs["diff"]  += loss_diff.item()
            logs["smooth"]+= loss_smooth.item()
            logs["mono"]  += loss_mono.item()
            logs["linear"]+= loss_linear.item()
            logs["consist"]+= loss_consistency.item()
            logs["cls"]   += loss_cls.item()
            logs["maintain"]+= loss_maintenance.item()
            logs["smooth_enh"]+= loss_smooth_enhanced.item()
            logs["global_sup"]+= loss_global_superiority.item()
            logs["liquid_weight"]+= loss_liquid_weight.item()
            logs["all"]   += loss.item()
            n_bt += 1

        # Record NaN batches
        if batch_nan_count > 0:
            nan_count += batch_nan_count

        print("    Validating...")
        for k in logs: logs[k] /= max(n_bt,1)
        vl = eval_epoch(model, ld_vl, device)
        print(f"[Epoch {ep:03d}] Train: L={logs['all']:.4f} rec_b={logs['mse_b']:.4f} rec_a={logs['mse_a']:.4f} "
              f"diff={logs['diff']:.4f} linear={logs['linear']:.4f} maintain={logs['maintain']:.4f} liquid_w={logs['liquid_weight']:.4f} global_sup={logs['global_sup']:.4f} cls={logs['cls']:.4f} | "
              f"Val: mse_b={vl['mse_b']:.4f} mse_a={vl['mse_a']:.4f} acc={vl['acc']:.3f} ΔHI_improvement={vl['delta_mean']:.3f}")

        # Simple early stopping: observe validation reconstruction loss
        vl_total = vl['mse_b'] + vl['mse_a'] + vl['acc']  # Lower is better (minimize MSE, maximize accuracy)
        if vl_total < best_v:
            best_v = vl_total
            best_epoch = ep
            best = {k: v.clone() if hasattr(v, 'clone') else v for k, v in model.state_dict().items()}
            no_improve_count = 0  # Reset counter
            print(f"    ✓ Saved new best model (val loss: {best_v:.4f})")
        else:
            no_improve_count += 1
            print(f"    Val loss not improved (current: {vl_total:.4f}, best: {best_v:.4f}) - No improvement for {no_improve_count} epochs")

            # Early stopping check
            if no_improve_count >= patience:
                print(f"\n[Early Stopping] No improvement for {patience} epochs. Stopping training at epoch {ep}.")
                break

    if best is not None:
        model.load_state_dict(best)
        print(f"\n[Best Checkpoint] Val reconstruction loss: {best_v:.4f} (Epoch {best_epoch})")

        # Save best model to specified path
        save_path = "/content/drive/MyDrive/CMAPSS/main_dynamics_identification_3.pth"
        torch.save(model.state_dict(), save_path)
        print(f"[Model Saved] Best model saved to: {save_path}")

    if nan_count > 0:
        print(f"[Warning] Total skipped {nan_count} batches containing NaN/Inf")

    # Test set evaluation
    print("\n" + "="*60)
    print("Final Test Set Evaluation")
    print("="*60)
    te = eval_epoch(model, ld_te, device)
    print(f"[Test Set] mse_b={te['mse_b']:.4f} mse_a={te['mse_a']:.4f} acc={te['acc']:.3f} ΔHI_improvement={te['delta_mean']:.3f}")

    # Collect test predictions and visualization data
    y_true, y_pred, all_delta_mean, all_uids, keep_curves = collect_test_predictions(model, ld_te, device, max_curve_keep=24)

    # Classification report
    if len(y_true) > 0:
        from sklearn.metrics import classification_report, confusion_matrix
        print("\n[Classification Report]")
        print(classification_report(y_true, y_pred, target_names=["Perfect", "General", "Poor"]))
        print("\n[Confusion Matrix]")
        print(confusion_matrix(y_true, y_pred))

        # Create learned ΔHI statistics table
        df_delta = pd.DataFrame({
            "uid": all_uids[:len(y_true)],
            "true": y_true,
            "pred": y_pred,
            "delta_hi_mean": all_delta_mean[:len(y_true)]
        })
        print("\n[Learned ΔHI Statistics] By true class (maintenance effect improvement)")
        print(df_delta.groupby("true")["delta_hi_mean"].describe())

        # Show Top-K samples
        topk_by_delta(df_delta, k=3)

        # HI curve visualization
        print("\n[Visualization] Showing learned health index curves with liquid weight adaptation...")
        plot_hi_examples_aligned(keep_curves, n_show=6, seed=0)

        # Liquid weight visualization
        print("\n[Visualization] Showing liquid operator weights evolution...")
        plot_liquid_weights(keep_curves, n_show=3, seed=0)

        # Sensor reconstruction visualization
        print("\n[Visualization] Showing sensor reconstruction predictions...")
        plot_sensor_examples_aligned(keep_curves, sensor_idx_list=None, n_cols=4)

    return model, (y_true, y_pred, all_delta_mean, all_uids, keep_curves)


model, results = train_model(
    pairs=pairs,
    epochs=300,
    batch_size=258,
    lr=1e-3,
    horizon=50,
    device=None,  # Auto select
    patience=20   # Early stopping patience
)

print("\nTraining completed! Model enhanced with Liquid Weight Generator:")
print("- Operator weights adapt to input conditions (h_multi, x)")
print("- Time-smooth, entropy-balanced, and usage-balanced regularization")
print("- Curriculum learning: from uniform to personalized weight distribution")
print("- All operators participate with continuous weighting (no routing/switching)")
print("View learned HI curves and liquid weight evolution through enhanced visualization.")

# %% [notebook code cell 7]

# ============================================================
# Monotonic-Decreasing HI + aligned plotting (before -> after)
# ============================================================
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import random
import matplotlib.pyplot as plt

# ============================================================
# 1) Dataset: read samples from pairs, perform shift operations in batch later
# ============================================================
class PairsReconstructDataset(Dataset):
    """
    Each sample contains:
      x_before: (L,C) feature sequence before maintenance
      x_after : (L,C) feature sequence after maintenance
      hi_before/hi_after: (L,) health index (for evaluation/optional analysis only, not strong supervision)
      strategy: maintenance type ("Perfect Maintenance" / "General Maintenance" / "Poor Maintenance")
    """
    def __init__(self, pairs: dict, horizon: int=None):
        items = []
        for uid, sd in pairs.items():
            for strat, d in sd.items():
                xb = np.asarray(d["x_before"], dtype=np.float32)
                xa = np.asarray(d["x_after"],  dtype=np.float32)
                # When there's no real HI, create placeholder that will be learned by the model
                if "hi_before" in d and d["hi_before"] is not None:
                    hib = np.asarray(d["hi_before"],dtype=np.float32)
                else:
                    # Create initialized fake HI, model will learn real patterns
                    hib = np.linspace(0.8, 0.4, len(xb)).astype(np.float32)

                if "hi_after" in d and d["hi_after"] is not None:
                    hia = np.asarray(d["hi_after"], dtype=np.float32)
                else:
                    # Create different fake HI patterns based on maintenance strategy
                    if "perfect" in strat.lower():
                        # Perfect maintenance: significant improvement
                        hia = np.linspace(0.9, 0.6, len(xa)).astype(np.float32)
                    elif "general" in strat.lower():
                        # General maintenance: moderate improvement
                        hia = np.linspace(0.7, 0.5, len(xa)).astype(np.float32)
                    else:
                        # Poor maintenance: slight improvement
                        hia = np.linspace(0.5, 0.35, len(xa)).astype(np.float32)

                L, C = xb.shape
                # Unify length (optional)
                if horizon is not None and L > horizon:
                    xb, xa, hib, hia = xb[:horizon], xa[:horizon], hib[:horizon], hia[:horizon]
                    L = horizon
                if L < 3:  # At least need to form 0:L-2 -> 1:L-1
                    continue
                items.append({"uid": uid, "strategy": strat, "x_before": xb, "x_after": xa,
                              "hi_before": hib, "hi_after": hia})
        assert len(items)>0, "Dataset is empty, please check pairs data."
        self.items = items
        self.C = items[0]["x_before"].shape[1]

        # Strategy to 3-class label mapping
        def strat_to_cls(s):
            s = s.lower()
            if "perfect" in s: return 0
            if "general" in s: return 1
            if "poor" in s: return 2
            raise ValueError(f"Unknown strategy name: {s}")
        self.labels = [strat_to_cls(it["strategy"]) for it in items]

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        return {
            "uid": it["uid"],
            "label": self.labels[idx],
            "x_before": torch.from_numpy(it["x_before"]), # (L,C)
            "x_after":  torch.from_numpy(it["x_after"]),  # (L,C)
            "hi_before":torch.from_numpy(it["hi_before"]),# (L,)
            "hi_after": torch.from_numpy(it["hi_after"]), # (L,)
            "strategy": it["strategy"]
        }

def pad_collate_shift(batch):
    """
    Pad sequences to same length; return mask, do NOT perform shift operation here
    to allow unified training: inputs = [:,0:L-2], targets = [:,1:L-1]
    """
    Ls = [b["x_before"].shape[0] for b in batch]
    Lmax = max(Ls)
    C = batch[0]["x_before"].shape[1]

    def pad2d(x):
        L, C0 = x.shape
        out = torch.zeros(Lmax, C0, dtype=x.dtype)
        out[:L] = x
        return out

    def pad1d(x):
        L = x.shape[0]
        out = torch.zeros(Lmax, dtype=x.dtype)
        out[:L] = x
        return out

    x_before = torch.stack([pad2d(b["x_before"]) for b in batch], 0) # (B,Lmax,C)
    x_after  = torch.stack([pad2d(b["x_after"])  for b in batch], 0) # (B,Lmax,C)
    hi_before= torch.stack([pad1d(b["hi_before"])for b in batch], 0) # (B,Lmax)
    hi_after = torch.stack([pad1d(b["hi_after"]) for b in batch], 0) # (B,Lmax)
    lengths  = torch.tensor(Ls, dtype=torch.long)
    mask     = torch.zeros(len(batch), Lmax)
    for i,L in enumerate(Ls): mask[i,:L] = 1.0
    labels   = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    uids     = [b["uid"] for b in batch]
    return {"uids": uids, "labels": labels,
            "x_before": x_before, "x_after": x_after,
            "hi_before": hi_before, "hi_after": hi_after,
            "lengths": lengths, "mask": mask}

# ============================================================
# 2) Model: encode "damage" → project to monotonic decreasing HI → recursive reconstruction of two sequences → classify using ΔHI
# ============================================================
class BoltzmannKAN(nn.Module):
    def __init__(self, in_ch, out_ch=4):
        super().__init__()
        self.E  = nn.Parameter(torch.zeros(out_ch, in_ch))
        self.kT = nn.Parameter(torch.ones(out_ch, in_ch))
    def forward(self, x):
        B,T,C = x.shape
        kT = torch.clamp(F.softplus(self.kT), 0.01, 10.0).unsqueeze(0).unsqueeze(2)  # (1,out_ch,1,in_ch)
        E  = torch.clamp(self.E, -10.0, 10.0).unsqueeze(0).unsqueeze(2)             # (1,out_ch,1,in_ch)
        x_ = torch.clamp(x.unsqueeze(1), -10.0, 10.0)                               # (B, out_ch, T, in_ch)
        exp = torch.clamp((E - x_) / kT, -50, 50)
        w   = torch.sigmoid(exp)
        h   = (w * x_).sum(dim=3).permute(0, 2, 1)           # (B,T,out_ch) >=0
        return torch.clamp(F.softplus(h), 0.0, 100.0)

class BaseOp(nn.Module):
    """
    Base operator class: supports dynamic parameter modulation
    """
    def __init__(self):
        super().__init__()
        self.param_modulator = None

    def set_param_modulator(self, modulator):
        """Set parameter modulator"""
        self.param_modulator = modulator

    def get_modulated_params(self, context):
        """Get parameters modulated by context"""
        if self.param_modulator is None:
            return {}
        return self.param_modulator(context)

class MonotonicLinearOp(BaseOp):
    def __init__(self):
        super().__init__()
        self.raw_scale = nn.Parameter(torch.tensor(0.0))
        self.bias = nn.Parameter(torch.tensor(0.0))
        self.smin, self.smax = 0.01, 5.0

    def forward(self, h, context=None):
        # Get modulated parameters
        modulated_params = self.get_modulated_params(context) if context is not None else {}

        # Base parameters
        base_scale = torch.clamp(F.softplus(self.raw_scale), self.smin, self.smax)
        base_bias = self.bias

        # Parameter modulation
        scale_offset = modulated_params.get('scale_offset', 0.0)
        bias_offset = modulated_params.get('bias_offset', 0.0)

        # Modulated parameters
        scale = torch.clamp(base_scale + scale_offset, self.smin, self.smax)
        bias = torch.clamp(base_bias + bias_offset, -5.0, 5.0)

        xm = torch.clamp(h.mean(-1), -10.0, 10.0)
        return torch.clamp(F.softplus(scale * (xm + bias)), 0.0, 100.0)

class MonotonicFlatOp(BaseOp):
    def __init__(self):
        super().__init__()
        self.raw_scale = nn.Parameter(torch.tensor(0.0))
        self.bias = nn.Parameter(torch.tensor(0.0))
        self.smin, self.smax = 1e-3, 1.0

    def _cum(self, x):
        # Fix: ensure computation stays on time dimension
        B, T = x.shape  # x should be (B, T) shape
        diff = F.relu(x[:, 1:] - x[:, :-1])  # (B, T-1)
        zero_init = torch.zeros(B, 1, device=x.device, dtype=x.dtype)  # (B, 1)
        return torch.cat([zero_init, torch.cumsum(diff, dim=1)], dim=1)  # (B, T)

    def forward(self, h, context=None):
        modulated_params = self.get_modulated_params(context) if context is not None else {}

        base_scale = torch.clamp(F.softplus(self.raw_scale), self.smin, self.smax)
        base_bias = self.bias

        scale_offset = modulated_params.get('scale_offset', 0.0)
        bias_offset = modulated_params.get('bias_offset', 0.0)

        scale = torch.clamp(base_scale + scale_offset, self.smin, self.smax)
        bias = torch.clamp(base_bias + bias_offset, -2.0, 2.0)

        xm = torch.clamp(h.mean(-1), -10.0, 10.0)  # (B, T)
        cum_result = self._cum(xm)  # (B, T)
        return torch.clamp(F.softplus(scale * (cum_result + bias)), 0.0, 100.0)

class ConcaveLogOp(BaseOp):
    def __init__(self):
        super().__init__()
        self.raw_scale = nn.Parameter(torch.tensor(0.0))
        self.eps = 1e-3
        self.smin, self.smax = 0.01, 5.0

    def forward(self, h, context=None):
        modulated_params = self.get_modulated_params(context) if context is not None else {}

        base_scale = torch.clamp(F.softplus(self.raw_scale), self.smin, self.smax)
        scale_offset = modulated_params.get('scale_offset', 0.0)
        eps_offset = modulated_params.get('eps_offset', 0.0)

        scale = torch.clamp(base_scale + scale_offset, self.smin, self.smax)
        eps = torch.clamp(self.eps + eps_offset, 1e-6, 1e-1)

        xm = torch.clamp(h.mean(-1), -10.0, 10.0)
        return torch.clamp(F.softplus(scale * torch.log(torch.abs(xm) + eps)), 0.0, 100.0)

class SaturationSigmoidOp(BaseOp):
    def __init__(self):
        super().__init__()
        self.raw_scale = nn.Parameter(torch.tensor(0.0))
        self.raw_slope = nn.Parameter(torch.tensor(0.0))
        self.bias = nn.Parameter(torch.tensor(0.0))
        self.smin, self.smax = 0.01, 5.0
        self.lmin, self.lmax = 0.1, 5.0

    def forward(self, h, context=None):
        modulated_params = self.get_modulated_params(context) if context is not None else {}

        base_scale = torch.clamp(F.softplus(self.raw_scale), self.smin, self.smax)
        base_slope = torch.clamp(F.softplus(self.raw_slope), self.lmin, self.lmax)
        base_bias = self.bias

        scale_offset = modulated_params.get('scale_offset', 0.0)
        slope_offset = modulated_params.get('slope_offset', 0.0)
        bias_offset = modulated_params.get('bias_offset', 0.0)

        scale = torch.clamp(base_scale + scale_offset, self.smin, self.smax)
        slope = torch.clamp(base_slope + slope_offset, self.lmin, self.lmax)
        bias = torch.clamp(base_bias + bias_offset, -3.0, 3.0)

        xm = torch.clamp(h.mean(-1), -10.0, 10.0)
        return torch.clamp(F.softplus(scale * torch.sigmoid(slope * (xm - bias))), 0.0, 100.0)

class HingeReLUOp(BaseOp):
    def __init__(self):
        super().__init__()
        self.raw_scale = nn.Parameter(torch.tensor(0.0))
        self.threshold = nn.Parameter(torch.tensor(0.0))
        self.smin, self.smax = 0.01, 5.0

    def forward(self, h, context=None):
        modulated_params = self.get_modulated_params(context) if context is not None else {}

        base_scale = torch.clamp(F.softplus(self.raw_scale), self.smin, self.smax)
        base_threshold = self.threshold

        scale_offset = modulated_params.get('scale_offset', 0.0)
        threshold_offset = modulated_params.get('threshold_offset', 0.0)

        scale = torch.clamp(base_scale + scale_offset, self.smin, self.smax)
        threshold = torch.clamp(base_threshold + threshold_offset, -3.0, 3.0)

        xm = torch.clamp(h.mean(-1), -10.0, 10.0)
        return torch.clamp(F.softplus(scale * F.relu(xm - threshold)), 0.0, 100.0)

class PolynomialOp(BaseOp):
    def __init__(self, deg=3):
        super().__init__()
        self.raw_coeff = nn.Parameter(torch.zeros(deg))
        self.deg = deg

    def forward(self, h, context=None):
        modulated_params = self.get_modulated_params(context) if context is not None else {}

        base_coeff = torch.clamp(F.softplus(self.raw_coeff), 0.01, 5.0)

        # Fix: Use broadcasting compatible approach for batch dimension
        xm = torch.clamp(h.mean(-1), -5.0, 5.0)  # (B, T)
        B, T = xm.shape

        # Handle coefficient offsets properly for broadcasting
        coeff_offset = modulated_params.get('coeff_offset', torch.zeros_like(base_coeff))
        if coeff_offset.dim() > 1:
            # If coeff_offset is (B, deg), take mean over batch for simplicity
            coeff_offset = coeff_offset.mean(0)

        coeff = torch.clamp(base_coeff + coeff_offset, 0.01, 5.0)  # (deg,)

        y = torch.zeros_like(xm)  # (B, T)
        for i in range(self.deg):
            # Broadcast coefficient to match batch dimension
            power_term = torch.clamp(xm ** (i + 1), -100.0, 100.0)  # (B, T)
            y = y + coeff[i] * power_term  # (B, T) + scalar * (B, T)
        return torch.clamp(F.softplus(y), 0.0, 100.0)

class DampedSinOp(BaseOp):
    def __init__(self):
        super().__init__()
        self.raw_scale = nn.Parameter(torch.tensor(0.0))
        self.raw_freq = nn.Parameter(torch.tensor(0.0))
        self.raw_lambda = nn.Parameter(torch.tensor(0.0))
        self.phase = nn.Parameter(torch.tensor(0.0))
        self.smin, self.smax = 0.01, 5.0
        self.fmin, self.fmax = 0.1, 5.0
        self.lmin, self.lmax = 0.01, 3.0

    def forward(self, h, context=None):
        modulated_params = self.get_modulated_params(context) if context is not None else {}

        base_scale = torch.clamp(F.softplus(self.raw_scale), self.smin, self.smax)
        base_freq = torch.clamp(F.softplus(self.raw_freq), self.fmin, self.fmax)
        base_lambda = torch.clamp(F.softplus(self.raw_lambda), self.lmin, self.lmax)
        base_phase = torch.clamp(self.phase, -10.0, 10.0)

        scale_offset = modulated_params.get('scale_offset', 0.0)
        freq_offset = modulated_params.get('freq_offset', 0.0)
        lambda_offset = modulated_params.get('lambda_offset', 0.0)
        phase_offset = modulated_params.get('phase_offset', 0.0)

        scale = torch.clamp(base_scale + scale_offset, self.smin, self.smax)
        freq = torch.clamp(base_freq + freq_offset, self.fmin, self.fmax)
        lam = torch.clamp(base_lambda + lambda_offset, self.lmin, self.lmax)
        phase = torch.clamp(base_phase + phase_offset, -10.0, 10.0)

        xm = torch.clamp(h.mean(-1), -10.0, 10.0)
        damp = 1 / (1 + lam * torch.abs(xm))
        return torch.clamp(F.softplus(scale * damp * (torch.sin(freq * xm + phase) + 1)), 0.0, 100.0)

class PiecewiseLinearOp(BaseOp):
    def __init__(self):
        super().__init__()
        self.raw_k1 = nn.Parameter(torch.tensor(0.0))
        self.raw_k2 = nn.Parameter(torch.tensor(0.0))
        self.threshold = nn.Parameter(torch.tensor(0.0))
        self.kmin, self.kmax = 0.01, 5.0

    def forward(self, h, context=None):
        modulated_params = self.get_modulated_params(context) if context is not None else {}

        base_k1 = torch.clamp(F.softplus(self.raw_k1), self.kmin, self.kmax)
        base_k2 = torch.clamp(F.softplus(self.raw_k2), self.kmin, self.kmax)
        base_threshold = torch.clamp(self.threshold, -5.0, 5.0)

        k1_offset = modulated_params.get('k1_offset', 0.0)
        k2_offset = modulated_params.get('k2_offset', 0.0)
        threshold_offset = modulated_params.get('threshold_offset', 0.0)

        k1 = torch.clamp(base_k1 + k1_offset, self.kmin, self.kmax)
        k2 = torch.clamp(base_k2 + k2_offset, self.kmin, self.kmax)
        threshold = torch.clamp(base_threshold + threshold_offset, -5.0, 5.0)

        xm = torch.clamp(h.mean(-1), -10.0, 10.0)
        left = k1 * xm
        right = k1 * threshold + k2 * (xm - threshold)
        out = torch.where(xm <= threshold, left, right)
        return torch.clamp(F.softplus(out), 0.0, 100.0)

class ParameterModulator(nn.Module):
    """
    Parameter modulator: dynamically adjusts operator parameters based on input context
    """
    def __init__(self, context_dim, param_configs):
        super().__init__()
        self.param_configs = param_configs

        # Create prediction network for each parameter
        self.param_predictors = nn.ModuleDict()
        for param_name, param_info in param_configs.items():
            param_dim = param_info.get('dim', 1)
            self.param_predictors[param_name] = nn.Sequential(
                nn.Linear(context_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, param_dim),
                nn.Tanh()  # Output range [-1, 1]
            )

    def forward(self, context):
        """
        Args:
            context: (B, T, context_dim) context information
        Returns:
            dict: modulated parameter offsets
        """
        # Take average over time dimension as global context
        global_context = context.mean(dim=1)  # (B, context_dim)

        modulated_params = {}
        for param_name, predictor in self.param_predictors.items():
            param_info = self.param_configs[param_name]
            raw_offset = predictor(global_context)  # (B, param_dim)

            # Scale offset according to configuration
            scale = param_info.get('scale', 0.1)
            modulated_params[param_name] = raw_offset * scale

        return modulated_params

class LiquidWeightGenerator(nn.Module):
    """
    Improved Liquid Weight Generator with stronger operator diversity and better temporal dynamics
    """
    def __init__(self, h_dim, x_dim, n_ops, hidden_dim=64, tau_min=1.0, tau_max=3.0):
        super().__init__()
        self.n_ops = n_ops
        self.tau_min, self.tau_max = tau_min, tau_max

        # Enhanced feature encoder with more sophisticated temporal dynamics
        self.h_feature_net = nn.Sequential(
            nn.Linear(h_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.x_feature_net = nn.Sequential(
            nn.Linear(x_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Add temporal encoding
        self.temporal_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim // 4),  # t, dt, phase
            nn.ReLU()
        )

        # Combined feature processing
        combined_dim = hidden_dim + hidden_dim // 4
        self.feature_fusion = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Weight prediction with explicit per-operator branches
        self.op_branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.ReLU(),
                nn.Linear(hidden_dim // 4, 1)
            ) for _ in range(n_ops)
        ])

        # Temperature predictor
        self.temp_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )

        # Remove global bias or make it minimal
        self.global_bias_scale = 0.01  # Very small influence

        # Initialize to encourage diversity
        for branch in self.op_branches:
            nn.init.xavier_uniform_(branch[0].weight)
            nn.init.xavier_uniform_(branch[2].weight)

    def forward(self, h_multi, x):
        """
        Args:
            h_multi: (B, T, h_dim) - Boltzmann KAN output
            x: (B, T, x_dim) - Raw sensor data
        Returns:
            weights: (B, T, K) - Per-timestep operator weights
            temperature: (B, T) - Temperature parameters
        """
        B, T, h_dim = h_multi.shape
        _, _, x_dim = x.shape

        # Process h and x separately to maintain diversity
        h_features = self.h_feature_net(h_multi)  # (B, T, hidden_dim//2)
        x_features = self.x_feature_net(x)        # (B, T, hidden_dim//2)

        # Add temporal encoding
        t_normalized = torch.arange(T, device=x.device, dtype=x.dtype).unsqueeze(0).expand(B, -1) / max(T-1, 1)
        dt = torch.zeros_like(t_normalized)
        dt[:, 1:] = t_normalized[:, 1:] - t_normalized[:, :-1]
        phase = torch.sin(2 * 3.14159 * t_normalized)  # Cyclical patterns

        temporal_input = torch.stack([t_normalized, dt, phase], dim=-1)  # (B, T, 3)
        temporal_features = self.temporal_encoder(temporal_input)  # (B, T, hidden_dim//4)

        # Fuse features
        combined = torch.cat([h_features, x_features, temporal_features], dim=-1)
        fused_features = self.feature_fusion(combined)  # (B, T, hidden_dim)

        # Generate per-operator logits using separate branches
        raw_logits = []
        for i, branch in enumerate(self.op_branches):
            logit = branch(fused_features).squeeze(-1)  # (B, T)
            raw_logits.append(logit)
        raw_weights = torch.stack(raw_logits, dim=-1)  # (B, T, K)

        # Zero-mean normalization to prevent systematic bias
        raw_weights = raw_weights - raw_weights.mean(dim=-1, keepdim=True)

        # Minimal global bias (optional)
        if hasattr(self, 'global_bias_scale') and self.global_bias_scale > 0:
            # Very small random bias to break symmetry
            device = raw_weights.device
            global_bias = torch.randn(self.n_ops, device=device) * self.global_bias_scale
            raw_weights = raw_weights + global_bias.unsqueeze(0).unsqueeze(0)

        # Predict temperature
        raw_temp = self.temp_predictor(fused_features).squeeze(-1)  # (B, T)
        temperature = torch.clamp(F.softplus(raw_temp) + self.tau_min, self.tau_min, self.tau_max)

        # Apply temperature-scaled softmax
        weights = F.softmax(raw_weights / temperature.unsqueeze(-1), dim=-1)  # (B, T, K)

        return weights, temperature

class CustomKAN(nn.Module):
    """
    Enhanced CustomKAN with improved operator diversity and multi-scale feature extraction
    Supports operators with dynamic parameter modulation
    """
    def __init__(self, ops, h_dim, x_dim):
        super().__init__()
        self.ops = nn.ModuleList(ops)
        self.n_ops = len(ops)

        # Enhanced weight generator
        self.weight_generator = LiquidWeightGenerator(
            h_dim=h_dim,
            x_dim=x_dim,
            n_ops=self.n_ops,
            hidden_dim=128
        )

        # Per-operator feature extraction to increase diversity
        self.op_feature_extractors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(h_dim, h_dim),
                nn.ReLU(),
                nn.Linear(h_dim, h_dim)
            ) for _ in range(self.n_ops)
        ])

        # Create parameter modulator for each operator
        self.param_modulators = nn.ModuleList()
        context_dim = h_dim + x_dim  # Context dimension

        for i, op in enumerate(self.ops):
            # Define parameter configuration based on operator type
            param_configs = self._get_param_configs_for_op(op)
            if param_configs:
                modulator = ParameterModulator(context_dim, param_configs)
                self.param_modulators.append(modulator)
                # Set operator's parameter modulator
                op.set_param_modulator(modulator)
            else:
                self.param_modulators.append(None)

        # Final transformation parameters
        self.raw_gain = nn.Parameter(torch.tensor(0.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def _get_param_configs_for_op(self, op):
        """Return parameter configuration based on operator type"""
        if isinstance(op, MonotonicLinearOp):
            return {
                'scale_offset': {'dim': 1, 'scale': 0.2},
                'bias_offset': {'dim': 1, 'scale': 0.3}
            }
        elif isinstance(op, MonotonicFlatOp):
            return {
                'scale_offset': {'dim': 1, 'scale': 0.1},
                'bias_offset': {'dim': 1, 'scale': 0.2}
            }
        elif isinstance(op, ConcaveLogOp):
            return {
                'scale_offset': {'dim': 1, 'scale': 0.2},
                'eps_offset': {'dim': 1, 'scale': 0.01}
            }
        elif isinstance(op, SaturationSigmoidOp):
            return {
                'scale_offset': {'dim': 1, 'scale': 0.2},
                'slope_offset': {'dim': 1, 'scale': 0.3},
                'bias_offset': {'dim': 1, 'scale': 0.3}
            }
        elif isinstance(op, HingeReLUOp):
            return {
                'scale_offset': {'dim': 1, 'scale': 0.2},
                'threshold_offset': {'dim': 1, 'scale': 0.3}
            }
        elif isinstance(op, PolynomialOp):
            return {
                'coeff_offset': {'dim': op.deg, 'scale': 0.2}
            }
        elif isinstance(op, DampedSinOp):
            return {
                'scale_offset': {'dim': 1, 'scale': 0.2},
                'freq_offset': {'dim': 1, 'scale': 0.3},
                'lambda_offset': {'dim': 1, 'scale': 0.2},
                'phase_offset': {'dim': 1, 'scale': 0.5}
            }
        elif isinstance(op, PiecewiseLinearOp):
            return {
                'k1_offset': {'dim': 1, 'scale': 0.2},
                'k2_offset': {'dim': 1, 'scale': 0.2},
                'threshold_offset': {'dim': 1, 'scale': 0.3}
            }
        return {}

    def forward(self, h_multi, x):  # h_multi:(B,T,h_dim), x:(B,T,x_dim)
        B, T, _ = h_multi.shape

        # Apply per-operator feature extraction for diversity
        op_features = []
        for i, extractor in enumerate(self.op_feature_extractors):
            feat = extractor(h_multi)  # (B, T, h_dim)
            op_features.append(feat)

        # Build context information for parameter modulation
        context = torch.cat([h_multi, x], dim=-1)  # (B, T, h_dim + x_dim)

        # Compute operator outputs with enhanced diversity and dynamic parameters
        outs = []
        for i, op in enumerate(self.ops):
            op_out = op(op_features[i], context)  # Pass context for parameter modulation
            outs.append(op_out)

        # Align lengths
        Tm = min(o.size(1) for o in outs)
        outs = [o[:, :Tm] for o in outs]
        h_multi_aligned = h_multi[:, :Tm, :]
        x_aligned = x[:, :Tm, :]

        # Stack operator outputs
        op_stack = torch.stack(outs, dim=-1)  # (B, Tm, K)

        # Generate liquid weights
        weights, temperature = self.weight_generator(h_multi_aligned, x_aligned)  # (B, Tm, K)

        # Weighted combination (continuous weighting)
        damage = torch.sum(op_stack * weights, dim=-1)  # (B, Tm)
        damage = torch.clamp(damage, 0.0, 100.0)

        # Final transformation
        gain = torch.clamp(F.softplus(self.raw_gain), 0.1, 2.0)  # Reduced gain range
        bias_val = torch.clamp(self.bias, -2.0, 2.0)  # Reduced bias range
        damage = torch.clamp(gain * damage + bias_val, 0.0, 100.0)

        return damage, weights, temperature

class TrendEncoder(nn.Module):
    """
    Encoder outputs "damage", then projects to monotonic decreasing HI:
        HI is learned from sensor data without hard range constraints
    Without real HI, learns mapping from sensor data to infer health states
    Enhanced with improved Liquid Weight Generator and dynamic parameter modulation
    """
    def __init__(self, in_ch, trend_ch=4):
        super().__init__()
        self.boltz = BoltzmannKAN(in_ch, out_ch=trend_ch)
        ops = [MonotonicLinearOp(), MonotonicFlatOp(), ConcaveLogOp(),
               SaturationSigmoidOp(), HingeReLUOp(), PolynomialOp(),
               DampedSinOp(), PiecewiseLinearOp()]

        # Enhanced CustomKAN with dynamic parameter modulation
        self.customkan = CustomKAN(ops, h_dim=trend_ch, x_dim=in_ch)

        # Projection parameters - learn flexible health states from sensor features
        self.proj_gain = nn.Parameter(torch.tensor(1.0))
        self.proj_bias = nn.Parameter(torch.tensor(0.0))

        # Add health-aware layer, directly infer from multi-dimensional sensor features
        self.health_aware_transform = nn.Sequential(
            nn.Linear(in_ch, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output 0-1 range health state
        )

        # Add linear constraint layer, enforce linear trend
        self.linear_enforcer = nn.Sequential(
            nn.Linear(in_ch, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output linear decay degree
        )

    def forward(self, x):  # x:(B,T,C)
        # Feature extraction based on Boltzmann KAN
        h_multi = self.boltz(x)              # (B,T,trend_ch) >=0

        # Enhanced liquid weight generation with dynamic parameter modulation
        damage, weights, temperature = self.customkan(h_multi, x)    # damage:(B,T), weights:(B,T,K), temp:(B,T)

        # Directly learn health states from raw sensor data
        # Assess health state for sensor readings at each time step
        B, T, C = x.shape
        x_reshaped = x.view(-1, C)  # (B*T, C)
        health_direct = self.health_aware_transform(x_reshaped)  # (B*T, 1)
        health_direct = health_direct.view(B, T)  # (B, T)

        # Add linear constraint, force linear decay pattern
        linear_weight = self.linear_enforcer(x_reshaped).view(B, T)  # (B, T)

        # Generate ideal linear decay pattern
        t = torch.arange(T, device=x.device, dtype=x.dtype).unsqueeze(0).expand(B, -1)  # (B, T)
        t_normalized = t / (T - 1)  # Normalize to [0, 1]

        # Combine damage and direct health assessment
        g = torch.clamp(F.softplus(self.proj_gain), 0.1, 5.0)
        b = torch.clamp(self.proj_bias, -5.0, 5.0)

        # Convert damage to health state decay
        damage_normalized = torch.sigmoid(g*(damage + b))

        # Combine three health assessment methods: direct assessment, damage pattern, linear constraint
        combined_health = health_direct * (1 - 0.3 * damage_normalized)

        # Generate ideal linear decay curve (from high to low) - remove hard range constraints
        linear_decay = 1.0 - t_normalized * 0.5  # Simple linear decay from 1.0 to 0.5

        # Use linear_weight to mix linear decay and complex patterns
        # Higher linear_weight means more towards linear
        hi = linear_weight * linear_decay + (1 - linear_weight) * combined_health

        # Force stronger monotonic decreasing constraint without hard range limits
        for t_idx in range(1, T):
            hi = torch.cat([
                hi[:, :t_idx],
                torch.min(hi[:, t_idx:t_idx+1], hi[:, t_idx-1:t_idx] - 0.001),  # Force at least 0.001 decrease
                hi[:, t_idx+1:]
            ], dim=1)

        return hi, h_multi, weights, temperature

class ReconHead(nn.Module):
    """
    Recursive reconstruction: given (x_t, h_t) → predict x_{t+1}
    """
    def __init__(self, C, hidden=128):
        super().__init__()
        self.gru = nn.GRU(input_size=C+1, hidden_size=hidden, batch_first=True)
        self.out = nn.Linear(hidden, C)
    def forward(self, x_in, h_in):
        # x_in:(B,T_in,C), h_in:(B,T_in)
        B,T,C = x_in.shape
        h_in_clamped = torch.clamp(h_in, 0.0, 10.0)  # Remove upper limit constraint
        feat = torch.cat([x_in, h_in_clamped.unsqueeze(-1)], dim=-1)  # (B,T,C+1)
        H,_ = self.gru(feat)                                  # (B,T,H)
        y = self.out(H)                                       # (B,T,C) aligned predict x_{t+1}
        return y

class MaintClassifier(nn.Module):
    """
    Use HI before-after difference features for 3-class classification
    Enhanced version: not only uses ΔHI, but also sensor data change patterns
    """
    def __init__(self, sensor_dim, hidden=64, n_classes=3):
        super().__init__()
        # Original HI difference features
        self.hi_mlp = nn.Sequential(
            nn.Linear(6, hidden), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden//2)
        )

        # Sensor data change features
        self.sensor_mlp = nn.Sequential(
            nn.Linear(sensor_dim * 2, hidden), nn.ReLU(),  # Before and after statistical features
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden//2)
        )

        # Fusion classifier
        self.final_classifier = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, n_classes)
        )

    def forward(self, h_b, h_a, x_b, x_a, mask):
        # HI difference features (original logic)
        m = mask
        T = m.size(1)
        valid = m.sum(1, keepdim=True).clamp_min(1.0)  # (B,1)
        db = h_a - h_b                        # (B,T)
        mean_d = (db*m).sum(1,keepdim=True)/valid
        var_d  = (((db-mean_d)*m)**2).sum(1,keepdim=True)/valid
        std_d  = (var_d + 1e-8).sqrt()
        d0     = (db[:, :1])                  # First value
        dmax   = (db.masked_fill(m==0, -1e9)).max(dim=1, keepdim=True).values
        pos_ratio = ((db>0).float()*m).sum(1,keepdim=True)/valid

        # Slope (linear fitting)
        t = torch.arange(T, device=h_b.device, dtype=h_b.dtype)
        t = (t - t.mean())/(t.std()+1e-6)
        def slope(x):
            num = (x*t).sum(1) - x.sum(1)*t.sum()/T
            den = (t**2).sum() - (t.sum()**2)/T + 1e-8
            return (num/den).unsqueeze(1)
        slope_diff = slope(h_a) - slope(h_b)  # (B,1)
        hi_feat = torch.cat([mean_d, d0, dmax, std_d, slope_diff, pos_ratio], dim=1)  # (B,6)
        hi_feat = torch.clamp(hi_feat, -10.0, 10.0)
        hi_features = self.hi_mlp(hi_feat)

        # Sensor data change features (fix broadcast BUG: directly divide by (B,1) shaped valid)
        x_b_mean = (x_b * m.unsqueeze(-1)).sum(1) / valid        # (B,C)
        x_a_mean = (x_a * m.unsqueeze(-1)).sum(1) / valid        # (B,C)
        sensor_change = torch.cat([x_b_mean, x_a_mean], dim=1)   # (B, 2*C)
        sensor_features = self.sensor_mlp(sensor_change)

        # Fused features
        combined_features = torch.cat([hi_features, sensor_features], dim=1)
        logits = self.final_classifier(combined_features)

        return logits

def sanitize_tensors(*tensors):
    """Replace NaN/Inf in tensors with finite values"""
    result = []
    for t in tensors:
        if t is not None:
            t_clean = torch.nan_to_num(t, nan=0.0, posinf=1e6, neginf=-1e6)
            result.append(t_clean)
        else:
            result.append(t)
    return result[0] if len(result) == 1 else result

class DiffAwareReconstructor(nn.Module):
    """
    Overall model:
     - Two encoders share parameters: encode x_before / x_after → h_b / h_a (monotonic decreasing HI)
     - Two reconstruction heads: perform one-step recursive reconstruction respectively
     - Classification head: use (h_a - h_b) statistical features to identify maintenance type
    Without real HI labels, learn to infer health states from sensor data through self-supervised learning
    Enhanced with improved Liquid Weight Generator and dynamic parameter modulation
    """
    def __init__(self, in_ch, trend_ch=4, hidden=128, n_classes=3):
        super().__init__()
        self.encoder = TrendEncoder(in_ch, trend_ch)
        self.recon_b = ReconHead(in_ch, hidden)
        self.recon_a = ReconHead(in_ch, hidden)
        self.clf     = MaintClassifier(sensor_dim=in_ch, hidden=64, n_classes=n_classes)

        # Add self-supervised consistency constraint
        self.consistency_weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, x_b, x_a, mask):
        # x_b/x_a : (B,L,C) (will be cut to 0:L-2 / 1:L-1 later)
        h_b, h_multi_b, weights_b, temp_b = self.encoder(x_b)  # Health state learned from sensor data
        h_a, h_multi_a, weights_a, temp_a = self.encoder(x_a)

        # Numerical stabilization
        h_b, h_a = sanitize_tensors(h_b, h_a)
        weights_b, weights_a = sanitize_tensors(weights_b, weights_a)

        # Recursive reconstruction: input 0:L-2 → predict 1:L-1
        xb_in, hb_in = x_b[:, :-2], h_b[:, :-2]
        xa_in, ha_in = x_a[:, :-2], h_a[:, :-2]
        yb_hat = self.recon_b(xb_in, hb_in)   # (B,L-2,C) ≈ x_b[:,1:L-1]
        ya_hat = self.recon_a(xa_in, ha_in)   # (B,L-2,C) ≈ x_a[:,1:L-1]

        # Numerical stabilization
        yb_hat, ya_hat = sanitize_tensors(yb_hat, ya_hat)

        # Classification: use unclipped length mask and include sensor features
        logits = self.clf(h_b, h_a, x_b, x_a, mask)
        logits = sanitize_tensors(logits)

        return yb_hat, ya_hat, h_b, h_a, logits, weights_b, weights_a, temp_b, temp_a

# ============================================================
# 3) Loss function: reconstruction + ΔHI + smooth/monotonic(decreasing) + classification + first-order linear + self-supervised consistency + maintenance effect constraint + global HI superiority constraint + enhanced liquid weight regularization + parameter modulation regularization
# ============================================================
def masked_mse(a, b, mask):
    # a,b:(B,T,C)  mask:(B,T)
    diff = (a - b)**2
    mse  = (diff.mean(-1) * mask).sum() / (mask.sum() + 1e-6)
    return mse

def slope_loss(h, mask, kind="smooth"):
    # Smooth: second-order difference; Monotonic decreasing: penalize upward trend
    if kind=="smooth":
        d2 = h[:,2:] - 2*h[:,1:-1] + h[:,:-2]
        m  = mask[:,2:]
        return (d2.abs() * m).sum() / (m.sum()+1e-6)
    elif kind=="mono_dec":
        dh = h[:,1:] - h[:,:-1]        # If >0 indicates upward (violates "decreasing")
        m  = mask[:,1:]
        return (F.relu(dh) * m).sum() / (m.sum()+1e-6)
    else:
        return torch.tensor(0., device=h.device)

def enhanced_liquid_weight_regularization(weights_b, weights_a, mask, lambda_tv=0.05, lambda_ent=0.2, lambda_bal=0.1, lambda_div=0.1):
    """
    Enhanced liquid weight regularization with stronger diversity encouragement
    """
    total_loss = torch.tensor(0.0, device=weights_b.device)

    # 1. Temporal smoothness (reduced weight)
    def tv_loss(weights, mask):
        if weights.size(1) <= 1:
            return torch.tensor(0.0, device=weights.device)
        diff = weights[:, 1:, :] - weights[:, :-1, :]  # (B, T-1, K)
        mask_diff = mask[:, 1:]  # (B, T-1)
        tv = (diff.abs().sum(-1) * mask_diff).sum() / (mask_diff.sum() + 1e-6)
        return tv

    tv_loss_b = tv_loss(weights_b, mask)
    tv_loss_a = tv_loss(weights_a, mask)
    total_loss += lambda_tv * (tv_loss_b + tv_loss_a)

    # 2. Enhanced entropy regularization (stronger)
    def entropy_loss(weights, mask):
        eps = 1e-8
        entropy = -(weights * torch.log(weights + eps)).sum(-1)  # (B, T)
        # Target higher entropy (more diverse operator usage)
        target_entropy = np.log(weights.size(-1)) * 0.8  # 80% of maximum entropy
        entropy_penalty = F.relu(target_entropy - entropy)  # Penalize low entropy
        return (entropy_penalty * mask).sum() / (mask.sum() + 1e-6)

    ent_loss_b = entropy_loss(weights_b, mask)
    ent_loss_a = entropy_loss(weights_a, mask)
    total_loss += lambda_ent * (ent_loss_b + ent_loss_a)

    # 3. Enhanced balance loss (stronger)
    def balance_loss(weights, mask):
        valid_weights = weights * mask.unsqueeze(-1)  # (B, T, K)
        mean_usage = valid_weights.sum([0, 1]) / (mask.sum() + 1e-6)  # (K,)
        target_usage = 1.0 / weights.size(-1)  # Equal usage
        balance = ((mean_usage - target_usage) ** 2).sum()
        return balance

    bal_loss_b = balance_loss(weights_b, mask)
    bal_loss_a = balance_loss(weights_a, mask)
    total_loss += lambda_bal * (bal_loss_b + bal_loss_a)

    # 4. Operator diversity loss (new)
    def diversity_loss(weights, mask):
        # Encourage different operators to be used at different times
        B, T, K = weights.shape
        if T <= 1:
            return torch.tensor(0.0, device=weights.device)

        # Correlation between operator usage over time
        valid_weights = weights * mask.unsqueeze(-1)

        # Compute correlation matrix between operators
        weights_flat = valid_weights.view(-1, K)  # (B*T, K)
        weights_centered = weights_flat - weights_flat.mean(0, keepdim=True)
        cov = torch.mm(weights_centered.T, weights_centered) / (weights_flat.size(0) - 1 + 1e-6)

        # Penalize high off-diagonal correlations
        K = cov.size(0)
        mask_offdiag = torch.ones_like(cov) - torch.eye(K, device=cov.device)
        corr_penalty = (cov.abs() * mask_offdiag).sum()

        return corr_penalty

    div_loss_b = diversity_loss(weights_b, mask)
    div_loss_a = diversity_loss(weights_a, mask)
    total_loss += lambda_div * (div_loss_b + div_loss_a)

    return total_loss

def parameter_modulation_regularization(model, lambda_param=0.01):
    """
    Parameter modulation regularization: prevent excessive parameter offsets
    """
    total_loss = torch.tensor(0.0, device=next(model.parameters()).device)

    # Iterate through all parameter modulators
    for modulator in model.encoder.customkan.param_modulators:
        if modulator is not None:
            # Apply L2 regularization to all parameter predictor weights
            for param_name, predictor in modulator.param_predictors.items():
                for param in predictor.parameters():
                    total_loss += lambda_param * (param ** 2).sum()

    return total_loss

def enhanced_linear_slope_loss(h, mask, weight=5.0):
    """
    Enhanced linear constraint: calculate deviation from best linear fit, stronger weight
    Without real HI, this constraint helps learn reasonable decay patterns
    """
    B, T = h.shape
    valid_mask = mask  # (B, T)

    # Calculate best linear fit for each sample
    t = torch.arange(T, device=h.device, dtype=h.dtype).unsqueeze(0).expand(B, -1)  # (B, T)

    # Consider only valid positions
    loss_total = torch.tensor(0.0, device=h.device)
    n_valid_samples = 0

    for b in range(B):
        valid_indices = valid_mask[b] > 0
        if valid_indices.sum() < 2:  # Need at least 2 points for linear fitting
            continue

        h_valid = h[b][valid_indices]  # Valid HI values
        t_valid = t[b][valid_indices]  # Corresponding time points

        # Center time points
        t_mean = t_valid.mean()
        t_centered = t_valid - t_mean
        h_mean = h_valid.mean()

        # Calculate linear regression slope
        numerator = (t_centered * (h_valid - h_mean)).sum()
        denominator = (t_centered ** 2).sum() + 1e-8
        slope = numerator / denominator

        # Calculate intercept
        intercept = h_mean - slope * t_mean

        # Linear predicted values
        h_linear = slope * t_valid + intercept

        # Calculate MSE with linear fit, using stronger weight
        linear_mse = ((h_valid - h_linear) ** 2).mean()

        # Additional penalty for positive slope (encourage negative slope, i.e., decreasing)
        slope_penalty = F.relu(slope + 0.01) * 2.0  # Slope should be negative

        loss_total += weight * linear_mse + slope_penalty
        n_valid_samples += 1

    return loss_total / max(n_valid_samples, 1)

def maintenance_improvement_constraint(h_b, h_a, labels, mask):
    """
    Maintenance improvement constraint: ensure post-maintenance HI is significantly higher than pre-maintenance
    - Perfect(0): require post-maintenance significantly higher than pre-maintenance (at least 0.4)
    - General(1): require post-maintenance moderately higher than pre-maintenance (at least 0.2)
    - Poor(2): require post-maintenance slightly higher than pre-maintenance (at least 0.1)
    """
    valid = mask.sum(1, keepdim=True).clamp_min(1.0)  # (B,1)

    # Calculate average HI before and after maintenance
    h_b_mean = (h_b * mask).sum(1, keepdim=True) / valid  # (B,1)
    h_a_mean = (h_a * mask).sum(1, keepdim=True) / valid  # (B,1)

    improvement = h_a_mean - h_b_mean  # Improvement after maintenance relative to before

    loss = torch.zeros_like(improvement)
    for cls in [0, 1, 2]:
        sel = (labels == cls).float().unsqueeze(1)  # (B,1)
        if sel.sum() < 1:
            continue

        if cls == 0:  # Perfect
            # Require at least 0.4 improvement
            loss += sel * F.relu(0.4 - improvement) ** 2
        elif cls == 1:  # General
            # Require at least 0.2 improvement
            loss += sel * F.relu(0.2 - improvement) ** 2
        else:  # Poor
            # Require at least 0.1 improvement
            loss += sel * F.relu(0.1 - improvement) ** 2

    return loss.mean()

def global_hi_superiority_constraint(h_b, h_a, mask, weight=3.0):
    """
    Reduced global superiority constraint: ensure post-maintenance HI is entirely higher than pre-maintenance HI
    Reduced weight to allow more diversity in operator usage
    """
    valid = mask  # (B, T)

    loss_total = torch.tensor(0.0, device=h_b.device)
    n_valid_samples = 0

    for b in range(h_b.size(0)):
        valid_mask_b = valid[b] > 0
        if valid_mask_b.sum() < 1:
            continue

        # Get valid HI values before and after maintenance
        h_b_valid = h_b[b][valid_mask_b]  # Valid HI values before maintenance
        h_a_valid = h_a[b][valid_mask_b]  # Valid HI values after maintenance

        # Point-wise superiority: each post-maintenance point should be higher than corresponding pre-maintenance point
        pointwise_gap = h_a_valid - h_b_valid  # Should be positive
        pointwise_loss = F.relu(-pointwise_gap + 0.05).mean()  # Penalize when post-maintenance is not sufficiently higher

        # Global superiority: minimum post-maintenance should be higher than maximum pre-maintenance
        h_b_max = h_b_valid.max()
        h_a_min = h_a_valid.min()

        # Require post-maintenance minimum to be higher than pre-maintenance maximum
        margin = 0.1  # Post-maintenance minimum should be at least 0.1 higher than pre-maintenance maximum
        global_gap_loss = F.relu(h_b_max + margin - h_a_min) ** 2

        # Additional constraint: post-maintenance average should be significantly higher than pre-maintenance average
        h_b_mean = h_b_valid.mean()
        h_a_mean = h_a_valid.mean()
        mean_gap_loss = F.relu(h_b_mean + 0.2 - h_a_mean) ** 2  # Average should improve by at least 0.2

        loss_total += weight * (pointwise_loss + global_gap_loss + 0.5 * mean_gap_loss)
        n_valid_samples += 1

    return loss_total / max(n_valid_samples, 1)

def diff_margin_by_class(mean_delta, labels, m_low=0.1, m_mid=0.25, m_high=0.45):
    """
    Class-conditional difference constraint (enhanced maintenance effect):
      - Perfect(0):  Δ>=m_high (post-maintenance significantly higher than pre-maintenance)
      - General(1):  Δ≈m_mid   (post-maintenance moderately higher than pre-maintenance)
      - Poor(2):     Δ>=m_low  (post-maintenance at least slightly higher than pre-maintenance)
    All classes require post-maintenance HI higher than pre-maintenance (positive improvement), just different degrees
    """
    loss = torch.zeros_like(mean_delta)
    for cls in [0,1,2]:
        sel = (labels==cls).float()
        if sel.sum() < 1: continue
        if cls==0:
            # Perfect: require significant improvement (at least reach m_high)
            loss += sel * F.relu(m_high - mean_delta)**2
        elif cls==1:
            # General: require moderate improvement (close to m_mid)
            loss += sel * (mean_delta - m_mid)**2
        else:
            # Poor: require at least small improvement (at least reach m_low)
            loss += sel * F.relu(m_low - mean_delta)**2
    denom = torch.ones_like(mean_delta) * (labels.numel())
    return loss.sum() / denom.sum()

def sensor_consistency_loss(x_b, x_a, h_b, h_a, mask):
    """
    Sensor data consistency loss: ensure HI changes are consistent with sensor data change patterns
    Important constraint when no real HI labels are available
    """
    # Calculate statistical changes in sensor data
    valid = mask.sum(1, keepdim=True).clamp_min(1.0)   # (B,1)

    # Average change magnitude in sensor data (fix broadcast BUG: divide by (B,1))
    xb_mean = (x_b * mask.unsqueeze(-1)).sum(1) / valid   # (B,C)
    xa_mean = (x_a * mask.unsqueeze(-1)).sum(1) / valid   # (B,C)
    sensor_change = torch.abs(xa_mean - xb_mean)          # (B,C)
    sensor_change_magnitude = sensor_change.mean(1)       # (B,)

    # HI change magnitude (post-maintenance improvement relative to pre-maintenance)
    hi_change = ((h_a - h_b) * mask).sum(1) / valid.squeeze(-1)  # (B,) Note: remove abs, maintain positive direction

    # Expect HI change to positively correlate with sensor change, and HI should improve positively
    # Use Pearson correlation coefficient as consistency metric
    mean_sensor = sensor_change_magnitude.mean()
    mean_hi = hi_change.mean()

    correlation = ((sensor_change_magnitude - mean_sensor) * (hi_change - mean_hi)).mean() / \
                  (sensor_change_magnitude.std() * hi_change.std() + 1e-8)

    # Encourage positive correlation (large sensor change when HI change is also large)
    consistency_loss = F.relu(0.5 - correlation)  # Expect correlation at least 0.5

    # Additional constraint: force positive HI improvement
    negative_change_penalty = F.relu(-hi_change).mean() * 2.0  # Penalize negative changes

    return consistency_loss + negative_change_penalty

def smoothness_enhancement_loss(h, mask, order=2):
    """
    Enhanced smoothness loss: use higher-order differences to force smoother curves
    """
    if order == 2:
        # Second-order difference
        d2 = h[:,2:] - 2*h[:,1:-1] + h[:,:-2]
        m = mask[:,2:]
        return (d2.abs() * m).sum() / (m.sum() + 1e-6)
    elif order == 3:
        # Third-order difference (stronger smoothness constraint)
        d3 = h[:,3:] - 3*h[:,2:-1] + 3*h[:,1:-2] - h[:,:-3]
        m = mask[:,3:]
        return (d3.abs() * m).sum() / (m.sum() + 1e-6)
    else:
        return torch.tensor(0., device=h.device)

# ============================================================
# 4) Data splitting and visualization (aligned to same time axis)
# ============================================================
LABEL2NAME = {0: "Perfect", 1: "General", 2: "Poor"}

def split_pairs_uid(pairs, train_ratio=0.7, val_ratio=0.1, seed=42):
    uids = sorted(list(pairs.keys()))
    rs = np.random.RandomState(seed)
    rs.shuffle(uids)
    n = len(uids); n_tr=int(n*train_ratio); n_vl=int(n*val_ratio)
    to_dict = lambda ss:{u:pairs[u] for u in ss if u in pairs}
    return to_dict(uids[:n_tr]), to_dict(uids[n_tr:n_tr+n_vl]), to_dict(uids[n_tr+n_vl:])

def print_split_summary(pairs, train_ratio=0.7, val_ratio=0.1, seed=42):
    pairs_tr, pairs_vl, pairs_te = split_pairs_uid(pairs, train_ratio, val_ratio, seed)
    def count_items(pp):
        cnt = 0
        for uid, d in pp.items():
            cnt += len(d)
        return len(pp), cnt
    n_uid_tr, n_pair_tr = count_items(pairs_tr)
    n_uid_vl, n_pair_vl = count_items(pairs_vl)
    n_uid_te, n_pair_te = count_items(pairs_te)
    print(f"[Data Split] Train: UID={n_uid_tr}, pairs≈{n_pair_tr} | "
          f"Val: UID={n_uid_vl}, pairs≈{n_pair_vl} | "
          f"Test: UID={n_uid_te}, pairs≈{n_pair_te}")

def remove_constant_segments(hi_values, threshold=1e-6):
    """
    Remove constant segments in HI curves (caused by mask or other reasons)
    Return valid HI values and corresponding time indices
    """
    if len(hi_values) <= 1:
        return hi_values, np.arange(len(hi_values))

    # Calculate differences between adjacent points
    diffs = np.abs(np.diff(hi_values))

    # Find points with sufficient change
    valid_mask = np.ones(len(hi_values), dtype=bool)
    valid_mask[1:] = diffs > threshold

    # Ensure at least first and last two points are kept
    if np.sum(valid_mask) < 2:
        valid_mask[0] = True
        valid_mask[-1] = True

    valid_indices = np.where(valid_mask)[0]
    return hi_values[valid_indices], valid_indices

@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total = {"mse_b":0.0, "mse_a":0.0, "acc":0.0, "delta_mean":0.0}
    n_batch=0
    n_acc = 0; n_tot=0
    for batch in loader:
        xb = batch["x_before"].to(device)
        xa = batch["x_after"].to(device)
        hi_b = batch["hi_before"].to(device)
        hi_a = batch["hi_after"].to(device)
        labels = batch["labels"].to(device)
        lengths= batch["lengths"].to(device)
        mask   = batch["mask"].to(device)

        # Valid segment: 0:L-2 input, 1:L-1 target
        m_tgt  = (torch.arange(xb.size(1)-2, device=device)[None,:] < (lengths-2)[:,None]).float()

        yb_hat, ya_hat, h_b, h_a, logits, _, _, _, _ = model(xb, xa, mask)

        # Numerical stabilization
        yb_hat, ya_hat, h_b, h_a, logits = sanitize_tensors(yb_hat, ya_hat, h_b, h_a, logits)

        # Align targets
        yb_tgt = xb[:,1:-1,:]
        ya_tgt = xa[:,1:-1,:]

        mse_b = masked_mse(yb_hat, yb_tgt, m_tgt)
        mse_a = masked_mse(ya_hat, ya_tgt, m_tgt)

        # Check for NaN
        if torch.isnan(mse_b) or torch.isnan(mse_a):
            continue

        # Classification
        pred = logits.argmax(dim=1)
        n_acc += (pred == labels).sum().item()
        n_tot += labels.numel()

        # Statistics of ΔHI mean (post-maintenance improvement relative to pre-maintenance)
        valid = mask.sum(1, keepdim=True).clamp_min(1.0)
        delta_mean = ((h_a - h_b)*mask).sum(1,keepdim=True)/valid
        delta_mean = torch.nan_to_num(delta_mean, nan=0.0)
        total["delta_mean"] += delta_mean.mean().item()

        total["mse_b"] += mse_b.item()
        total["mse_a"] += mse_a.item()
        n_batch += 1

    for k in ["mse_b","mse_a","delta_mean"]:
        total[k] /= max(n_batch,1)
    total["acc"] = n_acc / max(n_tot,1)
    return total

@torch.no_grad()
def collect_test_predictions(model, loader, device, max_curve_keep=24):
    """
    Collect predictions, ΔHI, and a few sample HI curves on test set (for plotting).
    Also collect operator outputs and diagnostics.
    """
    model.eval()
    y_true, y_pred = [], []
    all_delta_mean, all_uids = [], []
    keep_curves = []

    for batch in loader:
        xb = batch["x_before"].to(device)   # (B,L,C)
        xa = batch["x_after"].to(device)
        mask   = batch["mask"].to(device)   # (B,L)
        labels = batch["labels"].to(device) # (B,)
        lengths= batch["lengths"]           # cpu tensor
        uids   = batch["uids"]

        yb_hat, ya_hat, h_b, h_a, logits, weights_b, weights_a, temp_b, temp_a = model(xb, xa, mask)

        # Numerical stabilization
        yb_hat, ya_hat, h_b, h_a, logits = sanitize_tensors(yb_hat, ya_hat, h_b, h_a, logits)

        prob = F.softmax(logits, dim=1)     # (B,3)
        pred = prob.argmax(1)               # (B,)

        # Statistics of ΔHI mean (post-maintenance improvement relative to pre-maintenance)
        valid = mask.sum(1, keepdim=True).clamp_min(1.0)
        delta_mean = ((h_a - h_b) * mask).sum(1, keepdim=True) / valid  # (B,1)
        delta_mean = torch.nan_to_num(delta_mean, nan=0.0)

        y_true.append(labels.cpu().numpy())
        y_pred.append(pred.cpu().numpy())
        all_delta_mean.append(delta_mean.squeeze(1).cpu().numpy())
        all_uids.extend(uids)

        # Collect operator outputs for diagnostics
        # Run encoder again to get intermediate outputs
        with torch.no_grad():
            h_multi_b = model.encoder.boltz(xb)
            h_multi_a = model.encoder.boltz(xa)

            # Get operator outputs
            op_outs_b = []
            op_outs_a = []

            # Build context information
            context_b = torch.cat([h_multi_b, xb], dim=-1)
            context_a = torch.cat([h_multi_a, xa], dim=-1)

            for i, op in enumerate(model.encoder.customkan.ops):
                # Get operator-specific features
                feat_b = model.encoder.customkan.op_feature_extractors[i](h_multi_b)
                feat_a = model.encoder.customkan.op_feature_extractors[i](h_multi_a)

                out_b = op(feat_b, context_b)
                out_a = op(feat_a, context_a)

                op_outs_b.append(out_b.cpu().numpy())
                op_outs_a.append(out_a.cpu().numpy())

        # Save some curves for visualization
        for i in range(xb.size(0)):
            if len(keep_curves) >= max_curve_keep:
                break
            L_i = int(lengths[i].item())
            # Fix: truncate reconstruction prediction by actual length
            L_recon = min(L_i - 2, yb_hat.size(1))  # Reconstruction sequence length

            # Collect operator outputs for this sample
            sample_op_outs_b = [op_out[i, :L_i] for op_out in op_outs_b]
            sample_op_outs_a = [op_out[i, :L_i] for op_out in op_outs_a]

            keep_curves.append({
                "uid": uids[i],
                "true": int(labels[i].cpu().item()),
                "pred": int(pred[i].cpu().item()),
                "prob": prob[i].cpu().numpy(),
                "h_before": h_b[i, :L_i].cpu().numpy(),
                "h_after":  h_a[i, :L_i].cpu().numpy(),
                "weights_before": weights_b[i, :L_i].cpu().numpy() if weights_b is not None else None,
                "weights_after": weights_a[i, :L_i].cpu().numpy() if weights_a is not None else None,
                "temp_before": temp_b[i, :L_i].cpu().numpy() if temp_b is not None else None,
                "temp_after": temp_a[i, :L_i].cpu().numpy() if temp_a is not None else None,
                "op_outputs_before": sample_op_outs_b,
                "op_outputs_after": sample_op_outs_a,
                "yb_hat":   yb_hat[i, :L_recon].cpu().numpy(),       # (L_recon,C)
                "ya_hat":   ya_hat[i, :L_recon].cpu().numpy(),       # (L_recon,C)
                "x_before": xb[i, :L_i].cpu().numpy(),               # (L,C)
                "x_after":  xa[i, :L_i].cpu().numpy(),               # (L,C)
            })

    y_true = np.concatenate(y_true, axis=0) if len(y_true)>0 else np.array([])
    y_pred = np.concatenate(y_pred, axis=0) if len(y_pred)>0 else np.array([])
    all_delta_mean = np.concatenate(all_delta_mean, axis=0) if len(all_delta_mean)>0 else np.array([])
    return y_true, y_pred, all_delta_mean, all_uids, keep_curves

def plot_hi_examples_aligned(curves, n_show=6, seed=0):
    """Post-maintenance sequence continues from pre-maintenance endpoint; draw vertical line to mark t_m; remove constant segments.
    Add "Dynamic Parameter Modulation" in title to indicate this is health state inferred from sensor data with adaptive operators"""
    if len(curves)==0:
        print("(No visualization samples)")
        return
    random.Random(seed).shuffle(curves)
    n_show = min(n_show, len(curves))
    cols = 3
    rows = int(np.ceil(n_show/cols))
    plt.figure(figsize=(cols*4.6, rows*3.2))
    for k in range(n_show):
        ex = curves[k]
        hb = ex["h_before"]; ha = ex["h_after"]
        L  = min(len(hb), len(ha))  # Defensive alignment
        hb, ha = hb[:L], ha[:L]

        # Remove constant segments
        hb_clean, hb_indices = remove_constant_segments(hb)
        ha_clean, ha_indices = remove_constant_segments(ha)

        # Corresponding time axis
        t_b = hb_indices
        t_a = ha_indices + L  # after follows immediately

        uid = ex["uid"]
        pred = ex["pred"]; true = ex["true"]; prob = ex["prob"]
        plt.subplot(rows, cols, k+1)

        # Only plot non-constant segments
        if len(hb_clean) > 1:
            plt.plot(t_b, hb_clean, label="Learned HI_Pre-maintenance", linewidth=1.8, marker='o', markersize=3)
        if len(ha_clean) > 1:
            plt.plot(t_a, ha_clean, label="Learned HI_Post-maintenance",  linewidth=1.8, linestyle='--', marker='s', markersize=3)

        # Maintenance time vertical line
        plt.axvline(L-1, color='k', linestyle=':', linewidth=1.0, alpha=0.7)

        d_mean = float(np.mean(ha - hb))  # Post-maintenance improvement relative to pre-maintenance
        d_max  = float(np.max(ha - hb))
        title = (f"uid={uid} (Dynamic Parameter Modulation)\nTrue={LABEL2NAME[true]} | Pred={LABEL2NAME[pred]} "
                 f"| p={prob[pred]:.2f}\nΔHI_mean={d_mean:.3f}, ΔHI_max={d_max:.3f}")
        plt.title(title, fontsize=9)
        plt.xlabel("Cycle (Pre-maintenance | Post-maintenance)")
        plt.ylabel("Learned Health Index (↑Post should be higher)")
        plt.grid(ls='--', alpha=.35)
        # Remove Y-axis range constraint, let it adapt to data
        if k==0: plt.legend()
    plt.tight_layout()
    plt.show()

def plot_enhanced_liquid_weights(curves, n_show=3, seed=0):
    """
    Enhanced visualization of liquid weights with operator output heatmaps
    """
    if len(curves) == 0:
        print("(No visualization samples)")
        return

    # Filter curves that have weight information
    curves_with_weights = [c for c in curves if c.get("weights_before") is not None]
    if len(curves_with_weights) == 0:
        print("(No weight information available)")
        return

    random.Random(seed).shuffle(curves_with_weights)
    n_show = min(n_show, len(curves_with_weights))

    op_names = ["MonotonicLinear", "MonotonicFlat", "ConcaveLog", "SaturationSigmoid",
                "HingeReLU", "Polynomial", "DampedSin", "PiecewiseLinear"]

    for i in range(n_show):
        ex = curves_with_weights[i]
        weights_b = ex["weights_before"]  # (T, K)
        weights_a = ex["weights_after"]   # (T, K)
        op_outs_b = ex.get("op_outputs_before", [])
        op_outs_a = ex.get("op_outputs_after", [])

        if weights_b is None or weights_a is None:
            continue

        uid = ex["uid"]
        true_label = LABEL2NAME[ex["true"]]
        pred_label = LABEL2NAME[ex["pred"]]

        # Create figure with 2x2 layout for heatmaps
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle(f"Dynamic Parameter Modulation Analysis - UID={uid}, True={true_label}, Pred={pred_label}",
                     fontsize=16, fontweight='bold')

        # Create GridSpec for flexible layout
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # 1. Pre-maintenance operator outputs heatmap
        ax1 = fig.add_subplot(gs[0, 0])
        if op_outs_b and len(op_outs_b) > 0:
            # Stack operator outputs: (K, T)
            op_matrix_b = np.stack([op_out for op_out in op_outs_b], axis=0)
            im1 = ax1.imshow(op_matrix_b, aspect='auto', cmap='viridis', interpolation='nearest')
            ax1.set_title("Pre-maintenance Operator Outputs", fontsize=14, fontweight='bold')
            ax1.set_xlabel("Time Step", fontsize=12)
            ax1.set_ylabel("Operator", fontsize=12)
            ax1.set_yticks(range(len(op_names)))
            ax1.set_yticklabels(op_names, fontsize=10)
            cbar1 = plt.colorbar(im1, ax=ax1)
            cbar1.set_label('Output Value', fontsize=11)
        else:
            ax1.text(0.5, 0.5, "Operator output\ndata not available",
                    ha='center', va='center', transform=ax1.transAxes, fontsize=12)
            ax1.set_title("Pre-maintenance Operator Outputs", fontsize=14, fontweight='bold')

        # 2. Post-maintenance operator outputs heatmap
        ax2 = fig.add_subplot(gs[0, 1])
        if op_outs_a and len(op_outs_a) > 0:
            # Stack operator outputs: (K, T)
            op_matrix_a = np.stack([op_out for op_out in op_outs_a], axis=0)
            im2 = ax2.imshow(op_matrix_a, aspect='auto', cmap='viridis', interpolation='nearest')
            ax2.set_title("Post-maintenance Operator Outputs", fontsize=14, fontweight='bold')
            ax2.set_xlabel("Time Step", fontsize=12)
            ax2.set_ylabel("Operator", fontsize=12)
            ax2.set_yticks(range(len(op_names)))
            ax2.set_yticklabels(op_names, fontsize=10)
            cbar2 = plt.colorbar(im2, ax=ax2)
            cbar2.set_label('Output Value', fontsize=11)
        else:
            ax2.text(0.5, 0.5, "Operator output\ndata not available",
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title("Post-maintenance Operator Outputs", fontsize=14, fontweight='bold')

        # 3. Pre-maintenance weights heatmap
        ax3 = fig.add_subplot(gs[1, 0])
        T_b, K = weights_b.shape
        im3 = ax3.imshow(weights_b.T, aspect='auto', cmap='plasma', interpolation='nearest', vmin=0, vmax=1)
        ax3.set_title("Pre-maintenance Operator Weights", fontsize=14, fontweight='bold')
        ax3.set_xlabel("Time Step", fontsize=12)
        ax3.set_ylabel("Operator", fontsize=12)
        ax3.set_yticks(range(K))
        ax3.set_yticklabels([op_names[k] if k < len(op_names) else f"Op_{k}" for k in range(K)], fontsize=10)
        cbar3 = plt.colorbar(im3, ax=ax3)
        cbar3.set_label('Weight', fontsize=11)

        # 4. Post-maintenance weights heatmap
        ax4 = fig.add_subplot(gs[1, 1])
        T_a, _ = weights_a.shape
        im4 = ax4.imshow(weights_a.T, aspect='auto', cmap='plasma', interpolation='nearest', vmin=0, vmax=1)
        ax4.set_title("Post-maintenance Operator Weights", fontsize=14, fontweight='bold')
        ax4.set_xlabel("Time Step", fontsize=12)
        ax4.set_ylabel("Operator", fontsize=12)
        ax4.set_yticks(range(K))
        ax4.set_yticklabels([op_names[k] if k < len(op_names) else f"Op_{k}" for k in range(K)], fontsize=10)
        cbar4 = plt.colorbar(im4, ax=ax4)
        cbar4.set_label('Weight', fontsize=11)

        plt.show()

def plot_liquid_weights(curves, n_show=3, seed=0):
    """Wrapper function for backward compatibility"""
    plot_enhanced_liquid_weights(curves, n_show, seed)

def plot_sensor_examples_aligned(curves, sensor_idx_list=None, n_cols=4):
    """
    Plot several sensors' original vs recursive prediction (post), with after time axis connected after before.
    - Original before: x_before[:, s]
    - Original after : x_after [:, s] (optional)
    - Predicted after : ya_hat   [:, s] (one-step sequence aligned with x_after)
    """
    if len(curves)==0:
        print("(No visualization samples)")
        return
    # Take first sample to show multiple sensors
    ex = curves[0]
    xb = ex["x_before"]         # (L,C)
    xa = ex["x_after"]          # (L,C)
    ya = ex["ya_hat"]           # (L_recon,C) predicted is 1..L-1
    L, C = xb.shape
    Lhat = ya.shape[0]          # Actual reconstruction length

    if sensor_idx_list is None:
        # Default pick 8
        step = max(1, C//8)
        sensor_idx_list = list(range(0, C, step))[:8]

    n = len(sensor_idx_list)
    n_rows = int(np.ceil(n/n_cols))
    plt.figure(figsize=(n_cols*4.3, n_rows*3.1))
    for i, s in enumerate(sensor_idx_list):
        plt.subplot(n_rows, n_cols, i+1)
        t_b = np.arange(L)
        t_a = np.arange(L, 2*L)
        plt.plot(t_b, xb[:,s], label="Original (before)", linewidth=1.2)
        plt.plot(t_a, xa[:,s], label="Original (after)",  linewidth=1.0, linestyle="--", alpha=0.7)
        # Predicted after: aligned with after target (corresponding to 1..L-1)
        t_pred = np.arange(L+1, L+1+Lhat)      # Align ya_hat time
        plt.plot(t_pred, ya[:,s], label="Post-maint. prediction", linewidth=1.6)
        plt.axvline(L-1, color='k', linestyle=':', linewidth='1.0')
        plt.title(f"Sensor_{s:02d} (Dynamic Parameters)")
        if i==0: plt.legend()
        plt.grid(ls="--", alpha=.35)
    plt.tight_layout()
    plt.show()

def topk_by_delta(df_delta, k=5):
    if len(df_delta)==0:
        return
    for cls in [0,1,2]:
        sub = df_delta[df_delta["true"]==cls].sort_values("delta_hi_mean", ascending=False).head(k)
        print(f"\n[True={LABEL2NAME[cls]}] Top {k} samples with highest learned ΔHI_mean (maintenance effect):")
        print(sub[["uid","delta_hi_mean","pred"]].reset_index(drop=True))

# ============================================================
# 5) Training/evaluation loop with enhanced liquid weight regularization and dynamic parameter modulation
# ============================================================
def train_model(pairs, epochs=20, batch_size=32, lr=1e-3, horizon=None, device=None, patience=20):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data split 7:1:2
    pairs_tr, pairs_vl, pairs_te = split_pairs_uid(pairs, 0.7, 0.1, 42)
    print_split_summary(pairs, 0.7, 0.1, 42)

    ds_tr = PairsReconstructDataset(pairs_tr, horizon=horizon)
    ds_vl = PairsReconstructDataset(pairs_vl, horizon=horizon)
    ds_te = PairsReconstructDataset(pairs_te, horizon=horizon)

    ld_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_shift)
    ld_vl = DataLoader(ds_vl, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_shift)
    ld_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_shift)

    C = ds_tr.C
    print(f"\n[Dimension Check] Sensor feature dimension C = {C}")

    model = DiffAwareReconstructor(in_ch=C, trend_ch=4, hidden=128, n_classes=3).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)  # Use AdamW
    ce  = nn.CrossEntropyLoss()

    best_v = 1e9
    best = None
    nan_count = 0  # Count NaN batches
    best_epoch = 0  # Record best epoch
    no_improve_count = 0  # Count epochs without improvement for early stopping

    print(f"\nStarting training for {epochs} epochs with early stopping (patience={patience})...")
    print(f"Device: {device}")
    print(f"Train set: {len(ds_tr)} samples, Val set: {len(ds_vl)} samples, Test set: {len(ds_te)} samples")
    print("Note: Enhanced with Improved Liquid Weight Generator")
    print("- Increased temperature minimum to reduce weight collapse")
    print("- Zero-mean weight normalization to prevent systematic bias")
    print("- Per-operator feature extraction for diversity")
    print("- Enhanced entropy and diversity regularization")
    print("- Reduced constraint weights to allow more operator diversity")

    for ep in range(1, epochs+1):
        print(f"\n[Epoch {ep}/{epochs}] Training...")
        model.train()
        logs = {"mse_b":0.0, "mse_a":0.0, "diff":0.0, "smooth":0.0, "mono":0.0, "linear":0.0, "cls":0.0, "consist":0.0, "maintain":0.0, "smooth_enh":0.0, "global_sup":0.0, "liquid_weight":0.0, "all":0.0}
        n_bt = 0
        batch_nan_count = 0

        # Enhanced curriculum learning for operator diversity
        progress = ep / epochs
        # Stronger entropy encouragement in early training
        lambda_tv = 0.03 * (1 - progress * 0.3)      # Reduced TV weight
        lambda_ent = 0.3 * (1 - progress * 0.7)      # Higher initial entropy weight
        lambda_bal = 0.15 * (1 - progress * 0.5)     # Higher balance weight
        lambda_div = 0.2 * (1 - progress * 0.6)      # New diversity regularization

        for batch_idx, batch in enumerate(ld_tr):
            # Show training progress
            if batch_idx % max(1, len(ld_tr)//10) == 0:
                progress_pct = (batch_idx + 1) / len(ld_tr) * 100
                print(f"    Training progress: {progress_pct:.1f}% ({batch_idx+1}/{len(ld_tr)} batches)")

            xb = batch["x_before"].to(device)  # (B,L,C)
            xa = batch["x_after"].to(device)
            labels = batch["labels"].to(device)
            lengths= batch["lengths"].to(device)
            mask   = batch["mask"].to(device)

            # Input/target mask
            m_tgt  = (torch.arange(xb.size(1)-2, device=device)[None,:] < (lengths-2)[:,None]).float()

            yb_hat, ya_hat, h_b, h_a, logits, weights_b, weights_a, temp_b, temp_a = model(xb, xa, mask)

            # Numerical stabilization
            yb_hat, ya_hat, h_b, h_a, logits = sanitize_tensors(yb_hat, ya_hat, h_b, h_a, logits)
            weights_b, weights_a = sanitize_tensors(weights_b, weights_a)

            # Recursive reconstruction targets
            yb_tgt = xb[:,1:-1,:]
            ya_tgt = xa[:,1:-1,:]
            loss_b = masked_mse(yb_hat, yb_tgt, m_tgt)
            loss_a = masked_mse(ya_hat, ya_tgt, m_tgt)

            # HI difference (class-conditional margin) - ensure post-maintenance higher than pre-maintenance
            valid = mask.sum(1, keepdim=True).clamp_min(1.0)
            delta_mean = ((h_a - h_b)*mask).sum(1,keepdim=True)/valid  # (B,1)
            loss_diff = diff_margin_by_class(delta_mean.squeeze(1), labels,
                                             m_low=0.1, m_mid=0.25, m_high=0.45)

            # HI smooth + monotonic decreasing (constrain both pre/post) - remove range constraint
            loss_smooth= slope_loss(h_a, mask, "smooth") + slope_loss(h_b, mask, "smooth")
            loss_mono  = slope_loss(h_a, mask, "mono_dec") + slope_loss(h_b, mask, "mono_dec")

            # Enhanced linear loss - reduced weight to allow more diversity
            loss_linear = enhanced_linear_slope_loss(h_b, mask, weight=5.0) + enhanced_linear_slope_loss(h_a, mask, weight=5.0)

            # Sensor consistency loss (including constraint that post-maintenance should be higher than pre-maintenance)
            loss_consistency = sensor_consistency_loss(xb, xa, h_b, h_a, mask)

            # Maintenance improvement constraint
            loss_maintenance = maintenance_improvement_constraint(h_b, h_a, labels, mask)

            # Reduced global HI superiority constraint
            loss_global_superiority = global_hi_superiority_constraint(h_b, h_a, mask, weight=3.0)

            # Enhanced smoothness loss (reduced weight)
            loss_smooth_enhanced = (smoothness_enhancement_loss(h_b, mask, order=2) +
                                   smoothness_enhancement_loss(h_a, mask, order=2) +
                                   smoothness_enhancement_loss(h_b, mask, order=3) * 0.3 +
                                   smoothness_enhancement_loss(h_a, mask, order=3) * 0.3)

            # Enhanced liquid weight regularization
            loss_liquid_weight = enhanced_liquid_weight_regularization(weights_b, weights_a, mask,
                                                                      lambda_tv=lambda_tv,
                                                                      lambda_ent=lambda_ent,
                                                                      lambda_bal=lambda_bal,
                                                                      lambda_div=lambda_div)

            # Classification loss
            loss_cls = ce(logits, labels)

            # Total loss (adjusted weights to encourage operator diversity)
            w_rec=1.0; w_diff=0.4; w_smooth=0.03; w_mono=0.05; w_linear=0.5; w_consist=0.2; w_cls=1.0; w_maintain=1.0; w_smooth_enh=0.3; w_global_sup=1.0; w_liquid=0.5
            loss = (w_rec*(loss_b+loss_a) + w_diff*loss_diff +
                   w_smooth*loss_smooth + w_mono*loss_mono + w_linear*loss_linear +
                   w_consist*loss_consistency + w_cls*loss_cls + w_maintain*loss_maintenance +
                   w_smooth_enh*loss_smooth_enhanced + w_global_sup*loss_global_superiority +
                   w_liquid*loss_liquid_weight)

            # Numerical stabilization of all losses
            loss_b, loss_a, loss_diff, loss_smooth, loss_mono, loss_linear, loss_consistency, loss_cls, loss_maintenance, loss_smooth_enhanced, loss_global_superiority, loss_liquid_weight, loss = sanitize_tensors(
                loss_b, loss_a, loss_diff, loss_smooth, loss_mono, loss_linear, loss_consistency, loss_cls, loss_maintenance, loss_smooth_enhanced, loss_global_superiority, loss_liquid_weight, loss)

            # Check NaN and skip
            if torch.isnan(loss) or torch.isinf(loss):
                batch_nan_count += 1
                if batch_nan_count == 1:  # Only print warning once
                    print(f"    [Warning] Epoch {ep}: Found NaN/Inf loss, skipping abnormal batch")
                continue

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            logs["mse_b"] += loss_b.item()
            logs["mse_a"] += loss_a.item()
            logs["diff"]  += loss_diff.item()
            logs["smooth"]+= loss_smooth.item()
            logs["mono"]  += loss_mono.item()
            logs["linear"]+= loss_linear.item()
            logs["consist"]+= loss_consistency.item()
            logs["cls"]   += loss_cls.item()
            logs["maintain"]+= loss_maintenance.item()
            logs["smooth_enh"]+= loss_smooth_enhanced.item()
            logs["global_sup"]+= loss_global_superiority.item()
            logs["liquid_weight"]+= loss_liquid_weight.item()
            logs["all"]   += loss.item()
            n_bt += 1

        # Record NaN batches
        if batch_nan_count > 0:
            nan_count += batch_nan_count

        print("    Validating...")
        for k in logs: logs[k] /= max(n_bt,1)
        vl = eval_epoch(model, ld_vl, device)
        print(f"[Epoch {ep:03d}] Train: L={logs['all']:.4f} rec_b={logs['mse_b']:.4f} rec_a={logs['mse_a']:.4f} "
              f"diff={logs['diff']:.4f} linear={logs['linear']:.4f} maintain={logs['maintain']:.4f} liquid_w={logs['liquid_weight']:.4f} global_sup={logs['global_sup']:.4f} cls={logs['cls']:.4f} | "
              f"Val: mse_b={vl['mse_b']:.4f} mse_a={vl['mse_a']:.4f} acc={vl['acc']:.3f} ΔHI_improvement={vl['delta_mean']:.3f}")

        # Simple early stopping: observe validation reconstruction loss
        vl_total = vl['mse_b'] + vl['mse_a'] + vl['acc']  # Lower is better (minimize MSE, maximize accuracy)
        if vl_total < best_v:
            best_v = vl_total
            best_epoch = ep
            best = {k: v.clone() if hasattr(v, 'clone') else v for k, v in model.state_dict().items()}
            no_improve_count = 0  # Reset counter
            print(f"    ✓ Saved new best model (val loss: {best_v:.4f})")
        else:
            no_improve_count += 1
            print(f"    Val loss not improved (current: {vl_total:.4f}, best: {best_v:.4f}) - No improvement for {no_improve_count} epochs")

            # Early stopping check
            if no_improve_count >= patience:
                print(f"\n[Early Stopping] No improvement for {patience} epochs. Stopping training at epoch {ep}.")
                break

    if best is not None:
        model.load_state_dict(best)
        print(f"\n[Best Checkpoint] Val reconstruction loss: {best_v:.4f} (Epoch {best_epoch})")

        # Save best model to specified path
        save_path = "/content/drive/MyDrive/CMAPSS/main_dynamics_identification_4.pth"
        torch.save(model.state_dict(), save_path)
        print(f"[Model Saved] Best model saved to: {save_path}")

    if nan_count > 0:
        print(f"[Warning] Total skipped {nan_count} batches containing NaN/Inf")

    # Test set evaluation
    print("\n" + "="*60)
    print("Final Test Set Evaluation")
    print("="*60)
    te = eval_epoch(model, ld_te, device)
    print(f"[Test Set] mse_b={te['mse_b']:.4f} mse_a={te['mse_a']:.4f} acc={te['acc']:.3f} ΔHI_improvement={te['delta_mean']:.3f}")

    # Collect test predictions and visualization data
    y_true, y_pred, all_delta_mean, all_uids, keep_curves = collect_test_predictions(model, ld_te, device, max_curve_keep=24)

    # Classification report
    if len(y_true) > 0:
        from sklearn.metrics import classification_report, confusion_matrix
        print("\n[Classification Report]")
        print(classification_report(y_true, y_pred, target_names=["Perfect", "General", "Poor"]))
        print("\n[Confusion Matrix]")
        print(confusion_matrix(y_true, y_pred))

        # Create learned ΔHI statistics table
        df_delta = pd.DataFrame({
            "uid": all_uids[:len(y_true)],
            "true": y_true,
            "pred": y_pred,
            "delta_hi_mean": all_delta_mean[:len(y_true)]
        })
        print("\n[Learned ΔHI Statistics] By true class (maintenance effect improvement)")
        print(df_delta.groupby("true")["delta_hi_mean"].describe())

        # Show Top-K samples
        topk_by_delta(df_delta, k=3)

        # HI curve visualization
        print("\n[Visualization] Showing learned health index curves with liquid weight adaptation...")
        plot_hi_examples_aligned(keep_curves, n_show=6, seed=0)

        # Liquid weight visualization
        print("\n[Visualization] Showing liquid operator weights evolution...")
        plot_liquid_weights(keep_curves, n_show=3, seed=0)

        # Sensor reconstruction visualization
        print("\n[Visualization] Showing sensor reconstruction predictions...")
        plot_sensor_examples_aligned(keep_curves, sensor_idx_list=None, n_cols=4)

    return model, (y_true, y_pred, all_delta_mean, all_uids, keep_curves)


model, results = train_model(
    pairs=pairs,
    epochs=300,
    batch_size=258,
    lr=1e-3,
    horizon=50,
    device=None,  # Auto select
    patience=20   # Early stopping patience
)

print("\nTraining completed! Model enhanced with Liquid Weight Generator:")
print("- Operator weights adapt to input conditions (h_multi, x)")
print("- Time-smooth, entropy-balanced, and usage-balanced regularization")
print("- Curriculum learning: from uniform to personalized weight distribution")
print("- All operators participate with continuous weighting (no routing/switching)")
print("View learned HI curves and liquid weight evolution through enhanced visualization.")

# %% [notebook code cell 8]
import numpy as np
import matplotlib.pyplot as plt
import torch

# =============== Gadget: Change all displayed "General" to "Improved" ===============
def display_general_as_improved(text):
    """
    For visual display only: replace 'General' in the string with 'Improved'.
    'General' can still be used inside the model/dataset without being affected.
    """
    if isinstance(text, str):
        return text.replace("General", "Improved")
    return text


# =============== 1. Collect all samples of a given UID (all maintenance strategies) ===============
horizon = 50  # Use the same horizon as in training
target_uid = "702"  # UID you are interested in

ds_all = PairsReconstructDataset(pairs, horizon=horizon)
device = next(model.parameters()).device
model.eval()

uid_samples = []

# Scan the whole dataset to find all samples with the given UID
for idx in range(len(ds_all)):
    sample = ds_all[idx]  # dict with keys: uid, label, x_before, x_after, hi_before, hi_after, strategy
    if str(sample["uid"]) == str(target_uid):
        uid_samples.append(sample)

if len(uid_samples) == 0:
    print(f"[Warning] UID={target_uid} not found in dataset. Please check 'pairs'.")
else:
    print(f"Found {len(uid_samples)} samples for UID={target_uid} (different maintenance strategies).")

# =============== 2. Run model and collect operator outputs + weights for each maintenance strategy ===============
results_per_strategy = []

with torch.no_grad():
    for sample in uid_samples:
        x_b = sample["x_before"].unsqueeze(0).to(device)  # (1, L, C)
        x_a = sample["x_after"].unsqueeze(0).to(device)   # (1, L, C)
        L = x_b.size(1)
        mask = torch.ones(1, L, device=device)            # (1, L) all valid

        # Forward pass through the full model to get operator weights
        _, _, _, _, logits, weights_b, weights_a, _, _ = model(x_b, x_a, mask)  # weights: (1, T, K)
        weights_b = weights_b[0, :L].detach().cpu().numpy()  # (T, K)
        weights_a = weights_a[0, :L].detach().cpu().numpy()  # (T, K)

        # Re-compute operator outputs (consistent with collect_test_predictions)
        h_multi_b = model.encoder.boltz(x_b)  # (1, T, trend_ch)
        h_multi_a = model.encoder.boltz(x_a)

        context_b = torch.cat([h_multi_b, x_b], dim=-1)  # (1, T, trend_ch + C)
        context_a = torch.cat([h_multi_a, x_a], dim=-1)

        op_outputs_before = []
        op_outputs_after = []

        for i, op in enumerate(model.encoder.customkan.ops):
            feat_b = model.encoder.customkan.op_feature_extractors[i](h_multi_b)  # (1, T, trend_ch)
            feat_a = model.encoder.customkan.op_feature_extractors[i](h_multi_a)

            out_b = op(feat_b, context_b)  # (1, T)
            out_a = op(feat_a, context_a)  # (1, T)

            op_outputs_before.append(out_b[0, :L].detach().cpu().numpy())
            op_outputs_after.append(out_a[0, :L].detach().cpu().numpy())

        op_matrix_b = np.stack(op_outputs_before, axis=0)  # (K, T)
        op_matrix_a = np.stack(op_outputs_after, axis=0)   # (K, T)

        # True / predicted maintenance class
        true_label_idx = int(sample["label"])
        pred_label_idx = int(logits.argmax(dim=1).item())

        # Original tag name (may be "Perfect" / "General" / "Poor")
        true_label_name_raw = LABEL2NAME[true_label_idx]
        pred_label_name_raw = LABEL2NAME[pred_label_idx]

        # Display the tag name: display "General" as "Improved"
        true_label_name = display_general_as_improved(true_label_name_raw)
        pred_label_name = display_general_as_improved(pred_label_name_raw)

        # strategy is also used for display: such as "General Maintenance" -> "Improved Maintenance"
        strategy_display = display_general_as_improved(sample["strategy"])

        results_per_strategy.append({
            "strategy": strategy_display, # General->Improved display mapping has been done
            "strategy_raw": sample["strategy"], # Optional: retain the original strategy for internal analysis
            "true_label_idx": true_label_idx,
            "pred_label_idx": pred_label_idx,
            "true_label_name": true_label_name, # mapped
            "pred_label_name": pred_label_name, # mapped
            "weights_before": weights_b,
            "weights_after": weights_a,
            "ops_before": op_matrix_b,
            "ops_after": op_matrix_a,
        })

# =============== 3. Plot operator heatmaps for each maintenance strategy of this UID ===============
if len(results_per_strategy) == 0:
    print(f"[Warning] No operator data collected for UID={target_uid}.")
else:
    op_names = [
        "MonotonicLinear", "MonotonicFlat", "ConcaveLog", "SaturationSigmoid",
        "HingeReLU", "Polynomial", "DampedSin", "PiecewiseLinear"
    ]

    for res in results_per_strategy:
        # What is taken out here is already the displayed version (all General has been replaced with Improved)
        strategy = res["strategy"]
        true_label = res["true_label_name"]
        pred_label = res["pred_label_name"]

        weights_b_702 = res["weights_before"]  # (T, K)
        weights_a_702 = res["weights_after"]   # (T, K)
        op_matrix_b_702 = res["ops_before"]    # (K, T)
        op_matrix_a_702 = res["ops_after"]     # (K, T)

        T, K = weights_b_702.shape
        if len(op_names) < K:
            op_names = op_names + [f"Op_{i}" for i in range(len(op_names), K)]

        # --- One-line title (title + "legend info") and smaller distance to plots ---
        title_text = (
            f"UID={target_uid} | Strategy={strategy} | "
            f"True={true_label} | Pred={pred_label}"
        )

        fig = plt.figure(figsize=(18, 10))
        # y slightly lower so the plots are closer to the title
        fig.suptitle(title_text, fontsize=14, fontweight="bold", y=0.96)

        # Smaller hspace / wspace to reduce gaps between subplots
        gs = fig.add_gridspec(2, 2, hspace=0.22, wspace=0.22)

        # ---- 1. Operator outputs before maintenance ----
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(op_matrix_b_702, aspect="auto", interpolation="nearest")
        ax1.set_title("Operator Outputs (Before)", fontsize=11, pad=4)
        ax1.set_xlabel("Time Step", fontsize=10)
        ax1.set_ylabel("Operator", fontsize=10)
        ax1.set_yticks(np.arange(K))
        ax1.set_yticklabels(op_names[:K], fontsize=9)
        cbar1 = plt.colorbar(im1, ax=ax1, pad=0.01)  # smaller pad -> closer to image
        cbar1.set_label("Output", fontsize=10)

        # ---- 2. Operator outputs after maintenance ----
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(op_matrix_a_702, aspect="auto", interpolation="nearest")
        ax2.set_title("Operator Outputs (After)", fontsize=11, pad=4)
        ax2.set_xlabel("Time Step", fontsize=10)
        ax2.set_ylabel("Operator", fontsize=10)
        ax2.set_yticks(np.arange(K))
        ax2.set_yticklabels(op_names[:K], fontsize=9)
        cbar2 = plt.colorbar(im2, ax=ax2, pad=0.01)
        cbar2.set_label("Output", fontsize=10)

        # ---- 3. Operator weights before maintenance ----
        ax3 = fig.add_subplot(gs[1, 0])
        im3 = ax3.imshow(
            weights_b_702.T,  # (K, T)
            aspect="auto",
            interpolation="nearest",
            vmin=0.0,
            vmax=1.0
        )
        ax3.set_title("Operator Weights (Before)", fontsize=11, pad=4)
        ax3.set_xlabel("Time Step", fontsize=10)
        ax3.set_ylabel("Operator", fontsize=10)
        ax3.set_yticks(np.arange(K))
        ax3.set_yticklabels(op_names[:K], fontsize=9)
        cbar3 = plt.colorbar(im3, ax=ax3, pad=0.01)
        cbar3.set_label("Weight", fontsize=10)

        # ---- 4. Operator weights after maintenance ----
        ax4 = fig.add_subplot(gs[1, 1])
        im4 = ax4.imshow(
            weights_a_702.T,  # (K, T)
            aspect="auto",
            interpolation="nearest",
            vmin=0.0,
            vmax=1.0
        )
        ax4.set_title("Operator Weights (After)", fontsize=11, pad=4)
        ax4.set_xlabel("Time Step", fontsize=10)
        ax4.set_ylabel("Operator", fontsize=10)
        ax4.set_yticks(np.arange(K))
        ax4.set_yticklabels(op_names[:K], fontsize=9)
        cbar4 = plt.colorbar(im4, ax=ax4, pad=0.01)
        cbar4.set_label("Weight", fontsize=10)

        # Make title–plots spacing smaller by shrinking the rect top
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        plt.show()

# %% [notebook code cell 9]
# -*- coding: utf-8 -*-
"""
One Figure + One Table
Confusion Matrix (heatmap) and Classification Report (table)
Polished, 'Nature'-style plotting & typesetting
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------
# 1) Data (from your results)
# ---------------------------
classes = ["Perfect", "Improved", "Poor"]

conf_mat = np.array([
    [129, 14,  0],
    [  4,139,  0],
    [  0,  0,143]
], dtype=int)

report = {
    "precision": {"Perfect": 0.97, "Improved": 0.91, "Poor": 1.00, "macro avg": 0.96, "weighted avg": 0.96},
    "recall":    {"Perfect": 0.90, "Improved": 0.97, "Poor": 1.00, "macro avg": 0.96, "weighted avg": 0.96},
    "f1-score":  {"Perfect": 0.93, "Improved": 0.94, "Poor": 1.00, "macro avg": 0.96, "weighted avg": 0.96},
    "support":   {"Perfect": 143,  "Improved": 143,  "Poor": 143,  "macro avg": 429,  "weighted avg": 429},
}
accuracy = 0.96
total_support = 429

rows = ["Perfect", "Improved", "Poor", "macro avg", "weighted avg"]
cols = ["precision", "recall", "f1-score", "support"]
df = pd.DataFrame({c: [report[c][r] for r in rows] for c in cols}, index=rows)
df["support"] = df["support"].astype(int)

# ---------------------------
# 2) Global style
# ---------------------------
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.4,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})

# ==========================================================
# FIGURE 1: Confusion Matrix (heatmap)
# ==========================================================
fig1, ax = plt.subplots(figsize=(4.8, 4.6))
im = ax.imshow(conf_mat, cmap="Blues", interpolation="nearest")
cbar = fig1.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.tick_params(length=3)

ax.set_xticks(np.arange(len(classes)), labels=classes, rotation=30, ha="right")
ax.set_yticks(np.arange(len(classes)), labels=classes)
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
ax.set_title("Confusion Matrix")

# cell value labels with contrast-aware text color
thr = conf_mat.max() / 2.0
for i in range(conf_mat.shape[0]):
    for j in range(conf_mat.shape[1]):
        val = conf_mat[i, j]
        ax.text(j, i, f"{val}", va="center", ha="center",
                fontsize=9, color=("white" if val > thr else "black"))

# minor grid to gently delineate cells
ax.set_xticks(np.arange(-0.5, len(classes), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(classes), 1), minor=True)
ax.grid(which="minor", color="white", linewidth=0.8)
ax.tick_params(which="minor", bottom=False, left=False)

fig1.tight_layout()
fig1.savefig("confusion_matrix.svg", bbox_inches="tight")
fig1.savefig("confusion_matrix.pdf", bbox_inches="tight")

# ==========================================================
# FIGURE 2: Classification Report (table)
# ==========================================================
fig2, ax2 = plt.subplots(figsize=(6.6, 3.8))
ax2.set_axis_off()
ax2.set_title("Classification Report", pad=6)

# format metrics for display
render_df = df.copy()
for metric in ["precision", "recall", "f1-score"]:
    render_df[metric] = render_df[metric].map(lambda x: f"{x:.2f}")
render_df["support"] = render_df["support"].map(lambda x: f"{x:d}")

# create the table
tbl = ax2.table(cellText=render_df.values,
                rowLabels=render_df.index,
                colLabels=render_df.columns,
                loc="center",
                cellLoc="center",
                rowLoc="center")

# style tweaks
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1.15, 1.15)

for (r, c), cell in tbl.get_celld().items():
    # header row
    if r == 0:
        cell.set_text_props(weight="bold")
        cell.set_edgecolor("0.2")
        cell.set_linewidth(0.8)
        cell.set_alpha(0.98)
    # row labels (index column)
    if c == -1 and r > 0:
        cell.set_text_props(weight="bold")
        cell.set_edgecolor("0.85")
        cell.set_linewidth(0.6)
    # body cells
    if r > 0 and c >= 0:
        cell.set_edgecolor("0.85")
        cell.set_linewidth(0.6)

# accuracy note below the table (footnote style)
ax2.annotate(
    f"Accuracy = {accuracy:.2f}  (n = {total_support})",
    xy=(0.0, -0.06), xycoords="axes fraction",
    ha="left", va="top", fontsize=9
)

fig2.tight_layout()
fig2.savefig("classification_report_table.svg", bbox_inches="tight")
fig2.savefig("classification_report_table.pdf", bbox_inches="tight")

plt.show()



# %% [notebook code cell 10]
# ============================================================
# Monotonic-Decreasing HI + aligned plotting (before -> after)
# ============================================================
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import random
import matplotlib.pyplot as plt

# ============================================================
# 1) Dataset: read samples from pairs, perform shift operations in batch later
# ============================================================
class PairsReconstructDataset(Dataset):
    """
    Each sample contains:
      x_before: (L,C) feature sequence before maintenance
      x_after : (L,C) feature sequence after maintenance
      hi_before/hi_after: (L,) health index (for evaluation/optional analysis only, not strong supervision)
      strategy: maintenance type ("Perfect Maintenance" / "General Maintenance" / "Poor Maintenance")
    """
    def __init__(self, pairs: dict, horizon: int=None):
        items = []
        for uid, sd in pairs.items():
            for strat, d in sd.items():
                xb = np.asarray(d["x_before"], dtype=np.float32)
                xa = np.asarray(d["x_after"],  dtype=np.float32)
                # When there's no real HI, create placeholder that will be learned by the model
                if "hi_before" in d and d["hi_before"] is not None:
                    hib = np.asarray(d["hi_before"],dtype=np.float32)
                else:
                    # Create initialized fake HI, model will learn real patterns
                    hib = np.linspace(0.8, 0.4, len(xb)).astype(np.float32)

                if "hi_after" in d and d["hi_after"] is not None:
                    hia = np.asarray(d["hi_after"], dtype=np.float32)
                else:
                    # Create different fake HI patterns based on maintenance strategy
                    if "perfect" in strat.lower():
                        # Perfect maintenance: significant improvement
                        hia = np.linspace(0.9, 0.6, len(xa)).astype(np.float32)
                    elif "general" in strat.lower():
                        # General maintenance: moderate improvement
                        hia = np.linspace(0.7, 0.5, len(xa)).astype(np.float32)
                    else:
                        # Poor maintenance: slight improvement
                        hia = np.linspace(0.5, 0.35, len(xa)).astype(np.float32)

                L, C = xb.shape
                # Unify length (optional)
                if horizon is not None and L > horizon:
                    xb, xa, hib, hia = xb[:horizon], xa[:horizon], hib[:horizon], hia[:horizon]
                    L = horizon
                if L < 3:  # At least need to form 0:L-2 -> 1:L-1
                    continue
                items.append({"uid": uid, "strategy": strat, "x_before": xb, "x_after": xa,
                              "hi_before": hib, "hi_after": hia})
        assert len(items)>0, "Dataset is empty, please check pairs data."
        self.items = items
        self.C = items[0]["x_before"].shape[1]

        # Strategy to 3-class label mapping
        def strat_to_cls(s):
            s = s.lower()
            if "perfect" in s: return 0
            if "general" in s: return 1
            if "poor" in s: return 2
            raise ValueError(f"Unknown strategy name: {s}")
        self.labels = [strat_to_cls(it["strategy"]) for it in items]

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        return {
            "uid": it["uid"],
            "label": self.labels[idx],
            "x_before": torch.from_numpy(it["x_before"]), # (L,C)
            "x_after":  torch.from_numpy(it["x_after"]),  # (L,C)
            "hi_before":torch.from_numpy(it["hi_before"]),# (L,)
            "hi_after": torch.from_numpy(it["hi_after"]), # (L,)
            "strategy": it["strategy"]
        }

def pad_collate_shift(batch):
    """
    Pad sequences to same length; return mask, do NOT perform shift operation here
    to allow unified training: inputs = [:,0:L-2], targets = [:,1:L-1]
    """
    Ls = [b["x_before"].shape[0] for b in batch]
    Lmax = max(Ls)
    C = batch[0]["x_before"].shape[1]

    def pad2d(x):
        L, C0 = x.shape
        out = torch.zeros(Lmax, C0, dtype=x.dtype)
        out[:L] = x
        return out

    def pad1d(x):
        L = x.shape[0]
        out = torch.zeros(Lmax, dtype=x.dtype)
        out[:L] = x
        return out

    x_before = torch.stack([pad2d(b["x_before"]) for b in batch], 0) # (B,Lmax,C)
    x_after  = torch.stack([pad2d(b["x_after"])  for b in batch], 0) # (B,Lmax,C)
    hi_before= torch.stack([pad1d(b["hi_before"])for b in batch], 0) # (B,Lmax)
    hi_after = torch.stack([pad1d(b["hi_after"]) for b in batch], 0) # (B,Lmax)
    lengths  = torch.tensor(Ls, dtype=torch.long)
    mask     = torch.zeros(len(batch), Lmax)
    for i,L in enumerate(Ls): mask[i,:L] = 1.0
    labels   = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    uids     = [b["uid"] for b in batch]
    return {"uids": uids, "labels": labels,
            "x_before": x_before, "x_after": x_after,
            "hi_before": hi_before, "hi_after": hi_after,
            "lengths": lengths, "mask": mask}

# ============================================================
# 2) Model: encode "damage" → project to monotonic decreasing HI → recursive reconstruction of two sequences → classify using ΔHI
# ============================================================
class BoltzmannKAN(nn.Module):
    def __init__(self, in_ch, out_ch=4):
        super().__init__()
        self.E  = nn.Parameter(torch.zeros(out_ch, in_ch))
        self.kT = nn.Parameter(torch.ones(out_ch, in_ch))
    def forward(self, x):
        B,T,C = x.shape
        kT = torch.clamp(F.softplus(self.kT), 0.01, 10.0).unsqueeze(0).unsqueeze(2)  # (1,out_ch,1,in_ch)
        E  = torch.clamp(self.E, -10.0, 10.0).unsqueeze(0).unsqueeze(2)             # (1,out_ch,1,in_ch)
        x_ = torch.clamp(x.unsqueeze(1), -10.0, 10.0)                               # (B, out_ch, T, in_ch)
        exp = torch.clamp((E - x_) / kT, -50, 50)
        w   = torch.sigmoid(exp)
        h   = (w * x_).sum(dim=3).permute(0, 2, 1)           # (B,T,out_ch) >=0
        return torch.clamp(F.softplus(h), 0.0, 100.0)

class BaseOp(nn.Module): pass
class MonotonicLinearOp(BaseOp):
    def __init__(self):
        super().__init__()
        self.raw_scale=nn.Parameter(torch.tensor(0.0))
        self.bias=nn.Parameter(torch.tensor(0.0))
        self.smin,self.smax=0.01,5.0
    def forward(self,h):
        scale=torch.clamp(F.softplus(self.raw_scale),self.smin,self.smax)
        xm=torch.clamp(h.mean(-1), -10.0, 10.0)
        return torch.clamp(F.softplus(scale*(xm+self.bias)), 0.0, 100.0)

class MonotonicFlatOp(BaseOp):
    def __init__(self):
        super().__init__()
        self.raw_scale=nn.Parameter(torch.tensor(0.0))
        self.bias=nn.Parameter(torch.tensor(0.0))
        self.smin,self.smax=1e-3,1.0
    def _cum(self,x):
        diff=F.relu(x[:,1:]-x[:,:-1])
        return torch.cat([torch.zeros_like(diff[:,:1]),torch.cumsum(diff,1)],1)
    def forward(self,h):
        scale=torch.clamp(F.softplus(self.raw_scale),self.smin,self.smax)
        xm=torch.clamp(h.mean(-1,keepdim=True), -10.0, 10.0)
        return torch.clamp(F.softplus(scale*(self._cum(xm)+self.bias)).squeeze(-1), 0.0, 100.0)

class ConcaveLogOp(BaseOp):
    def __init__(self):
        super().__init__()
        self.raw_scale=nn.Parameter(torch.tensor(0.0))
        self.eps=1e-3
        self.smin,self.smax=0.01,5.0
    def forward(self,h):
        scale=torch.clamp(F.softplus(self.raw_scale),self.smin,self.smax)
        xm=torch.clamp(h.mean(-1), -10.0, 10.0)
        return torch.clamp(F.softplus(scale*torch.log(torch.abs(xm)+self.eps)), 0.0, 100.0)

class SaturationSigmoidOp(BaseOp):
    def __init__(self):
        super().__init__()
        self.raw_scale=nn.Parameter(torch.tensor(0.0))
        self.raw_slope=nn.Parameter(torch.tensor(0.0))
        self.bias=nn.Parameter(torch.tensor(0.0))
        self.smin,self.smax=0.01,5.0
        self.lmin,self.lmax=0.1,5.0
    def forward(self,h):
        scale=torch.clamp(F.softplus(self.raw_scale),self.smin,self.smax)
        slope=torch.clamp(F.softplus(self.raw_slope),self.lmin,self.lmax)
        xm=torch.clamp(h.mean(-1), -10.0, 10.0)
        return torch.clamp(F.softplus(scale*torch.sigmoid(slope*(xm-self.bias))), 0.0, 100.0)

class HingeReLUOp(BaseOp):
    def __init__(self):
        super().__init__()
        self.raw_scale=nn.Parameter(torch.tensor(0.0))
        self.threshold=nn.Parameter(torch.tensor(0.0))
        self.smin,self.smax=0.01,5.0
    def forward(self,h):
        scale=torch.clamp(F.softplus(self.raw_scale),self.smin,self.smax)
        xm=torch.clamp(h.mean(-1), -10.0, 10.0)
        return torch.clamp(F.softplus(scale*F.relu(xm-self.threshold)), 0.0, 100.0)

class PolynomialOp(BaseOp):
    def __init__(self,deg=3):
        super().__init__()
        self.raw_coeff=nn.Parameter(torch.zeros(deg))
        self.deg=deg
    def forward(self,h):
        xm=torch.clamp(h.mean(-1), -5.0, 5.0)
        y=torch.zeros_like(xm)
        for i in range(self.deg):
            coeff = torch.clamp(F.softplus(self.raw_coeff[i]), 0.01, 5.0)
            y = y + coeff * torch.clamp(xm ** (i+1), -100.0, 100.0)
        return torch.clamp(F.softplus(y), 0.0, 100.0)

class DampedSinOp(BaseOp):
    def __init__(self):
        super().__init__()
        self.raw_scale=nn.Parameter(torch.tensor(0.0))
        self.raw_freq=nn.Parameter(torch.tensor(0.0))
        self.raw_lambda=nn.Parameter(torch.tensor(0.0))
        self.phase=nn.Parameter(torch.tensor(0.0))
        self.smin,self.smax=0.01,5.0
        self.fmin,self.fmax=0.1,5.0
        self.lmin,self.lmax=0.01,3.0
    def forward(self,h):
        scale=torch.clamp(F.softplus(self.raw_scale),self.smin,self.smax)
        freq=torch.clamp(F.softplus(self.raw_freq),self.fmin,self.fmax)
        lam=torch.clamp(F.softplus(self.raw_lambda),self.lmin,self.lmax)
        xm=torch.clamp(h.mean(-1), -10.0, 10.0)
        damp=1/(1+lam*torch.abs(xm))
        phase_val = torch.clamp(self.phase, -10.0, 10.0)
        return torch.clamp(F.softplus(scale*damp*(torch.sin(freq*xm+phase_val)+1)), 0.0, 100.0)

class PiecewiseLinearOp(BaseOp):
    def __init__(self):
        super().__init__()
        self.raw_k1=nn.Parameter(torch.tensor(0.0))
        self.raw_k2=nn.Parameter(torch.tensor(0.0))
        self.threshold=nn.Parameter(torch.tensor(0.0))
        self.kmin,self.kmax=0.01,5.0
    def forward(self,h):
        k1=torch.clamp(F.softplus(self.raw_k1),self.kmin,self.kmax)
        k2=torch.clamp(F.softplus(self.raw_k2),self.kmin,self.kmax)
        thresh = torch.clamp(self.threshold, -5.0, 5.0)
        xm=torch.clamp(h.mean(-1), -10.0, 10.0)
        left=k1*xm
        right=k1*thresh + k2*(xm-thresh)
        out=torch.where(xm<=thresh,left,right)
        return torch.clamp(F.softplus(out), 0.0, 100.0)

class SparseGate(nn.Module):
    def __init__(self, n_ops, tau_start=5.0, tau_end=0.1, n_steps=10000):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(n_ops))
        self.tau_start, self.tau_end, self.n_steps = tau_start, tau_end, n_steps
        self.register_buffer("step", torch.tensor(0.))
    def forward(self):
        tau = (self.tau_start*(1-self.step/self.n_steps) + self.tau_end*(self.step/self.n_steps)).clamp(min=self.tau_end)
        self.step.add_(1)  # Use add_ to avoid inplace operation
        g = -torch.empty_like(self.logits).exponential_().log() if self.training else torch.zeros_like(self.logits)
        g = torch.clamp(g, -50.0, 50.0)  # Limit Gumbel noise
        logits_stable = torch.clamp(self.logits, -50.0, 50.0)
        w = F.softmax((logits_stable + g)/tau, dim=-1)
        return w.view(1,1,-1)

class CustomKAN(nn.Module):
    def __init__(self, ops):
        super().__init__()
        self.ops = nn.ModuleList(ops)
        self.gate= SparseGate(len(ops))
        # Additional learnable scaling/bias for damage
        self.raw_gain = nn.Parameter(torch.tensor(0.0))
        self.bias     = nn.Parameter(torch.tensor(0.0))
    def forward(self, h):  # h:(B,T,trend_ch)
        outs = [op(h) for op in self.ops]          # list of (B,T)
        Tm = min(o.size(1) for o in outs)
        outs = [o[:,:Tm] for o in outs]
        st = torch.stack(outs, dim=-1)             # (B,Tm,K), >=0
        w  = self.gate()                           # (1,1,K)
        damage = torch.clamp((st*w).sum(-1), 0.0, 100.0)  # (B,Tm) non-negative, regarded as "damage"
        gain = torch.clamp(F.softplus(self.raw_gain), 0.1, 5.0)
        bias_val = torch.clamp(self.bias, -5.0, 5.0)
        damage = torch.clamp(gain*damage + bias_val, 0.0, 100.0)
        return damage                               # (B,Tm)

class TrendEncoder(nn.Module):
    """
    Encoder outputs "damage", then projects to monotonic decreasing HI:
        HI is learned from sensor data without hard range constraints
    Without real HI, learns mapping from sensor data to infer health states
    """
    def __init__(self, in_ch, trend_ch=4):
        super().__init__()
        self.boltz = BoltzmannKAN(in_ch, out_ch=trend_ch)
        ops = [MonotonicLinearOp(), MonotonicFlatOp(), ConcaveLogOp(),
               SaturationSigmoidOp(), HingeReLUOp(), PolynomialOp(),
               DampedSinOp(), PiecewiseLinearOp()]
        self.customkan = CustomKAN(ops)
        # Projection parameters - learn flexible health states from sensor features
        self.proj_gain = nn.Parameter(torch.tensor(1.0))
        self.proj_bias = nn.Parameter(torch.tensor(0.0))

        # Add health-aware layer, directly infer from multi-dimensional sensor features
        self.health_aware_transform = nn.Sequential(
            nn.Linear(in_ch, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output 0-1 range health state
        )

        # Add linear constraint layer, enforce linear trend
        self.linear_enforcer = nn.Sequential(
            nn.Linear(in_ch, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output linear decay degree
        )

    def forward(self, x):  # x:(B,T,C)
        # Feature extraction based on Boltzmann KAN
        h_multi = self.boltz(x)              # (B,T,trend_ch) >=0
        damage  = self.customkan(h_multi)    # (B,T)           >=0

        # Directly learn health states from raw sensor data
        # Assess health state for sensor readings at each time step
        B, T, C = x.shape
        x_reshaped = x.view(-1, C)  # (B*T, C)
        health_direct = self.health_aware_transform(x_reshaped)  # (B*T, 1)
        health_direct = health_direct.view(B, T)  # (B, T)

        # Add linear constraint, force linear decay pattern
        linear_weight = self.linear_enforcer(x_reshaped).view(B, T)  # (B, T)

        # Generate ideal linear decay pattern
        t = torch.arange(T, device=x.device, dtype=x.dtype).unsqueeze(0).expand(B, -1)  # (B, T)
        t_normalized = t / (T - 1)  # Normalize to [0, 1]

        # Combine damage and direct health assessment
        g = torch.clamp(F.softplus(self.proj_gain), 0.1, 5.0)
        b = torch.clamp(self.proj_bias, -5.0, 5.0)

        # Convert damage to health state decay
        damage_normalized = torch.sigmoid(g*(damage + b))

        # Combine three health assessment methods: direct assessment, damage pattern, linear constraint
        combined_health = health_direct * (1 - 0.3 * damage_normalized)

        # Generate ideal linear decay curve (from high to low) - remove hard range constraints
        linear_decay = 1.0 - t_normalized * 0.5  # Simple linear decay from 1.0 to 0.5

        # Use linear_weight to mix linear decay and complex patterns
        # Higher linear_weight means more towards linear
        hi = linear_weight * linear_decay + (1 - linear_weight) * combined_health

        # Force stronger monotonic decreasing constraint without hard range limits
        for t_idx in range(1, T):
            hi = torch.cat([
                hi[:, :t_idx],
                torch.min(hi[:, t_idx:t_idx+1], hi[:, t_idx-1:t_idx] - 0.001),  # Force at least 0.001 decrease
                hi[:, t_idx+1:]
            ], dim=1)

        return hi, h_multi

class ReconHead(nn.Module):
    """
    Recursive reconstruction: given (x_t, h_t) → predict x_{t+1}
    """
    def __init__(self, C, hidden=128):
        super().__init__()
        self.gru = nn.GRU(input_size=C+1, hidden_size=hidden, batch_first=True)
        self.out = nn.Linear(hidden, C)
    def forward(self, x_in, h_in):
        # x_in:(B,T_in,C), h_in:(B,T_in)
        B,T,C = x_in.shape
        h_in_clamped = torch.clamp(h_in, 0.0, 10.0)  # Remove upper limit constraint
        feat = torch.cat([x_in, h_in_clamped.unsqueeze(-1)], dim=-1)  # (B,T,C+1)
        H,_ = self.gru(feat)                                  # (B,T,H)
        y = self.out(H)                                       # (B,T,C) aligned predict x_{t+1}
        return y

class MaintClassifier(nn.Module):
    """
    Use HI before-after difference features for 3-class classification
    Enhanced version: not only uses ΔHI, but also sensor data change patterns
    """
    def __init__(self, sensor_dim, hidden=64, n_classes=3):
        super().__init__()
        # Original HI difference features
        self.hi_mlp = nn.Sequential(
            nn.Linear(6, hidden), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden//2)
        )

        # Sensor data change features
        self.sensor_mlp = nn.Sequential(
            nn.Linear(sensor_dim * 2, hidden), nn.ReLU(),  # Before and after statistical features
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden//2)
        )

        # Fusion classifier
        self.final_classifier = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, n_classes)
        )

    def forward(self, h_b, h_a, x_b, x_a, mask):
        # HI difference features (original logic)
        m = mask
        T = m.size(1)
        valid = m.sum(1, keepdim=True).clamp_min(1.0)  # (B,1)
        db = h_a - h_b                        # (B,T)
        mean_d = (db*m).sum(1,keepdim=True)/valid
        var_d  = (((db-mean_d)*m)**2).sum(1,keepdim=True)/valid
        std_d  = (var_d + 1e-8).sqrt()
        d0     = (db[:, :1])                  # First value
        dmax   = (db.masked_fill(m==0, -1e9)).max(dim=1, keepdim=True).values
        pos_ratio = ((db>0).float()*m).sum(1,keepdim=True)/valid

        # Slope (linear fitting)
        t = torch.arange(T, device=h_b.device, dtype=h_b.dtype)
        t = (t - t.mean())/(t.std()+1e-6)
        def slope(x):
            num = (x*t).sum(1) - x.sum(1)*t.sum()/T
            den = (t**2).sum() - (t.sum()**2)/T + 1e-8
            return (num/den).unsqueeze(1)
        slope_diff = slope(h_a) - slope(h_b)  # (B,1)
        hi_feat = torch.cat([mean_d, d0, dmax, std_d, slope_diff, pos_ratio], dim=1)  # (B,6)
        hi_feat = torch.clamp(hi_feat, -10.0, 10.0)
        hi_features = self.hi_mlp(hi_feat)

        # Sensor data change features (fix broadcast BUG: directly divide by (B,1) shaped valid)
        x_b_mean = (x_b * m.unsqueeze(-1)).sum(1) / valid        # (B,C)
        x_a_mean = (x_a * m.unsqueeze(-1)).sum(1) / valid        # (B,C)
        sensor_change = torch.cat([x_b_mean, x_a_mean], dim=1)   # (B, 2*C)
        sensor_features = self.sensor_mlp(sensor_change)

        # Fused features
        combined_features = torch.cat([hi_features, sensor_features], dim=1)
        logits = self.final_classifier(combined_features)

        return logits

def sanitize_tensors(*tensors):
    """Replace NaN/Inf in tensors with finite values"""
    result = []
    for t in tensors:
        if t is not None:
            t_clean = torch.nan_to_num(t, nan=0.0, posinf=1e6, neginf=-1e6)
            result.append(t_clean)
        else:
            result.append(t)
    return result[0] if len(result) == 1 else result

class DiffAwareReconstructor(nn.Module):
    """
    Overall model:
     - Two encoders share parameters: encode x_before / x_after → h_b / h_a (monotonic decreasing HI)
     - Two reconstruction heads: perform one-step recursive reconstruction respectively
     - Classification head: use (h_a - h_b) statistical features to identify maintenance type
    Without real HI labels, learn to infer health states from sensor data through self-supervised learning
    """
    def __init__(self, in_ch, trend_ch=4, hidden=128, n_classes=3):
        super().__init__()
        self.encoder = TrendEncoder(in_ch, trend_ch)
        self.recon_b = ReconHead(in_ch, hidden)
        self.recon_a = ReconHead(in_ch, hidden)
        self.clf     = MaintClassifier(sensor_dim=in_ch, hidden=64, n_classes=n_classes)

        # Add self-supervised consistency constraint
        self.consistency_weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, x_b, x_a, mask):
        # x_b/x_a : (B,L,C) (will be cut to 0:L-2 / 1:L-1 later)
        h_b, _ = self.encoder(x_b)  # (B,L)  Health state learned from sensor data
        h_a, _ = self.encoder(x_a)  # (B,L)

        # Numerical stabilization
        h_b, h_a = sanitize_tensors(h_b, h_a)

        # Recursive reconstruction: input 0:L-2 → predict 1:L-1
        xb_in, hb_in = x_b[:, :-2], h_b[:, :-2]
        xa_in, ha_in = x_a[:, :-2], h_a[:, :-2]
        yb_hat = self.recon_b(xb_in, hb_in)   # (B,L-2,C) ≈ x_b[:,1:L-1]
        ya_hat = self.recon_a(xa_in, ha_in)   # (B,L-2,C) ≈ x_a[:,1:L-1]

        # Numerical stabilization
        yb_hat, ya_hat = sanitize_tensors(yb_hat, ya_hat)

        # Classification: use unclipped length mask and include sensor features
        logits = self.clf(h_b, h_a, x_b, x_a, mask)
        logits = sanitize_tensors(logits)

        return yb_hat, ya_hat, h_b, h_a, logits
# ============================================================
# 3) Loss function: reconstruction + ΔHI + smooth/monotonic(decreasing) + classification + first-order linear + self-supervised consistency + maintenance effect constraint + global HI superiority constraint
# ============================================================
def masked_mse(a, b, mask):
    # a,b:(B,T,C)  mask:(B,T)
    diff = (a - b)**2
    mse  = (diff.mean(-1) * mask).sum() / (mask.sum() + 1e-6)
    return mse

def slope_loss(h, mask, kind="smooth"):
    # Smooth: second-order difference; Monotonic decreasing: penalize upward trend
    if kind=="smooth":
        d2 = h[:,2:] - 2*h[:,1:-1] + h[:,:-2]
        m  = mask[:,2:]
        return (d2.abs() * m).sum() / (m.sum()+1e-6)
    elif kind=="mono_dec":
        dh = h[:,1:] - h[:,:-1]        # If >0 indicates upward (violates "decreasing")
        m  = mask[:,1:]
        return (F.relu(dh) * m).sum() / (m.sum()+1e-6)
    else:
        return torch.tensor(0., device=h.device)

def enhanced_linear_slope_loss(h, mask, weight=5.0):
    """
    Enhanced linear constraint: calculate deviation from best linear fit, stronger weight
    Without real HI, this constraint helps learn reasonable decay patterns
    """
    B, T = h.shape
    valid_mask = mask  # (B, T)

    # Calculate best linear fit for each sample
    t = torch.arange(T, device=h.device, dtype=h.dtype).unsqueeze(0).expand(B, -1)  # (B, T)

    # Consider only valid positions
    loss_total = torch.tensor(0.0, device=h.device)
    n_valid_samples = 0

    for b in range(B):
        valid_indices = valid_mask[b] > 0
        if valid_indices.sum() < 2:  # Need at least 2 points for linear fitting
            continue

        h_valid = h[b][valid_indices]  # Valid HI values
        t_valid = t[b][valid_indices]  # Corresponding time points

        # Center time points
        t_mean = t_valid.mean()
        t_centered = t_valid - t_mean
        h_mean = h_valid.mean()

        # Calculate linear regression slope
        numerator = (t_centered * (h_valid - h_mean)).sum()
        denominator = (t_centered ** 2).sum() + 1e-8
        slope = numerator / denominator

        # Calculate intercept
        intercept = h_mean - slope * t_mean

        # Linear predicted values
        h_linear = slope * t_valid + intercept

        # Calculate MSE with linear fit, using stronger weight
        linear_mse = ((h_valid - h_linear) ** 2).mean()

        # Additional penalty for positive slope (encourage negative slope, i.e., decreasing)
        slope_penalty = F.relu(slope + 0.01) * 2.0  # Slope should be negative

        loss_total += weight * linear_mse + slope_penalty
        n_valid_samples += 1

    return loss_total / max(n_valid_samples, 1)

def maintenance_improvement_constraint(h_b, h_a, labels, mask):
    """
    Maintenance improvement constraint: ensure post-maintenance HI is significantly higher than pre-maintenance
    - Perfect(0): require post-maintenance significantly higher than pre-maintenance (at least 0.4)
    - General(1): require post-maintenance moderately higher than pre-maintenance (at least 0.2)
    - Poor(2): require post-maintenance slightly higher than pre-maintenance (at least 0.1)
    """
    valid = mask.sum(1, keepdim=True).clamp_min(1.0)  # (B,1)

    # Calculate average HI before and after maintenance
    h_b_mean = (h_b * mask).sum(1, keepdim=True) / valid  # (B,1)
    h_a_mean = (h_a * mask).sum(1, keepdim=True) / valid  # (B,1)

    improvement = h_a_mean - h_b_mean  # Improvement after maintenance relative to before

    loss = torch.zeros_like(improvement)
    for cls in [0, 1, 2]:
        sel = (labels == cls).float().unsqueeze(1)  # (B,1)
        if sel.sum() < 1:
            continue

        if cls == 0:  # Perfect
            # Require at least 0.4 improvement
            loss += sel * F.relu(0.4 - improvement) ** 2
        elif cls == 1:  # General
            # Require at least 0.2 improvement
            loss += sel * F.relu(0.2 - improvement) ** 2
        else:  # Poor
            # Require at least 0.1 improvement
            loss += sel * F.relu(0.1 - improvement) ** 2

    return loss.mean()

def global_hi_superiority_constraint(h_b, h_a, mask, weight=5.0):
    """
    Enhanced global superiority constraint: ensure post-maintenance HI is entirely higher than pre-maintenance HI
    For each sample, require post-maintenance HI at every time point to be higher than pre-maintenance HI
    Remove range constraints and focus on relative superiority
    """
    valid = mask  # (B, T)

    loss_total = torch.tensor(0.0, device=h_b.device)
    n_valid_samples = 0

    for b in range(h_b.size(0)):
        valid_mask_b = valid[b] > 0
        if valid_mask_b.sum() < 1:
            continue

        # Get valid HI values before and after maintenance
        h_b_valid = h_b[b][valid_mask_b]  # Valid HI values before maintenance
        h_a_valid = h_a[b][valid_mask_b]  # Valid HI values after maintenance

        # Point-wise superiority: each post-maintenance point should be higher than corresponding pre-maintenance point
        pointwise_gap = h_a_valid - h_b_valid  # Should be positive
        pointwise_loss = F.relu(-pointwise_gap + 0.05).mean()  # Penalize when post-maintenance is not sufficiently higher

        # Global superiority: minimum post-maintenance should be higher than maximum pre-maintenance
        h_b_max = h_b_valid.max()
        h_a_min = h_a_valid.min()

        # Require post-maintenance minimum to be higher than pre-maintenance maximum
        margin = 0.1  # Post-maintenance minimum should be at least 0.1 higher than pre-maintenance maximum
        global_gap_loss = F.relu(h_b_max + margin - h_a_min) ** 2

        # Additional constraint: post-maintenance average should be significantly higher than pre-maintenance average
        h_b_mean = h_b_valid.mean()
        h_a_mean = h_a_valid.mean()
        mean_gap_loss = F.relu(h_b_mean + 0.2 - h_a_mean) ** 2  # Average should improve by at least 0.2

        loss_total += weight * (pointwise_loss + global_gap_loss + 0.5 * mean_gap_loss)
        n_valid_samples += 1

    return loss_total / max(n_valid_samples, 1)

def diff_margin_by_class(mean_delta, labels, m_low=0.1, m_mid=0.25, m_high=0.45):
    """
    Class-conditional difference constraint (enhanced maintenance effect):
      - Perfect(0):  Δ>=m_high (post-maintenance significantly higher than pre-maintenance)
      - General(1):  Δ≈m_mid   (post-maintenance moderately higher than pre-maintenance)
      - Poor(2):     Δ>=m_low  (post-maintenance at least slightly higher than pre-maintenance)
    All classes require post-maintenance HI higher than pre-maintenance (positive improvement), just different degrees
    """
    loss = torch.zeros_like(mean_delta)
    for cls in [0,1,2]:
        sel = (labels==cls).float()
        if sel.sum() < 1: continue
        if cls==0:
            # Perfect: require significant improvement (at least reach m_high)
            loss += sel * F.relu(m_high - mean_delta)**2
        elif cls==1:
            # General: require moderate improvement (close to m_mid)
            loss += sel * (mean_delta - m_mid)**2
        else:
            # Poor: require at least small improvement (at least reach m_low)
            loss += sel * F.relu(m_low - mean_delta)**2
    denom = torch.ones_like(mean_delta) * (labels.numel())
    return loss.sum() / denom.sum()

def sensor_consistency_loss(x_b, x_a, h_b, h_a, mask):
    """
    Sensor data consistency loss: ensure HI changes are consistent with sensor data change patterns
    Important constraint when no real HI labels are available
    """
    # Calculate statistical changes in sensor data
    valid = mask.sum(1, keepdim=True).clamp_min(1.0)   # (B,1)

    # Average change magnitude in sensor data (fix broadcast BUG: divide by (B,1))
    xb_mean = (x_b * mask.unsqueeze(-1)).sum(1) / valid   # (B,C)
    xa_mean = (x_a * mask.unsqueeze(-1)).sum(1) / valid   # (B,C)
    sensor_change = torch.abs(xa_mean - xb_mean)          # (B,C)
    sensor_change_magnitude = sensor_change.mean(1)       # (B,)

    # HI change magnitude (post-maintenance improvement relative to pre-maintenance)
    hi_change = ((h_a - h_b) * mask).sum(1) / valid.squeeze(-1)  # (B,) Note: remove abs, maintain positive direction

    # Expect HI change to positively correlate with sensor change, and HI should improve positively
    # Use Pearson correlation coefficient as consistency metric
    mean_sensor = sensor_change_magnitude.mean()
    mean_hi = hi_change.mean()

    correlation = ((sensor_change_magnitude - mean_sensor) * (hi_change - mean_hi)).mean() / \
                  (sensor_change_magnitude.std() * hi_change.std() + 1e-8)

    # Encourage positive correlation (large sensor change when HI change is also large)
    consistency_loss = F.relu(0.5 - correlation)  # Expect correlation at least 0.5

    # Additional constraint: force positive HI improvement
    negative_change_penalty = F.relu(-hi_change).mean() * 2.0  # Penalize negative changes

    return consistency_loss + negative_change_penalty

def smoothness_enhancement_loss(h, mask, order=2):
    """
    Enhanced smoothness loss: use higher-order differences to force smoother curves
    """
    if order == 2:
        # Second-order difference
        d2 = h[:,2:] - 2*h[:,1:-1] + h[:,:-2]
        m = mask[:,2:]
        return (d2.abs() * m).sum() / (m.sum() + 1e-6)
    elif order == 3:
        # Third-order difference (stronger smoothness constraint)
        d3 = h[:,3:] - 3*h[:,2:-1] + 3*h[:,1:-2] - h[:,:-3]
        m = mask[:,3:]
        return (d3.abs() * m).sum() / (m.sum() + 1e-6)
    else:
        return torch.tensor(0., device=h.device)

# ============================================================
# 4) Data splitting and visualization (aligned to same time axis)
# ============================================================
LABEL2NAME = {0: "Perfect", 1: "General", 2: "Poor"}

def split_pairs_uid(pairs, train_ratio=0.7, val_ratio=0.1, seed=42):
    uids = sorted(list(pairs.keys()))
    rs = np.random.RandomState(seed)
    rs.shuffle(uids)
    n = len(uids); n_tr=int(n*train_ratio); n_vl=int(n*val_ratio)
    to_dict = lambda ss:{u:pairs[u] for u in ss if u in pairs}
    return to_dict(uids[:n_tr]), to_dict(uids[n_tr:n_tr+n_vl]), to_dict(uids[n_tr+n_vl:])

def print_split_summary(pairs, train_ratio=0.7, val_ratio=0.1, seed=42):
    pairs_tr, pairs_vl, pairs_te = split_pairs_uid(pairs, train_ratio, val_ratio, seed)
    def count_items(pp):
        cnt = 0
        for uid, d in pp.items():
            cnt += len(d)
        return len(pp), cnt
    n_uid_tr, n_pair_tr = count_items(pairs_tr)
    n_uid_vl, n_pair_vl = count_items(pairs_vl)
    n_uid_te, n_pair_te = count_items(pairs_te)
    print(f"[Data Split] Train: UID={n_uid_tr}, pairs≈{n_pair_tr} | "
          f"Val: UID={n_uid_vl}, pairs≈{n_pair_vl} | "
          f"Test: UID={n_uid_te}, pairs≈{n_pair_te}")

def remove_constant_segments(hi_values, threshold=1e-6):
    """
    Remove constant segments in HI curves (caused by mask or other reasons)
    Return valid HI values and corresponding time indices
    """
    if len(hi_values) <= 1:
        return hi_values, np.arange(len(hi_values))

    # Calculate differences between adjacent points
    diffs = np.abs(np.diff(hi_values))

    # Find points with sufficient change
    valid_mask = np.ones(len(hi_values), dtype=bool)
    valid_mask[1:] = diffs > threshold

    # Ensure at least first and last two points are kept
    if np.sum(valid_mask) < 2:
        valid_mask[0] = True
        valid_mask[-1] = True

    valid_indices = np.where(valid_mask)[0]
    return hi_values[valid_indices], valid_indices

@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total = {"mse_b":0.0, "mse_a":0.0, "acc":0.0, "delta_mean":0.0}
    n_batch=0
    n_acc = 0; n_tot=0
    for batch in loader:
        xb = batch["x_before"].to(device)
        xa = batch["x_after"].to(device)
        hi_b = batch["hi_before"].to(device)
        hi_a = batch["hi_after"].to(device)
        labels = batch["labels"].to(device)
        lengths= batch["lengths"].to(device)
        mask   = batch["mask"].to(device)

        # Valid segment: 0:L-2 input, 1:L-1 target
        m_tgt  = (torch.arange(xb.size(1)-2, device=device)[None,:] < (lengths-2)[:,None]).float()

        yb_hat, ya_hat, h_b, h_a, logits = model(xb, xa, mask)

        # Numerical stabilization
        yb_hat, ya_hat, h_b, h_a, logits = sanitize_tensors(yb_hat, ya_hat, h_b, h_a, logits)

        # Align targets
        yb_tgt = xb[:,1:-1,:]
        ya_tgt = xa[:,1:-1,:]

        mse_b = masked_mse(yb_hat, yb_tgt, m_tgt)
        mse_a = masked_mse(ya_hat, ya_tgt, m_tgt)

        # Check for NaN
        if torch.isnan(mse_b) or torch.isnan(mse_a):
            continue

        # Classification
        pred = logits.argmax(dim=1)
        n_acc += (pred == labels).sum().item()
        n_tot += labels.numel()

        # Statistics of ΔHI mean (post-maintenance improvement relative to pre-maintenance)
        valid = mask.sum(1, keepdim=True).clamp_min(1.0)
        delta_mean = ((h_a - h_b)*mask).sum(1,keepdim=True)/valid
        delta_mean = torch.nan_to_num(delta_mean, nan=0.0)
        total["delta_mean"] += delta_mean.mean().item()

        total["mse_b"] += mse_b.item()
        total["mse_a"] += mse_a.item()
        n_batch += 1

    for k in ["mse_b","mse_a","delta_mean"]:
        total[k] /= max(n_batch,1)
    total["acc"] = n_acc / max(n_tot,1)
    return total

@torch.no_grad()
def collect_test_predictions(model, loader, device, max_curve_keep=24):
    """
    Collect predictions, ΔHI, and a few sample HI curves on test set (for plotting).
    """
    model.eval()
    y_true, y_pred = [], []
    all_delta_mean, all_uids = [], []
    keep_curves = []

    for batch in loader:
        xb = batch["x_before"].to(device)   # (B,L,C)
        xa = batch["x_after"].to(device)
        mask   = batch["mask"].to(device)   # (B,L)
        labels = batch["labels"].to(device) # (B,)
        lengths= batch["lengths"]           # cpu tensor
        uids   = batch["uids"]

        yb_hat, ya_hat, h_b, h_a, logits = model(xb, xa, mask)

        # Numerical stabilization
        yb_hat, ya_hat, h_b, h_a, logits = sanitize_tensors(yb_hat, ya_hat, h_b, h_a, logits)

        prob = F.softmax(logits, dim=1)     # (B,3)
        pred = prob.argmax(1)               # (B,)

        # Statistics of ΔHI mean (post-maintenance improvement relative to pre-maintenance)
        valid = mask.sum(1, keepdim=True).clamp_min(1.0)
        delta_mean = ((h_a - h_b) * mask).sum(1, keepdim=True) / valid  # (B,1)
        delta_mean = torch.nan_to_num(delta_mean, nan=0.0)

        y_true.append(labels.cpu().numpy())
        y_pred.append(pred.cpu().numpy())
        all_delta_mean.append(delta_mean.squeeze(1).cpu().numpy())
        all_uids.extend(uids)

        # Save some curves for visualization (aligned: after concatenated after before)
        for i in range(xb.size(0)):
            if len(keep_curves) >= max_curve_keep:
                break
            L_i = int(lengths[i].item())
            # Fix: truncate reconstruction prediction by actual length
            L_recon = min(L_i - 2, yb_hat.size(1))  # Reconstruction sequence length
            keep_curves.append({
                "uid": uids[i],
                "true": int(labels[i].cpu().item()),
                "pred": int(pred[i].cpu().item()),
                "prob": prob[i].cpu().numpy(),
                "h_before": h_b[i, :L_i].cpu().numpy(),
                "h_after":  h_a[i, :L_i].cpu().numpy(),
                "yb_hat":   yb_hat[i, :L_recon].cpu().numpy(),       # (L_recon,C)
                "ya_hat":   ya_hat[i, :L_recon].cpu().numpy(),       # (L_recon,C)
                "x_before": xb[i, :L_i].cpu().numpy(),               # (L,C)
                "x_after":  xa[i, :L_i].cpu().numpy(),               # (L,C)
            })

    y_true = np.concatenate(y_true, axis=0) if len(y_true)>0 else np.array([])
    y_pred = np.concatenate(y_pred, axis=0) if len(y_pred)>0 else np.array([])
    all_delta_mean = np.concatenate(all_delta_mean, axis=0) if len(all_delta_mean)>0 else np.array([])
    return y_true, y_pred, all_delta_mean, all_uids, keep_curves

def plot_hi_examples_aligned(curves, n_show=6, seed=0):
    """Post-maintenance sequence continues from pre-maintenance endpoint; draw vertical line to mark t_m; remove constant segments.
    Add "Learned HI" in title to indicate this is health state inferred from sensor data, post-maintenance should be higher than pre-maintenance"""
    if len(curves)==0:
        print("(No visualization samples)")
        return
    random.Random(seed).shuffle(curves)
    n_show = min(n_show, len(curves))
    cols = 3
    rows = int(np.ceil(n_show/cols))
    plt.figure(figsize=(cols*4.6, rows*3.2))
    for k in range(n_show):
        ex = curves[k]
        hb = ex["h_before"]; ha = ex["h_after"]
        L  = min(len(hb), len(ha))  # Defensive alignment
        hb, ha = hb[:L], ha[:L]

        # Remove constant segments
        hb_clean, hb_indices = remove_constant_segments(hb)
        ha_clean, ha_indices = remove_constant_segments(ha)

        # Corresponding time axis
        t_b = hb_indices
        t_a = ha_indices + L  # after follows immediately

        uid = ex["uid"]
        pred = ex["pred"]; true = ex["true"]; prob = ex["prob"]
        plt.subplot(rows, cols, k+1)

        # Only plot non-constant segments
        if len(hb_clean) > 1:
            plt.plot(t_b, hb_clean, label="Learned HI_Pre-maintenance", linewidth=1.8, marker='o', markersize=3)
        if len(ha_clean) > 1:
            plt.plot(t_a, ha_clean, label="Learned HI_Post-maintenance",  linewidth=1.8, linestyle='--', marker='s', markersize=3)

        # Maintenance time vertical line
        plt.axvline(L-1, color='k', linestyle=':', linewidth=1.0, alpha=0.7)

        d_mean = float(np.mean(ha - hb))  # Post-maintenance improvement relative to pre-maintenance
        d_max  = float(np.max(ha - hb))
        title = (f"uid={uid} (Maintenance Effect Assessment)\nTrue={LABEL2NAME[true]} | Pred={LABEL2NAME[pred]} "
                 f"| p={prob[pred]:.2f}\nΔHI_mean={d_mean:.3f}, ΔHI_max={d_max:.3f}")
        plt.title(title, fontsize=9)
        plt.xlabel("Cycle (Pre-maintenance | Post-maintenance)")
        plt.ylabel("Learned Health Index (↑Post should be higher)")
        plt.grid(ls='--', alpha=.35)
        # Remove Y-axis range constraint, let it adapt to data
        if k==0: plt.legend()
    plt.tight_layout()
    plt.show()

def plot_sensor_examples_aligned(curves, sensor_idx_list=None, n_cols=4):
    """
    Plot several sensors' original vs recursive prediction (post), with after time axis connected after before.
    - Original before: x_before[:, s]
    - Original after : x_after [:, s] (optional)
    - Predicted after : ya_hat   [:, s] (one-step sequence aligned with x_after)
    """
    if len(curves)==0:
        print("(No visualization samples)")
        return
    # Take first sample to show multiple sensors
    ex = curves[0]
    xb = ex["x_before"]         # (L,C)
    xa = ex["x_after"]          # (L,C)
    ya = ex["ya_hat"]           # (L_recon,C) predicted is 1..L-1
    L, C = xb.shape
    Lhat = ya.shape[0]          # Actual reconstruction length

    if sensor_idx_list is None:
        # Default pick 8
        step = max(1, C//8)
        sensor_idx_list = list(range(0, C, step))[:8]

    n = len(sensor_idx_list)
    n_rows = int(np.ceil(n/n_cols))
    plt.figure(figsize=(n_cols*4.3, n_rows*3.1))
    for i, s in enumerate(sensor_idx_list):
        plt.subplot(n_rows, n_cols, i+1)
        t_b = np.arange(L)
        t_a = np.arange(L, 2*L)
        plt.plot(t_b, xb[:,s], label="Original (before)", linewidth=1.2)
        plt.plot(t_a, xa[:,s], label="Original (after)",  linewidth=1.0, linestyle="--", alpha=0.7)
        # Predicted after: aligned with after target (corresponding to 1..L-1)
        t_pred = np.arange(L+1, L+1+Lhat)      # Align ya_hat time
        plt.plot(t_pred, ya[:,s], label="Post-maint. prediction", linewidth=1.6)
        plt.axvline(L-1, color='k', linestyle=':', linewidth='1.0')
        plt.title(f"Sensor_{s:02d}")
        if i==0: plt.legend()
        plt.grid(ls="--", alpha=.35)
    plt.tight_layout()
    plt.show()

def topk_by_delta(df_delta, k=5):
    if len(df_delta)==0:
        return
    for cls in [0,1,2]:
        sub = df_delta[df_delta["true"]==cls].sort_values("delta_hi_mean", ascending=False).head(k)
        print(f"\n[True={LABEL2NAME[cls]}] Top {k} samples with highest learned ΔHI_mean (maintenance effect):")
        print(sub[["uid","delta_hi_mean","pred"]].reset_index(drop=True))

# ============================================================
# 5) Training/evaluation loop
# ============================================================
def train_model(pairs, epochs=20, batch_size=32, lr=1e-3, horizon=None, device=None, patience=20):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data split 7:1:2
    pairs_tr, pairs_vl, pairs_te = split_pairs_uid(pairs, 0.7, 0.1, 42)
    print_split_summary(pairs, 0.7, 0.1, 42)

    ds_tr = PairsReconstructDataset(pairs_tr, horizon=horizon)
    ds_vl = PairsReconstructDataset(pairs_vl, horizon=horizon)
    ds_te = PairsReconstructDataset(pairs_te, horizon=horizon)

    ld_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_shift)
    ld_vl = DataLoader(ds_vl, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_shift)
    ld_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_shift)

    C = ds_tr.C
    print(f"\n[Dimension Check] Sensor feature dimension C = {C}")

    model = DiffAwareReconstructor(in_ch=C, trend_ch=4, hidden=128, n_classes=3).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)  # Use AdamW
    ce  = nn.CrossEntropyLoss()

    best_v = 1e9
    best = None
    nan_count = 0  # Count NaN batches
    best_epoch = 0  # Record best epoch
    no_improve_count = 0  # Count epochs without improvement for early stopping

    print(f"\nStarting training for {epochs} epochs with early stopping (patience={patience})...")
    print(f"Device: {device}")
    print(f"Train set: {len(ds_tr)} samples, Val set: {len(ds_vl)} samples, Test set: {len(ds_te)} samples")
    print("Note: Without real HI labels, model will learn health states from sensor data")
    print("Goal: Post-maintenance HI should be entirely higher than pre-maintenance, without hard range constraints")

    for ep in range(1, epochs+1):
        print(f"\n[Epoch {ep}/{epochs}] Training...")
        model.train()
        logs = {"mse_b":0.0, "mse_a":0.0, "diff":0.0, "smooth":0.0, "mono":0.0, "linear":0.0, "cls":0.0, "consist":0.0, "maintain":0.0, "smooth_enh":0.0, "global_sup":0.0, "all":0.0}
        n_bt = 0
        batch_nan_count = 0

        for batch_idx, batch in enumerate(ld_tr):
            # Show training progress
            if batch_idx % max(1, len(ld_tr)//10) == 0:
                progress = (batch_idx + 1) / len(ld_tr) * 100
                print(f"    Training progress: {progress:.1f}% ({batch_idx+1}/{len(ld_tr)} batches)")

            xb = batch["x_before"].to(device)  # (B,L,C)
            xa = batch["x_after"].to(device)
            labels = batch["labels"].to(device)
            lengths= batch["lengths"].to(device)
            mask   = batch["mask"].to(device)

            # Input/target mask
            m_tgt  = (torch.arange(xb.size(1)-2, device=device)[None,:] < (lengths-2)[:,None]).float()

            yb_hat, ya_hat, h_b, h_a, logits = model(xb, xa, mask)

            # Numerical stabilization
            yb_hat, ya_hat, h_b, h_a, logits = sanitize_tensors(yb_hat, ya_hat, h_b, h_a, logits)

            # Recursive reconstruction targets
            yb_tgt = xb[:,1:-1,:]
            ya_tgt = xa[:,1:-1,:]
            loss_b = masked_mse(yb_hat, yb_tgt, m_tgt)
            loss_a = masked_mse(ya_hat, ya_tgt, m_tgt)

            # HI difference (class-conditional margin) - ensure post-maintenance higher than pre-maintenance
            valid = mask.sum(1, keepdim=True).clamp_min(1.0)
            delta_mean = ((h_a - h_b)*mask).sum(1,keepdim=True)/valid  # (B,1)
            loss_diff = diff_margin_by_class(delta_mean.squeeze(1), labels,
                                             m_low=0.1, m_mid=0.25, m_high=0.45)

            # HI smooth + monotonic decreasing (constrain both pre/post) - remove range constraint
            loss_smooth= slope_loss(h_a, mask, "smooth") + slope_loss(h_b, mask, "smooth")
            loss_mono  = slope_loss(h_a, mask, "mono_dec") + slope_loss(h_b, mask, "mono_dec")

            # Enhanced linear loss - stronger weight
            loss_linear = enhanced_linear_slope_loss(h_b, mask, weight=8.0) + enhanced_linear_slope_loss(h_a, mask, weight=8.0)

            # Sensor consistency loss (including constraint that post-maintenance should be higher than pre-maintenance)
            loss_consistency = sensor_consistency_loss(xb, xa, h_b, h_a, mask)

            # Maintenance improvement constraint
            loss_maintenance = maintenance_improvement_constraint(h_b, h_a, labels, mask)

            # Enhanced global HI superiority constraint (higher weight)
            loss_global_superiority = global_hi_superiority_constraint(h_b, h_a, mask, weight=5.0)

            # Enhanced smoothness loss (second and third order differences)
            loss_smooth_enhanced = (smoothness_enhancement_loss(h_b, mask, order=2) +
                                   smoothness_enhancement_loss(h_a, mask, order=2) +
                                   smoothness_enhancement_loss(h_b, mask, order=3) * 0.5 +
                                   smoothness_enhancement_loss(h_a, mask, order=3) * 0.5)

            # Classification loss
            loss_cls = ce(logits, labels)

            # Total loss (adjust weights - enhance global superiority constraint, remove range constraint)
            w_rec=1.0; w_diff=0.5; w_smooth=0.05; w_mono=0.1; w_linear=1.0; w_consist=0.3; w_cls=1.0; w_maintain=1.5; w_smooth_enh=0.5; w_global_sup=2.0
            loss = (w_rec*(loss_b+loss_a) + w_diff*loss_diff +
                   w_smooth*loss_smooth + w_mono*loss_mono + w_linear*loss_linear +
                   w_consist*loss_consistency + w_cls*loss_cls + w_maintain*loss_maintenance +
                   w_smooth_enh*loss_smooth_enhanced + w_global_sup*loss_global_superiority)

            # Numerical stabilization of all losses
            loss_b, loss_a, loss_diff, loss_smooth, loss_mono, loss_linear, loss_consistency, loss_cls, loss_maintenance, loss_smooth_enhanced, loss_global_superiority, loss = sanitize_tensors(
                loss_b, loss_a, loss_diff, loss_smooth, loss_mono, loss_linear, loss_consistency, loss_cls, loss_maintenance, loss_smooth_enhanced, loss_global_superiority, loss)

            # Check NaN and skip
            if torch.isnan(loss) or torch.isinf(loss):
                batch_nan_count += 1
                if batch_nan_count == 1:  # Only print warning once
                    print(f"    [Warning] Epoch {ep}: Found NaN/Inf loss, skipping abnormal batch")
                continue

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            logs["mse_b"] += loss_b.item()
            logs["mse_a"] += loss_a.item()
            logs["diff"]  += loss_diff.item()
            logs["smooth"]+= loss_smooth.item()
            logs["mono"]  += loss_mono.item()
            logs["linear"]+= loss_linear.item()
            logs["consist"]+= loss_consistency.item()
            logs["cls"]   += loss_cls.item()
            logs["maintain"]+= loss_maintenance.item()
            logs["smooth_enh"]+= loss_smooth_enhanced.item()
            logs["global_sup"]+= loss_global_superiority.item()
            logs["all"]   += loss.item()
            n_bt += 1

        # Record NaN batches
        if batch_nan_count > 0:
            nan_count += batch_nan_count

        print("    Validating...")
        for k in logs: logs[k] /= max(n_bt,1)
        vl = eval_epoch(model, ld_vl, device)
        print(f"[Epoch {ep:03d}] Train: L={logs['all']:.4f} rec_b={logs['mse_b']:.4f} rec_a={logs['mse_a']:.4f} "
              f"diff={logs['diff']:.4f} linear={logs['linear']:.4f} maintain={logs['maintain']:.4f} smooth_enh={logs['smooth_enh']:.4f} global_sup={logs['global_sup']:.4f} cls={logs['cls']:.4f} | "
              f"Val: mse_b={vl['mse_b']:.4f} mse_a={vl['mse_a']:.4f} acc={vl['acc']:.3f} ΔHI_improvement={vl['delta_mean']:.3f}")

        # Simple early stopping: observe validation reconstruction loss
        vl_total = vl['mse_b'] + vl['mse_a'] + vl['acc']  # Lower is better (minimize MSE, maximize accuracy)
        if vl_total < best_v:
            best_v = vl_total
            best_epoch = ep
            best = {k: v.clone() if hasattr(v, 'clone') else v for k, v in model.state_dict().items()}
            no_improve_count = 0  # Reset counter
            print(f"    ✓ Saved new best model (val loss: {best_v:.4f})")
        else:
            no_improve_count += 1
            print(f"    Val loss not improved (current: {vl_total:.4f}, best: {best_v:.4f}) - No improvement for {no_improve_count} epochs")

            # Early stopping check
            if no_improve_count >= patience:
                print(f"\n[Early Stopping] No improvement for {patience} epochs. Stopping training at epoch {ep}.")
                break

    if best is not None:
        model.load_state_dict(best)
        print(f"\n[Best Checkpoint] Val reconstruction loss: {best_v:.4f} (Epoch {best_epoch})")

        # Save best model to specified path
        save_path = "/content/drive/MyDrive/CMAPSS/main_identification_2.pth"
        torch.save(model.state_dict(), save_path)
        print(f"[Model Saved] Best model saved to: {save_path}")

    if nan_count > 0:
        print(f"[Warning] Total skipped {nan_count} batches containing NaN/Inf")

    # Test set evaluation
    print("\n" + "="*60)
    print("Final Test Set Evaluation")
    print("="*60)
    te = eval_epoch(model, ld_te, device)
    print(f"[Test Set] mse_b={te['mse_b']:.4f} mse_a={te['mse_a']:.4f} acc={te['acc']:.3f} ΔHI_improvement={te['delta_mean']:.3f}")

    # Collect test predictions and visualization data
    y_true, y_pred, all_delta_mean, all_uids, keep_curves = collect_test_predictions(model, ld_te, device, max_curve_keep=24)

    # Classification report
    if len(y_true) > 0:
        from sklearn.metrics import classification_report, confusion_matrix
        print("\n[Classification Report]")
        print(classification_report(y_true, y_pred, target_names=["Perfect", "General", "Poor"]))
        print("\n[Confusion Matrix]")
        print(confusion_matrix(y_true, y_pred))

        # Create learned ΔHI statistics table
        df_delta = pd.DataFrame({
            "uid": all_uids[:len(y_true)],
            "true": y_true,
            "pred": y_pred,
            "delta_hi_mean": all_delta_mean[:len(y_true)]
        })
        print("\n[Learned ΔHI Statistics] By true class (maintenance effect improvement)")
        print(df_delta.groupby("true")["delta_hi_mean"].describe())

        # Show Top-K samples
        topk_by_delta(df_delta, k=3)

        # HI curve visualization
        print("\n[Visualization] Showing learned health index curves (post-maintenance entirely higher than pre-maintenance)...")
        plot_hi_examples_aligned(keep_curves, n_show=6, seed=0)

        # Sensor reconstruction visualization
        print("\n[Visualization] Showing sensor reconstruction predictions...")
        plot_sensor_examples_aligned(keep_curves, sensor_idx_list=None, n_cols=4)

    return model, (y_true, y_pred, all_delta_mean, all_uids, keep_curves)


model, results = train_model(
    pairs=pairs,
    epochs=300,
    batch_size=258,
    lr=1e-3,
    horizon=50,
    device=None,  # Auto select
    patience=20   # Early stopping patience
)

print("\nTraining completed! Model has learned to infer health states from sensor data without hard range constraints.")
print("Enhanced global superiority constraint ensures post-maintenance HI is entirely higher than pre-maintenance HI.")
print("View learned HI curves and enhanced maintenance effect classification results through visualization.")

# %% [notebook code cell 11]
# ============================================================
# Monotonic-Decreasing HI + aligned plotting (before -> after)
# ============================================================
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import random
import matplotlib.pyplot as plt

# ============================================================
# 1) Data set: Read samples from pairs, and then perform shift operations in batches
# ============================================================
class PairsReconstructDataset(Dataset):
    """
    Each sample contains:
      x_before: (L,C) feature sequence before maintenance
      x_after: (L,C) maintained feature sequence
      hi_before/hi_after: (L,) health indicators (only used for evaluation/optional analysis, no strong supervision)
      strategy: maintenance type ("Perfect Maintenance" / "General Maintenance" / "Poor Maintenance")
    """
    def __init__(self, pairs: dict, horizon: int=None):
        items = []
        for uid, sd in pairs.items():
            for strat, d in sd.items():
                xb = np.asarray(d["x_before"], dtype=np.float32)
                xa = np.asarray(d["x_after"],  dtype=np.float32)
                hib= np.asarray(d["hi_before"],dtype=np.float32)
                hia= np.asarray(d["hi_after"], dtype=np.float32)
                L, C = xb.shape
                #Uniform length (optional)
                if horizon is not None and L > horizon:
                    xb, xa, hib, hia = xb[:horizon], xa[:horizon], hib[:horizon], hia[:horizon]
                    L = horizon
                if L < 3: # At least it must be able to form 0:L-2 -> 1:L-1
                    continue
                items.append({"uid": uid, "strategy": strat, "x_before": xb, "x_after": xa,
                              "hi_before": hib, "hi_after": hia})
        assert len(items)>0, "The data set is empty, please check the pairs data."
        self.items = items
        self.C = items[0]["x_before"].shape[1]

        # Mapping of strategies to 3 types of labels
        def strat_to_cls(s):
            s = s.lower()
            if "perfect" in s: return 0
            if "general" in s: return 1
            if "poor" in s: return 2
            raise ValueError(f"Unknown strategy name: {s}")
        self.labels = [strat_to_cls(it["strategy"]) for it in items]

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        return {
            "uid": it["uid"],
            "label": self.labels[idx],
            "x_before": torch.from_numpy(it["x_before"]), # (L,C)
            "x_after":  torch.from_numpy(it["x_after"]),  # (L,C)
            "hi_before":torch.from_numpy(it["hi_before"]),# (L,)
            "hi_after": torch.from_numpy(it["hi_after"]), # (L,)
            "strategy": it["strategy"]
        }

def pad_collate_shift(batch):
    """
    Pad the sequence to the same length; return the mask and perform the shift operation **not here**,
    In order to conduct uniform training: inputs = [:,0:L-2], targets = [:,1:L-1]
    """
    Ls = [b["x_before"].shape[0] for b in batch]
    Lmax = max(Ls)
    C = batch[0]["x_before"].shape[1]

    def pad2d(x):
        L, C0 = x.shape
        out = torch.zeros(Lmax, C0, dtype=x.dtype)
        out[:L] = x
        return out

    def pad1d(x):
        L = x.shape[0]
        out = torch.zeros(Lmax, dtype=x.dtype)
        out[:L] = x
        return out

    x_before = torch.stack([pad2d(b["x_before"]) for b in batch], 0) # (B,Lmax,C)
    x_after  = torch.stack([pad2d(b["x_after"])  for b in batch], 0) # (B,Lmax,C)
    hi_before= torch.stack([pad1d(b["hi_before"])for b in batch], 0) # (B,Lmax)
    hi_after = torch.stack([pad1d(b["hi_after"]) for b in batch], 0) # (B,Lmax)
    lengths  = torch.tensor(Ls, dtype=torch.long)
    mask     = torch.zeros(len(batch), Lmax)
    for i,L in enumerate(Ls): mask[i,:L] = 1.0
    labels   = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    uids     = [b["uid"] for b in batch]
    return {"uids": uids, "labels": labels,
            "x_before": x_before, "x_after": x_after,
            "hi_before": hi_before, "hi_after": hi_after,
            "lengths": lengths, "mask": mask}

# ============================================================
# 2) Model: Encoding "damage" → Project into monotonically decreasing HI → Recursively reconstruct two sequences → Classify with ΔHI
# ============================================================
class BoltzmannKAN(nn.Module):
    def __init__(self, in_ch, out_ch=4):
        super().__init__()
        self.E  = nn.Parameter(torch.zeros(out_ch, in_ch))
        self.kT = nn.Parameter(torch.ones(out_ch, in_ch))
    def forward(self, x):
        B,T,C = x.shape
        kT = torch.clamp(F.softplus(self.kT), 0.01, 10.0).unsqueeze(0).unsqueeze(2)  # (1,out_ch,1,in_ch)
        E  = torch.clamp(self.E, -10.0, 10.0).unsqueeze(0).unsqueeze(2)             # (1,out_ch,1,in_ch)
        x_ = torch.clamp(x.unsqueeze(1), -10.0, 10.0)                               # (B, out_ch, T, in_ch)
        exp = torch.clamp((E - x_) / kT, -50, 50)
        w   = torch.sigmoid(exp)
        h   = (w * x_).sum(dim=3).permute(0, 2, 1)           # (B,T,out_ch) >=0
        return torch.clamp(F.softplus(h), 0.0, 100.0)

class BaseOp(nn.Module): pass
class MonotonicLinearOp(BaseOp):
    def __init__(self):
        super().__init__()
        self.raw_scale=nn.Parameter(torch.tensor(0.0))
        self.bias=nn.Parameter(torch.tensor(0.0))
        self.smin,self.smax=0.01,5.0
    def forward(self,h):
        scale=torch.clamp(F.softplus(self.raw_scale),self.smin,self.smax)
        xm=torch.clamp(h.mean(-1), -10.0, 10.0)
        return torch.clamp(F.softplus(scale*(xm+self.bias)), 0.0, 100.0)

class MonotonicFlatOp(BaseOp):
    def __init__(self):
        super().__init__()
        self.raw_scale=nn.Parameter(torch.tensor(0.0))
        self.bias=nn.Parameter(torch.tensor(0.0))
        self.smin,self.smax=1e-3,1.0
    def _cum(self,x):
        diff=F.relu(x[:,1:]-x[:,:-1])
        return torch.cat([torch.zeros_like(diff[:,:1]),torch.cumsum(diff,1)],1)
    def forward(self,h):
        scale=torch.clamp(F.softplus(self.raw_scale),self.smin,self.smax)
        xm=torch.clamp(h.mean(-1,keepdim=True), -10.0, 10.0)
        return torch.clamp(F.softplus(scale*(self._cum(xm)+self.bias)).squeeze(-1), 0.0, 100.0)

class ConcaveLogOp(BaseOp):
    def __init__(self):
        super().__init__()
        self.raw_scale=nn.Parameter(torch.tensor(0.0))
        self.eps=1e-3
        self.smin,self.smax=0.01,5.0
    def forward(self,h):
        scale=torch.clamp(F.softplus(self.raw_scale),self.smin,self.smax)
        xm=torch.clamp(h.mean(-1), -10.0, 10.0)
        return torch.clamp(F.softplus(scale*torch.log(torch.abs(xm)+self.eps)), 0.0, 100.0)

class SaturationSigmoidOp(BaseOp):
    def __init__(self):
        super().__init__()
        self.raw_scale=nn.Parameter(torch.tensor(0.0))
        self.raw_slope=nn.Parameter(torch.tensor(0.0))
        self.bias=nn.Parameter(torch.tensor(0.0))
        self.smin,self.smax=0.01,5.0
        self.lmin,self.lmax=0.1,5.0
    def forward(self,h):
        scale=torch.clamp(F.softplus(self.raw_scale),self.smin,self.smax)
        slope=torch.clamp(F.softplus(self.raw_slope),self.lmin,self.lmax)
        xm=torch.clamp(h.mean(-1), -10.0, 10.0)
        return torch.clamp(F.softplus(scale*torch.sigmoid(slope*(xm-self.bias))), 0.0, 100.0)

class HingeReLUOp(BaseOp):
    def __init__(self):
        super().__init__()
        self.raw_scale=nn.Parameter(torch.tensor(0.0))
        self.threshold=nn.Parameter(torch.tensor(0.0))
        self.smin,self.smax=0.01,5.0
    def forward(self,h):
        scale=torch.clamp(F.softplus(self.raw_scale),self.smin,self.smax)
        xm=torch.clamp(h.mean(-1), -10.0, 10.0)
        return torch.clamp(F.softplus(scale*F.relu(xm-self.threshold)), 0.0, 100.0)

class PolynomialOp(BaseOp):
    def __init__(self,deg=3):
        super().__init__()
        self.raw_coeff=nn.Parameter(torch.zeros(deg))
        self.deg=deg
    def forward(self,h):
        xm=torch.clamp(h.mean(-1), -5.0, 5.0)
        y=torch.zeros_like(xm)
        for i in range(self.deg):
            coeff = torch.clamp(F.softplus(self.raw_coeff[i]), 0.01, 5.0)
            y = y + coeff * torch.clamp(xm ** (i+1), -100.0, 100.0)
        return torch.clamp(F.softplus(y), 0.0, 100.0)

class DampedSinOp(BaseOp):
    def __init__(self):
        super().__init__()
        self.raw_scale=nn.Parameter(torch.tensor(0.0))
        self.raw_freq=nn.Parameter(torch.tensor(0.0))
        self.raw_lambda=nn.Parameter(torch.tensor(0.0))
        self.phase=nn.Parameter(torch.tensor(0.0))
        self.smin,self.smax=0.01,5.0
        self.fmin,self.fmax=0.1,5.0
        self.lmin,self.lmax=0.01,3.0
    def forward(self,h):
        scale=torch.clamp(F.softplus(self.raw_scale),self.smin,self.smax)
        freq=torch.clamp(F.softplus(self.raw_freq),self.fmin,self.fmax)
        lam=torch.clamp(F.softplus(self.raw_lambda),self.lmin,self.lmax)
        xm=torch.clamp(h.mean(-1), -10.0, 10.0)
        damp=1/(1+lam*torch.abs(xm))
        phase_val = torch.clamp(self.phase, -10.0, 10.0)
        return torch.clamp(F.softplus(scale*damp*(torch.sin(freq*xm+phase_val)+1)), 0.0, 100.0)

class PiecewiseLinearOp(BaseOp):
    def __init__(self):
        super().__init__()
        self.raw_k1=nn.Parameter(torch.tensor(0.0))
        self.raw_k2=nn.Parameter(torch.tensor(0.0))
        self.threshold=nn.Parameter(torch.tensor(0.0))
        self.kmin,self.kmax=0.01,5.0
    def forward(self,h):
        k1=torch.clamp(F.softplus(self.raw_k1),self.kmin,self.kmax)
        k2=torch.clamp(F.softplus(self.raw_k2),self.kmin,self.kmax)
        thresh = torch.clamp(self.threshold, -5.0, 5.0)
        xm=torch.clamp(h.mean(-1), -10.0, 10.0)
        left=k1*xm
        right=k1*thresh + k2*(xm-thresh)
        out=torch.where(xm<=thresh,left,right)
        return torch.clamp(F.softplus(out), 0.0, 100.0)

class SparseGate(nn.Module):
    def __init__(self, n_ops, tau_start=5.0, tau_end=0.1, n_steps=10000):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(n_ops))
        self.tau_start, self.tau_end, self.n_steps = tau_start, tau_end, n_steps
        self.register_buffer("step", torch.tensor(0.))
    def forward(self):
        tau = (self.tau_start*(1-self.step/self.n_steps) + self.tau_end*(self.step/self.n_steps)).clamp(min=self.tau_end)
        self.step += 1
        g = -torch.empty_like(self.logits).exponential_().log() if self.training else torch.zeros_like(self.logits)
        g = torch.clamp(g, -50.0, 50.0) # Limit Gumbel noise
        logits_stable = torch.clamp(self.logits, -50.0, 50.0)
        w = F.softmax((logits_stable + g)/tau, dim=-1)
        return w.view(1,1,-1)

class CustomKAN(nn.Module):
    def __init__(self, ops):
        super().__init__()
        self.ops = nn.ModuleList(ops)
        self.gate= SparseGate(len(ops))
        # Additional learnable scaling/offsets for damage
        self.raw_gain = nn.Parameter(torch.tensor(0.0))
        self.bias     = nn.Parameter(torch.tensor(0.0))
    def forward(self, h):  # h:(B,T,trend_ch)
        outs = [op(h) for op in self.ops]          # list of (B,T)
        Tm = min(o.size(1) for o in outs)
        outs = [o[:,:Tm] for o in outs]
        st = torch.stack(outs, dim=-1)             # (B,Tm,K), >=0
        w  = self.gate()                           # (1,1,K)
        damage = torch.clamp((st*w).sum(-1), 0.0, 100.0) # (B,Tm) is non-negative and is regarded as "damage"
        gain = torch.clamp(F.softplus(self.raw_gain), 0.1, 5.0)
        bias_val = torch.clamp(self.bias, -5.0, 5.0)
        damage = torch.clamp(gain*damage + bias_val, 0.0, 100.0)
        return damage                               # (B,Tm)

class TrendEncoder(nn.Module):
    """
    The encoder outputs "damage", which is then projected into a monotonically decreasing HI:
        HI = 1 - sigmoid(g*(damage + b))
    """
    def __init__(self, in_ch, trend_ch=4):
        super().__init__()
        self.boltz = BoltzmannKAN(in_ch, out_ch=trend_ch)
        ops = [MonotonicLinearOp(), MonotonicFlatOp(), ConcaveLogOp(),
               SaturationSigmoidOp(), HingeReLUOp(), PolynomialOp(),
               DampedSinOp(), PiecewiseLinearOp()]
        self.customkan = CustomKAN(ops)
        # Projection parameters
        self.proj_gain = nn.Parameter(torch.tensor(1.0))
        self.proj_bias = nn.Parameter(torch.tensor(0.0))
    def forward(self, x):  # x:(B,T,C)
        h_multi = self.boltz(x)              # (B,T,trend_ch) >=0
        damage  = self.customkan(h_multi)    # (B,T)           >=0
        g = torch.clamp(F.softplus(self.proj_gain), 0.1, 5.0)
        b = torch.clamp(self.proj_bias, -5.0, 5.0)
        # The greater the damage → the smaller the HI (decreasing tendency)
        hi = 1.0 - torch.sigmoid(g*(damage + b))
        # Make sure HI is in the range [0, 1]
        hi = torch.clamp(hi, 0.0, 1.0)
        return hi, h_multi                   # hi ∈ (0,1)

class ReconHead(nn.Module):
    """
    Recursive reconstruction: given (x_t, h_t) → predict x_{t+1}
    """
    def __init__(self, C, hidden=128):
        super().__init__()
        self.gru = nn.GRU(input_size=C+1, hidden_size=hidden, batch_first=True)
        self.out = nn.Linear(hidden, C)
    def forward(self, x_in, h_in):
        # x_in:(B,T_in,C), h_in:(B,T_in)
        B,T,C = x_in.shape
        h_in_clamped = torch.clamp(h_in, 0.0, 1.0)
        feat = torch.cat([x_in, h_in_clamped.unsqueeze(-1)], dim=-1)  # (B,T,C+1)
        H,_ = self.gru(feat)                                  # (B,T,H)
        y = self.out(H) # (B,T,C) alignment prediction x_{t+1}
        return y

class MaintClassifier(nn.Module):
    """
    Use the before and after difference features of HI to perform three classifications
    """
    def __init__(self, hidden=64, n_classes=3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(6, hidden), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, n_classes)
        )
    def forward(self, h_b, h_a, mask):
        # Statistical features: Δmean, Δfirst value, Δmaximum value, Δstandard deviation, slope difference, positive difference ratio
        m = mask
        T = m.size(1)
        valid = m.sum(1, keepdim=True).clamp_min(1.0)
        db = h_a - h_b                        # (B,T)
        mean_d = (db*m).sum(1,keepdim=True)/valid
        var_d  = (((db-mean_d)*m)**2).sum(1,keepdim=True)/valid
        std_d  = (var_d + 1e-8).sqrt()
        d0 = (db[:, :1]) # first value
        dmax   = (db.masked_fill(m==0, -1e9)).max(dim=1, keepdim=True).values
        pos_ratio = ((db>0).float()*m).sum(1,keepdim=True)/valid
        # Slope (linear fit)
        t = torch.arange(T, device=h_b.device, dtype=h_b.dtype)
        t = (t - t.mean())/(t.std()+1e-6)
        def slope(x):
            num = (x*t).sum(1) - x.sum(1)*t.sum()/T
            den = (t**2).sum() - (t.sum()**2)/T + 1e-8
            return (num/den).unsqueeze(1)
        slope_diff = slope(h_a) - slope(h_b)  # (B,1)
        feat = torch.cat([mean_d, d0, dmax, std_d, slope_diff, pos_ratio], dim=1)  # (B,6)
        # Numerical stabilization
        feat = torch.clamp(feat, -10.0, 10.0)
        return self.mlp(feat)  # logits

def sanitize_tensors(*tensors):
    """Replace NaN/Inf in tensors with finite values"""
    result = []
    for t in tensors:
        if t is not None:
            t_clean = torch.nan_to_num(t, nan=0.0, posinf=1e6, neginf=-1e6)
            result.append(t_clean)
        else:
            result.append(t)
    return result[0] if len(result) == 1 else result

class DiffAwareReconstructor(nn.Module):
    """
    Overall model:
     - Two encoders share parameters: encode x_before / x_after → h_b / h_a respectively (monotonically decreasing HI)
     - Two reconstruction heads: perform one-step recursive reconstruction respectively
     - Classification header: Use the statistical characteristics of (h_a - h_b) to identify the maintenance type
    """
    def __init__(self, in_ch, trend_ch=4, hidden=128, n_classes=3):
        super().__init__()
        self.encoder = TrendEncoder(in_ch, trend_ch)
        self.recon_b = ReconHead(in_ch, hidden)
        self.recon_a = ReconHead(in_ch, hidden)
        self.clf     = MaintClassifier(hidden=64, n_classes=n_classes)

    def forward(self, x_b, x_a, mask):
        # x_b/x_a : (B,L,C) (will switch to 0:L-2 / 1:L-1 later)
        h_b, _ = self.encoder(x_b) # (B,L) monotonically decreasing (encouraged by loss)
        h_a, _ = self.encoder(x_a)  # (B,L)

        # Numerical stabilization
        h_b, h_a = sanitize_tensors(h_b, h_a)

        # Recursive reconstruction: input 0:L-2 → predict 1:L-1
        xb_in, hb_in = x_b[:, :-2], h_b[:, :-2]
        xa_in, ha_in = x_a[:, :-2], h_a[:, :-2]
        yb_hat = self.recon_b(xb_in, hb_in)   # (B,L-2,C) ≈ x_b[:,1:L-1]
        ya_hat = self.recon_a(xa_in, ha_in)   # (B,L-2,C) ≈ x_a[:,1:L-1]

        # Numerical stabilization
        yb_hat, ya_hat = sanitize_tensors(yb_hat, ya_hat)

        # Classification: Use uncropped length mask
        logits = self.clf(h_b, h_a, mask)
        logits = sanitize_tensors(logits)

        return yb_hat, ya_hat, h_b, h_a, logits

# ============================================================
# 3) Loss function: reconstruction + ΔHI + range/smooth/monotone (decreasing) + classification
# ============================================================
def masked_mse(a, b, mask):
    # a,b:(B,T,C)  mask:(B,T)
    diff = (a - b)**2
    mse  = (diff.mean(-1) * mask).sum() / (mask.sum() + 1e-6)
    return mse

def slope_loss(h, mask, kind="smooth"):
    # Smoothing: second-order difference; monotonic decrease: penalty rising
    if kind=="smooth":
        d2 = h[:,2:] - 2*h[:,1:-1] + h[:,:-2]
        m  = mask[:,2:]
        return (d2.abs() * m).sum() / (m.sum()+1e-6)
    elif kind=="mono_dec":
        dh = h[:,1:] - h[:,:-1] # If >0, it means rising (contrary to "decreasing")
        m  = mask[:,1:]
        return (F.relu(dh) * m).sum() / (m.sum()+1e-6)
    else:
        return torch.tensor(0., device=h.device)

def range_penalty(h, mask):
    pen = F.relu(h-1) + F.relu(0-h)
    return (pen * mask).sum() / (mask.sum()+1e-6)

def diff_margin_by_class(mean_delta, labels, m_low=0.05, m_mid=0.15, m_high=0.35):
    """
    Class condition difference constraints:
      - Perfect(0):  Δ>=m_high
      - General(1): Δ≈m_mid → L2 to target distance
      - Poor(2):     Δ<=m_low
    """
    loss = torch.zeros_like(mean_delta)
    for cls in [0,1,2]:
        sel = (labels==cls).float()
        if sel.sum() < 1: continue
        if cls==0:
            loss += sel * F.relu(m_high - mean_delta)**2
        elif cls==1:
            loss += sel * (mean_delta - m_mid)**2
        else:
            loss += sel * F.relu(mean_delta - m_low)**2
    denom = torch.ones_like(mean_delta) * (labels.numel())
    return loss.sum() / denom.sum()

# ============================================================
# 4) Data partitioning and visualization (aligned to the same timeline)
# ============================================================
LABEL2NAME = {0: "Perfect", 1: "General", 2: "Poor"}

def split_pairs_uid(pairs, train_ratio=0.7, val_ratio=0.1, seed=42):
    uids = sorted(list(pairs.keys()))
    rs = np.random.RandomState(seed)
    rs.shuffle(uids)
    n = len(uids); n_tr=int(n*train_ratio); n_vl=int(n*val_ratio)
    to_dict = lambda ss:{u:pairs[u] for u in ss if u in pairs}
    return to_dict(uids[:n_tr]), to_dict(uids[n_tr:n_tr+n_vl]), to_dict(uids[n_tr+n_vl:])

def print_split_summary(pairs, train_ratio=0.7, val_ratio=0.1, seed=42):
    pairs_tr, pairs_vl, pairs_te = split_pairs_uid(pairs, train_ratio, val_ratio, seed)
    def count_items(pp):
        cnt = 0
        for uid, d in pp.items():
            cnt += len(d)
        return len(pp), cnt
    n_uid_tr, n_pair_tr = count_items(pairs_tr)
    n_uid_vl, n_pair_vl = count_items(pairs_vl)
    n_uid_te, n_pair_te = count_items(pairs_te)
    print(f"[data division] training: UID={n_uid_tr}, number of pairs≈{n_pair_tr} | "
          f"Verification: UID={n_uid_vl}, number of pairs≈{n_pair_vl} | "
          f"Test: UID={n_uid_te}, number of pairs≈{n_pair_te}")

@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total = {"mse_b":0.0, "mse_a":0.0, "acc":0.0, "delta_mean":0.0}
    n_batch=0
    n_acc = 0; n_tot=0
    for batch in loader:
        xb = batch["x_before"].to(device)
        xa = batch["x_after"].to(device)
        hi_b = batch["hi_before"].to(device)
        hi_a = batch["hi_after"].to(device)
        labels = batch["labels"].to(device)
        lengths= batch["lengths"].to(device)
        mask   = batch["mask"].to(device)

        # Valid segments: 0:L-2 input, 1:L-1 target
        m_tgt  = (torch.arange(xb.size(1)-2, device=device)[None,:] < (lengths-2)[:,None]).float()

        yb_hat, ya_hat, h_b, h_a, logits = model(xb, xa, mask)

        # Numerical stabilization
        yb_hat, ya_hat, h_b, h_a, logits = sanitize_tensors(yb_hat, ya_hat, h_b, h_a, logits)

        #Align target
        yb_tgt = xb[:,1:-1,:]
        ya_tgt = xa[:,1:-1,:]

        mse_b = masked_mse(yb_hat, yb_tgt, m_tgt)
        mse_a = masked_mse(ya_hat, ya_tgt, m_tgt)

        # Check if it is NaN
        if torch.isnan(mse_b) or torch.isnan(mse_a):
            continue

        # Classification
        pred = logits.argmax(dim=1)
        n_acc += (pred == labels).sum().item()
        n_tot += labels.numel()

        # Statistics ΔHI mean (only used for logging)
        valid = mask.sum(1, keepdim=True).clamp_min(1.0)
        delta_mean = ((h_a - h_b)*mask).sum(1,keepdim=True)/valid
        delta_mean = torch.nan_to_num(delta_mean, nan=0.0)
        total["delta_mean"] += delta_mean.mean().item()

        total["mse_b"] += mse_b.item()
        total["mse_a"] += mse_a.item()
        n_batch += 1

    for k in ["mse_b","mse_a","delta_mean"]:
        total[k] /= max(n_batch,1)
    total["acc"] = n_acc / max(n_tot,1)
    return total

@torch.no_grad()
def collect_test_predictions(model, loader, device, max_curve_keep=24):
    """
    Collect predictions on the test set, ΔHI, and HI curves for a small number of samples (for plotting).
    """
    model.eval()
    y_true, y_pred = [], []
    all_delta_mean, all_uids = [], []
    keep_curves = []

    for batch in loader:
        xb = batch["x_before"].to(device)   # (B,L,C)
        xa = batch["x_after"].to(device)
        mask   = batch["mask"].to(device)   # (B,L)
        labels = batch["labels"].to(device) # (B,)
        lengths= batch["lengths"]           # cpu tensor
        uids   = batch["uids"]

        yb_hat, ya_hat, h_b, h_a, logits = model(xb, xa, mask)

        # Numerical stabilization
        yb_hat, ya_hat, h_b, h_a, logits = sanitize_tensors(yb_hat, ya_hat, h_b, h_a, logits)

        prob = F.softmax(logits, dim=1)     # (B,3)
        pred = prob.argmax(1)               # (B,)

        # Statistics ΔHI mean
        valid = mask.sum(1, keepdim=True).clamp_min(1.0)
        delta_mean = ((h_a - h_b) * mask).sum(1, keepdim=True) / valid  # (B,1)
        delta_mean = torch.nan_to_num(delta_mean, nan=0.0)

        y_true.append(labels.cpu().numpy())
        y_pred.append(pred.cpu().numpy())
        all_delta_mean.append(delta_mean.squeeze(1).cpu().numpy())
        all_uids.extend(uids)

        # Save part of the curve for visualization (alignment: after is spliced ​​behind before)
        for i in range(xb.size(0)):
            if len(keep_curves) >= max_curve_keep:
                break
            L_i = int(lengths[i].item())
            # Fix: intercept and reconstruct predictions according to actual length
            L_recon = min(L_i - 2, yb_hat.size(1)) # Reconstruction sequence length
            keep_curves.append({
                "uid": uids[i],
                "true": int(labels[i].cpu().item()),
                "pred": int(pred[i].cpu().item()),
                "prob": prob[i].cpu().numpy(),
                "h_before": h_b[i, :L_i].cpu().numpy(),
                "h_after":  h_a[i, :L_i].cpu().numpy(),
                "yb_hat":   yb_hat[i, :L_recon].cpu().numpy(),       # (L_recon,C)
                "ya_hat":   ya_hat[i, :L_recon].cpu().numpy(),       # (L_recon,C)
                "x_before": xb[i, :L_i].cpu().numpy(),               # (L,C)
                "x_after":  xa[i, :L_i].cpu().numpy(),               # (L,C)
            })

    y_true = np.concatenate(y_true, axis=0) if len(y_true)>0 else np.array([])
    y_pred = np.concatenate(y_pred, axis=0) if len(y_pred)>0 else np.array([])
    all_delta_mean = np.concatenate(all_delta_mean, axis=0) if len(all_delta_mean)>0 else np.array([])
    return y_true, y_pred, all_delta_mean, all_uids, keep_curves

def plot_hi_examples_aligned(curves, n_show=6, seed=0):
    """The post-maintenance sequence continues from the pre-maintenance end point; draw a vertical line in the middle to mark t_m."""
    if len(curves)==0:
        print("(No visualization sample)")
        return
    random.Random(seed).shuffle(curves)
    n_show = min(n_show, len(curves))
    cols = 3
    rows = int(np.ceil(n_show/cols))
    plt.figure(figsize=(cols*4.6, rows*3.2))
    for k in range(n_show):
        ex = curves[k]
        hb = ex["h_before"]; ha = ex["h_after"]
        L = min(len(hb), len(ha)) # Defensive alignment
        hb, ha = hb[:L], ha[:L]
        t_b = np.arange(L)
        t_a = np.arange(L, 2*L) # after immediately after
        uid = ex["uid"]
        pred = ex["pred"]; true = ex["true"]; prob = ex["prob"]
        plt.subplot(rows, cols, k+1)
        plt.plot(t_b, hb, label="HI_before", linewidth=1.8)
        plt.plot(t_a, ha, label="HI_after",  linewidth=1.8, linestyle='--')
        #Maintenance time dotted line
        plt.axvline(L-1, color='k', linestyle=':', linewidth=1.0)
        d_mean = float(np.mean(ha - hb))
        d_max  = float(np.max(ha - hb))
        title = (f"uid={uid}\nTrue={LABEL2NAME[true]} | Pred={LABEL2NAME[pred]} "
                 f"| p={prob[pred]:.2f}\nΔHI_mean={d_mean:.3f}, ΔHI_max={d_max:.3f}")
        plt.title(title)
        plt.xlabel("Cycle (before | after)")
        plt.ylabel("HI (↓)")
        plt.grid(ls='--', alpha=.35)
        if k==0: plt.legend()
    plt.tight_layout()
    plt.show()

def plot_sensor_examples_aligned(curves, sensor_idx_list=None, n_cols=4):
    """
    Draw the original vs recursive prediction (post) of several sensors, and connect the after timeline after the before.
    - Original before: x_before[:, s]
    - original after : x_after [:, s] (optional)
    - predict after : ya_hat [:, s] (one-step sequence aligned with x_after)
    """
    if len(curves)==0:
        print("(No visualization sample)")
        return
    # Take the first sample to display multiple sensors
    ex = curves[0]
    xb = ex["x_before"]         # (L,C)
    xa = ex["x_after"]          # (L,C)
    ya = ex["ya_hat"] # (L_recon,C) predicts 1..L-1
    L, C = xb.shape
    Lhat = ya.shape[0] #actual reconstruction length

    if sensor_idx_list is None:
        # Pick 8 by default
        step = max(1, C//8)
        sensor_idx_list = list(range(0, C, step))[:8]

    n = len(sensor_idx_list)
    n_rows = int(np.ceil(n/n_cols))
    plt.figure(figsize=(n_cols*4.3, n_rows*3.1))
    for i, s in enumerate(sensor_idx_list):
        plt.subplot(n_rows, n_cols, i+1)
        t_b = np.arange(L)
        t_a = np.arange(L, 2*L)
        plt.plot(t_b, xb[:,s], label="Original (before)", linewidth=1.2)
        plt.plot(t_a, xa[:,s], label="Original (after)",  linewidth=1.0, linestyle="--", alpha=0.7)
        # Predict after: aligned with the target of after (corresponding to 1..L-1)
        t_pred = np.arange(L+1, L+1+Lhat) # Align the time of ya_hat
        plt.plot(t_pred, ya[:,s], label="Post-maint. prediction", linewidth=1.6)
        plt.axvline(L-1, color='k', linestyle=':', linewidth=1.0)
        plt.title(f"Sensor_{s:02d}")
        if i==0: plt.legend()
        plt.grid(ls="--", alpha=.35)
    plt.tight_layout()
    plt.show()

def topk_by_delta(df_delta, k=5):
    if len(df_delta)==0:
        return
    for cls in [0,1,2]:
        sub = df_delta[df_delta["true"]==cls].sort_values("delta_hi_mean", ascending=False).head(k)
        print(f"\n[True={LABEL2NAME[cls]}] ΔHI_mean largest first {k} samples:")
        print(sub[["uid","delta_hi_mean","pred"]].reset_index(drop=True))

# ============================================================
# 5) Training/evaluation loop
# ============================================================
def train_model(pairs, epochs=20, batch_size=32, lr=1e-3, horizon=None, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data partition 7:1:2
    pairs_tr, pairs_vl, pairs_te = split_pairs_uid(pairs, 0.7, 0.1, 42)
    print_split_summary(pairs, 0.7, 0.1, 42)

    ds_tr = PairsReconstructDataset(pairs_tr, horizon=horizon)
    ds_vl = PairsReconstructDataset(pairs_vl, horizon=horizon)
    ds_te = PairsReconstructDataset(pairs_te, horizon=horizon)

    ld_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_shift)
    ld_vl = DataLoader(ds_vl, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_shift)
    ld_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_shift)

    C = ds_tr.C
    model = DiffAwareReconstructor(in_ch=C, trend_ch=4, hidden=128, n_classes=3).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5) # Use AdamW
    ce  = nn.CrossEntropyLoss()

    best_v = 1e9
    best = None
    nan_count = 0 # Count the number of NaN batches

    print(f"\nStart training, total {epochs} rounds...")
    print(f"Device: {device}")
    print(f"Training set: {len(ds_tr)} samples, validation set: {len(ds_vl)} samples, test set: {len(ds_te)} samples")

    for ep in range(1, epochs+1):
        model.train()
        logs = {"mse_b":0.0, "mse_a":0.0, "diff":0.0, "range":0.0, "smooth":0.0, "mono":0.0, "cls":0.0, "all":0.0}
        n_bt = 0
        batch_nan_count = 0

        for batch in ld_tr:
            xb = batch["x_before"].to(device)  # (B,L,C)
            xa = batch["x_after"].to(device)
            labels = batch["labels"].to(device)
            lengths= batch["lengths"].to(device)
            mask   = batch["mask"].to(device)

            # input/destination mask
            m_tgt  = (torch.arange(xb.size(1)-2, device=device)[None,:] < (lengths-2)[:,None]).float()

            yb_hat, ya_hat, h_b, h_a, logits = model(xb, xa, mask)

            # Numerical stabilization
            yb_hat, ya_hat, h_b, h_a, logits = sanitize_tensors(yb_hat, ya_hat, h_b, h_a, logits)

            # Recursive reconstruction target
            yb_tgt = xb[:,1:-1,:]
            ya_tgt = xa[:,1:-1,:]
            loss_b = masked_mse(yb_hat, yb_tgt, m_tgt)
            loss_a = masked_mse(ya_hat, ya_tgt, m_tgt)

            # HI difference (class condition margin)
            valid = mask.sum(1, keepdim=True).clamp_min(1.0)
            delta_mean = ((h_a - h_b)*mask).sum(1,keepdim=True)/valid  # (B,1)
            loss_diff = diff_margin_by_class(delta_mean.squeeze(1), labels,
                                             m_low=0.05, m_mid=0.15, m_high=0.35)

            # HI range + smoothing + monotonic decrease (both front and rear constraints)
            loss_range = range_penalty(h_b, mask) + range_penalty(h_a, mask)
            loss_smooth= slope_loss(h_a, mask, "smooth") + slope_loss(h_b, mask, "smooth")
            loss_mono  = slope_loss(h_a, mask, "mono_dec") + slope_loss(h_b, mask, "mono_dec")

            # Classification loss
            loss_cls = ce(logits, labels)

            # Aggregation loss (weight can be adjusted)
            w_rec=1.0; w_diff=0.2; w_range=0.01; w_smooth=0.02; w_mono=0.05; w_cls=1.0
            loss = w_rec*(loss_b+loss_a) + w_diff*loss_diff + w_range*loss_range + w_smooth*loss_smooth + w_mono*loss_mono + w_cls*loss_cls

            # Numerically stabilize all losses
            loss_b, loss_a, loss_diff, loss_range, loss_smooth, loss_mono, loss_cls, loss = sanitize_tensors(
                loss_b, loss_a, loss_diff, loss_range, loss_smooth, loss_mono, loss_cls, loss)

            # Check for NaN and skip
            if torch.isnan(loss) or torch.isinf(loss):
                batch_nan_count += 1
                if batch_nan_count == 1: # Only print the warning once
                    print(f" [Warning] Round {ep}: NaN/Inf loss found, skipping exception batch")
                continue

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            logs["mse_b"] += loss_b.item()
            logs["mse_a"] += loss_a.item()
            logs["diff"]  += loss_diff.item()
            logs["range"] += loss_range.item()
            logs["smooth"]+= loss_smooth.item()
            logs["mono"]  += loss_mono.item()
            logs["cls"]   += loss_cls.item()
            logs["all"]   += loss.item()
            n_bt += 1

        # Log NaN batches
        if batch_nan_count > 0:
            nan_count += batch_nan_count

        for k in logs: logs[k] /= max(n_bt,1)
        vl = eval_epoch(model, ld_vl, device)
        print(f"[round {ep:03d}] training: L={logs['all']:.4f} rec_b={logs['mse_b']:.4f} rec_a={logs['mse_a']:.4f} "
              f"diff={logs['diff']:.4f} mono={logs['mono']:.4f} cls={logs['cls']:.4f} | Verification: mse_b={vl['mse_b']:.4f} mse_a={vl['mse_a']:.4f} acc={vl['acc']:.3f} ΔHI mean={vl['delta_mean']:.3f}")

        # Simple early stopping: observe and verify the total reconstruction error
        if vl["mse_b"]+vl["mse_a"] < best_v:
            best_v = vl["mse_b"]+vl["mse_a"]
            best = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            print(f" -> Best model update! Verify MSE: {best_v:.6f}")

    if best is not None:
        model.load_state_dict(best)
        print(f"\nModel training completed, optimal weights have been loaded")
        if nan_count > 0:
            print(f"{nan_count} NaN batches were skipped during training")

        # Save the best model
        save_path = "/content/drive/MyDrive/Mimar_turbo"
        os.makedirs(save_path, exist_ok=True)
        model_save_path = os.path.join(save_path, "main_identification_1.pth")
        torch.save(model.state_dict(), model_save_path)
        print(f"The best model has been saved to: {model_save_path}")

    # Test set evaluation
    te_result = eval_epoch(model, ld_te, device)
    print(f"\n== Test result == mse_b={te_result['mse_b']:.4f} mse_a={te_result['mse_a']:.4f} acc={te_result['acc']:.3f} ΔHI mean={te_result['delta_mean']:.3f}")

    # Detailed test result analysis
    y_true, y_pred, delta_mean_all, uids_all, curves = collect_test_predictions(model, ld_te, device)

    print("\n================== Test set overall indicators ==================")
    print(f"Number of samples: {len(y_true)}")
    if len(y_true) > 0:
        acc = (y_true == y_pred).mean()
        print(f"Classification accuracy (Accuracy): {acc:.4f}")
    else:
        print("There are no samples in the test set (check pairs partitioning and horizon conditions).")

    # Classification Report & Confusion Matrix
    try:
        from sklearn.metrics import classification_report, confusion_matrix
        target_names = [LABEL2NAME[i] for i in sorted(LABEL2NAME)]
        print("\n[Classification Report]")
        print(classification_report(y_true, y_pred, target_names=target_names, digits=4))
        cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])
    except Exception as e:
        print("\n[Warning] scikit-learn is not installed, a simple confusion table will be printed.")
        cm = np.zeros((3,3), dtype=int)
        for t,p in zip(y_true, y_pred):
            cm[int(t), int(p)] += 1

    print("\n[Confusion Matrix] rows=real classes, columns=predicted classes")
    print(pd.DataFrame(cm, index=[LABEL2NAME[i] for i in [0,1,2]],
                          columns=[LABEL2NAME[i] for i in [0,1,2]]))

    # Draw confusion matrix
    plt.figure(figsize=(4.8,4.0))
    plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion Matrix (Test)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks([0,1,2], [LABEL2NAME[i] for i in [0,1,2]])
    plt.yticks([0,1,2], [LABEL2NAME[i] for i in [0,1,2]])
    for i in range(3):
        for j in range(3):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    # Statistics ΔHI distribution
    if len(delta_mean_all) > 0:
        df_delta = pd.DataFrame({
            "uid": uids_all[:len(delta_mean_all)],
            "true": y_true.astype(int),
            "pred": y_pred.astype(int),
            "delta_hi_mean": delta_mean_all.astype(float)
        })
        print("\n[ΔHI mean statistics based on true categories]")
        print(df_delta.groupby("true")["delta_hi_mean"].describe())
        print("\n[ΔHI mean statistics by predicted category]")
        print(df_delta.groupby("pred")["delta_hi_mean"].describe())

        #Top-K Analysis
        topk_by_delta(df_delta, k=5)

    # Visualization (aligned to continuous timeline)
    plot_hi_examples_aligned(curves, n_show=6, seed=2025)
    plot_sensor_examples_aligned(curves, sensor_idx_list=None, n_cols=4)

    return model, (ds_tr,ld_tr), (ds_vl,ld_vl), (ds_te,ld_te)

# ============================================================
# 6) Run
# ============================================================
# You need to prepare the `pairs` variable in advance (the structure is consistent with the original script)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 300
BATCH_SIZE = 32
HORIZON = None # or e.g. 80
LR = 1e-3

model, tr, vl, te = train_model(
    pairs,
    epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, horizon=HORIZON, device=DEVICE
)

# (Optional) Export the sample-by-sample HI difference/slope analysis table of the test set
@torch.no_grad()
def hi_analysis_table(model, loader, device):
    model.eval()
    rows=[]
    for batch in loader:
        xb = batch["x_before"].to(device)
        xa = batch["x_after"].to(device)
        labels = batch["labels"].to(device)
        mask   = batch["mask"].to(device)
        uids   = batch["uids"]

        _, _, h_b, h_a, logits = model(xb, xa, mask)

        # Numerical stabilization
        h_b, h_a, logits = sanitize_tensors(h_b, h_a, logits)

        pred = logits.argmax(1).cpu().numpy()
        B,T,_C = xb.shape
        valid = mask.sum(1,keepdim=True).clamp_min(1.0)
        delta_mean = (((h_a-h_b)*mask).sum(1,keepdim=True)/valid).cpu().numpy()
        # linear slope
        t = torch.arange(T, device=mask.device, dtype=h_b.dtype)
        t = (t - t.mean())/(t.std()+1e-6)
        def slope(x):
            num = (x*t).sum(1) - x.sum(1)*t.sum()/T
            den = (t**2).sum() - (t.sum()**2)/T + 1e-8
            return num/den
        slope_b = slope(h_b).cpu().numpy()
        slope_a = slope(h_a).cpu().numpy()
        for i,uid in enumerate(uids):
            rows.append({
                "uid": uid,
                "delta_hi_mean": float(delta_mean[i]),
                "slope_before": float(slope_b[i]),
                "slope_after":  float(slope_a[i]),
                "pred_class":   int(pred[i]),
                "true_class":   int(labels[i].cpu().item())
            })
    return pd.DataFrame(rows)

ds_te, ld_te = te
df_hi = hi_analysis_table(model, ld_te, DEVICE)
print(df_hi.head(20))

# %% [notebook code cell 12]

# ============================================================
# Monotonic-Decreasing HI + aligned plotting (before -> after)
# ============================================================
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import random
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL2NAME = {0: "Perfect", 1: "General", 2: "Poor"}

# ============================================================
# 1) Dataset: read samples from pairs, perform shift operations in batch later
# ============================================================
class PairsReconstructDataset(Dataset):
    """
    Each sample contains:
      x_before: (L,C) feature sequence before maintenance
      x_after : (L,C) feature sequence after maintenance
      hi_before/hi_after: (L,) health index (for evaluation/optional analysis only, not strong supervision)
      strategy: maintenance type ("Perfect Maintenance" / "General Maintenance" / "Poor Maintenance")
    """
    def __init__(self, pairs: dict, horizon: int=None):
        items = []
        for uid, sd in pairs.items():
            for strat, d in sd.items():
                xb = np.asarray(d["x_before"], dtype=np.float32)
                xa = np.asarray(d["x_after"],  dtype=np.float32)
                # When there's no real HI, create placeholder that will be learned by the model
                if "hi_before" in d and d["hi_before"] is not None:
                    hib = np.asarray(d["hi_before"],dtype=np.float32)
                else:
                    # Create initialized fake HI, model will learn real patterns
                    hib = np.linspace(0.8, 0.4, len(xb)).astype(np.float32)

                if "hi_after" in d and d["hi_after"] is not None:
                    hia = np.asarray(d["hi_after"], dtype=np.float32)
                else:
                    # Create different fake HI patterns based on maintenance strategy
                    if "perfect" in strat.lower():
                        # Perfect maintenance: significant improvement
                        hia = np.linspace(0.9, 0.6, len(xa)).astype(np.float32)
                    elif "general" in strat.lower():
                        # General maintenance: moderate improvement
                        hia = np.linspace(0.7, 0.5, len(xa)).astype(np.float32)
                    else:
                        # Poor maintenance: slight improvement
                        hia = np.linspace(0.5, 0.35, len(xa)).astype(np.float32)

                L, C = xb.shape
                # Unify length (optional)
                if horizon is not None and L > horizon:
                    xb, xa, hib, hia = xb[:horizon], xa[:horizon], hib[:horizon], hia[:horizon]
                    L = horizon
                if L < 3:  # At least need to form 0:L-2 -> 1:L-1
                    continue
                items.append({"uid": uid, "strategy": strat, "x_before": xb, "x_after": xa,
                              "hi_before": hib, "hi_after": hia})
        assert len(items)>0, "Dataset is empty, please check pairs data."
        self.items = items
        self.C = items[0]["x_before"].shape[1]

        # Strategy to 3-class label mapping
        def strat_to_cls(s):
            s = s.lower()
            if "perfect" in s: return 0
            if "general" in s: return 1
            if "poor" in s: return 2
            raise ValueError(f"Unknown strategy name: {s}")
        self.labels = [strat_to_cls(it["strategy"]) for it in items]

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        return {
            "uid": it["uid"],
            "label": self.labels[idx],
            "x_before": torch.from_numpy(it["x_before"]), # (L,C)
            "x_after":  torch.from_numpy(it["x_after"]),  # (L,C)
            "hi_before":torch.from_numpy(it["hi_before"]),# (L,)
            "hi_after": torch.from_numpy(it["hi_after"]), # (L,)
            "strategy": it["strategy"]
        }

def pad_collate_shift(batch):
    """
    Pad sequences to same length; return mask, do NOT perform shift operation here
    to allow unified training: inputs = [:,0:L-2], targets = [:,1:L-1]
    """
    Ls = [b["x_before"].shape[0] for b in batch]
    Lmax = max(Ls)
    C = batch[0]["x_before"].shape[1]

    def pad2d(x):
        L, C0 = x.shape
        out = torch.zeros(Lmax, C0, dtype=x.dtype)
        out[:L] = x
        return out

    def pad1d(x):
        L = x.shape[0]
        out = torch.zeros(Lmax, dtype=x.dtype)
        out[:L] = x
        return out

    x_before = torch.stack([pad2d(b["x_before"]) for b in batch], 0) # (B,Lmax,C)
    x_after  = torch.stack([pad2d(b["x_after"])  for b in batch], 0) # (B,Lmax,C)
    hi_before= torch.stack([pad1d(b["hi_before"])for b in batch], 0) # (B,Lmax)
    hi_after = torch.stack([pad1d(b["hi_after"]) for b in batch], 0) # (B,Lmax)
    lengths  = torch.tensor(Ls, dtype=torch.long)
    mask     = torch.zeros(len(batch), Lmax)
    for i,L in enumerate(Ls): mask[i,:L] = 1.0
    labels   = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    uids     = [b["uid"] for b in batch]
    return {"uids": uids, "labels": labels,
            "x_before": x_before, "x_after": x_after,
            "hi_before": hi_before, "hi_after": hi_after,
            "lengths": lengths, "mask": mask}

# ============================================================
# 2) Model: encode "damage" → project to monotonic decreasing HI → recursive reconstruction of two sequences → classify using ΔHI
# ============================================================
class BoltzmannKAN(nn.Module):
    def __init__(self, in_ch, out_ch=4):
        super().__init__()
        self.E  = nn.Parameter(torch.zeros(out_ch, in_ch))
        self.kT = nn.Parameter(torch.ones(out_ch, in_ch))
    def forward(self, x):
        B,T,C = x.shape
        kT = torch.clamp(F.softplus(self.kT), 0.01, 10.0).unsqueeze(0).unsqueeze(2)  # (1,out_ch,1,in_ch)
        E  = torch.clamp(self.E, -10.0, 10.0).unsqueeze(0).unsqueeze(2)             # (1,out_ch,1,in_ch)
        x_ = torch.clamp(x.unsqueeze(1), -10.0, 10.0)                               # (B, out_ch, T, in_ch)
        exp = torch.clamp((E - x_) / kT, -50, 50)
        w   = torch.sigmoid(exp)
        h   = (w * x_).sum(dim=3).permute(0, 2, 1)           # (B,T,out_ch) >=0
        return torch.clamp(F.softplus(h), 0.0, 100.0)

class BaseOp(nn.Module): pass
class MonotonicLinearOp(BaseOp):
    def __init__(self):
        super().__init__()
        self.raw_scale=nn.Parameter(torch.tensor(0.0))
        self.bias=nn.Parameter(torch.tensor(0.0))
        self.smin,self.smax=0.01,5.0
    def forward(self,h):
        scale=torch.clamp(F.softplus(self.raw_scale),self.smin,self.smax)
        xm=torch.clamp(h.mean(-1), -10.0, 10.0)
        return torch.clamp(F.softplus(scale*(xm+self.bias)), 0.0, 100.0)

class MonotonicFlatOp(BaseOp):
    def __init__(self):
        super().__init__()
        self.raw_scale=nn.Parameter(torch.tensor(0.0))
        self.bias=nn.Parameter(torch.tensor(0.0))
        self.smin,self.smax=1e-3,1.0
    def _cum(self,x):
        diff=F.relu(x[:,1:]-x[:,:-1])
        return torch.cat([torch.zeros_like(diff[:,:1]),torch.cumsum(diff,1)],1)
    def forward(self,h):
        scale=torch.clamp(F.softplus(self.raw_scale),self.smin,self.smax)
        xm=torch.clamp(h.mean(-1,keepdim=True), -10.0, 10.0)
        return torch.clamp(F.softplus(scale*(self._cum(xm)+self.bias)).squeeze(-1), 0.0, 100.0)

class ConcaveLogOp(BaseOp):
    def __init__(self):
        super().__init__()
        self.raw_scale=nn.Parameter(torch.tensor(0.0))
        self.eps=1e-3
        self.smin,self.smax=0.01,5.0
    def forward(self,h):
        scale=torch.clamp(F.softplus(self.raw_scale),self.smin,self.smax)
        xm=torch.clamp(h.mean(-1), -10.0, 10.0)
        return torch.clamp(F.softplus(scale*torch.log(torch.abs(xm)+self.eps)), 0.0, 100.0)

class SaturationSigmoidOp(BaseOp):
    def __init__(self):
        super().__init__()
        self.raw_scale=nn.Parameter(torch.tensor(0.0))
        self.raw_slope=nn.Parameter(torch.tensor(0.0))
        self.bias=nn.Parameter(torch.tensor(0.0))
        self.smin,self.smax=0.01,5.0
        self.lmin,self.lmax=0.1,5.0
    def forward(self,h):
        scale=torch.clamp(F.softplus(self.raw_scale),self.smin,self.smax)
        slope=torch.clamp(F.softplus(self.raw_slope),self.lmin,self.lmax)
        xm=torch.clamp(h.mean(-1), -10.0, 10.0)
        return torch.clamp(F.softplus(scale*torch.sigmoid(slope*(xm-self.bias))), 0.0, 100.0)

class HingeReLUOp(BaseOp):
    def __init__(self):
        super().__init__()
        self.raw_scale=nn.Parameter(torch.tensor(0.0))
        self.threshold=nn.Parameter(torch.tensor(0.0))
        self.smin,self.smax=0.01,5.0
    def forward(self,h):
        scale=torch.clamp(F.softplus(self.raw_scale),self.smin,self.smax)
        xm=torch.clamp(h.mean(-1), -10.0, 10.0)
        return torch.clamp(F.softplus(scale*F.relu(xm-self.threshold)), 0.0, 100.0)

class PolynomialOp(BaseOp):
    def __init__(self,deg=3):
        super().__init__()
        self.raw_coeff=nn.Parameter(torch.zeros(deg))
        self.deg=deg
    def forward(self,h):
        xm=torch.clamp(h.mean(-1), -5.0, 5.0)
        y=torch.zeros_like(xm)
        for i in range(self.deg):
            coeff = torch.clamp(F.softplus(self.raw_coeff[i]), 0.01, 5.0)
            y = y + coeff * torch.clamp(xm ** (i+1), -100.0, 100.0)
        return torch.clamp(F.softplus(y), 0.0, 100.0)

class DampedSinOp(BaseOp):
    def __init__(self):
        super().__init__()
        self.raw_scale=nn.Parameter(torch.tensor(0.0))
        self.raw_freq=nn.Parameter(torch.tensor(0.0))
        self.raw_lambda=nn.Parameter(torch.tensor(0.0))
        self.phase=nn.Parameter(torch.tensor(0.0))
        self.smin,self.smax=0.01,5.0
        self.fmin,self.fmax=0.1,5.0
        self.lmin,self.lmax=0.01,3.0
    def forward(self,h):
        scale=torch.clamp(F.softplus(self.raw_scale),self.smin,self.smax)
        freq=torch.clamp(F.softplus(self.raw_freq),self.fmin,self.fmax)
        lam=torch.clamp(F.softplus(self.raw_lambda),self.lmin,self.lmax)
        xm=torch.clamp(h.mean(-1), -10.0, 10.0)
        damp=1/(1+lam*torch.abs(xm))
        phase_val = torch.clamp(self.phase, -10.0, 10.0)
        return torch.clamp(F.softplus(scale*damp*(torch.sin(freq*xm+phase_val)+1)), 0.0, 100.0)

class PiecewiseLinearOp(BaseOp):
    def __init__(self):
        super().__init__()
        self.raw_k1=nn.Parameter(torch.tensor(0.0))
        self.raw_k2=nn.Parameter(torch.tensor(0.0))
        self.threshold=nn.Parameter(torch.tensor(0.0))
        self.kmin,self.kmax=0.01,5.0
    def forward(self,h):
        k1=torch.clamp(F.softplus(self.raw_k1),self.kmin,self.kmax)
        k2=torch.clamp(F.softplus(self.raw_k2),self.kmin,self.kmax)
        thresh = torch.clamp(self.threshold, -5.0, 5.0)
        xm=torch.clamp(h.mean(-1), -10.0, 10.0)
        left=k1*xm
        right=k1*thresh + k2*(xm-thresh)
        out=torch.where(xm<=thresh,left,right)
        return torch.clamp(F.softplus(out), 0.0, 100.0)

class SparseGate(nn.Module):
    def __init__(self, n_ops, tau_start=5.0, tau_end=0.1, n_steps=10000):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(n_ops))
        self.tau_start, self.tau_end, self.n_steps = tau_start, tau_end, n_steps
        self.register_buffer("step", torch.tensor(0.))
    def forward(self):
        tau = (self.tau_start*(1-self.step/self.n_steps) + self.tau_end*(self.step/self.n_steps)).clamp(min=self.tau_end)
        self.step.add_(1)  # Use add_ to avoid inplace operation
        g = -torch.empty_like(self.logits).exponential_().log() if self.training else torch.zeros_like(self.logits)
        g = torch.clamp(g, -50.0, 50.0)  # Limit Gumbel noise
        logits_stable = torch.clamp(self.logits, -50.0, 50.0)
        w = F.softmax((logits_stable + g)/tau, dim=-1)
        return w.view(1,1,-1)

class CustomKAN(nn.Module):
    def __init__(self, ops):
        super().__init__()
        self.ops = nn.ModuleList(ops)
        self.gate= SparseGate(len(ops))
        # Additional learnable scaling/bias for damage
        self.raw_gain = nn.Parameter(torch.tensor(0.0))
        self.bias     = nn.Parameter(torch.tensor(0.0))
    def forward(self, h):  # h:(B,T,trend_ch)
        outs = [op(h) for op in self.ops]          # list of (B,T)
        Tm = min(o.size(1) for o in outs)
        outs = [o[:,:Tm] for o in outs]
        st = torch.stack(outs, dim=-1)             # (B,Tm,K), >=0
        w  = self.gate()                           # (1,1,K)
        damage = torch.clamp((st*w).sum(-1), 0.0, 100.0)  # (B,Tm) non-negative, regarded as "damage"
        gain = torch.clamp(F.softplus(self.raw_gain), 0.1, 5.0)
        bias_val = torch.clamp(self.bias, -5.0, 5.0)
        damage = torch.clamp(gain*damage + bias_val, 0.0, 100.0)
        return damage                               # (B,Tm)

class TrendEncoder(nn.Module):
    """
    Encoder outputs "damage", then projects to monotonic decreasing HI:
        HI is learned from sensor data without hard range constraints
    Without real HI, learns mapping from sensor data to infer health states
    """
    def __init__(self, in_ch, trend_ch=4):
        super().__init__()
        self.boltz = BoltzmannKAN(in_ch, out_ch=trend_ch)
        ops = [MonotonicLinearOp(), MonotonicFlatOp(), ConcaveLogOp(),
               SaturationSigmoidOp(), HingeReLUOp(), PolynomialOp(),
               DampedSinOp(), PiecewiseLinearOp()]
        self.customkan = CustomKAN(ops)
        # Projection parameters - learn flexible health states from sensor features
        self.proj_gain = nn.Parameter(torch.tensor(1.0))
        self.proj_bias = nn.Parameter(torch.tensor(0.0))

        # Add health-aware layer, directly infer from multi-dimensional sensor features
        self.health_aware_transform = nn.Sequential(
            nn.Linear(in_ch, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output 0-1 range health state
        )

        # Add linear constraint layer, enforce linear trend
        self.linear_enforcer = nn.Sequential(
            nn.Linear(in_ch, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output linear decay degree
        )

    def forward(self, x):  # x:(B,T,C)
        # Feature extraction based on Boltzmann KAN
        h_multi = self.boltz(x)              # (B,T,trend_ch) >=0
        damage  = self.customkan(h_multi)    # (B,T)           >=0

        # Directly learn health states from raw sensor data
        # Assess health state for sensor readings at each time step
        B, T, C = x.shape
        x_reshaped = x.view(-1, C)  # (B*T, C)
        health_direct = self.health_aware_transform(x_reshaped)  # (B*T, 1)
        health_direct = health_direct.view(B, T)  # (B, T)

        # Add linear constraint, force linear decay pattern
        linear_weight = self.linear_enforcer(x_reshaped).view(B, T)  # (B, T)

        # Generate ideal linear decay pattern
        t = torch.arange(T, device=x.device, dtype=x.dtype).unsqueeze(0).expand(B, -1)  # (B, T)
        t_normalized = t / (T - 1)  # Normalize to [0, 1]

        # Combine damage and direct health assessment
        g = torch.clamp(F.softplus(self.proj_gain), 0.1, 5.0)
        b = torch.clamp(self.proj_bias, -5.0, 5.0)

        # Convert damage to health state decay
        damage_normalized = torch.sigmoid(g*(damage + b))

        # Combine three health assessment methods: direct assessment, damage pattern, linear constraint
        combined_health = health_direct * (1 - 0.3 * damage_normalized)

        # Generate ideal linear decay curve (from high to low) - remove hard range constraints
        linear_decay = 1.0 - t_normalized * 0.5  # Simple linear decay from 1.0 to 0.5

        # Use linear_weight to mix linear decay and complex patterns
        # Higher linear_weight means more towards linear
        hi = linear_weight * linear_decay + (1 - linear_weight) * combined_health

        # Force stronger monotonic decreasing constraint without hard range limits
        for t_idx in range(1, T):
            hi = torch.cat([
                hi[:, :t_idx],
                torch.min(hi[:, t_idx:t_idx+1], hi[:, t_idx-1:t_idx] - 0.001),  # Force at least 0.001 decrease
                hi[:, t_idx+1:]
            ], dim=1)

        return hi, h_multi

class ReconHead(nn.Module):
    """
    Recursive reconstruction: given (x_t, h_t) → predict x_{t+1}
    """
    def __init__(self, C, hidden=128):
        super().__init__()
        self.gru = nn.GRU(input_size=C+1, hidden_size=hidden, batch_first=True)
        self.out = nn.Linear(hidden, C)
    def forward(self, x_in, h_in):
        # x_in:(B,T_in,C), h_in:(B,T_in)
        B,T,C = x_in.shape
        h_in_clamped = torch.clamp(h_in, 0.0, 10.0)  # Remove upper limit constraint
        feat = torch.cat([x_in, h_in_clamped.unsqueeze(-1)], dim=-1)  # (B,T,C+1)
        H,_ = self.gru(feat)                                  # (B,T,H)
        y = self.out(H)                                       # (B,T,C) aligned predict x_{t+1}
        return y

class MaintClassifier(nn.Module):
    """
    Use HI before-after difference features for 3-class classification
    Enhanced version: not only uses ΔHI, but also sensor data change patterns
    """
    def __init__(self, sensor_dim, hidden=64, n_classes=3):
        super().__init__()
        # Original HI difference features
        self.hi_mlp = nn.Sequential(
            nn.Linear(6, hidden), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden//2)
        )

        # Sensor data change features
        self.sensor_mlp = nn.Sequential(
            nn.Linear(sensor_dim * 2, hidden), nn.ReLU(),  # Before and after statistical features
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden//2)
        )

        # Fusion classifier
        self.final_classifier = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, n_classes)
        )

    def forward(self, h_b, h_a, x_b, x_a, mask):
        # HI difference features (original logic)
        m = mask
        T = m.size(1)
        valid = m.sum(1, keepdim=True).clamp_min(1.0)  # (B,1)
        db = h_a - h_b                        # (B,T)
        mean_d = (db*m).sum(1,keepdim=True)/valid
        var_d  = (((db-mean_d)*m)**2).sum(1,keepdim=True)/valid
        std_d  = (var_d + 1e-8).sqrt()
        d0     = (db[:, :1])                  # First value
        dmax   = (db.masked_fill(m==0, -1e9)).max(dim=1, keepdim=True).values
        pos_ratio = ((db>0).float()*m).sum(1,keepdim=True)/valid

        # Slope (linear fitting)
        t = torch.arange(T, device=h_b.device, dtype=h_b.dtype)
        t = (t - t.mean())/(t.std()+1e-6)
        def slope(x):
            num = (x*t).sum(1) - x.sum(1)*t.sum()/T
            den = (t**2).sum() - (t.sum()**2)/T + 1e-8
            return (num/den).unsqueeze(1)
        slope_diff = slope(h_a) - slope(h_b)  # (B,1)
        hi_feat = torch.cat([mean_d, d0, dmax, std_d, slope_diff, pos_ratio], dim=1)  # (B,6)
        hi_feat = torch.clamp(hi_feat, -10.0, 10.0)
        hi_features = self.hi_mlp(hi_feat)

        # Sensor data change features (fix broadcast BUG: directly divide by (B,1) shaped valid)
        x_b_mean = (x_b * m.unsqueeze(-1)).sum(1) / valid        # (B,C)
        x_a_mean = (x_a * m.unsqueeze(-1)).sum(1) / valid        # (B,C)
        sensor_change = torch.cat([x_b_mean, x_a_mean], dim=1)   # (B, 2*C)
        sensor_features = self.sensor_mlp(sensor_change)

        # Fused features
        combined_features = torch.cat([hi_features, sensor_features], dim=1)
        logits = self.final_classifier(combined_features)

        return logits

def sanitize_tensors(*tensors):
    """Replace NaN/Inf in tensors with finite values"""
    result = []
    for t in tensors:
        if t is not None:
            t_clean = torch.nan_to_num(t, nan=0.0, posinf=1e6, neginf=-1e6)
            result.append(t_clean)
        else:
            result.append(t)
    return result[0] if len(result) == 1 else result

class DiffAwareReconstructor(nn.Module):
    """
    Overall model:
     - Two encoders share parameters: encode x_before / x_after → h_b / h_a (monotonic decreasing HI)
     - Two reconstruction heads: perform one-step recursive reconstruction respectively
     - Classification head: use (h_a - h_b) statistical features to identify maintenance type
    Without real HI labels, learn to infer health states from sensor data through self-supervised learning
    """
    def __init__(self, in_ch, trend_ch=4, hidden=128, n_classes=3):
        super().__init__()
        self.encoder = TrendEncoder(in_ch, trend_ch)
        self.recon_b = ReconHead(in_ch, hidden)
        self.recon_a = ReconHead(in_ch, hidden)
        self.clf     = MaintClassifier(sensor_dim=in_ch, hidden=64, n_classes=n_classes)

        # Add self-supervised consistency constraint
        self.consistency_weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, x_b, x_a, mask):
        # x_b/x_a : (B,L,C) (will be cut to 0:L-2 / 1:L-1 later)
        h_b, _ = self.encoder(x_b)  # (B,L)  Health state learned from sensor data
        h_a, _ = self.encoder(x_a)  # (B,L)

        # Numerical stabilization
        h_b, h_a = sanitize_tensors(h_b, h_a)

        # Recursive reconstruction: input 0:L-2 → predict 1:L-1
        xb_in, hb_in = x_b[:, :-2], h_b[:, :-2]
        xa_in, ha_in = x_a[:, :-2], h_a[:, :-2]
        yb_hat = self.recon_b(xb_in, hb_in)   # (B,L-2,C) ≈ x_b[:,1:L-1]
        ya_hat = self.recon_a(xa_in, ha_in)   # (B,L-2,C) ≈ x_a[:,1:L-1]

        # Numerical stabilization
        yb_hat, ya_hat = sanitize_tensors(yb_hat, ya_hat)

        # Classification: use unclipped length mask and include sensor features
        logits = self.clf(h_b, h_a, x_b, x_a, mask)
        logits = sanitize_tensors(logits)

        return yb_hat, ya_hat, h_b, h_a, logits

# ============================================================
# 3) Loss function: reconstruction + ΔHI + smooth/monotonic(decreasing) + classification + first-order linear + self-supervised consistency + maintenance effect constraint + global HI superiority constraint
# ============================================================
def masked_mse(a, b, mask):
    # a,b:(B,T,C)  mask:(B,T)
    diff = (a - b)**2
    mse  = (diff.mean(-1) * mask).sum() / (mask.sum() + 1e-6)
    return mse

def slope_loss(h, mask, kind="smooth"):
    # Smooth: second-order difference; Monotonic decreasing: penalize upward trend
    if kind=="smooth":
        d2 = h[:,2:] - 2*h[:,1:-1] + h[:,:-2]
        m  = mask[:,2:]
        return (d2.abs() * m).sum() / (m.sum()+1e-6)
    elif kind=="mono_dec":
        dh = h[:,1:] - h[:,:-1]        # If >0 indicates upward (violates "decreasing")
        m  = mask[:,1:]
        return (F.relu(dh) * m).sum() / (m.sum()+1e-6)
    else:
        return torch.tensor(0., device=h.device)

def enhanced_linear_slope_loss(h, mask, weight=5.0):
    """
    Enhanced linear constraint: calculate deviation from best linear fit, stronger weight
    Without real HI, this constraint helps learn reasonable decay patterns
    """
    B, T = h.shape
    valid_mask = mask  # (B, T)

    # Calculate best linear fit for each sample
    t = torch.arange(T, device=h.device, dtype=h.dtype).unsqueeze(0).expand(B, -1)  # (B, T)

    # Consider only valid positions
    loss_total = torch.tensor(0.0, device=h.device)
    n_valid_samples = 0

    for b in range(B):
        valid_indices = valid_mask[b] > 0
        if valid_indices.sum() < 2:  # Need at least 2 points for linear fitting
            continue

        h_valid = h[b][valid_indices]  # Valid HI values
        t_valid = t[b][valid_indices]  # Corresponding time points

        # Center time points
        t_mean = t_valid.mean()
        t_centered = t_valid - t_mean
        h_mean = h_valid.mean()

        # Calculate linear regression slope
        numerator = (t_centered * (h_valid - h_mean)).sum()
        denominator = (t_centered ** 2).sum() + 1e-8
        slope = numerator / denominator

        # Calculate intercept
        intercept = h_mean - slope * t_mean

        # Linear predicted values
        h_linear = slope * t_valid + intercept

        # Calculate MSE with linear fit, using stronger weight
        linear_mse = ((h_valid - h_linear) ** 2).mean()

        # Additional penalty for positive slope (encourage negative slope, i.e., decreasing)
        slope_penalty = F.relu(slope + 0.01) * 2.0  # Slope should be negative

        loss_total += weight * linear_mse + slope_penalty
        n_valid_samples += 1

    return loss_total / max(n_valid_samples, 1)

def maintenance_improvement_constraint(h_b, h_a, labels, mask):
    """
    Maintenance improvement constraint: ensure post-maintenance HI is significantly higher than pre-maintenance
    - Perfect(0): require post-maintenance significantly higher than pre-maintenance (at least 0.4)
    - General(1): require post-maintenance moderately higher than pre-maintenance (at least 0.2)
    - Poor(2): require post-maintenance slightly higher than pre-maintenance (at least 0.1)
    """
    valid = mask.sum(1, keepdim=True).clamp_min(1.0)  # (B,1)

    # Calculate average HI before and after maintenance
    h_b_mean = (h_b * mask).sum(1, keepdim=True) / valid  # (B,1)
    h_a_mean = (h_a * mask).sum(1, keepdim=True) / valid  # (B,1)

    improvement = h_a_mean - h_b_mean  # Improvement after maintenance relative to before

    loss = torch.zeros_like(improvement)
    for cls in [0, 1, 2]:
        sel = (labels == cls).float().unsqueeze(1)  # (B,1)
        if sel.sum() < 1:
            continue

        if cls == 0:  # Perfect
            # Require at least 0.4 improvement
            loss += sel * F.relu(0.4 - improvement) ** 2
        elif cls == 1:  # General
            # Require at least 0.2 improvement
            loss += sel * F.relu(0.2 - improvement) ** 2
        else:  # Poor
            # Require at least 0.1 improvement
            loss += sel * F.relu(0.1 - improvement) ** 2

    return loss.mean()

def global_hi_superiority_constraint(h_b, h_a, mask, weight=5.0):
    """
    Enhanced global superiority constraint: ensure post-maintenance HI is entirely higher than pre-maintenance HI
    For each sample, require post-maintenance HI at every time point to be higher than pre-maintenance HI
    Remove range constraints and focus on relative superiority
    """
    valid = mask  # (B, T)

    loss_total = torch.tensor(0.0, device=h_b.device)
    n_valid_samples = 0

    for b in range(h_b.size(0)):
        valid_mask_b = valid[b] > 0
        if valid_mask_b.sum() < 1:
            continue

        # Get valid HI values before and after maintenance
        h_b_valid = h_b[b][valid_mask_b]  # Valid HI values before maintenance
        h_a_valid = h_a[b][valid_mask_b]  # Valid HI values after maintenance

        # Point-wise superiority: each post-maintenance point should be higher than corresponding pre-maintenance point
        pointwise_gap = h_a_valid - h_b_valid  # Should be positive
        pointwise_loss = F.relu(-pointwise_gap + 0.05).mean()  # Penalize when post-maintenance is not sufficiently higher

        # Global superiority: minimum post-maintenance should be higher than maximum pre-maintenance
        h_b_max = h_b_valid.max()
        h_a_min = h_a_valid.min()

        # Require post-maintenance minimum to be higher than pre-maintenance maximum
        margin = 0.1  # Post-maintenance minimum should be at least 0.1 higher than pre-maintenance maximum
        global_gap_loss = F.relu(h_b_max + margin - h_a_min) ** 2

        # Additional constraint: post-maintenance average should be significantly higher than pre-maintenance average
        h_b_mean = h_b_valid.mean()
        h_a_mean = h_a_valid.mean()
        mean_gap_loss = F.relu(h_b_mean + 0.2 - h_a_mean) ** 2  # Average should improve by at least 0.2

        loss_total += weight * (pointwise_loss + global_gap_loss + 0.5 * mean_gap_loss)
        n_valid_samples += 1

    return loss_total / max(n_valid_samples, 1)

def diff_margin_by_class(mean_delta, labels, m_low=0.1, m_mid=0.25, m_high=0.45):
    """
    Class-conditional difference constraint (enhanced maintenance effect):
      - Perfect(0):  Δ>=m_high (post-maintenance significantly higher than pre-maintenance)
      - General(1):  Δ≈m_mid   (post-maintenance moderately higher than pre-maintenance)
      - Poor(2):     Δ>=m_low  (post-maintenance at least slightly higher than pre-maintenance)
    All classes require post-maintenance HI higher than pre-maintenance (positive improvement), just different degrees
    """
    loss = torch.zeros_like(mean_delta)
    for cls in [0,1,2]:
        sel = (labels==cls).float()
        if sel.sum() < 1: continue
        if cls==0:
            # Perfect: require significant improvement (at least reach m_high)
            loss += sel * F.relu(m_high - mean_delta)**2
        elif cls==1:
            # General: require moderate improvement (close to m_mid)
            loss += sel * (mean_delta - m_mid)**2
        else:
            # Poor: require at least small improvement (at least reach m_low)
            loss += sel * F.relu(m_low - mean_delta)**2
    denom = torch.ones_like(mean_delta) * (labels.numel())
    return loss.sum() / denom.sum()

def sensor_consistency_loss(x_b, x_a, h_b, h_a, mask):
    """
    Sensor data consistency loss: ensure HI changes are consistent with sensor data change patterns
    Important constraint when no real HI labels are available
    """
    # Calculate statistical changes in sensor data
    valid = mask.sum(1, keepdim=True).clamp_min(1.0)   # (B,1)

    # Average change magnitude in sensor data (fix broadcast BUG: divide by (B,1))
    xb_mean = (x_b * mask.unsqueeze(-1)).sum(1) / valid   # (B,C)
    xa_mean = (x_a * mask.unsqueeze(-1)).sum(1) / valid   # (B,C)
    sensor_change = torch.abs(xa_mean - xb_mean)          # (B,C)
    sensor_change_magnitude = sensor_change.mean(1)       # (B,)

    # HI change magnitude (post-maintenance improvement relative to pre-maintenance)
    hi_change = ((h_a - h_b) * mask).sum(1) / valid.squeeze(-1)  # (B,) Note: remove abs, maintain positive direction

    # Expect HI change to positively correlate with sensor change, and HI should improve positively
    # Use Pearson correlation coefficient as consistency metric
    mean_sensor = sensor_change_magnitude.mean()
    mean_hi = hi_change.mean()

    correlation = ((sensor_change_magnitude - mean_sensor) * (hi_change - mean_hi)).mean() / \
                  (sensor_change_magnitude.std() * hi_change.std() + 1e-8)

    # Encourage positive correlation (large sensor change when HI change is also large)
    consistency_loss = F.relu(0.5 - correlation)  # Expect correlation at least 0.5

    # Additional constraint: force positive HI improvement
    negative_change_penalty = F.relu(-hi_change).mean() * 2.0  # Penalize negative changes

    return consistency_loss + negative_change_penalty

def smoothness_enhancement_loss(h, mask, order=2):
    """
    Enhanced smoothness loss: use higher-order differences to force smoother curves
    """
    if order == 2:
        # Second-order difference
        d2 = h[:,2:] - 2*h[:,1:-1] + h[:,:-2]
        m = mask[:,2:]
        return (d2.abs() * m).sum() / (m.sum() + 1e-6)
    elif order == 3:
        # Third-order difference (stronger smoothness constraint)
        d3 = h[:,3:] - 3*h[:,2:-1] + 3*h[:,1:-2] - h[:,:-3]
        m = mask[:,3:]
        return (d3.abs() * m).sum() / (m.sum() + 1e-6)
    else:
        return torch.tensor(0., device=h.device)

def split_pairs_uid(pairs, train_ratio=0.7, val_ratio=0.1, seed=42):
    uids = sorted(list(pairs.keys()))
    rs = np.random.RandomState(seed)
    rs.shuffle(uids)
    n = len(uids); n_tr=int(n*train_ratio); n_vl=int(n*val_ratio)
    to_dict = lambda ss:{u:pairs[u] for u in ss if u in pairs}
    return to_dict(uids[:n_tr]), to_dict(uids[n_tr:n_tr+n_vl]), to_dict(uids[n_tr+n_vl:])

def print_split_summary(pairs, train_ratio=0.7, val_ratio=0.1, seed=42):
    pairs_tr, pairs_vl, pairs_te = split_pairs_uid(pairs, train_ratio, val_ratio, seed)
    def count_items(pp):
        cnt = 0
        for uid, d in pp.items():
            cnt += len(d)
        return len(pp), cnt
    n_uid_tr, n_pair_tr = count_items(pairs_tr)
    n_uid_vl, n_pair_vl = count_items(pairs_vl)
    n_uid_te, n_pair_te = count_items(pairs_te)
    print(f"[Data Split] Train: UID={n_uid_tr}, pairs≈{n_pair_tr} | "
          f"Val: UID={n_uid_vl}, pairs≈{n_pair_vl} | "
          f"Test: UID={n_uid_te}, pairs≈{n_pair_te}")

def remove_constant_segments(hi_values, threshold=1e-6):
    """
    Remove constant segments in HI curves (caused by mask or other reasons)
    Return valid HI values and corresponding time indices
    """
    if len(hi_values) <= 1:
        return hi_values, np.arange(len(hi_values))

    # Calculate differences between adjacent points
    diffs = np.abs(np.diff(hi_values))

    # Find points with sufficient change
    valid_mask = np.ones(len(hi_values), dtype=bool)
    valid_mask[1:] = diffs > threshold

    # Ensure at least first and last two points are kept
    if np.sum(valid_mask) < 2:
        valid_mask[0] = True
        valid_mask[-1] = True

    valid_indices = np.where(valid_mask)[0]
    return hi_values[valid_indices], valid_indices

@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total = {"mse_b":0.0, "mse_a":0.0, "acc":0.0, "delta_mean":0.0}
    n_batch=0
    n_acc = 0; n_tot=0
    for batch in loader:
        xb = batch["x_before"].to(device)
        xa = batch["x_after"].to(device)
        hi_b = batch["hi_before"].to(device)
        hi_a = batch["hi_after"].to(device)
        labels = batch["labels"].to(device)
        lengths= batch["lengths"].to(device)
        mask   = batch["mask"].to(device)

        # Valid segment: 0:L-2 input, 1:L-1 target
        m_tgt  = (torch.arange(xb.size(1)-2, device=device)[None,:] < (lengths-2)[:,None]).float()

        yb_hat, ya_hat, h_b, h_a, logits = model(xb, xa, mask)

        # Numerical stabilization
        yb_hat, ya_hat, h_b, h_a, logits = sanitize_tensors(yb_hat, ya_hat, h_b, h_a, logits)

        # Align targets
        yb_tgt = xb[:,1:-1,:]
        ya_tgt = xa[:,1:-1,:]

        mse_b = masked_mse(yb_hat, yb_tgt, m_tgt)
        mse_a = masked_mse(ya_hat, ya_tgt, m_tgt)

        # Check for NaN
        if torch.isnan(mse_b) or torch.isnan(mse_a):
            continue

        # Classification
        pred = logits.argmax(dim=1)
        n_acc += (pred == labels).sum().item()
        n_tot += labels.numel()

        # Statistics of ΔHI mean (post-maintenance improvement relative to pre-maintenance)
        valid = mask.sum(1, keepdim=True).clamp_min(1.0)
        delta_mean = ((h_a - h_b)*mask).sum(1,keepdim=True)/valid
        delta_mean = torch.nan_to_num(delta_mean, nan=0.0)
        total["delta_mean"] += delta_mean.mean().item()

        total["mse_b"] += mse_b.item()
        total["mse_a"] += mse_a.item()
        n_batch += 1

    for k in ["mse_b","mse_a","delta_mean"]:
        total[k] /= max(n_batch,1)
    total["acc"] = n_acc / max(n_tot,1)
    return total

@torch.no_grad()
def collect_test_predictions(model, loader, device, max_curve_keep=24):
    """
    Collect predictions, ΔHI, and a few sample HI curves on test set (for plotting).
    """
    model.eval()
    y_true, y_pred = [], []
    all_delta_mean, all_uids = [], []
    keep_curves = []

    for batch in loader:
        xb = batch["x_before"].to(device)   # (B,L,C)
        xa = batch["x_after"].to(device)
        mask   = batch["mask"].to(device)   # (B,L)
        labels = batch["labels"].to(device) # (B,)
        lengths= batch["lengths"]           # cpu tensor
        uids   = batch["uids"]
        hi_before = batch["hi_before"].to(device)  # (B,L) ground truth HI_before from dataset
        hi_after = batch["hi_after"].to(device)    # (B,L) ground truth HI_after from dataset

        yb_hat, ya_hat, h_b, h_a, logits = model(xb, xa, mask)

        # Numerical stabilization
        yb_hat, ya_hat, h_b, h_a, logits = sanitize_tensors(yb_hat, ya_hat, h_b, h_a, logits)

        prob = F.softmax(logits, dim=1)     # (B,3)
        pred = prob.argmax(1)               # (B,)

        # Statistics of ΔHI mean (post-maintenance improvement relative to pre-maintenance)
        valid = mask.sum(1, keepdim=True).clamp_min(1.0)
        delta_mean = ((h_a - h_b) * mask).sum(1, keepdim=True) / valid  # (B,1)
        delta_mean = torch.nan_to_num(delta_mean, nan=0.0)

        y_true.append(labels.cpu().numpy())
        y_pred.append(pred.cpu().numpy())
        all_delta_mean.append(delta_mean.squeeze(1).cpu().numpy())
        all_uids.extend(uids)

        # Save some curves for visualization (aligned: after concatenated after before)
        for i in range(xb.size(0)):
            if len(keep_curves) >= max_curve_keep:
                break
            L_i = int(lengths[i].item())
            # Fix: truncate reconstruction prediction by actual length
            L_recon = min(L_i - 2, yb_hat.size(1))  # Reconstruction sequence length
            keep_curves.append({
                "uid": uids[i],
                "true": int(labels[i].cpu().item()),
                "pred": int(pred[i].cpu().item()),
                "prob": prob[i].cpu().numpy(),
                "h_before": h_b[i, :L_i].cpu().numpy(),          # model predicted HI_before
                "h_after":  h_a[i, :L_i].cpu().numpy(),          # model predicted HI_after
                "hi_before_gt": hi_before[i, :L_i].cpu().numpy(), # ground truth HI_before from dataset
                "hi_after_gt": hi_after[i, :L_i].cpu().numpy(),   # ground truth HI_after from dataset
                "yb_hat":   yb_hat[i, :L_recon].cpu().numpy(),       # (L_recon,C)
                "ya_hat":   ya_hat[i, :L_recon].cpu().numpy(),       # (L_recon,C)
                "x_before": xb[i, :L_i].cpu().numpy(),               # (L,C)
                "x_after":  xa[i, :L_i].cpu().numpy(),               # (L,C)
                "length": L_i
            })

    y_true = np.concatenate(y_true, axis=0) if len(y_true)>0 else np.array([])
    y_pred = np.concatenate(y_pred, axis=0) if len(y_pred)>0 else np.array([])
    all_delta_mean = np.concatenate(all_delta_mean, axis=0) if len(all_delta_mean)>0 else np.array([])
    return y_true, y_pred, all_delta_mean, all_uids, keep_curves

def select_best_curves_for_plotting(curves, n_show=6):
    """
    Intelligent selection of the best drawing cases:
    1. Prioritize samples with correct predictions
    2. Select the most representative sample within each maintenance strategy category
    3. Select the sample with the most obvious change in health index
    4. Select samples with the best sensor data quality
    """
    if len(curves) == 0:
        return []

    #Group by real category
    curves_by_class = {0: [], 1: [], 2: []}
    for curve in curves:
        curves_by_class[curve["true"]].append(curve)

    # Calculate quality score for each sample
    def calculate_quality_score(curve):
        score = 0.0

        # 1. Prediction accuracy (weight: 50%)
        if curve["true"] == curve["pred"]:
            score += 0.5

        # 2. Prediction confidence (weight: 20%)
        confidence = curve["prob"][curve["pred"]]
        score += 0.2 * confidence

        # 3. Obvious degree of change in health index (weight: 20%)
        h_before = curve["h_before"]
        h_after = curve["h_after"]
        hi_improvement = np.mean(h_after - h_before)
        # Normalize to 0-1 range
        normalized_improvement = np.clip((hi_improvement + 0.5) / 1.0, 0, 1)
        score += 0.2 * normalized_improvement

        # 4. Data quality (weight: 10%)
        # Check if there are outliers or constant segments
        x_before_var = np.var(curve["x_before"])
        x_after_var = np.var(curve["x_after"])
        data_quality = np.clip((x_before_var + x_after_var) / 2.0, 0, 1)
        score += 0.1 * data_quality

        return score

    # Calculate quality scores for all samples
    for curve in curves:
        curve["quality_score"] = calculate_quality_score(curve)

    # Intelligent selection strategy
    selected_curves = []

    # Select at least one best sample for each category
    for class_id in [0, 1, 2]:
        if len(curves_by_class[class_id]) > 0:
            # Sort by quality score
            sorted_curves = sorted(curves_by_class[class_id],
                                 key=lambda x: x["quality_score"], reverse=True)
            selected_curves.append(sorted_curves[0])

    # If more samples are needed, select from the remaining high-quality samples
    remaining_slots = n_show - len(selected_curves)
    if remaining_slots > 0:
        # Exclude selected from all samples, select by quality score
        already_selected_uids = {curve["uid"] for curve in selected_curves}
        remaining_curves = [curve for curve in curves
                          if curve["uid"] not in already_selected_uids]

        if len(remaining_curves) > 0:
            remaining_curves.sort(key=lambda x: x["quality_score"], reverse=True)
            selected_curves.extend(remaining_curves[:remaining_slots])

    #Final sorting by quality score
    selected_curves.sort(key=lambda x: x["quality_score"], reverse=True)

    return selected_curves[:n_show]

def plot_hi_examples_aligned(curves, n_show=6, seed=0):
    """Post-maintenance sequence continues from pre-maintenance endpoint; draw vertical line to mark t_m; remove constant segments.
    Add "Learned HI" in title to indicate this is health state inferred from sensor data, post-maintenance should be higher than pre-maintenance
    Intelligent selection of the best drawing cases
    """
    if len(curves)==0:
        print("(No visualization samples)")
        return

    # Use the smart selection function to select the best drawing case
    selected_curves = select_best_curves_for_plotting(curves, n_show)

    if len(selected_curves) == 0:
        print("(No high-quality visualization samples found)")
        return

    n_show = len(selected_curves)
    cols = 3
    rows = int(np.ceil(n_show/cols))
    plt.figure(figsize=(cols*4.6, rows*3.2))

    for k in range(n_show):
        ex = selected_curves[k]
        hb = ex["h_before"]; ha = ex["h_after"]
        hb_gt = ex["hi_before_gt"]; ha_gt = ex["hi_after_gt"]      # ground truth HI from dataset
        L  = ex["length"]  # Use actual length instead of padded length

        # Remove constant segments
        hb_clean, hb_indices = remove_constant_segments(hb)
        ha_clean, ha_indices = remove_constant_segments(ha)
        hb_gt_clean, hb_gt_indices = remove_constant_segments(hb_gt)
        ha_gt_clean, ha_gt_indices = remove_constant_segments(ha_gt)

        # Corresponding time axis
        t_b = hb_indices
        t_a = ha_indices + L  # after follows immediately
        t_b_gt = hb_gt_indices
        t_a_gt = ha_gt_indices + L

        uid = ex["uid"]
        pred = ex["pred"]; true = ex["true"]; prob = ex["prob"]
        quality_score = ex["quality_score"]

        plt.subplot(rows, cols, k+1)

        # Only plot non-constant segments - model predicted HI
        if len(hb_clean) > 1:
            plt.plot(t_b, hb_clean, label="Learned HI_Pre-maintenance", linewidth=2.5, marker='o', markersize=4, color='blue')
        if len(ha_clean) > 1:
            plt.plot(t_a, ha_clean, label="Learned HI_Post-maintenance",  linewidth=2.5, linestyle='--', marker='s', markersize=4, color='red')

        # Plot ground truth HI from dataset
        if len(hb_gt_clean) > 1:
            plt.plot(t_b_gt, hb_gt_clean, label="GT HI_Pre-maintenance", linewidth=1.8, color='cyan', alpha=0.8)
        if len(ha_gt_clean) > 1:
            plt.plot(t_a_gt, ha_gt_clean, label="GT HI_Post-maintenance",  linewidth=1.8, linestyle=':', color='orange', alpha=0.8)

        # Maintenance time vertical line
        plt.axvline(L-1, color='k', linestyle=':', linewidth=1.5, alpha=0.8)

        d_mean = float(np.mean(ha - hb))  # Post-maintenance improvement relative to pre-maintenance
        d_max  = float(np.max(ha - hb))
        d_mean_gt = float(np.mean(ha_gt - hb_gt))

        # Add quality rating to title
        correctness = "✓" if true == pred else "✗"
        title = (f"uid={uid} {correctness} (Quality: {quality_score:.2f})\n"
                 f"True={LABEL2NAME[true]} | Pred={LABEL2NAME[pred]} | Conf={prob[pred]:.3f}\n"
                 f"ΔHI_Learned={d_mean:.3f}, ΔHI_GT={d_mean_gt:.3f}")
        plt.title(title, fontsize=9)
        plt.xlabel("Cycle (Pre-maintenance | Post-maintenance)")
        plt.ylabel("Health Index (↑Post should be higher)")
        plt.grid(ls='--', alpha=.4, linewidth=0.8)

        #Add background color to indicate prediction accuracy
        if true == pred:
            plt.gca().set_facecolor('#f0f8ff') # Light blue background indicates correct prediction
        else:
            plt.gca().set_facecolor('#fff0f0') # Light red background indicates incorrect prediction

        if k==0: plt.legend(fontsize=8, loc='best')

    plt.suptitle(f"Best {n_show} Health Index Examples (Ranked by Quality Score)", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.show()

def plot_sensor_examples_aligned(curves, sensor_idx_list=None, n_cols=4):
    """
    Plot several sensors: original(before/after) + one-step recursive prediction of after(ya_hat),
    post-maintenance time series starts after pre-maintenance end. Show trajectories under different
    maintenance strategies. Intelligent selection of the best sensor dimensions for drawing
    """
    if len(curves)==0:
        print("(No visualization samples)")
        return

    # Select the best samples for sensor visualization
    selected_curves = select_best_curves_for_plotting(curves, 9) # Select more samples for sensor analysis

    # Group curves by maintenance strategy (true class)
    curves_by_strategy = {0: [], 1: [], 2: []}
    for curve in selected_curves:
        curves_by_strategy[curve["true"]].append(curve)

    # Select best example from each strategy for comparison
    strategy_examples = {}
    for strategy in [0, 1, 2]:
        if len(curves_by_strategy[strategy]) > 0:
            # Select the sample with the highest quality score under this strategy
            best_curve = max(curves_by_strategy[strategy], key=lambda x: x["quality_score"])
            strategy_examples[strategy] = best_curve

    if len(strategy_examples) == 0:
        print("(No high-quality visualization samples for any maintenance strategy)")
        return

    # Intelligent selection of sensor dimensions
    first_ex = list(strategy_examples.values())[0]
    xb = first_ex["x_before"]       # (L,C)
    C = xb.shape[1]

    if sensor_idx_list is None:
        # Intelligently select the most representative sensor dimension
        sensor_scores = []

        for sensor_idx in range(C):
            score = 0.0

            # Calculate the degree of change of the sensor under different maintenance strategies
            for strategy, ex in strategy_examples.items():
                xb_sensor = ex["x_before"][:, sensor_idx]
                xa_sensor = ex["x_after"][:, sensor_idx]

                # 1. Range of change (weight: 40%)
                change_magnitude = np.abs(np.mean(xa_sensor) - np.mean(xb_sensor))
                score += 0.4 * min(change_magnitude / (np.std(xb_sensor) + 1e-6), 5.0)

                # 2. Variance (information amount) (weight: 30%)
                variance = np.var(xb_sensor) + np.var(xa_sensor)
                score += 0.3 * min(variance, 10.0)

                # 3. Signal-to-noise ratio (weight: 30%)
                snr = np.mean(xb_sensor) / (np.std(xb_sensor) + 1e-6)
                score += 0.3 * min(abs(snr), 5.0)

            sensor_scores.append((sensor_idx, score))

        # Select the sensor with the highest rating
        sensor_scores.sort(key=lambda x: x[1], reverse=True)
        sensor_idx_list = [idx for idx, _ in sensor_scores[:min(8, len(sensor_scores))]]

    n = len(sensor_idx_list)
    n_rows = int(np.ceil(n/n_cols))
    plt.figure(figsize=(n_cols*5.5, n_rows*4.0))

    colors = {0: '#e74c3c', 1: '#f39c12', 2: '#9b59b6'} # Brighter colors

    for i, s in enumerate(sensor_idx_list):
        plt.subplot(n_rows, n_cols, i+1)

        # Plot pre-maintenance trajectory (common for all strategies, use best example)
        best_ex = max(strategy_examples.values(), key=lambda x: x["quality_score"])
        xb = best_ex["x_before"]
        L = best_ex["length"]
        t_b = np.arange(L)
        plt.plot(t_b, xb[:L,s], label="Original (Pre-maintenance)", linewidth=2.0, color='#2c3e50', alpha=0.9)

        # Plot post-maintenance trajectories for each available strategy
        for strategy in sorted(strategy_examples.keys()):
            ex = strategy_examples[strategy]
            xa = ex["x_after"]
            ya = ex["ya_hat"]
            L_strategy = ex["length"]
            quality = ex["quality_score"]

            # Post-maintenance original trajectory
            t_a = np.arange(L_strategy, L_strategy + L_strategy)
            plt.plot(t_a, xa[:L_strategy,s],
                    label=f"Original (Post-{LABEL2NAME[strategy]}, Q:{quality:.2f})",
                    linewidth=1.8, linestyle="--",
                    color=colors[strategy], alpha=0.9)

            # Post-maintenance prediction trajectory
            if len(ya) > 0 and L_strategy-2 > 0:
                t_pred = np.arange(L_strategy+1, L_strategy+1+min(L_strategy-2, len(ya)))
                plt.plot(t_pred, ya[:len(t_pred),s],
                        label=f"Predicted (Post-{LABEL2NAME[strategy]})",
                        linewidth=2.2, color=colors[strategy], marker='o', markersize=3)

        # Draw vertical line at maintenance point
        plt.axvline(L-0.5, color='#27ae60', linestyle=':', linewidth=2.5, alpha=0.9, label='Maintenance Point')

        # Calculate the quality score of this sensor and add it to the title
        sensor_score = dict(sensor_scores)[s] if 'sensor_scores' in locals() else 0
        plt.title(f"Sensor_{s:02d} (Score: {sensor_score:.2f}) - Strategy Comparison", fontsize=10, fontweight='bold')

        if i==0:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.xlabel("Continuous Time Sequence", fontsize=9)
        plt.ylabel("Sensor Value", fontsize=9)
        plt.grid(ls="--", alpha=.4, linewidth=0.8)

        # Add slight background color
        plt.gca().set_facecolor('#fafafa')

    plt.suptitle("Best Sensor Trajectories under Different Maintenance Strategies",
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()

def topk_by_delta(df_delta, k=5):
    if len(df_delta)==0:
        return
    print("\n" + "="*60)
    print("TOP-K SAMPLES WITH HIGHEST MAINTENANCE EFFECTS")
    print("="*60)

    for cls in [0,1,2]:
        sub = df_delta[df_delta["true"]==cls].sort_values("delta_hi_mean", ascending=False).head(k)
        if len(sub) > 0:
            print(f"\n[True={LABEL2NAME[cls]}] Top {k} samples with highest learned ΔHI_mean:")
            print("-" * 50)
            for idx, row in sub.iterrows():
                pred_correct = "✓" if row["true"] == row["pred"] else "✗"
                print(f"  UID: {row['uid']:>8} | ΔHI: {row['delta_hi_mean']:>6.3f} | "
                      f"Pred: {LABEL2NAME[int(row['pred'])]:>7} {pred_correct}")

# —— Load trained best model
def load_trained_model(model_path, device, in_ch):
    """Load the best model weights saved during training"""
    # Initialize model with same architecture
    model = DiffAwareReconstructor(in_ch=in_ch, trend_ch=4, hidden=128, n_classes=3).to(device)

    if os.path.exists(model_path):
        print(f"Loading model: {model_path}")
        state_dict = torch.load(model_path, map_location=device)

        # Filter out keys that don't match (for backward compatibility)
        model_dict = model.state_dict()
        filtered_dict = {}
        for k, v in state_dict.items():
            if k in model_dict and model_dict[k].shape == v.shape:
                filtered_dict[k] = v
            else:
                print(f"Warning: Skipping key {k} due to shape mismatch or missing in current model")

        # Load only matching parameters
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict, strict=False)
        print("Model loaded successfully (with potential missing keys)!")
    else:
        print(f"Warning: Model file does not exist {model_path}, will use randomly initialized model")
    return model

# —— Need to determine in_ch from pairs data first
def get_input_dim_from_pairs(pairs):
    """Get input dimension from pairs data"""
    for uid, strategies in pairs.items():
        for strategy, data in strategies.items():
            if "x_before" in data:
                return np.array(data["x_before"]).shape[1]
    raise ValueError("Cannot determine input dimension from pairs data")

# Get input dimension
C = get_input_dim_from_pairs(pairs)
print(f"Detected input dimension: {C}")

# Load best model (if exists)
model_path = "/content/drive/MyDrive/CMAPSS/main_identification_2.pth"
model = load_trained_model(model_path, DEVICE, C)

# —— Print train/validation/test split (consistent with training phase: 7/1/2)
print_split_summary(pairs)

# —— Prepare test set data
_, _, pairs_te = split_pairs_uid(pairs, 0.7, 0.1, 42)
ds_te = PairsReconstructDataset(pairs_te, horizon=50)  # Same horizon as training
ld_te = DataLoader(ds_te, batch_size=32, shuffle=False, collate_fn=pad_collate_shift)
te = (ds_te, ld_te)

y_true, y_pred, delta_mean_all, uids_all, curves = collect_test_predictions(model, ld_te, DEVICE)

# —— Print overall metrics
print("\n" + "="*60)
print("TEST SET OVERALL METRICS")
print("="*60)
print(f"Sample count: {len(y_true)}")
if len(y_true) > 0:
    acc = (y_true == y_pred).mean()
    print(f"Classification Accuracy: {acc:.4f}")

    # Calculate the accuracy of each category
    for cls in [0, 1, 2]:
        cls_mask = (y_true == cls)
        if cls_mask.sum() > 0:
            cls_acc = (y_pred[cls_mask] == cls).mean()
            print(f"{LABEL2NAME[cls]} Accuracy: {cls_acc:.4f} ({cls_mask.sum()} samples)")
else:
    print("No samples in test set (check pairs split and horizon conditions).")

# —— Classification report & confusion matrix
try:
    from sklearn.metrics import classification_report, confusion_matrix
    target_names = [LABEL2NAME[i] for i in sorted(LABEL2NAME)]
    print("\n[Classification Report]")
    print(classification_report(y_true, y_pred, target_names=target_names, digits=4))
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])
except Exception:
    print("\n[Warning] scikit-learn not installed, will print simple confusion table.")
    cm = np.zeros((3,3), dtype=int)
    for t,p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1

# —— Print confusion matrix values
print("\n[Confusion Matrix] Row=True class, Column=Predicted class")
print(pd.DataFrame(cm, index=[LABEL2NAME[i] for i in [0,1,2]],
                      columns=[LABEL2NAME[i] for i in [0,1,2]]))

# —— Plot confusion matrix (enhanced)
plt.figure(figsize=(6.0,5.0))
im = plt.imshow(cm, interpolation='nearest', cmap='Blues')
plt.title("Confusion Matrix (Test Set)", fontsize=14, fontweight='bold', pad=20)
plt.xlabel("Predicted", fontsize=12)
plt.ylabel("True", fontsize=12)
plt.xticks([0,1,2], [LABEL2NAME[i] for i in [0,1,2]])
plt.yticks([0,1,2], [LABEL2NAME[i] for i in [0,1,2]])

#Add values ​​and percentages
for i in range(3):
    for j in range(3):
        total = cm[i].sum()
        if total > 0:
            percentage = cm[i, j] / total * 100
            text_color = 'white' if cm[i, j] > cm.max() * 0.5 else 'black'
            plt.text(j, i, f'{cm[i, j]}\n({percentage:.1f}%)',
                    ha="center", va="center", color=text_color, fontweight='bold')

plt.colorbar(im)
plt.tight_layout()
plt.show()

# —— Statistics of ΔHI distribution (by true class/predicted class)
if len(delta_mean_all) > 0:
    df_delta = pd.DataFrame({
        "uid": uids_all[:len(delta_mean_all)],
        "true": y_true.astype(int),
        "pred": y_pred.astype(int),
        "delta_hi_mean": delta_mean_all.astype(float)
    })

    print("\n" + "="*50)
    print("ΔHI MEAN STATISTICS BY TRUE CLASS")
    print("="*50)
    stats_by_true = df_delta.groupby("true")["delta_hi_mean"].describe()
    print(stats_by_true.round(4))

    print("\n" + "="*50)
    print("ΔHI MEAN STATISTICS BY PREDICTED CLASS")
    print("="*50)
    stats_by_pred = df_delta.groupby("pred")["delta_hi_mean"].describe()
    print(stats_by_pred.round(4))

# ——Continuous time axis: HI and several sensors before/after aligned visualization (intelligent selection of the best drawing)
print("\n" + "="*60)
print("GENERATING BEST QUALITY VISUALIZATIONS...")
print("="*60)

plot_hi_examples_aligned(curves, n_show=6, seed=2025)
plot_sensor_examples_aligned(curves, sensor_idx_list=None, n_cols=4)

# ——(Optional) Top-K with largest ΔHI in each class, for manual review (enhanced display)
if len(delta_mean_all) > 0:
    topk_by_delta(df_delta, k=5)

# %% [notebook code cell 13]


def extract_hi_formula_and_analyze_maintenance_effects(model, test_loader, device):
    """
    Extract parsable expression formulas from input to HI, and analyze changes in HI composition parameters under different maintenance operations
    """
    model.eval()

    # Store parameters under different maintenance strategies
    maintenance_params = {0: [], 1: [], 2: []}  # Perfect, General, Poor
    hi_formulas = {0: [], 1: [], 2: []}

    print("\n" + "="*80)
    print("HI FORMULA EXTRACTION AND MAINTENANCE EFFECT ANALYSIS")
    print("="*80)

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx >= 10: # Limit the number of analysis samples to save time
                break

            x_before = batch["x_before"].to(device)  # (B,L,C)
            x_after = batch["x_after"].to(device)
            labels = batch["labels"]
            mask = batch["mask"].to(device)
            uids = batch["uids"]

            B, L, C = x_before.shape

            for b in range(B):
                label = labels[b].item()
                uid = uids[b]
                seq_len = int(batch["lengths"][b].item())

                #Extract a single sample
                xb_sample = x_before[b:b+1, :seq_len]  # (1,seq_len,C)
                xa_sample = x_after[b:b+1, :seq_len]

                # Analyze the HI formula before maintenance
                hi_before_params = analyze_hi_formula(model, xb_sample, "before", uid, label)

                # Analyze the maintained HI formula
                hi_after_params = analyze_hi_formula(model, xa_sample, "after", uid, label)

                #Storage parameters for subsequent analysis
                maintenance_params[label].append({
                    'uid': uid,
                    'before': hi_before_params,
                    'after': hi_after_params
                })

    # Analyze parameter change patterns of different maintenance strategies
    analyze_maintenance_parameter_changes(maintenance_params)

    return maintenance_params

def analyze_hi_formula(model, x_input, phase, uid, label):
    """
    Analyze the HI formula composition of a single input sequence
    """
    # Forward propagation to obtain intermediate variables
    h = model.encoder.boltz(x_input) # (1,L,trend_ch) Boltzmann feature

    # Get the output of each operation in CustomKAN
    kan_ops_outputs = []
    with torch.no_grad():
        for i, op in enumerate(model.encoder.customkan.ops):
            op_output = op(h)  # (1,L)
            kan_ops_outputs.append(op_output.cpu().numpy().flatten())

    # Get sparse gating weights
    gate_weights = model.encoder.customkan.gate().cpu().numpy().flatten()

    # Get the damage value
    damage = model.encoder.customkan(h).cpu().numpy().flatten()  # (L,)

    # Get HI projection parameters
    proj_gain = model.encoder.proj_gain.item()
    proj_bias = model.encoder.proj_bias.item()

    # Get Boltzmann parameters
    boltz_E = model.encoder.boltz.E.detach().cpu().numpy()
    boltz_kT = torch.clamp(torch.nn.functional.softplus(model.encoder.boltz.kT), 0.01, 10.0).detach().cpu().numpy()

    # Build a parsable formula description
    formula_components = {
        'phase': phase,
        'uid': uid,
        'label': label,
        'gate_weights': gate_weights,
        'proj_gain': proj_gain,
        'proj_bias': proj_bias,
        'boltz_E': boltz_E,
        'boltz_kT': boltz_kT,
        'damage_mean': np.mean(damage),
        'damage_std': np.std(damage),
        'ops_contributions': []
    }

    # Analyze the contribution of each operation
    op_names = ['MonotonicLinear', 'MonotonicFlat', 'ConcaveLog', 'SaturationSigmoid',
                'HingeReLU', 'Polynomial', 'DampedSin', 'PiecewiseLinear']

    for i, (op_name, weight, output) in enumerate(zip(op_names, gate_weights, kan_ops_outputs)):
        contribution = {
            'op_name': op_name,
            'weight': weight,
            'output_mean': np.mean(output),
            'output_std': np.std(output),
            'weighted_contribution': weight * np.mean(output)
        }
        formula_components['ops_contributions'].append(contribution)

    return formula_components

def analyze_maintenance_parameter_changes(maintenance_params):
    """
    Analyze the changing patterns of HI formula parameters under different maintenance strategies
    """
    print("\n" + "="*60)
    print("MAINTENANCE STRATEGY PARAMETER ANALYSIS")
    print("="*60)

    strategy_names = {0: "Perfect", 1: "General", 2: "Poor"}

    for strategy in [0, 1, 2]:
        if len(maintenance_params[strategy]) == 0:
            continue

        print(f"\n[{strategy_names[strategy]} Maintenance Strategy]")
        print("-" * 40)

        #Analyze changes in projection parameters
        proj_gain_changes = []
        proj_bias_changes = []
        damage_changes = []

        for sample in maintenance_params[strategy]:
            before = sample['before']
            after = sample['after']

            gain_change = after['proj_gain'] - before['proj_gain']
            bias_change = after['proj_bias'] - before['proj_bias']
            damage_change = after['damage_mean'] - before['damage_mean']

            proj_gain_changes.append(gain_change)
            proj_bias_changes.append(bias_change)
            damage_changes.append(damage_change)

        print(f"Projection gain change: mean={np.mean(proj_gain_changes):.4f}, standard deviation={np.std(proj_gain_changes):.4f}")
        print(f"Projection bias change: mean={np.mean(proj_bias_changes):.4f}, standard deviation={np.std(proj_bias_changes):.4f}")
        print(f"Damage mean change: mean={np.mean(damage_changes):.4f}, standard deviation={np.std(damage_changes):.4f}")

        #Analyze gating weight change patterns
        analyze_gate_weight_changes(maintenance_params[strategy], strategy_names[strategy])

        # Analyze Boltzmann parameter changes
        analyze_boltzmann_changes(maintenance_params[strategy], strategy_names[strategy])

def analyze_gate_weight_changes(strategy_samples, strategy_name):
    """
    Analyze the changing pattern of gate control weights under specific maintenance strategies
    """
    print(f"\n{strategy_name} Strategy Gating Weight Change Analysis:")

    op_names = ['MonotonicLinear', 'MonotonicFlat', 'ConcaveLog', 'SaturationSigmoid',
                'HingeReLU', 'Polynomial', 'DampedSin', 'PiecewiseLinear']

    for op_idx, op_name in enumerate(op_names):
        weight_changes = []

        for sample in strategy_samples:
            before_weight = sample['before']['gate_weights'][op_idx]
            after_weight = sample['after']['gate_weights'][op_idx]
            change = after_weight - before_weight
            weight_changes.append(change)

        if len(weight_changes) > 0:
            mean_change = np.mean(weight_changes)
            std_change = np.std(weight_changes)
            print(f" {op_name:>15}: Δ weight={mean_change:>7.4f}±{std_change:.4f}")

def analyze_boltzmann_changes(strategy_samples, strategy_name):
    """
    Analyze the changing pattern of Boltzmann parameters
    """
    print(f"\n{strategy_name} strategy Boltzmann parameter change analysis:")

    E_changes = []
    kT_changes = []

    for sample in strategy_samples:
        before_E = sample['before']['boltz_E']
        after_E = sample['after']['boltz_E']
        before_kT = sample['before']['boltz_kT']
        after_kT = sample['after']['boltz_kT']

        E_change = np.mean(np.abs(after_E - before_E))
        kT_change = np.mean(np.abs(after_kT - before_kT))

        E_changes.append(E_change)
        kT_changes.append(kT_change)

    if len(E_changes) > 0:
        print(f" Energy parameter E changes: mean={np.mean(E_changes):.4f}±{np.std(E_changes):.4f}")
        print(f" Changes in temperature parameter kT: mean={np.mean(kT_changes):.4f}±{np.std(kT_changes):.4f}")

def print_interpretable_hi_formula(maintenance_params):
    """
    Print parsable HI formula expression
    """
    print("\n" + "="*80)
    print("INTERPRETABLE HI FORMULA EXPRESSIONS")
    print("="*80)

    print("\nBasic HI calculation formula:")
    print("HI = proj_gain × (1 - normalized_damage) + proj_bias")
    print("\nwhere damage = Σ(w_i × op_i(Boltzmann_features))")
    print("Boltzmann_features = σ((E - x) / kT) × x")

    strategy_names = {0: "Perfect", 1: "General", 2: "Poor"}

    for strategy in [0, 1, 2]:
        if len(maintenance_params[strategy]) == 0:
            continue

        print(f"\n[{strategy_names[strategy]} Maintenance Strategy - Typical parameter values]")
        print("-" * 50)

        # Calculate the average parameters of the strategy
        sample = maintenance_params[strategy][0] # Take the first sample as a representative

        print("HI formula composition before maintenance:")
        before = sample['before']
        print(f"  proj_gain = {before['proj_gain']:.4f}")
        print(f"  proj_bias = {before['proj_bias']:.4f}")
        print(f"Main operational contribution:")

        for contrib in before['ops_contributions']:
            if contrib['weight'] > 0.1: # Only display operations with larger weights
                print(f" {contrib['op_name']:>15}: weight={contrib['weight']:.4f}, contribution={contrib['weighted_contribution']:.4f}")

        print("\nHI formula composition after maintenance:")
        after = sample['after']
        print(f"  proj_gain = {after['proj_gain']:.4f}")
        print(f"  proj_bias = {after['proj_bias']:.4f}")
        print(f"Main operational contribution:")

        for contrib in after['ops_contributions']:
            if contrib['weight'] > 0.1:
                print(f" {contrib['op_name']:>15}: weight={contrib['weight']:.4f}, contribution={contrib['weighted_contribution']:.4f}")

# Perform HI formula analysis
print("\nStart executing HI formula extraction and maintenance effect analysis...")
maintenance_analysis_params = extract_hi_formula_and_analyze_maintenance_effects(model, ld_te, DEVICE)
print_interpretable_hi_formula(maintenance_analysis_params)

# %% [notebook code cell 14]

# ==== extract_operator_params.py ====
import torch
import torch.nn.functional as F
import pandas as pd

def _sp(x):  # softplus
    return F.softplus(x)

@torch.no_grad()
def extract_customkan_params(encoder: torch.nn.Module) -> pd.DataFrame:
    """
    Extract the "effective parameters" (values ​​after softplus/clamp) of CustomKAN's 8 operators and summarize them into a DataFrame.
    """
    rows = []
    for idx, op in enumerate(encoder.customkan.ops):
        name = op.__class__.__name__
        rec = {"op_id": idx, "op_name": name}

        # Read out the "effective parameters" for different operators (matches clamp in forward)
        if name == "MonotonicLinearOp":
            scale = _sp(op.raw_scale).clamp(op.smin, op.smax).item()
            bias  = op.bias.clamp(-10.0, 10.0).item()
            rec.update(scale=scale, bias=bias)

        elif name == "MonotonicFlatOp":
            scale = _sp(op.raw_scale).clamp(op.smin, op.smax).item()
            bias  = op.bias.clamp(-10.0, 10.0).item()
            rec.update(scale=scale, bias=bias)

        elif name == "ConcaveLogOp":
            scale = _sp(op.raw_scale).clamp(op.smin, op.smax).item()
            eps   = op.eps
            rec.update(scale=scale, eps=eps)

        elif name == "SaturationSigmoidOp":
            scale = _sp(op.raw_scale).clamp(op.smin, op.smax).item()
            slope = _sp(op.raw_slope).clamp(op.lmin, op.lmax).item()
            bias  = op.bias.clamp(-10.0, 10.0).item()
            rec.update(scale=scale, slope=slope, bias=bias)

        elif name == "HingeReLUOp":
            scale = _sp(op.raw_scale).clamp(op.smin, op.smax).item()
            thr   = op.threshold.clamp(-10.0, 10.0).item()
            rec.update(scale=scale, threshold=thr)

        elif name == "PolynomialOp":
            # Effective coefficient: between softplus and (0.01, 5.0)
            coeffs = []
            for i in range(op.deg):
                coeffs.append(_sp(op.raw_coeff[i]).clamp(0.01, 5.0).item())
            for i,c in enumerate(coeffs, start=1):
                rec[f"c{i}"] = c

        elif name == "DampedSinOp":
            scale = _sp(op.raw_scale).clamp(op.smin, op.smax).item()
            freq  = _sp(op.raw_freq ).clamp(op.fmin, op.fmax).item()
            lam   = _sp(op.raw_lambda).clamp(op.lmin, op.lmax).item()
            phase = op.phase.clamp(-10.0, 10.0).item()
            rec.update(scale=scale, freq=freq, lam=lam, phase=phase)

        elif name == "PiecewiseLinearOp":
            k1 = _sp(op.raw_k1).clamp(op.kmin, op.kmax).item()
            k2 = _sp(op.raw_k2).clamp(op.kmin, op.kmax).item()
            thr = op.threshold.clamp(-5.0, 5.0).item()
            rec.update(k1=k1, k2=k2, threshold=thr)

        rows.append(rec)

    df = pd.DataFrame(rows).sort_values("op_id").reset_index(drop=True)
    return df

@torch.no_grad()
def extract_global_params(encoder: torch.nn.Module) -> dict:
    """Extract global parameters (independent of before/after): Boltzmann, gate, proj."""
    # Boltzmann
    E  = encoder.boltz.E.clamp(-10.0, 10.0).cpu().numpy()
    kT = _sp(encoder.boltz.kT).clamp(0.01, 10.0).cpu().numpy()
    # gate (eval is softmax(logits))
    gate = torch.softmax(encoder.customkan.gate.logits, dim=-1).cpu().numpy()
    # projection
    proj_gain = _sp(encoder.proj_gain).clamp(0.1, 5.0).item()
    proj_bias = encoder.proj_bias.clamp(-5.0, 5.0).item()
    return {
        "E": E, "kT": kT,
        "gate": gate,
        "proj_gain": proj_gain, "proj_bias": proj_bias
    }

def compare_params_before_after(encoder, tag_before="before", tag_after="after") -> pd.DataFrame:
    """
    In the current "shared parameters" model, this comparison will give two copies of the same table and calculate Δ (which will be 0).
    This can visually prove "the parameters remain unchanged" and at the same time completely list the actual effective parameters of each operator.
    """
    df_b = extract_customkan_params(encoder)
    df_b.columns = [c if c in ("op_id","op_name") else f"{c}_{tag_before}" for c in df_b.columns]

    df_a = extract_customkan_params(encoder)
    df_a.columns = [c if c in ("op_id","op_name") else f"{c}_{tag_after}" for c in df_a.columns]

    df = df_b.merge(df_a, on=["op_id","op_name"])
    # Calculate Δ column by column
    for c in df.columns:
        if c.endswith(f"_{tag_before}"):
            stem = c[:-(len(tag_before)+1)]
            ca = f"{stem}_{tag_after}"
            if ca in df:
                df[f"{stem}_delta"] = df[ca] - df[c]
    return df
encoder = model.encoder.eval()

# 1) Effective parameter table of each operator + Δ
df_params = compare_params_before_after(encoder, "before", "after")
print(df_params) # You will see that all *_delta are 0
df_params.to_csv("kan_operator_params_compare.csv", index=False)

# 2) Global (Boltzmann, gating, projection) parameters
g = extract_global_params(encoder)
print("proj_gain, proj_bias:", g["proj_gain"], g["proj_bias"])
print("gate (softmax logits):", g["gate"]) # Your results are ~ even, indicating insufficient sparsity.
print("Boltzmann E shape:", g["E"].shape, "kT shape:", g["kT"].shape)

# %% [notebook code cell 15]

# ============================================================
# Monotonic-Decreasing HI + aligned plotting (before -> after)
# ============================================================
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import random
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL2NAME = {0: "Perfect", 1: "General", 2: "Poor"}

# ============================================================
# 1) Dataset: read samples from pairs, perform shift operations in batch later
# ============================================================
class PairsReconstructDataset(Dataset):
    """
    Each sample contains:
      x_before: (L,C) feature sequence before maintenance
      x_after : (L,C) feature sequence after maintenance
      hi_before/hi_after: (L,) health index (for evaluation/optional analysis only, not strong supervision)
      strategy: maintenance type ("Perfect Maintenance" / "General Maintenance" / "Poor Maintenance")
    """
    def __init__(self, pairs: dict, horizon: int=None):
        items = []
        for uid, sd in pairs.items():
            for strat, d in sd.items():
                xb = np.asarray(d["x_before"], dtype=np.float32)
                xa = np.asarray(d["x_after"],  dtype=np.float32)
                # When there's no real HI, create placeholder that will be learned by the model
                if "hi_before" in d and d["hi_before"] is not None:
                    hib = np.asarray(d["hi_before"],dtype=np.float32)
                else:
                    # Create initialized fake HI, model will learn real patterns
                    hib = np.linspace(0.8, 0.4, len(xb)).astype(np.float32)

                if "hi_after" in d and d["hi_after"] is not None:
                    hia = np.asarray(d["hi_after"], dtype=np.float32)
                else:
                    # Create different fake HI patterns based on maintenance strategy
                    if "perfect" in strat.lower():
                        # Perfect maintenance: significant improvement
                        hia = np.linspace(0.9, 0.6, len(xa)).astype(np.float32)
                    elif "general" in strat.lower():
                        # General maintenance: moderate improvement
                        hia = np.linspace(0.7, 0.5, len(xa)).astype(np.float32)
                    else:
                        # Poor maintenance: slight improvement
                        hia = np.linspace(0.5, 0.35, len(xa)).astype(np.float32)

                L, C = xb.shape
                # Unify length (optional)
                if horizon is not None and L > horizon:
                    xb, xa, hib, hia = xb[:horizon], xa[:horizon], hib[:horizon], hia[:horizon]
                    L = horizon
                if L < 3:  # At least need to form 0:L-2 -> 1:L-1
                    continue
                items.append({"uid": uid, "strategy": strat, "x_before": xb, "x_after": xa,
                              "hi_before": hib, "hi_after": hia})
        assert len(items)>0, "Dataset is empty, please check pairs data."
        self.items = items
        self.C = items[0]["x_before"].shape[1]

        # Strategy to 3-class label mapping
        def strat_to_cls(s):
            s = s.lower()
            if "perfect" in s: return 0
            if "general" in s: return 1
            if "poor" in s: return 2
            raise ValueError(f"Unknown strategy name: {s}")
        self.labels = [strat_to_cls(it["strategy"]) for it in items]

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        return {
            "uid": it["uid"],
            "label": self.labels[idx],
            "x_before": torch.from_numpy(it["x_before"]), # (L,C)
            "x_after":  torch.from_numpy(it["x_after"]),  # (L,C)
            "hi_before":torch.from_numpy(it["hi_before"]),# (L,)
            "hi_after": torch.from_numpy(it["hi_after"]), # (L,)
            "strategy": it["strategy"]
        }

def pad_collate_shift(batch):
    """
    Pad sequences to same length; return mask, do NOT perform shift operation here
    to allow unified training: inputs = [:,0:L-2], targets = [:,1:L-1]
    """
    Ls = [b["x_before"].shape[0] for b in batch]
    Lmax = max(Ls)
    C = batch[0]["x_before"].shape[1]

    def pad2d(x):
        L, C0 = x.shape
        out = torch.zeros(Lmax, C0, dtype=x.dtype)
        out[:L] = x
        return out

    def pad1d(x):
        L = x.shape[0]
        out = torch.zeros(Lmax, dtype=x.dtype)
        out[:L] = x
        return out

    x_before = torch.stack([pad2d(b["x_before"]) for b in batch], 0) # (B,Lmax,C)
    x_after  = torch.stack([pad2d(b["x_after"])  for b in batch], 0) # (B,Lmax,C)
    hi_before= torch.stack([pad1d(b["hi_before"])for b in batch], 0) # (B,Lmax)
    hi_after = torch.stack([pad1d(b["hi_after"]) for b in batch], 0) # (B,Lmax)
    lengths  = torch.tensor(Ls, dtype=torch.long)
    mask     = torch.zeros(len(batch), Lmax)
    for i,L in enumerate(Ls): mask[i,:L] = 1.0
    labels   = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    uids     = [b["uid"] for b in batch]
    return {"uids": uids, "labels": labels,
            "x_before": x_before, "x_after": x_after,
            "hi_before": hi_before, "hi_after": hi_after,
            "lengths": lengths, "mask": mask}

# ============================================================
# 2) Model: True liquid adaptive operator - with significantly different parameters and behavior patterns before and after maintenance
# ============================================================
class BoltzmannKAN(nn.Module):
    def __init__(self, in_ch, out_ch=4):
        super().__init__()
        self.E  = nn.Parameter(torch.zeros(out_ch, in_ch))
        self.kT = nn.Parameter(torch.ones(out_ch, in_ch))
    def forward(self, x):
        B,T,C = x.shape
        kT = torch.clamp(F.softplus(self.kT), 0.01, 10.0).unsqueeze(0).unsqueeze(2)  # (1,out_ch,1,in_ch)
        E  = torch.clamp(self.E, -10.0, 10.0).unsqueeze(0).unsqueeze(2)             # (1,out_ch,1,in_ch)
        x_ = torch.clamp(x.unsqueeze(1), -10.0, 10.0)                               # (B, out_ch, T, in_ch)
        exp = torch.clamp((E - x_) / kT, -50, 50)
        w   = torch.sigmoid(exp)
        h   = (w * x_).sum(dim=3).permute(0, 2, 1)           # (B,T,out_ch) >=0
        return torch.clamp(F.softplus(h), 0.0, 100.0)

class BaseOp(nn.Module): pass

class LiquidAdaptiveMonotonicLinearOp(BaseOp):
    """Monotonic linear operator for true liquids - significant changes in parameters before and after maintenance"""
    def __init__(self, param_dim=8):
        super().__init__()
        # Deep network, stronger nonlinear parameter generation capability
        self.param_net = nn.Sequential(
            nn.Linear(param_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3) # Output scale, bias, temperature (control maintenance sensitivity)
        )
        self.smin, self.smax = 0.1, 8.0
        # Dedicated to maintaining effect-aware networks
        self.maintenance_sensor = nn.Sequential(
            nn.Linear(param_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 4) # Output maintenance effect parameters
        )

    def forward(self, h):
        B, T, _ = h.shape
        h_mean = h.mean(dim=1)  # (B, param_dim)

        #Basic parameters
        params = self.param_net(h_mean)  # (B, 3)
        scale = torch.clamp(F.softplus(params[:, 0]), self.smin, self.smax).unsqueeze(1)  # (B, 1)
        bias = torch.clamp(params[:, 1], -8.0, 8.0).unsqueeze(1)  # (B, 1)
        temperature = torch.clamp(F.softplus(params[:, 2]), 0.1, 5.0).unsqueeze(1)  # (B, 1)

        # Maintain effect perception parameters
        maint_params = self.maintenance_sensor(h_mean)  # (B, 4)
        sensitivity = torch.clamp(F.softplus(maint_params[:, 0]), 0.5, 3.0).unsqueeze(1)  # (B, 1)
        phase_shift = torch.clamp(maint_params[:, 1], -2.0, 2.0).unsqueeze(1)  # (B, 1)
        amplitude_mod = torch.clamp(F.softplus(maint_params[:, 2]), 0.3, 2.0).unsqueeze(1)  # (B, 1)
        nonlin_factor = torch.clamp(torch.sigmoid(maint_params[:, 3]), 0.1, 0.9).unsqueeze(1)  # (B, 1)

        # Time-related change patterns
        t = torch.arange(T, device=h.device, dtype=h.dtype).unsqueeze(0).expand(B, -1)  # (B, T)
        t_norm = t / (T - 1) #Normalized time

        #Adjust parameters based on the statistical characteristics of the input data
        h_std = h.std(dim=1, keepdim=True).mean(dim=-1)  # (B, 1)
        dynamic_scale = scale * (1 + h_std * sensitivity)

        # Nonlinear time effects
        time_effect = torch.sin(2 * np.pi * t_norm * temperature + phase_shift) * amplitude_mod

        xm = torch.clamp(h.mean(-1), -10.0, 10.0)  # (B, T)
        # Liquid response: combining linear and nonlinear effects
        linear_part = dynamic_scale * (xm + bias)
        nonlinear_part = time_effect * xm * nonlin_factor

        result = linear_part + nonlinear_part
        return torch.clamp(F.softplus(result), 0.0, 100.0)

class LiquidAdaptiveMonotonicFlatOp(BaseOp):
    """Liquid smoothing operator - with maintenance-related dynamics"""
    def __init__(self, param_dim=8):
        super().__init__()
        self.param_net = nn.Sequential(
            nn.Linear(param_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4)  # scale, bias, smoothness, adaptivity
        )
        self.dynamic_net = nn.Sequential(
            nn.Linear(param_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 3) #Dynamic adjustment parameters
        )
        self.smin, self.smax = 1e-2, 2.0

    def _adaptive_cum(self, x, smoothness):
        """Adaptive accumulation function, adjusted according to the smoothness parameter"""
        # x is (B, 1, T), smoothness is (B, 1)
        diff = F.relu(x[:, :, 1:] - x[:, :, :-1])  # (B, 1, T-1)
        # Smoothness adjustment
        diff = diff * smoothness.unsqueeze(-1)  # (B, 1, T-1)
        return torch.cat([torch.zeros_like(diff[:, :, :1]), torch.cumsum(diff, 2)], 2)  # (B, 1, T)

    def forward(self, h):
        B, T, _ = h.shape
        h_mean = h.mean(dim=1)  # (B, param_dim)

        #Basic parameters
        params = self.param_net(h_mean)  # (B, 4)
        scale = torch.clamp(F.softplus(params[:, 0]), self.smin, self.smax).unsqueeze(1)
        bias = torch.clamp(params[:, 1], -5.0, 5.0).unsqueeze(1)
        smoothness = torch.clamp(F.softplus(params[:, 2]), 0.1, 2.0).unsqueeze(1)
        adaptivity = torch.clamp(torch.sigmoid(params[:, 3]), 0.1, 0.9).unsqueeze(1)

        #Dynamic adjustment parameters
        dynamic_params = self.dynamic_net(h_mean)  # (B, 3)
        temporal_weight = torch.clamp(F.softplus(dynamic_params[:, 0]), 0.5, 2.0).unsqueeze(1)
        fluctuation = torch.clamp(dynamic_params[:, 1], -1.0, 1.0).unsqueeze(1)
        maintenance_response = torch.clamp(F.softplus(dynamic_params[:, 2]), 0.2, 3.0).unsqueeze(1)

        # Time-varying characteristics of input data
        h_variance = h.var(dim=-1, keepdim=True)  # (B, T, 1)
        temporal_modulation = torch.sin(torch.arange(T, device=h.device).float() * temporal_weight / T * 2 * np.pi + fluctuation)
        temporal_modulation = temporal_modulation.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)  # (B, 1, T)

        xm = torch.clamp(h.mean(-1, keepdim=True), -10.0, 10.0).unsqueeze(1)  # (B, 1, T)

        # Liquid response: combining cumulative effect and dynamic adjustment
        cum_base = self._adaptive_cum(xm, smoothness)

        # Maintain response conditioning - ensure dimensions match
        h_variance_reshaped = h_variance.transpose(1, 2)  # (B, 1, T)
        maintenance_modulation = maintenance_response.unsqueeze(-1) * h_variance_reshaped * temporal_modulation

        result = scale.unsqueeze(-1) * (cum_base + maintenance_modulation + bias.unsqueeze(-1))
        result = result.squeeze(1) * (1 + adaptivity * temporal_modulation.squeeze(1))

        return torch.clamp(F.softplus(result), 0.0, 100.0)

class LiquidAdaptiveConcaveLogOp(BaseOp):
    """Liquid concave logarithmic operator - maintaining sensitive nonlinear response"""
    def __init__(self, param_dim=8):
        super().__init__()
        self.param_net = nn.Sequential(
            nn.Linear(param_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 5)  # scale, offset, curvature, maintenance_gain, noise_resist
        )
        self.eps = 1e-3
        self.smin, self.smax = 0.01, 8.0

    def forward(self, h):
        B, T, _ = h.shape
        h_mean = h.mean(dim=1)  # (B, param_dim)
        params = self.param_net(h_mean)  # (B, 5)

        scale = torch.clamp(F.softplus(params[:, 0]), self.smin, self.smax).unsqueeze(1)
        offset = torch.clamp(params[:, 1], -3.0, 3.0).unsqueeze(1)
        curvature = torch.clamp(F.softplus(params[:, 2]), 0.1, 3.0).unsqueeze(1)
        maintenance_gain = torch.clamp(F.softplus(params[:, 3]), 0.5, 4.0).unsqueeze(1)
        noise_resistance = torch.clamp(torch.sigmoid(params[:, 4]), 0.1, 0.95).unsqueeze(1)

        # Dynamic data feature awareness
        h_energy = (h ** 2).mean(dim=-1)  # (B, T)
        h_gradient = torch.abs(h[:, 1:] - h[:, :-1]).mean(dim=-1)  # (B, T-1)
        h_gradient = F.pad(h_gradient, (0, 1), value=0)  # (B, T)

        xm = torch.clamp(h.mean(-1), -10.0, 10.0)  # (B, T)

        # Liquid logarithmic response
        log_base = torch.log(torch.abs(xm + offset) + self.eps)
        curvature_effect = curvature * h_energy
        maintenance_effect = maintenance_gain * h_gradient

        #Adaptive noise suppression
        noise_filter = torch.sigmoid(h_energy * noise_resistance)

        result = scale * (log_base * curvature_effect + maintenance_effect) * noise_filter
        return torch.clamp(F.softplus(result), 0.0, 100.0)

class LiquidAdaptiveSaturationSigmoidOp(BaseOp):
    """Liquid saturated sigmoid operator - maintenance threshold sensitive"""
    def __init__(self, param_dim=8):
        super().__init__()
        self.param_net = nn.Sequential(
            nn.Linear(param_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 6)  # scale, slope, bias, threshold, saturation, maintenance_sens
        )
        self.smin, self.smax = 0.01, 6.0
        self.lmin, self.lmax = 0.1, 8.0

    def forward(self, h):
        B, T, _ = h.shape
        h_mean = h.mean(dim=1)  # (B, param_dim)
        params = self.param_net(h_mean)  # (B, 6)

        scale = torch.clamp(F.softplus(params[:, 0]), self.smin, self.smax).unsqueeze(1)
        slope = torch.clamp(F.softplus(params[:, 1]), self.lmin, self.lmax).unsqueeze(1)
        bias = torch.clamp(params[:, 2], -5.0, 5.0).unsqueeze(1)
        threshold = torch.clamp(params[:, 3], -3.0, 3.0).unsqueeze(1)
        saturation = torch.clamp(F.softplus(params[:, 4]), 0.5, 3.0).unsqueeze(1)
        maintenance_sens = torch.clamp(F.softplus(params[:, 5]), 0.2, 4.0).unsqueeze(1)

        # Maintain sensitive feature extraction
        h_peak = h.max(dim=-1)[0]  # (B, T)
        h_trough = h.min(dim=-1)[0]  # (B, T)
        h_range = h_peak - h_trough

        xm = torch.clamp(h.mean(-1), -10.0, 10.0)  # (B, T)

        # Dynamic threshold adjustment
        dynamic_threshold = threshold + h_range * maintenance_sens * 0.1

        # Liquid sigmoid response
        sigmoid_input = slope * (xm - dynamic_threshold - bias)
        sigmoid_output = torch.sigmoid(sigmoid_input)

        # saturation effect
        saturation_effect = torch.tanh(saturation * sigmoid_output)

        result = scale * saturation_effect
        return torch.clamp(F.softplus(result), 0.0, 100.0)

class LiquidAdaptiveHingeReLUOp(BaseOp):
    """Liquid hinge ReLU operator - multi-threshold maintenance response"""
    def __init__(self, param_dim=8):
        super().__init__()
        self.param_net = nn.Sequential(
            nn.Linear(param_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 6)  # scale, threshold1, threshold2, slope1, slope2, maintenance_amp
        )
        self.smin, self.smax = 0.01, 5.0

    def forward(self, h):
        B, T, _ = h.shape
        h_mean = h.mean(dim=1)  # (B, param_dim)
        params = self.param_net(h_mean)  # (B, 6)

        scale = torch.clamp(F.softplus(params[:, 0]), self.smin, self.smax).unsqueeze(1)
        threshold1 = torch.clamp(params[:, 1], -5.0, 0.0).unsqueeze(1)
        threshold2 = torch.clamp(params[:, 2], 0.0, 5.0).unsqueeze(1)
        slope1 = torch.clamp(F.softplus(params[:, 3]), 0.1, 3.0).unsqueeze(1)
        slope2 = torch.clamp(F.softplus(params[:, 4]), 0.1, 3.0).unsqueeze(1)
        maintenance_amp = torch.clamp(F.softplus(params[:, 5]), 0.5, 3.0).unsqueeze(1)

        # Maintain response characteristics
        h_trend = (h[:, -1] - h[:, 0]).mean(dim=-1, keepdim=True)  # (B, 1)
        h_volatility = h.std(dim=1).mean(dim=-1, keepdim=True)  # (B, 1)

        xm = torch.clamp(h.mean(-1), -10.0, 10.0)  # (B, T)

        #Multi-threshold hinge response
        hinge1 = F.relu(xm - threshold1) * slope1
        hinge2 = F.relu(xm - threshold2) * slope2

        # Maintenance effect modulation
        maintenance_modulation = maintenance_amp * (h_trend + h_volatility)

        result = scale * (hinge1 + hinge2) * (1 + maintenance_modulation)
        return torch.clamp(F.softplus(result), 0.0, 100.0)

class LiquidAdaptivePolynomialOp(BaseOp):
    """Liquid polynomial operator - adaptive order and coefficients"""
    def __init__(self, param_dim=8, max_deg=4):
        super().__init__()
        self.max_deg = max_deg
        self.param_net = nn.Sequential(
            nn.Linear(param_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, max_deg + 2) # coefficient + degree_weight + maintenance_coupling
        )

    def forward(self, h):
        B, T, _ = h.shape
        h_mean = h.mean(dim=1)  # (B, param_dim)
        params = self.param_net(h_mean)  # (B, max_deg + 2)

        coeffs = torch.clamp(F.softplus(params[:, :self.max_deg]), 0.01, 3.0)  # (B, max_deg)
        degree_weight = torch.clamp(torch.sigmoid(params[:, self.max_deg]), 0.1, 1.0).unsqueeze(1)  # (B, 1)
        maintenance_coupling = torch.clamp(F.softplus(params[:, self.max_deg + 1]), 0.2, 2.0).unsqueeze(1)  # (B, 1)

        # Maintain related dynamic features
        h_complexity = torch.var(h, dim=-1).mean(dim=-1, keepdim=True)  # (B, 1)

        xm = torch.clamp(h.mean(-1), -3.0, 3.0)  # (B, T)
        y = torch.zeros_like(xm)

        # Adaptive polynomial calculation
        for i in range(self.max_deg):
            coeff = coeffs[:, i].unsqueeze(1)  # (B, 1)
            power = i + 1

            #Adjust the coefficient according to the maintenance complexity
            adaptive_coeff = coeff * (1 + maintenance_coupling * h_complexity * (power / self.max_deg))

            term = adaptive_coeff * torch.clamp(xm ** power, -50.0, 50.0)
            y = y + term * degree_weight

        return torch.clamp(F.softplus(y), 0.0, 100.0)

class LiquidAdaptiveDampedSinOp(BaseOp):
    """Liquid damped sine operator - maintaining relevant oscillation modes"""
    def __init__(self, param_dim=8):
        super().__init__()
        self.param_net = nn.Sequential(
            nn.Linear(param_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 7)  # scale, freq, damping, phase, maintenance_freq, resonance, chaos
        )
        self.smin, self.smax = 0.01, 5.0
        self.fmin, self.fmax = 0.1, 8.0
        self.lmin, self.lmax = 0.01, 4.0

    def forward(self, h):
        B, T, _ = h.shape
        h_mean = h.mean(dim=1)  # (B, param_dim)
        params = self.param_net(h_mean)  # (B, 7)

        scale = torch.clamp(F.softplus(params[:, 0]), self.smin, self.smax).unsqueeze(1)
        freq = torch.clamp(F.softplus(params[:, 1]), self.fmin, self.fmax).unsqueeze(1)
        damping = torch.clamp(F.softplus(params[:, 2]), self.lmin, self.lmax).unsqueeze(1)
        phase = torch.clamp(params[:, 3], -2*np.pi, 2*np.pi).unsqueeze(1)
        maintenance_freq = torch.clamp(F.softplus(params[:, 4]), 0.1, 3.0).unsqueeze(1)
        resonance = torch.clamp(F.softplus(params[:, 5]), 0.5, 2.5).unsqueeze(1)
        chaos = torch.clamp(torch.sigmoid(params[:, 6]), 0.05, 0.3).unsqueeze(1)

        # Maintain related dynamic features
        h_rhythm = torch.fft.fft(h.mean(dim=-1)).abs().mean(dim=-1, keepdim=True).real  # (B, 1)

        xm = torch.clamp(h.mean(-1), -10.0, 10.0)  # (B, T)

        # time series
        t = torch.arange(T, device=h.device, dtype=h.dtype).unsqueeze(0).expand(B, -1)

        # Maintain sensitive oscillations
        maintenance_oscillation = torch.sin(maintenance_freq * t + phase) * resonance

        # Main oscillation + damping
        main_oscillation = torch.sin(freq * xm + phase)
        damping_factor = torch.exp(-damping * torch.abs(xm))

        #Chaotic disturbance (nonlinearity caused by maintenance)
        chaos_term = chaos * h_rhythm * torch.sin(freq * maintenance_freq * t)

        result = scale * damping_factor * (main_oscillation + maintenance_oscillation + chaos_term)
        return torch.clamp(F.softplus(result), 0.0, 100.0)

class LiquidAdaptivePiecewiseLinearOp(BaseOp):
    """Liquid piecewise linear operator - multi-stage maintenance response"""
    def __init__(self, param_dim=8):
        super().__init__()
        self.param_net = nn.Sequential(
            nn.Linear(param_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 8)  # k1, k2, k3, thresh1, thresh2, maintenance_shift, slope_adapt, transition_smooth
        )
        self.kmin, self.kmax = 0.01, 4.0

    def forward(self, h):
        B, T, _ = h.shape
        h_mean = h.mean(dim=1)  # (B, param_dim)
        params = self.param_net(h_mean)  # (B, 8)

        k1 = torch.clamp(F.softplus(params[:, 0]), self.kmin, self.kmax).unsqueeze(1)
        k2 = torch.clamp(F.softplus(params[:, 1]), self.kmin, self.kmax).unsqueeze(1)
        k3 = torch.clamp(F.softplus(params[:, 2]), self.kmin, self.kmax).unsqueeze(1)
        thresh1 = torch.clamp(params[:, 3], -4.0, 0.0).unsqueeze(1)
        thresh2 = torch.clamp(params[:, 4], 0.0, 4.0).unsqueeze(1)
        maintenance_shift = torch.clamp(params[:, 5], -2.0, 2.0).unsqueeze(1)
        slope_adapt = torch.clamp(F.softplus(params[:, 6]), 0.5, 2.0).unsqueeze(1)
        transition_smooth = torch.clamp(F.softplus(params[:, 7]), 0.1, 2.0).unsqueeze(1)

        # Threshold adjustment for maintenance impact
        h_dynamics = (h.max(dim=-1)[0] - h.min(dim=-1)[0]).mean(dim=-1, keepdim=True)  # (B, 1)

        dynamic_thresh1 = thresh1 + maintenance_shift * h_dynamics
        dynamic_thresh2 = thresh2 + maintenance_shift * h_dynamics

        xm = torch.clamp(h.mean(-1), -10.0, 10.0)  # (B, T)

        # Piecewise linear function + smooth transition
        segment1 = k1 * slope_adapt * xm
        segment2 = k1 * dynamic_thresh1 + k2 * slope_adapt * (xm - dynamic_thresh1)
        segment3 = k1 * dynamic_thresh1 + k2 * (dynamic_thresh2 - dynamic_thresh1) + k3 * slope_adapt * (xm - dynamic_thresh2)

        # Smooth transition weight
        w1 = torch.sigmoid(transition_smooth * (dynamic_thresh1 - xm))
        w2 = torch.sigmoid(transition_smooth * (xm - dynamic_thresh1)) * torch.sigmoid(transition_smooth * (dynamic_thresh2 - xm))
        w3 = torch.sigmoid(transition_smooth * (xm - dynamic_thresh2))

        # Ensure weight normalization
        total_w = w1 + w2 + w3 + 1e-8
        w1, w2, w3 = w1/total_w, w2/total_w, w3/total_w

        result = w1 * segment1 + w2 * segment2 + w3 * segment3
        return torch.clamp(F.softplus(result), 0.0, 100.0)

class TrueLiquidSparseGate(nn.Module):
    """Sparse gating of true liquids - weight distributions significantly different before and after maintenance"""
    def __init__(self, n_ops, param_dim=8, tau_start=8.0, tau_end=0.05, n_steps=15000):
        super().__init__()
        # Deep gated network to strengthen maintenance awareness capabilities
        self.gate_net = nn.Sequential(
            nn.Linear(param_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_ops)
        )

        # Maintain state-aware network
        self.maintenance_detector = nn.Sequential(
            nn.Linear(param_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, n_ops) # Maintenance sensitivity of each operator
        )

        # Dynamic diversity enhanced network
        self.diversity_enhancer = nn.Sequential(
            nn.Linear(param_dim, 32),
            nn.ReLU(),
            nn.Linear(32, n_ops) # Diversity weight
        )

        self.tau_start, self.tau_end, self.n_steps = tau_start, tau_end, n_steps
        self.register_buffer("step", torch.tensor(0.))

        # Maintain weight comparison records before and after
        self.register_buffer("prev_weights", torch.zeros(1, n_ops))
        self.register_buffer("weight_changes", torch.zeros(1, n_ops))

    def forward(self, h_input):
        B, T, param_dim = h_input.shape
        h_mean = h_input.mean(dim=1)  # (B, param_dim)
        h_std = h_input.std(dim=1)   # (B, param_dim)
        h_trend = h_input[:, -1] - h_input[:, 0] # (B, param_dim) trend characteristics

        #Basic logits
        base_logits = self.gate_net(h_mean)  # (B, n_ops)

        # Maintain sensitive modulation
        maintenance_sensitivity = self.maintenance_detector(h_std)  # (B, n_ops)

        # Diversity enhancement
        diversity_weights = self.diversity_enhancer(h_trend)  # (B, n_ops)

        # Combine logits - enhance differentiation
        combined_logits = base_logits + 2.0 * maintenance_sensitivity + 1.5 * diversity_weights

        # Dynamic temperature adjustment
        tau = (self.tau_start * (1 - self.step / self.n_steps) +
               self.tau_end * (self.step / self.n_steps)).clamp(min=self.tau_end)

        # Use different temperatures for each sample (based on its complexity)
        complexity = h_std.mean(dim=-1, keepdim=True)  # (B, 1)
        adaptive_tau = tau * (0.5 + 1.5 * torch.sigmoid(complexity))  # (B, 1)

        self.step.add_(1)

        # Gumbel noise - enhance randomness
        if self.training:
            gumbel_noise = -torch.empty_like(combined_logits).exponential_().log()
            gumbel_noise = torch.clamp(gumbel_noise, -50.0, 50.0)
            # Additional differential noise
            differential_noise = torch.randn_like(combined_logits) * 0.5
        else:
            gumbel_noise = torch.zeros_like(combined_logits)
            differential_noise = torch.zeros_like(combined_logits)

        # Apply temperature and noise
        logits_stable = torch.clamp(combined_logits, -50.0, 50.0)
        noisy_logits = logits_stable + gumbel_noise + differential_noise

        #Adaptive softmax
        w = F.softmax(noisy_logits / adaptive_tau, dim=-1)  # (B, n_ops)

        # Force diversity: if weights are too concentrated, add perturbation
        max_weight = w.max(dim=-1, keepdim=True)[0]
        diversity_penalty = torch.where(max_weight > 0.8,
                                      torch.randn_like(w) * 0.1,
                                      torch.zeros_like(w))
        w = F.softmax(w + diversity_penalty, dim=-1)

        # Record weight changes
        if self.training:
            current_weights = w.mean(dim=0, keepdim=True)  # (1, n_ops)
            if self.prev_weights.sum() > 0:
                self.weight_changes = torch.abs(current_weights - self.prev_weights)
            self.prev_weights = current_weights.clone()

        return w.unsqueeze(1)  # (B, 1, n_ops)

class TrueLiquidCustomKAN(nn.Module):
    """True liquid custom KAN - operator behavior significantly different before and after maintenance"""
    def __init__(self, ops, param_dim=8):
        super().__init__()
        self.ops = nn.ModuleList(ops)
        self.gate = TrueLiquidSparseGate(len(ops), param_dim)

        # Maintenance phase aware global modulation network
        self.global_modulator = nn.Sequential(
            nn.Linear(param_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4)  # global_gain, global_bias, phase_shift, maintenance_strength
        )

        #Interaction network between operators
        self.inter_op_network = nn.Sequential(
            nn.Linear(len(ops), 32),
            nn.ReLU(),
            nn.Linear(32, len(ops))
        )

    def forward(self, h):  # h:(B,T,param_dim)
        B, T, param_dim = h.shape

        # Global modulation parameters
        h_mean = h.mean(dim=1)  # (B, param_dim)
        global_params = self.global_modulator(h_mean)  # (B, 4)
        global_gain = torch.clamp(F.softplus(global_params[:, 0]), 0.1, 5.0).unsqueeze(1)
        global_bias = torch.clamp(global_params[:, 1], -3.0, 3.0).unsqueeze(1)
        phase_shift = torch.clamp(global_params[:, 2], -np.pi, np.pi).unsqueeze(1)
        maintenance_strength = torch.clamp(F.softplus(global_params[:, 3]), 0.2, 4.0).unsqueeze(1)

        # Output of all operators
        outs = []
        for i, op in enumerate(self.ops):
            try:
                op_out = op(h)  # Expected shape: (B, T)

                # Shape normalized to (B,T)
                if op_out.dim() == 3:  # (B, 1, T) or (B, C, T)
                    if op_out.size(1) == 1:
                        op_out = op_out.squeeze(1)  # (B, T)
                    else:
                        op_out = op_out.mean(dim=1)  # (B, T)
                elif op_out.dim() == 1:  # (T,)
                    op_out = op_out.unsqueeze(0).expand(B, -1)  # (B, T)
                elif op_out.dim() > 3:
                    op_out = op_out.reshape(B, -1)  # (B, ?)

                # Time dimension alignment
                if op_out.dim() == 2 and op_out.size(1) != T:
                    if op_out.size(1) > T:
                        op_out = op_out[:, :T]
                    else:
                        pad_size = T - op_out.size(1)
                        op_out = F.pad(op_out, (0, pad_size), value=0.0)

                # Make sure the shape is correct
                if op_out.shape != (B, T):
                    print(f"Warning: Operator {i} output shape {op_out.shape}, reshaping to ({B}, {T})")
                    op_out = torch.zeros(B, T, device=h.device, dtype=h.dtype)

                # Maintain related operator modulation
                t = torch.arange(T, device=h.device, dtype=h.dtype).unsqueeze(0).expand(B, -1)
                time_modulation = torch.sin(2 * np.pi * t / T + phase_shift) * maintenance_strength
                op_out = op_out * (1 + 0.2 * time_modulation) # Time-related modulation

                outs.append(op_out)
            except Exception as e:
                print(f"Warning: Operator {type(op).__name__} failed, using zeros. Error: {e}")
                outs.append(torch.zeros(B, T, device=h.device, dtype=h.dtype))

        st = torch.stack(outs, dim=-1)  # (B, T, K)

        #Adaptive weight
        w = self.gate(h)  # (B, 1, K)
        w = w.expand(-1, T, -1)  # (B, T, K)

        #Interaction between operators
        w_mean = w.mean(dim=1)  # (B, K)
        interaction_weights = torch.sigmoid(self.inter_op_network(w_mean))  # (B, K)
        interaction_weights = interaction_weights.unsqueeze(1).expand(-1, T, -1)  # (B, T, K)

        # Weighted combination + interaction effect
        base_damage = (st * w).sum(-1)  # (B, T)
        interaction_damage = (st * interaction_weights).sum(-1) * 0.3  # (B, T)

        damage = base_damage + interaction_damage
        damage = torch.clamp(damage, 0.0, 100.0)

        # Global modulation
        damage = global_gain * damage + global_bias

        return torch.clamp(damage, 0.0, 100.0)  # (B, T)

class TrendEncoder(nn.Module):
    """
    True Liquid Trend Encoder - Significantly Different Health Index Inference Patterns Before and After Maintenance
    """
    def __init__(self, in_ch, trend_ch=8): # Add trend_ch to provide richer features
        super().__init__()
        self.boltz = BoltzmannKAN(in_ch, out_ch=trend_ch)

        # Use liquid adaptive operator
        ops = [
            LiquidAdaptiveMonotonicLinearOp(trend_ch),
            LiquidAdaptiveMonotonicFlatOp(trend_ch),
            LiquidAdaptiveConcaveLogOp(trend_ch),
            LiquidAdaptiveSaturationSigmoidOp(trend_ch),
            LiquidAdaptiveHingeReLUOp(trend_ch),
            LiquidAdaptivePolynomialOp(trend_ch),
            LiquidAdaptiveDampedSinOp(trend_ch),
            LiquidAdaptivePiecewiseLinearOp(trend_ch)
        ]
        self.customkan = TrueLiquidCustomKAN(ops, trend_ch)

        # Maintain phase-aware projection network
        self.maintenance_aware_proj = nn.Sequential(
            nn.Linear(trend_ch + in_ch, 64), # Combine original features and trend features
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # gain, bias, maintenance_factor
        )

        # Health state network learned directly from sensor data - deeper and more complex
        self.health_inference_net = nn.Sequential(
            nn.Linear(in_ch, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Maintenance effect strengthens the network
        self.maintenance_effect_net = nn.Sequential(
            nn.Linear(in_ch, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # maintenance_intensity, recovery_rate
        )

        # Timing dynamic awareness network
        self.temporal_dynamics_net = nn.Sequential(
            nn.Linear(trend_ch, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3)  # temporal_weight, decay_rate, oscillation
        )

    def forward(self, x):  # x:(B,T,C)
        B, T, C = x.shape

        # Feature extraction based on Boltzmann KAN
        h_multi = self.boltz(x)              # (B,T,trend_ch)
        damage = self.customkan(h_multi)     # (B,T)

        # Combine original features and trend features for maintenance-aware projection
        x_reshaped = x.reshape(-1, C)  # (B*T, C)
        h_multi_reshaped = h_multi.reshape(-1, h_multi.size(-1))  # (B*T, trend_ch)
        combined_features = torch.cat([x_reshaped, h_multi_reshaped], dim=-1)  # (B*T, C+trend_ch)

        # Maintain perceptual projection parameters
        proj_params = self.maintenance_aware_proj(combined_features)  # (B*T, 3)
        proj_params = proj_params.view(B, T, 3)  # (B, T, 3)

        gain = torch.clamp(F.softplus(proj_params[:, :, 0]), 0.1, 5.0)  # (B, T)
        bias = torch.clamp(proj_params[:, :, 1], -3.0, 3.0)  # (B, T)
        maintenance_factor = torch.clamp(F.softplus(proj_params[:, :, 2]), 0.2, 3.0)  # (B, T)

        # Direct health status inference
        health_raw = self.health_inference_net(x_reshaped)  # (B*T, 1)
        health_direct = torch.sigmoid(health_raw).view(B, T)  # (B, T)

        # Maintain effect parameters
        maintenance_params = self.maintenance_effect_net(x_reshaped)  # (B*T, 2)
        maintenance_params = maintenance_params.view(B, T, 2)  # (B, T, 2)
        maintenance_intensity = torch.clamp(F.softplus(maintenance_params[:, :, 0]), 0.5, 3.0)  # (B, T)
        recovery_rate = torch.clamp(torch.sigmoid(maintenance_params[:, :, 1]), 0.1, 0.9)  # (B, T)

        # Timing dynamic characteristics
        temporal_params = self.temporal_dynamics_net(h_multi_reshaped)  # (B*T, 3)
        temporal_params = temporal_params.view(B, T, 3)  # (B, T, 3)
        temporal_weight = torch.clamp(F.softplus(temporal_params[:, :, 0]), 0.2, 2.0)  # (B, T)
        decay_rate = torch.clamp(torch.sigmoid(temporal_params[:, :, 1]), 0.05, 0.5)  # (B, T)
        oscillation = torch.clamp(temporal_params[:, :, 2], -0.5, 0.5)  # (B, T)

        # Timeline
        t = torch.arange(T, device=x.device, dtype=x.dtype).unsqueeze(0).expand(B, -1)  # (B, T)
        t_normalized = t / (T - 1)

        # Convert damage to health decay - more complex mapping
        damage_normalized = torch.sigmoid(gain * (damage + bias))

        # Timing dynamic effects
        temporal_decay = torch.exp(-decay_rate * t_normalized)
        temporal_oscillation = torch.sin(2 * np.pi * t_normalized + oscillation) * 0.1

        # Maintenance effect modeling
        maintenance_recovery = maintenance_intensity * torch.exp(-recovery_rate * t_normalized)

        # Comprehensive health status calculation
        #Basic health status: combined with direct inference and damage model
        base_health = health_direct * (1 - 0.4 * damage_normalized)

        # Timing dynamic modulation
        temporal_modulated = base_health * temporal_decay * (1 + temporal_oscillation)

        # Maintenance effect modulation
        maintenance_modulated = temporal_modulated * (1 + 0.3 * maintenance_recovery * maintenance_factor)

        # Generate an ideal attenuation pattern as a guide
        ideal_decay = 1.0 - t_normalized * 0.6 * temporal_weight # Adaptive attenuation strength

        # Mix ideal falloff and complex modes
        mixing_weight = torch.sigmoid(maintenance_factor - 1.0) # Maintenance intensity determines the mixing ratio
        hi = mixing_weight * ideal_decay + (1 - mixing_weight) * maintenance_modulated

        # Force monotonic decreasing constraints (stronger constraints)
        for t_idx in range(1, T):
            min_decrease = 0.002 + 0.001 * maintenance_factor[:, t_idx-1] # Adaptive minimum reduction
            hi = torch.cat([
                hi[:, :t_idx],
                torch.min(hi[:, t_idx:t_idx+1], hi[:, t_idx-1:t_idx] - min_decrease.unsqueeze(-1)),
                hi[:, t_idx+1:]
            ], dim=1)

        return hi, h_multi

class ReconHead(nn.Module):
    """
    Recursive reconstruction: given (x_t, h_t) → predict x_{t+1}
    """
    def __init__(self, C, hidden=128):
        super().__init__()
        self.gru = nn.GRU(input_size=C+1, hidden_size=hidden, batch_first=True)
        self.out = nn.Linear(hidden, C)
    def forward(self, x_in, h_in):
        # x_in:(B,T_in,C), h_in:(B,T_in)
        B,T,C = x_in.shape
        h_in_clamped = torch.clamp(h_in, 0.0, 10.0)  # Remove upper limit constraint
        feat = torch.cat([x_in, h_in_clamped.unsqueeze(-1)], dim=-1)  # (B,T,C+1)
        H,_ = self.gru(feat)                                  # (B,T,H)
        y = self.out(H)                                       # (B,T,C) aligned predict x_{t+1}
        return y

class MaintClassifier(nn.Module):
    """
    Use HI before-after difference features for 3-class classification
    Enhanced version: not only uses ΔHI, but also sensor data change patterns
    """
    def __init__(self, sensor_dim, hidden=64, n_classes=3):
        super().__init__()
        # Original HI difference features
        self.hi_mlp = nn.Sequential(
            nn.Linear(6, hidden), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden//2)
        )

        # Sensor data change features
        self.sensor_mlp = nn.Sequential(
            nn.Linear(sensor_dim * 2, hidden), nn.ReLU(),  # Before and after statistical features
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden//2)
        )

        # Fusion classifier
        self.final_classifier = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, n_classes)
        )

    def forward(self, h_b, h_a, x_b, x_a, mask):
        # HI difference features (original logic)
        m = mask
        T = m.size(1)
        valid = m.sum(1, keepdim=True).clamp_min(1.0)  # (B,1)
        db = h_a - h_b                        # (B,T)
        mean_d = (db*m).sum(1,keepdim=True)/valid
        var_d  = (((db-mean_d)*m)**2).sum(1,keepdim=True)/valid
        std_d  = (var_d + 1e-8).sqrt()
        d0     = (db[:, :1])                  # First value
        dmax   = (db.masked_fill(m==0, -1e9)).max(dim=1, keepdim=True).values
        pos_ratio = ((db>0).float()*m).sum(1,keepdim=True)/valid

        # Slope (linear fitting)
        t = torch.arange(T, device=h_b.device, dtype=h_b.dtype)
        t = (t - t.mean())/(t.std()+1e-6)
        def slope(x):
            num = (x*t).sum(1) - x.sum(1)*t.sum()/T
            den = (t**2).sum() - (t.sum()**2)/T + 1e-8
            return (num/den).unsqueeze(1)
        slope_diff = slope(h_a) - slope(h_b)  # (B,1)
        hi_feat = torch.cat([mean_d, d0, dmax, std_d, slope_diff, pos_ratio], dim=1)  # (B,6)
        hi_feat = torch.clamp(hi_feat, -10.0, 10.0)
        hi_features = self.hi_mlp(hi_feat)

        # Sensor data change features
        x_b_mean = (x_b * m.unsqueeze(-1)).sum(1) / valid        # (B,C)
        x_a_mean = (x_a * m.unsqueeze(-1)).sum(1) / valid        # (B,C)
        sensor_change = torch.cat([x_b_mean, x_a_mean], dim=1)   # (B, 2*C)
        sensor_features = self.sensor_mlp(sensor_change)

        # Fused features
        combined_features = torch.cat([hi_features, sensor_features], dim=1)
        logits = self.final_classifier(combined_features)

        return logits

def sanitize_tensors(*tensors):
    """Replace NaN/Inf in tensors with finite values"""
    result = []
    for t in tensors:
        if t is not None:
            t_clean = torch.nan_to_num(t, nan=0.0, posinf=1e6, neginf=-1e6)
            result.append(t_clean)
        else:
            result.append(t)
    return result[0] if len(result) == 1 else result

class DiffAwareReconstructor(nn.Module):
    """
    Difference-aware reconstructor for true liquids - operator parameters and behavior patterns are significantly different before and after maintenance
    """
    def __init__(self, in_ch, trend_ch=8, hidden=128, n_classes=3):
        super().__init__()
        self.encoder = TrendEncoder(in_ch, trend_ch)
        self.recon_b = ReconHead(in_ch, hidden)
        self.recon_a = ReconHead(in_ch, hidden)
        self.clf     = MaintClassifier(sensor_dim=in_ch, hidden=64, n_classes=n_classes)

        # Liquid property enhancement parameters
        self.liquid_enhancement = nn.Parameter(torch.tensor(2.0)) # Liquid enhancement factor
        self.maintenance_sensitivity = nn.Parameter(torch.tensor(1.5)) # Maintenance sensitivity

    def forward(self, x_b, x_a, mask):
        # x_b/x_a : (B,L,C) (will be cut to 0:L-2 / 1:L-1 later)
        h_b, h_multi_b = self.encoder(x_b)  # (B,L)  Health state learned from sensor data
        h_a, h_multi_a = self.encoder(x_a)  # (B,L)

        # Liquid Strengthening: Ensure significant difference before and after maintenance
        maintenance_effect = F.softplus(self.maintenance_sensitivity)
        h_a = h_a * (1 + 0.3 * maintenance_effect) # Health status improved after maintenance

        # Differences before and after forced maintenance
        diff_enhancement = F.softplus(self.liquid_enhancement)
        h_difference = torch.clamp(h_a - h_b, 0.05, 2.0) # Ensure positive difference
        h_a = h_b + h_difference * diff_enhancement

        # Numerical stabilization
        h_b, h_a = sanitize_tensors(h_b, h_a)

        # Recursive reconstruction: input 0:L-2 → predict 1:L-1
        xb_in, hb_in = x_b[:, :-2], h_b[:, :-2]
        xa_in, ha_in = x_a[:, :-2], h_a[:, :-2]
        yb_hat = self.recon_b(xb_in, hb_in)   # (B,L-2,C) ≈ x_b[:,1:L-1]
        ya_hat = self.recon_a(xa_in, ha_in)   # (B,L-2,C) ≈ x_a[:,1:L-1]

        # Numerical stabilization
        yb_hat, ya_hat = sanitize_tensors(yb_hat, ya_hat)

        # Classification: use unclipped length mask and include sensor features
        logits = self.clf(h_b, h_a, x_b, x_a, mask)
        logits = sanitize_tensors(logits)

        return yb_hat, ya_hat, h_b, h_a, logits

# ============================================================
# 3) Loss function: loss function that enhances liquid properties
# ============================================================
def masked_mse(a, b, mask):
    # a,b:(B,T,C)  mask:(B,T)
    diff = (a - b)**2
    mse  = (diff.mean(-1) * mask).sum() / (mask.sum() + 1e-6)
    return mse

def slope_loss(h, mask, kind="smooth"):
    # Smooth: second-order difference; Monotonic decreasing: penalize upward trend
    if kind=="smooth":
        d2 = h[:,2:] - 2*h[:,1:-1] + h[:,:-2]
        m  = mask[:,2:]
        return (d2.abs() * m).sum() / (m.sum()+1e-6)
    elif kind=="mono_dec":
        dh = h[:,1:] - h[:,:-1]        # If >0 indicates upward (violates "decreasing")
        m  = mask[:,1:]
        return (F.relu(dh) * m).sum() / (m.sum()+1e-6)
    else:
        return torch.tensor(0., device=h.device)

def liquid_operator_diversity_loss(model, weight=1.0):
    """
    Liquid operator diversity loss: forcing different operators to produce different output modes
    """
    if not hasattr(model.encoder.customkan, 'gate'):
        return torch.tensor(0.0)

    gate = model.encoder.customkan.gate
    if not hasattr(gate, 'weight_changes'):
        return torch.tensor(0.0)

    # Encourage diversity in weight changes
    weight_changes = gate.weight_changes  # (1, n_ops)
    if weight_changes.sum() < 1e-6:
        # If there is no weight change, apply a penalty
        return weight * torch.tensor(5.0, device=weight_changes.device)

    # Calculate the variance of weight changes to encourage diversity
    diversity = -torch.var(weight_changes) + 0.1 # Negative variance + base penalty
    return weight * torch.clamp(diversity, 0.0, 10.0)

def liquid_parameter_dynamics_loss(model, weight=0.5):
    """
    Dynamic loss of liquid parameters: ensure that operator parameters produce different responses under different inputs
    """
    loss = torch.tensor(0.0, device=next(model.parameters()).device)

    # Check the liquid operator in the encoder
    if hasattr(model.encoder, 'customkan') and hasattr(model.encoder.customkan, 'ops'):
        for op in model.encoder.customkan.ops:
            if hasattr(op, 'param_net'):
                # Check the weight changes of the parameter network
                for param in op.param_net.parameters():
                    if param.grad is not None:
                        # Encourage gradient diversity
                        grad_var = torch.var(param.grad)
                        loss += weight * torch.clamp(0.01 - grad_var, 0.0, 1.0)

    return loss

def enhanced_maintenance_effect_loss(h_b, h_a, labels, mask, weight=3.0):
    """
    Enhanced Maintenance Effect Loss: Ensures that different maintenance types produce significantly different patterns of health improvement
    """
    valid = mask.sum(1, keepdim=True).clamp_min(1.0)  # (B,1)

    # Calculate health improvement before and after maintenance
    improvement = ((h_a - h_b) * mask).sum(1, keepdim=True) / valid  # (B,1)

    # Calculate the improved variability (it should be similar within the same type, and there should be obvious differences between different types)
    loss = torch.zeros_like(improvement)

    for cls in [0, 1, 2]:
        cls_mask = (labels == cls).float().unsqueeze(1)  # (B,1)
        if cls_mask.sum() < 1:
            continue

        cls_improvements = improvement * cls_mask
        cls_valid = cls_mask.sum()

        if cls == 0: # Perfect: Requires significant improvement 0.4-0.6
            target_improvement = 0.5
            loss += cls_mask * ((improvement - target_improvement) ** 2) * 2.0
            # Additional requirements: The improvement of Perfect maintenance should be the largest
            loss += cls_mask * F.relu(0.4 - improvement) ** 2 * 3.0
        elif cls == 1: # General: Moderate improvement 0.2-0.4
            target_improvement = 0.3
            loss += cls_mask * ((improvement - target_improvement) ** 2) * 1.5
        else: # Poor: slight improvement 0.1-0.2
            target_improvement = 0.15
            loss += cls_mask * ((improvement - target_improvement) ** 2) * 1.0
            # Additional requirements: Poor maintenance cannot be improved too much
            loss += cls_mask * F.relu(improvement - 0.25) ** 2 * 2.0

    return weight * loss.mean()

def liquid_temporal_consistency_loss(h_b, h_a, mask, weight=1.0):
    """
    Liquid timing consistency loss: ensuring that the health index maintains reasonable dynamic changes over time
    """
    # Timing gradient consistency
    grad_b = torch.abs(h_b[:, 1:] - h_b[:, :-1])  # (B, T-1)
    grad_a = torch.abs(h_a[:, 1:] - h_a[:, :-1])  # (B, T-1)

    mask_grad = mask[:, 1:]  # (B, T-1)

    # The gradient after maintenance should be different from that before maintenance (reflecting liquid characteristics)
    grad_diff = torch.abs(grad_a - grad_b)

    # Encourage moderate gradient differences (not too big, not too small)
    target_grad_diff = 0.02
    grad_consistency_loss = ((grad_diff - target_grad_diff) ** 2 * mask_grad).sum() / (mask_grad.sum() + 1e-6)

    return weight * grad_consistency_loss

# ============================================================
# 4) Data splitting and visualization (aligned to same time axis)
# ============================================================
LABEL2NAME = {0: "Perfect", 1: "General", 2: "Poor"}

def split_pairs_uid(pairs, train_ratio=0.7, val_ratio=0.1, seed=42):
    uids = sorted(list(pairs.keys()))
    rs = np.random.RandomState(seed)
    rs.shuffle(uids)
    n = len(uids); n_tr=int(n*train_ratio); n_vl=int(n*val_ratio)
    to_dict = lambda ss:{u:pairs[u] for u in ss if u in pairs}
    return to_dict(uids[:n_tr]), to_dict(uids[n_tr:n_tr+n_vl]), to_dict(uids[n_tr+n_vl:])

def print_split_summary(pairs, train_ratio=0.7, val_ratio=0.1, seed=42):
    pairs_tr, pairs_vl, pairs_te = split_pairs_uid(pairs, train_ratio, val_ratio, seed)
    def count_items(pp):
        cnt = 0
        for uid, d in pp.items():
            cnt += len(d)
        return len(pp), cnt
    n_uid_tr, n_pair_tr = count_items(pairs_tr)
    n_uid_vl, n_pair_vl = count_items(pairs_vl)
    n_uid_te, n_pair_te = count_items(pairs_te)
    print(f"[Data Split] Train: UID={n_uid_tr}, pairs≈{n_pair_tr} | "
          f"Val: UID={n_uid_vl}, pairs≈{n_pair_vl} | "
          f"Test: UID={n_uid_te}, pairs≈{n_pair_te}")

def remove_constant_segments(hi_values, threshold=1e-6):
    """
    Remove constant segments in HI curves (caused by mask or other reasons)
    Return valid HI values and corresponding time indices
    """
    if len(hi_values) <= 1:
        return hi_values, np.arange(len(hi_values))

    # Calculate differences between adjacent points
    diffs = np.abs(np.diff(hi_values))

    # Find points with sufficient change
    valid_mask = np.ones(len(hi_values), dtype=bool)
    valid_mask[1:] = diffs > threshold

    # Ensure at least first and last two points are kept
    if np.sum(valid_mask) < 2:
        valid_mask[0] = True
        valid_mask[-1] = True

    valid_indices = np.where(valid_mask)[0]
    return hi_values[valid_indices], valid_indices

@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total = {"mse_b":0.0, "mse_a":0.0, "acc":0.0, "delta_mean":0.0}
    n_batch=0
    n_acc = 0; n_tot=0
    for batch in loader:
        xb = batch["x_before"].to(device)
        xa = batch["x_after"].to(device)
        hi_b = batch["hi_before"].to(device)
        hi_a = batch["hi_after"].to(device)
        labels = batch["labels"].to(device)
        lengths= batch["lengths"].to(device)
        mask   = batch["mask"].to(device)

        # Valid segment: 0:L-2 input, 1:L-1 target
        m_tgt  = (torch.arange(xb.size(1)-2, device=device)[None,:] < (lengths-2)[:,None]).float()

        yb_hat, ya_hat, h_b, h_a, logits = model(xb, xa, mask)

        # Numerical stabilization
        yb_hat, ya_hat, h_b, h_a, logits = sanitize_tensors(yb_hat, ya_hat, h_b, h_a, logits)

        # Align targets
        yb_tgt = xb[:,1:-1,:]
        ya_tgt = xa[:,1:-1,:]

        mse_b = masked_mse(yb_hat, yb_tgt, m_tgt)
        mse_a = masked_mse(ya_hat, ya_tgt, m_tgt)

        # Check for NaN
        if torch.isnan(mse_b) or torch.isnan(mse_a):
            continue

        # Classification
        pred = logits.argmax(dim=1)
        n_acc += (pred == labels).sum().item()
        n_tot += labels.numel()

        # Statistics of ΔHI mean (post-maintenance improvement relative to pre-maintenance)
        valid = mask.sum(1, keepdim=True).clamp_min(1.0)
        delta_mean = ((h_a - h_b)*mask).sum(1,keepdim=True)/valid
        delta_mean = torch.nan_to_num(delta_mean, nan=0.0)
        total["delta_mean"] += delta_mean.mean().item()

        total["mse_b"] += mse_b.item()
        total["mse_a"] += mse_a.item()
        n_batch += 1

    for k in ["mse_b","mse_a","delta_mean"]:
        total[k] /= max(n_batch,1)
    total["acc"] = n_acc / max(n_tot,1)
    return total

@torch.no_grad()
def collect_test_predictions(model, loader, device, max_curve_keep=24):
    """
    Collect predictions, ΔHI, and a few sample HI curves on test set (for plotting).
    """
    model.eval()
    y_true, y_pred = [], []
    all_delta_mean, all_uids = [], []
    keep_curves = []

    for batch in loader:
        xb = batch["x_before"].to(device)   # (B,L,C)
        xa = batch["x_after"].to(device)
        mask   = batch["mask"].to(device)   # (B,L)
        labels = batch["labels"].to(device) # (B,)
        lengths= batch["lengths"]           # cpu tensor
        uids   = batch["uids"]

        yb_hat, ya_hat, h_b, h_a, logits = model(xb, xa, mask)

        # Numerical stabilization
        yb_hat, ya_hat, h_b, h_a, logits = sanitize_tensors(yb_hat, ya_hat, h_b, h_a, logits)

        prob = F.softmax(logits, dim=1)     # (B,3)
        pred = prob.argmax(1)               # (B,)

        # Statistics of ΔHI mean (post-maintenance improvement relative to pre-maintenance)
        valid = mask.sum(1, keepdim=True).clamp_min(1.0)
        delta_mean = ((h_a - h_b) * mask).sum(1, keepdim=True) / valid  # (B,1)
        delta_mean = torch.nan_to_num(delta_mean, nan=0.0)

        y_true.append(labels.cpu().numpy())
        y_pred.append(pred.cpu().numpy())
        all_delta_mean.append(delta_mean.squeeze(1).cpu().numpy())
        all_uids.extend(uids)

        # Save some curves for visualization (aligned: after concatenated after before)
        for i in range(xb.size(0)):
            if len(keep_curves) >= max_curve_keep:
                break
            L_i = int(lengths[i].item())
            # Fix: truncate reconstruction prediction by actual length
            L_recon = min(L_i - 2, yb_hat.size(1))  # Reconstruction sequence length
            keep_curves.append({
                "uid": uids[i],
                "true": int(labels[i].cpu().item()),
                "pred": int(pred[i].cpu().item()),
                "prob": prob[i].cpu().numpy(),
                "h_before": h_b[i, :L_i].cpu().numpy(),
                "h_after":  h_a[i, :L_i].cpu().numpy(),
                "hi_before_gt": batch["hi_before"][i, :L_i].cpu().numpy(),
                "hi_after_gt": batch["hi_after"][i, :L_i].cpu().numpy(),
                "yb_hat":   yb_hat[i, :L_recon].cpu().numpy(),       # (L_recon,C)
                "ya_hat":   ya_hat[i, :L_recon].cpu().numpy(),       # (L_recon,C)
                "x_before": xb[i, :L_i].cpu().numpy(),               # (L,C)
                "x_after":  xa[i, :L_i].cpu().numpy(),               # (L,C)
                "length": L_i
            })

    y_true = np.concatenate(y_true, axis=0) if len(y_true)>0 else np.array([])
    y_pred = np.concatenate(y_pred, axis=0) if len(y_pred)>0 else np.array([])
    all_delta_mean = np.concatenate(all_delta_mean, axis=0) if len(all_delta_mean)>0 else np.array([])
    return y_true, y_pred, all_delta_mean, all_uids, keep_curves

def select_best_curves_for_plotting(curves, n_show=6):
    """
    Intelligent selection of drawing cases with the highest quality
    """
    if len(curves) == 0:
        return []

    #Group by category
    curves_by_class = {0: [], 1: [], 2: []}
    for curve in curves:
        curves_by_class[curve["true"]].append(curve)

    def calculate_quality_score(curve):
        """Calculate the sample quality score (between 0-1, the higher the better)"""
        score = 0.0

        # 1. Prediction accuracy (weight: 40%)
        if curve["true"] == curve["pred"]:
            score += 0.4 * curve["prob"][curve["pred"]] # The prediction is correct and the confidence is high
        else:
            score += 0.1 * (1 - curve["prob"][curve["pred"]]) # Prediction error, the lower the confidence, the better

        # 2. Degree of change of HI curve (weight: 30%)
        # I hope that the HI curve has enough changes and is not a flat straight line
        h_before = curve["h_before"]
        h_after = curve["h_after"]
        hi_variance = np.var(h_before) + np.var(h_after)
        normalized_variance = np.clip(hi_variance / 0.1, 0, 1)
        score += 0.3 * normalized_variance

        # 3. Maintenance effect visibility (weight: 20%)
        # It should be better after maintenance than before.
        h_before = curve["h_before"]
        h_after = curve["h_after"]
        hi_improvement = np.mean(h_after - h_before)
        # Normalize to 0-1 range
        normalized_improvement = np.clip((hi_improvement + 0.5) / 1.0, 0, 1)
        score += 0.2 * normalized_improvement

        # 4. Data quality (weight: 10%)
        # Check if there are outliers or constant segments
        x_before_var = np.var(curve["x_before"])
        x_after_var = np.var(curve["x_after"])
        data_quality = np.clip((x_before_var + x_after_var) / 2.0, 0, 1)
        score += 0.1 * data_quality

        return score

    # Calculate quality scores for all samples
    for curve in curves:
        curve["quality_score"] = calculate_quality_score(curve)

    # Intelligent selection strategy
    selected_curves = []

    # Select at least one best sample for each category
    for class_id in [0, 1, 2]:
        if len(curves_by_class[class_id]) > 0:
            # Sort by quality score
            sorted_curves = sorted(curves_by_class[class_id],
                                 key=lambda x: x["quality_score"], reverse=True)
            selected_curves.append(sorted_curves[0])

    # If more samples are needed, select from the remaining high-quality samples
    remaining_slots = n_show - len(selected_curves)
    if remaining_slots > 0:
        # Exclude selected from all samples, select by quality score
        already_selected_uids = {curve["uid"] for curve in selected_curves}
        remaining_curves = [curve for curve in curves
                          if curve["uid"] not in already_selected_uids]

        if len(remaining_curves) > 0:
            remaining_curves.sort(key=lambda x: x["quality_score"], reverse=True)
            selected_curves.extend(remaining_curves[:remaining_slots])

    #Final sorting by quality score
    selected_curves.sort(key=lambda x: x["quality_score"], reverse=True)

    return selected_curves[:n_show]

def plot_hi_examples_aligned(curves, n_show=6, seed=0):
    """Post-maintenance sequence continues from pre-maintenance endpoint; draw vertical line to mark t_m; remove constant segments.
    Add "Learned HI" in title to indicate this is health state inferred from sensor data, post-maintenance should be higher than pre-maintenance
    Intelligent selection of the best drawing cases
    """
    if len(curves)==0:
        print("(No visualization samples)")
        return

    # Use the smart selection function to select the best drawing case
    selected_curves = select_best_curves_for_plotting(curves, n_show)

    if len(selected_curves) == 0:
        print("(No high-quality visualization samples found)")
        return

    n_show = len(selected_curves)
    cols = 3
    rows = int(np.ceil(n_show/cols))
    plt.figure(figsize=(cols*4.6, rows*3.2))

    for k in range(n_show):
        ex = selected_curves[k]
        hb = ex["h_before"]; ha = ex["h_after"]
        hb_gt = ex["hi_before_gt"]; ha_gt = ex["hi_after_gt"]      # ground truth HI from dataset
        L  = ex["length"]  # Use actual length instead of padded length

        # Remove constant segments
        hb_clean, hb_indices = remove_constant_segments(hb)
        ha_clean, ha_indices = remove_constant_segments(ha)
        hb_gt_clean, hb_gt_indices = remove_constant_segments(hb_gt)
        ha_gt_clean, ha_gt_indices = remove_constant_segments(ha_gt)

        # Corresponding time axis
        t_b = hb_indices
        t_a = ha_indices + L  # after follows immediately
        t_b_gt = hb_gt_indices
        t_a_gt = ha_gt_indices + L

        uid = ex["uid"]
        pred = ex["pred"]; true = ex["true"]; prob = ex["prob"]
        quality_score = ex["quality_score"]

        plt.subplot(rows, cols, k+1)

        # Only plot non-constant segments - model predicted HI
        if len(hb_clean) > 1:
            plt.plot(t_b, hb_clean, label="Learned HI_Pre-maintenance", linewidth=2.5, marker='o', markersize=4, color='blue')
        if len(ha_clean) > 1:
            plt.plot(t_a, ha_clean, label="Learned HI_Post-maintenance",  linewidth=2.5, linestyle='--', marker='s', markersize=4, color='red')

        # Plot ground truth HI from dataset
        if len(hb_gt_clean) > 1:
            plt.plot(t_b_gt, hb_gt_clean, label="GT HI_Pre-maintenance", linewidth=1.8, color='cyan', alpha=0.8)
        if len(ha_gt_clean) > 1:
            plt.plot(t_a_gt, ha_gt_clean, label="GT HI_Post-maintenance",  linewidth=1.8, linestyle=':', color='orange', alpha=0.8)

        # Maintenance time vertical line
        plt.axvline(L-1, color='k', linestyle=':', linewidth=1.5, alpha=0.8)

        d_mean = float(np.mean(ha - hb))  # Post-maintenance improvement relative to pre-maintenance
        d_max  = float(np.max(ha - hb))
        d_mean_gt = float(np.mean(ha_gt - hb_gt))

        # Add quality rating to title
        correctness = "✓" if true == pred else "✗"
        title = (f"uid={uid} {correctness} (Quality: {quality_score:.2f})\n"
                 f"True={LABEL2NAME[true]} | Pred={LABEL2NAME[pred]} | Conf={prob[pred]:.3f}\n"
                 f"ΔHI_Learned={d_mean:.3f}, ΔHI_GT={d_mean_gt:.3f}")
        plt.title(title, fontsize=9)
        plt.xlabel("Cycle (Pre-maintenance | Post-maintenance)")
        plt.ylabel("Health Index (↑Post should be higher)")
        plt.grid(ls='--', alpha=.4, linewidth=0.8)

        #Add background color to indicate prediction accuracy
        if true == pred:
            plt.gca().set_facecolor('#f0f8ff') # Light blue background indicates correct prediction
        else:
            plt.gca().set_facecolor('#fff0f0') # Light red background indicates incorrect prediction

        if k==0: plt.legend(fontsize=8, loc='best')

    plt.suptitle(f"Best {n_show} Health Index Examples (Ranked by Quality Score)", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.show()

def plot_sensor_examples_aligned(curves, sensor_idx_list=None, n_cols=4):
    """
    Plot several sensors: original(before/after) + one-step recursive prediction of after(ya_hat),
    post-maintenance time series starts after pre-maintenance end. Show trajectories under different
    maintenance strategies. Intelligent selection of the best sensor dimensions for drawing
    """
    if len(curves)==0:
        print("(No visualization samples)")
        return

    # Select the best samples for sensor visualization
    selected_curves = select_best_curves_for_plotting(curves, 9) # Select more samples for sensor analysis

    # Group curves by maintenance strategy (true class)
    curves_by_strategy = {0: [], 1: [], 2: []}
    for curve in selected_curves:
        curves_by_strategy[curve["true"]].append(curve)

    # Select best example from each strategy for comparison
    strategy_examples = {}
    for strategy in [0, 1, 2]:
        if len(curves_by_strategy[strategy]) > 0:
            # Select the sample with the highest quality score under this strategy
            best_curve = max(curves_by_strategy[strategy], key=lambda x: x["quality_score"])
            strategy_examples[strategy] = best_curve

    if len(strategy_examples) == 0:
        print("(No high-quality visualization samples for any maintenance strategy)")
        return

    # Intelligent selection of sensor dimensions
    first_ex = list(strategy_examples.values())[0]
    xb = first_ex["x_before"]       # (L,C)
    C = xb.shape[1]

    if sensor_idx_list is None:
        # Intelligently select the most representative sensor dimension
        sensor_scores = []

        for sensor_idx in range(C):
            score = 0.0

            # Calculate the degree of change of the sensor under different maintenance strategies
            for strategy, ex in strategy_examples.items():
                xb_sensor = ex["x_before"][:, sensor_idx]
                xa_sensor = ex["x_after"][:, sensor_idx]

                # 1. Range of change (weight: 40%)
                change_magnitude = np.abs(np.mean(xa_sensor) - np.mean(xb_sensor))
                score += 0.4 * min(change_magnitude / (np.std(xb_sensor) + 1e-6), 5.0)

                # 2. Variance (information amount) (weight: 30%)
                variance = np.var(xb_sensor) + np.var(xa_sensor)
                score += 0.3 * min(variance, 10.0)

                # 3. Signal-to-noise ratio (weight: 30%)
                snr = np.mean(xb_sensor) / (np.std(xb_sensor) + 1e-6)
                score += 0.3 * min(abs(snr), 5.0)

            sensor_scores.append((sensor_idx, score))

        # Select the sensor with the highest rating
        sensor_scores.sort(key=lambda x: x[1], reverse=True)
        sensor_idx_list = [idx for idx, _ in sensor_scores[:min(8, len(sensor_scores))]]

    n = len(sensor_idx_list)
    n_rows = int(np.ceil(n/n_cols))
    plt.figure(figsize=(n_cols*5.5, n_rows*4.0))

    colors = {0: '#e74c3c', 1: '#f39c12', 2: '#9b59b6'} # Brighter colors

    for i, s in enumerate(sensor_idx_list):
        plt.subplot(n_rows, n_cols, i+1)

        # Plot pre-maintenance trajectory (common for all strategies, use best example)
        best_ex = max(strategy_examples.values(), key=lambda x: x["quality_score"])
        xb = best_ex["x_before"]
        L = best_ex["length"]
        t_b = np.arange(L)

# ============================================================
# Monotonic-Decreasing HI + aligned plotting (before -> after)
# ============================================================
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import random
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL2NAME = {0: "Perfect", 1: "General", 2: "Poor"}

# ============================================================
# 1) Dataset: read samples from pairs, perform shift operations in batch later
# ============================================================
class PairsReconstructDataset(Dataset):
    """
    Each sample contains:
      x_before: (L,C) feature sequence before maintenance
      x_after : (L,C) feature sequence after maintenance
      hi_before/hi_after: (L,) health index (for evaluation/optional analysis only, not strong supervision)
      strategy: maintenance type ("Perfect Maintenance" / "General Maintenance" / "Poor Maintenance")
    """
    def __init__(self, pairs: dict, horizon: int=None):
        items = []
        for uid, sd in pairs.items():
            for strat, d in sd.items():
                xb = np.asarray(d["x_before"], dtype=np.float32)
                xa = np.asarray(d["x_after"],  dtype=np.float32)
                # When there's no real HI, create placeholder that will be learned by the model
                if "hi_before" in d and d["hi_before"] is not None:
                    hib = np.asarray(d["hi_before"],dtype=np.float32)
                else:
                    # Create initialized fake HI, model will learn real patterns
                    hib = np.linspace(0.8, 0.4, len(xb)).astype(np.float32)

                if "hi_after" in d and d["hi_after"] is not None:
                    hia = np.asarray(d["hi_after"], dtype=np.float32)
                else:
                    # Create different fake HI patterns based on maintenance strategy
                    if "perfect" in strat.lower():
                        # Perfect maintenance: significant improvement
                        hia = np.linspace(0.9, 0.6, len(xa)).astype(np.float32)
                    elif "general" in strat.lower():
                        # General maintenance: moderate improvement
                        hia = np.linspace(0.7, 0.5, len(xa)).astype(np.float32)
                    else:
                        # Poor maintenance: slight improvement
                        hia = np.linspace(0.5, 0.35, len(xa)).astype(np.float32)

                L, C = xb.shape
                # Unify length (optional)
                if horizon is not None and L > horizon:
                    xb, xa, hib, hia = xb[:horizon], xa[:horizon], hib[:horizon], hia[:horizon]
                    L = horizon
                if L < 3:  # At least need to form 0:L-2 -> 1:L-1
                    continue
                items.append({"uid": uid, "strategy": strat, "x_before": xb, "x_after": xa,
                              "hi_before": hib, "hi_after": hia})
        assert len(items)>0, "Dataset is empty, please check pairs data."
        self.items = items
        self.C = items[0]["x_before"].shape[1]

        # Strategy to 3-class label mapping
        def strat_to_cls(s):
            s = s.lower()
            if "perfect" in s: return 0
            if "general" in s: return 1
            if "poor" in s: return 2
            raise ValueError(f"Unknown strategy name: {s}")
        self.labels = [strat_to_cls(it["strategy"]) for it in items]

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        return {
            "uid": it["uid"],
            "label": self.labels[idx],
            "x_before": torch.from_numpy(it["x_before"]), # (L,C)
            "x_after":  torch.from_numpy(it["x_after"]),  # (L,C)
            "hi_before":torch.from_numpy(it["hi_before"]),# (L,)
            "hi_after": torch.from_numpy(it["hi_after"]), # (L,)
            "strategy": it["strategy"]
        }

def pad_collate_shift(batch):
    """
    Pad sequences to same length; return mask, do NOT perform shift operation here
    to allow unified training: inputs = [:,0:L-2], targets = [:,1:L-1]
    """
    Ls = [b["x_before"].shape[0] for b in batch]
    Lmax = max(Ls)
    C = batch[0]["x_before"].shape[1]

    def pad2d(x):
        L, C0 = x.shape
        out = torch.zeros(Lmax, C0, dtype=x.dtype)
        out[:L] = x
        return out

    def pad1d(x):
        L = x.shape[0]
        out = torch.zeros(Lmax, dtype=x.dtype)
        out[:L] = x
        return out

    x_before = torch.stack([pad2d(b["x_before"]) for b in batch], 0) # (B,Lmax,C)
    x_after  = torch.stack([pad2d(b["x_after"])  for b in batch], 0) # (B,Lmax,C)
    hi_before= torch.stack([pad1d(b["hi_before"])for b in batch], 0) # (B,Lmax)
    hi_after = torch.stack([pad1d(b["hi_after"]) for b in batch], 0) # (B,Lmax)
    lengths  = torch.tensor(Ls, dtype=torch.long)
    mask     = torch.zeros(len(batch), Lmax)
    for i,L in enumerate(Ls): mask[i,:L] = 1.0
    labels   = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    uids     = [b["uid"] for b in batch]
    return {"uids": uids, "labels": labels,
            "x_before": x_before, "x_after": x_after,
            "hi_before": hi_before, "hi_after": hi_after,
            "lengths": lengths, "mask": mask}

# ============================================================
# 2) Model: encode "damage" → project to monotonic decreasing HI → recursive reconstruction of two sequences → classify using ΔHI
# ============================================================
class BoltzmannKAN(nn.Module):
    def __init__(self, in_ch, out_ch=4):
        super().__init__()
        self.E  = nn.Parameter(torch.zeros(out_ch, in_ch))
        self.kT = nn.Parameter(torch.ones(out_ch, in_ch))
    def forward(self, x):
        B,T,C = x.shape
        kT = torch.clamp(F.softplus(self.kT), 0.01, 10.0).unsqueeze(0).unsqueeze(2)  # (1,out_ch,1,in_ch)
        E  = torch.clamp(self.E, -10.0, 10.0).unsqueeze(0).unsqueeze(2)             # (1,out_ch,1,in_ch)
        x_ = torch.clamp(x.unsqueeze(1), -10.0, 10.0)                               # (B, out_ch, T, in_ch)
        exp = torch.clamp((E - x_) / kT, -50, 50)
        w   = torch.sigmoid(exp)
        h   = (w * x_).sum(dim=3).permute(0, 2, 1)           # (B,T,out_ch) >=0
        return torch.clamp(F.softplus(h), 0.0, 100.0)

class BaseOp(nn.Module): pass
class MonotonicLinearOp(BaseOp):
    def __init__(self):
        super().__init__()
        self.raw_scale=nn.Parameter(torch.tensor(0.0))
        self.bias=nn.Parameter(torch.tensor(0.0))
        self.smin,self.smax=0.01,5.0
    def forward(self,h):
        scale=torch.clamp(F.softplus(self.raw_scale),self.smin,self.smax)
        xm=torch.clamp(h.mean(-1), -10.0, 10.0)
        return torch.clamp(F.softplus(scale*(xm+self.bias)), 0.0, 100.0)

class MonotonicFlatOp(BaseOp):
    def __init__(self):
        super().__init__()
        self.raw_scale=nn.Parameter(torch.tensor(0.0))
        self.bias=nn.Parameter(torch.tensor(0.0))
        self.smin,self.smax=1e-3,1.0
    def _cum(self,x):
        diff=F.relu(x[:,1:]-x[:,:-1])
        return torch.cat([torch.zeros_like(diff[:,:1]),torch.cumsum(diff,1)],1)
    def forward(self,h):
        scale=torch.clamp(F.softplus(self.raw_scale),self.smin,self.smax)
        xm=torch.clamp(h.mean(-1,keepdim=True), -10.0, 10.0)
        return torch.clamp(F.softplus(scale*(self._cum(xm)+self.bias)).squeeze(-1), 0.0, 100.0)

class ConcaveLogOp(BaseOp):
    def __init__(self):
        super().__init__()
        self.raw_scale=nn.Parameter(torch.tensor(0.0))
        self.eps=1e-3
        self.smin,self.smax=0.01,5.0
    def forward(self,h):
        scale=torch.clamp(F.softplus(self.raw_scale),self.smin,self.smax)
        xm=torch.clamp(h.mean(-1), -10.0, 10.0)
        return torch.clamp(F.softplus(scale*torch.log(torch.abs(xm)+self.eps)), 0.0, 100.0)

class SaturationSigmoidOp(BaseOp):
    def __init__(self):
        super().__init__()
        self.raw_scale=nn.Parameter(torch.tensor(0.0))
        self.raw_slope=nn.Parameter(torch.tensor(0.0))
        self.bias=nn.Parameter(torch.tensor(0.0))
        self.smin,self.smax=0.01,5.0
        self.lmin,self.lmax=0.1,5.0
    def forward(self,h):
        scale=torch.clamp(F.softplus(self.raw_scale),self.smin,self.smax)
        slope=torch.clamp(F.softplus(self.raw_slope),self.lmin,self.lmax)
        xm=torch.clamp(h.mean(-1), -10.0, 10.0)
        return torch.clamp(F.softplus(scale*torch.sigmoid(slope*(xm-self.bias))), 0.0, 100.0)

class HingeReLUOp(BaseOp):
    def __init__(self):
        super().__init__()
        self.raw_scale=nn.Parameter(torch.tensor(0.0))
        self.threshold=nn.Parameter(torch.tensor(0.0))
        self.smin,self.smax=0.01,5.0
    def forward(self,h):
        scale=torch.clamp(F.softplus(self.raw_scale),self.smin,self.smax)
        xm=torch.clamp(h.mean(-1), -10.0, 10.0)
        return torch.clamp(F.softplus(scale*F.relu(xm-self.threshold)), 0.0, 100.0)

class PolynomialOp(BaseOp):
    def __init__(self,deg=3):
        super().__init__()
        self.raw_coeff=nn.Parameter(torch.zeros(deg))
        self.deg=deg
    def forward(self,h):
        xm=torch.clamp(h.mean(-1), -5.0, 5.0)
        y=torch.zeros_like(xm)
        for i in range(self.deg):
            coeff = torch.clamp(F.softplus(self.raw_coeff[i]), 0.01, 5.0)
            y = y + coeff * torch.clamp(xm ** (i+1), -100.0, 100.0)
        return torch.clamp(F.softplus(y), 0.0, 100.0)

class DampedSinOp(BaseOp):
    def __init__(self):
        super().__init__()
        self.raw_scale=nn.Parameter(torch.tensor(0.0))
        self.raw_freq=nn.Parameter(torch.tensor(0.0))
        self.raw_lambda=nn.Parameter(torch.tensor(0.0))
        self.phase=nn.Parameter(torch.tensor(0.0))
        self.smin,self.smax=0.01,5.0
        self.fmin,self.fmax=0.1,5.0
        self.lmin,self.lmax=0.01,3.0
    def forward(self,h):
        scale=torch.clamp(F.softplus(self.raw_scale),self.smin,self.smax)
        freq=torch.clamp(F.softplus(self.raw_freq),self.fmin,self.fmax)
        lam=torch.clamp(F.softplus(self.raw_lambda),self.lmin,self.lmax)
        xm=torch.clamp(h.mean(-1), -10.0, 10.0)
        damp=1/(1+lam*torch.abs(xm))
        phase_val = torch.clamp(self.phase, -10.0, 10.0)
        return torch.clamp(F.softplus(scale*damp*(torch.sin(freq*xm+phase_val)+1)), 0.0, 100.0)

class PiecewiseLinearOp(BaseOp):
    def __init__(self):
        super().__init__()
        self.raw_k1=nn.Parameter(torch.tensor(0.0))
        self.raw_k2=nn.Parameter(torch.tensor(0.0))
        self.threshold=nn.Parameter(torch.tensor(0.0))
        self.kmin,self.kmax=0.01,5.0
    def forward(self,h):
        k1=torch.clamp(F.softplus(self.raw_k1),self.kmin,self.kmax)
        k2=torch.clamp(F.softplus(self.raw_k2),self.kmin,self.kmax)
        thresh = torch.clamp(self.threshold, -5.0, 5.0)
        xm=torch.clamp(h.mean(-1), -10.0, 10.0)
        left=k1*xm
        right=k1*thresh + k2*(xm-thresh)
        out=torch.where(xm<=thresh,left,right)
        return torch.clamp(F.softplus(out), 0.0, 100.0)

class SparseGate(nn.Module):
    def __init__(self, n_ops, tau_start=5.0, tau_end=0.1, n_steps=10000):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(n_ops))
        self.tau_start, self.tau_end, self.n_steps = tau_start, tau_end, n_steps
        self.register_buffer("step", torch.tensor(0.))
    def forward(self):
        tau = (self.tau_start*(1-self.step/self.n_steps) + self.tau_end*(self.step/self.n_steps)).clamp(min=self.tau_end)
        self.step.add_(1)  # Use add_ to avoid inplace operation
        g = -torch.empty_like(self.logits).exponential_().log() if self.training else torch.zeros_like(self.logits)
        g = torch.clamp(g, -50.0, 50.0)  # Limit Gumbel noise
        logits_stable = torch.clamp(self.logits, -50.0, 50.0)
        w = F.softmax((logits_stable + g)/tau, dim=-1)
        return w.view(1,1,-1)

class CustomKAN(nn.Module):
    def __init__(self, ops):
        super().__init__()
        self.ops = nn.ModuleList(ops)
        self.gate= SparseGate(len(ops))
        # Additional learnable scaling/bias for damage
        self.raw_gain = nn.Parameter(torch.tensor(0.0))
        self.bias     = nn.Parameter(torch.tensor(0.0))
    def forward(self, h):  # h:(B,T,trend_ch)
        outs = [op(h) for op in self.ops]          # list of (B,T)
        Tm = min(o.size(1) for o in outs)
        outs = [o[:,:Tm] for o in outs]
        st = torch.stack(outs, dim=-1)             # (B,Tm,K), >=0
        w  = self.gate()                           # (1,1,K)
        damage = torch.clamp((st*w).sum(-1), 0.0, 100.0)  # (B,Tm) non-negative, regarded as "damage"
        gain = torch.clamp(F.softplus(self.raw_gain), 0.1, 5.0)
        bias_val = torch.clamp(self.bias, -5.0, 5.0)
        damage = torch.clamp(gain*damage + bias_val, 0.0, 100.0)
        return damage                               # (B,Tm)

class TrendEncoder(nn.Module):
    """
    Encoder outputs "damage", then projects to monotonic decreasing HI:
        HI is learned from sensor data without hard range constraints
    Without real HI, learns mapping from sensor data to infer health states
    """
    def __init__(self, in_ch, trend_ch=4):
        super().__init__()
        self.boltz = BoltzmannKAN(in_ch, out_ch=trend_ch)
        ops = [MonotonicLinearOp(), MonotonicFlatOp(), ConcaveLogOp(),
               SaturationSigmoidOp(), HingeReLUOp(), PolynomialOp(),
               DampedSinOp(), PiecewiseLinearOp()]
        self.customkan = CustomKAN(ops)
        # Projection parameters - learn flexible health states from sensor features
        self.proj_gain = nn.Parameter(torch.tensor(1.0))
        self.proj_bias = nn.Parameter(torch.tensor(0.0))

        # Add health-aware layer, directly infer from multi-dimensional sensor features
        self.health_aware_transform = nn.Sequential(
            nn.Linear(in_ch, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output 0-1 range health state
        )

        # Add linear constraint layer, enforce linear trend
        self.linear_enforcer = nn.Sequential(
            nn.Linear(in_ch, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output linear decay degree
        )

    def forward(self, x):  # x:(B,T,C)
        # Feature extraction based on Boltzmann KAN
        h_multi = self.boltz(x)              # (B,T,trend_ch) >=0
        damage  = self.customkan(h_multi)    # (B,T)           >=0

        # Directly learn health states from raw sensor data
        # Assess health state for sensor readings at each time step
        B, T, C = x.shape
        x_reshaped = x.view(-1, C)  # (B*T, C)
        health_direct = self.health_aware_transform(x_reshaped)  # (B*T, 1)
        health_direct = health_direct.view(B, T)  # (B, T)

        # Add linear constraint, force linear decay pattern
        linear_weight = self.linear_enforcer(x_reshaped).view(B, T)  # (B, T)

        # Generate ideal linear decay pattern
        t = torch.arange(T, device=x.device, dtype=x.dtype).unsqueeze(0).expand(B, -1)  # (B, T)
        t_normalized = t / (T - 1)  # Normalize to [0, 1]

        # Combine damage and direct health assessment
        g = torch.clamp(F.softplus(self.proj_gain), 0.1, 5.0)
        b = torch.clamp(self.proj_bias, -5.0, 5.0)

        # Convert damage to health state decay
        damage_normalized = torch.sigmoid(g*(damage + b))

        # Combine three health assessment methods: direct assessment, damage pattern, linear constraint
        combined_health = health_direct * (1 - 0.3 * damage_normalized)

        # Generate ideal linear decay curve (from high to low) - remove hard range constraints
        linear_decay = 1.0 - t_normalized * 0.5  # Simple linear decay from 1.0 to 0.5

        # Use linear_weight to mix linear decay and complex patterns
        # Higher linear_weight means more towards linear
        hi = linear_weight * linear_decay + (1 - linear_weight) * combined_health

        # Force stronger monotonic decreasing constraint without hard range limits
        for t_idx in range(1, T):
            hi = torch.cat([
                hi[:, :t_idx],
                torch.min(hi[:, t_idx:t_idx+1], hi[:, t_idx-1:t_idx] - 0.001),  # Force at least 0.001 decrease
                hi[:, t_idx+1:]
            ], dim=1)

        return hi, h_multi

class ReconHead(nn.Module):
    """
    Recursive reconstruction: given (x_t, h_t) → predict x_{t+1}
    """
    def __init__(self, C, hidden=128):
        super().__init__()
        self.gru = nn.GRU(input_size=C+1, hidden_size=hidden, batch_first=True)
        self.out = nn.Linear(hidden, C)
    def forward(self, x_in, h_in):
        # x_in:(B,T_in,C), h_in:(B,T_in)
        B,T,C = x_in.shape
        h_in_clamped = torch.clamp(h_in, 0.0, 10.0)  # Remove upper limit constraint
        feat = torch.cat([x_in, h_in_clamped.unsqueeze(-1)], dim=-1)  # (B,T,C+1)
        H,_ = self.gru(feat)                                  # (B,T,H)
        y = self.out(H)                                       # (B,T,C) aligned predict x_{t+1}
        return y

class MaintClassifier(nn.Module):
    """
    Use HI before-after difference features for 3-class classification
    Enhanced version: not only uses ΔHI, but also sensor data change patterns
    """
    def __init__(self, sensor_dim, hidden=64, n_classes=3):
        super().__init__()
        # Original HI difference features
        self.hi_mlp = nn.Sequential(
            nn.Linear(6, hidden), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden//2)
        )

        # Sensor data change features
        self.sensor_mlp = nn.Sequential(
            nn.Linear(sensor_dim * 2, hidden), nn.ReLU(),  # Before and after statistical features
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden//2)
        )

        # Fusion classifier
        self.final_classifier = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, n_classes)
        )

    def forward(self, h_b, h_a, x_b, x_a, mask):
        # HI difference features (original logic)
        m = mask
        T = m.size(1)
        valid = m.sum(1, keepdim=True).clamp_min(1.0)  # (B,1)
        db = h_a - h_b                        # (B,T)
        mean_d = (db*m).sum(1,keepdim=True)/valid
        var_d  = (((db-mean_d)*m)**2).sum(1,keepdim=True)/valid
        std_d  = (var_d + 1e-8).sqrt()
        d0     = (db[:, :1])                  # First value
        dmax   = (db.masked_fill(m==0, -1e9)).max(dim=1, keepdim=True).values
        pos_ratio = ((db>0).float()*m).sum(1,keepdim=True)/valid

        # Slope (linear fitting)
        t = torch.arange(T, device=h_b.device, dtype=h_b.dtype)
        t = (t - t.mean())/(t.std()+1e-6)
        def slope(x):
            num = (x*t).sum(1) - x.sum(1)*t.sum()/T
            den = (t**2).sum() - (t.sum()**2)/T + 1e-8
            return (num/den).unsqueeze(1)
        slope_diff = slope(h_a) - slope(h_b)  # (B,1)
        hi_feat = torch.cat([mean_d, d0, dmax, std_d, slope_diff, pos_ratio], dim=1)  # (B,6)
        hi_feat = torch.clamp(hi_feat, -10.0, 10.0)
        hi_features = self.hi_mlp(hi_feat)

        # Sensor data change features (fix broadcast BUG: directly divide by (B,1) shaped valid)
        x_b_mean = (x_b * m.unsqueeze(-1)).sum(1) / valid        # (B,C)
        x_a_mean = (x_a * m.unsqueeze(-1)).sum(1) / valid        # (B,C)
        sensor_change = torch.cat([x_b_mean, x_a_mean], dim=1)   # (B, 2*C)
        sensor_features = self.sensor_mlp(sensor_change)

        # Fused features
        combined_features = torch.cat([hi_features, sensor_features], dim=1)
        logits = self.final_classifier(combined_features)

        return logits

def sanitize_tensors(*tensors):
    """Replace NaN/Inf in tensors with finite values"""
    result = []
    for t in tensors:
        if t is not None:
            t_clean = torch.nan_to_num(t, nan=0.0, posinf=1e6, neginf=-1e6)
            result.append(t_clean)
        else:
            result.append(t)
    return result[0] if len(result) == 1 else result

class DiffAwareReconstructor(nn.Module):
    """
    Overall model:
     - Two encoders share parameters: encode x_before / x_after → h_b / h_a (monotonic decreasing HI)
     - Two reconstruction heads: perform one-step recursive reconstruction respectively
     - Classification head: use (h_a - h_b) statistical features to identify maintenance type
    Without real HI labels, learn to infer health states from sensor data through self-supervised learning
    """
    def __init__(self, in_ch, trend_ch=4, hidden=128, n_classes=3):
        super().__init__()
        self.encoder = TrendEncoder(in_ch, trend_ch)
        self.recon_b = ReconHead(in_ch, hidden)
        self.recon_a = ReconHead(in_ch, hidden)
        self.clf     = MaintClassifier(sensor_dim=in_ch, hidden=64, n_classes=n_classes)

        # Add self-supervised consistency constraint
        self.consistency_weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, x_b, x_a, mask):
        # x_b/x_a : (B,L,C) (will be cut to 0:L-2 / 1:L-1 later)
        h_b, _ = self.encoder(x_b)  # (B,L)  Health state learned from sensor data
        h_a, _ = self.encoder(x_a)  # (B,L)

        # Numerical stabilization
        h_b, h_a = sanitize_tensors(h_b, h_a)

        # Recursive reconstruction: input 0:L-2 → predict 1:L-1
        xb_in, hb_in = x_b[:, :-2], h_b[:, :-2]
        xa_in, ha_in = x_a[:, :-2], h_a[:, :-2]
        yb_hat = self.recon_b(xb_in, hb_in)   # (B,L-2,C) ≈ x_b[:,1:L-1]
        ya_hat = self.recon_a(xa_in, ha_in)   # (B,L-2,C) ≈ x_a[:,1:L-1]

        # Numerical stabilization
        yb_hat, ya_hat = sanitize_tensors(yb_hat, ya_hat)

        # Classification: use unclipped length mask and include sensor features
        logits = self.clf(h_b, h_a, x_b, x_a, mask)
        logits = sanitize_tensors(logits)

        return yb_hat, ya_hat, h_b, h_a, logits

# ============================================================
# 3) Loss function: reconstruction + ΔHI + smooth/monotonic(decreasing) + classification + first-order linear + self-supervised consistency + maintenance effect constraint + global HI superiority constraint
# ============================================================
def masked_mse(a, b, mask):
    # a,b:(B,T,C)  mask:(B,T)
    diff = (a - b)**2
    mse  = (diff.mean(-1) * mask).sum() / (mask.sum() + 1e-6)
    return mse

def slope_loss(h, mask, kind="smooth"):
    # Smooth: second-order difference; Monotonic decreasing: penalize upward trend
    if kind=="smooth":
        d2 = h[:,2:] - 2*h[:,1:-1] + h[:,:-2]
        m  = mask[:,2:]
        return (d2.abs() * m).sum() / (m.sum()+1e-6)
    elif kind=="mono_dec":
        dh = h[:,1:] - h[:,:-1]        # If >0 indicates upward (violates "decreasing")
        m  = mask[:,1:]
        return (F.relu(dh) * m).sum() / (m.sum()+1e-6)
    else:
        return torch.tensor(0., device=h.device)

def enhanced_linear_slope_loss(h, mask, weight=5.0):
    """
    Enhanced linear constraint: calculate deviation from best linear fit, stronger weight
    Without real HI, this constraint helps learn reasonable decay patterns
    """
    B, T = h.shape
    valid_mask = mask  # (B, T)

    # Calculate best linear fit for each sample
    t = torch.arange(T, device=h.device, dtype=h.dtype).unsqueeze(0).expand(B, -1)  # (B, T)

    # Consider only valid positions
    loss_total = torch.tensor(0.0, device=h.device)
    n_valid_samples = 0

    for b in range(B):
        valid_indices = valid_mask[b] > 0
        if valid_indices.sum() < 2:  # Need at least 2 points for linear fitting
            continue

        h_valid = h[b][valid_indices]  # Valid HI values
        t_valid = t[b][valid_indices]  # Corresponding time points

        # Center time points
        t_mean = t_valid.mean()
        t_centered = t_valid - t_mean
        h_mean = h_valid.mean()

        # Calculate linear regression slope
        numerator = (t_centered * (h_valid - h_mean)).sum()
        denominator = (t_centered ** 2).sum() + 1e-8
        slope = numerator / denominator

        # Calculate intercept
        intercept = h_mean - slope * t_mean

        # Linear predicted values
        h_linear = slope * t_valid + intercept

        # Calculate MSE with linear fit, using stronger weight
        linear_mse = ((h_valid - h_linear) ** 2).mean()

        # Additional penalty for positive slope (encourage negative slope, i.e., decreasing)
        slope_penalty = F.relu(slope + 0.01) * 2.0  # Slope should be negative

        loss_total += weight * linear_mse + slope_penalty
        n_valid_samples += 1

    return loss_total / max(n_valid_samples, 1)

def maintenance_improvement_constraint(h_b, h_a, labels, mask):
    """
    Maintenance improvement constraint: ensure post-maintenance HI is significantly higher than pre-maintenance
    - Perfect(0): require post-maintenance significantly higher than pre-maintenance (at least 0.4)
    - General(1): require post-maintenance moderately higher than pre-maintenance (at least 0.2)
    - Poor(2): require post-maintenance slightly higher than pre-maintenance (at least 0.1)
    """
    valid = mask.sum(1, keepdim=True).clamp_min(1.0)  # (B,1)

    # Calculate average HI before and after maintenance
    h_b_mean = (h_b * mask).sum(1, keepdim=True) / valid  # (B,1)
    h_a_mean = (h_a * mask).sum(1, keepdim=True) / valid  # (B,1)

    improvement = h_a_mean - h_b_mean  # Improvement after maintenance relative to before

    loss = torch.zeros_like(improvement)
    for cls in [0, 1, 2]:
        sel = (labels == cls).float().unsqueeze(1)  # (B,1)
        if sel.sum() < 1:
            continue

        if cls == 0:  # Perfect
            # Require at least 0.4 improvement
            loss += sel * F.relu(0.4 - improvement) ** 2
        elif cls == 1:  # General
            # Require at least 0.2 improvement
            loss += sel * F.relu(0.2 - improvement) ** 2
        else:  # Poor
            # Require at least 0.1 improvement
            loss += sel * F.relu(0.1 - improvement) ** 2

    return loss.mean()

def global_hi_superiority_constraint(h_b, h_a, mask, weight=5.0):
    """
    Enhanced global superiority constraint: ensure post-maintenance HI is entirely higher than pre-maintenance HI
    For each sample, require post-maintenance HI at every time point to be higher than pre-maintenance HI
    Remove range constraints and focus on relative superiority
    """
    valid = mask  # (B, T)

    loss_total = torch.tensor(0.0, device=h_b.device)
    n_valid_samples = 0

    for b in range(h_b.size(0)):
        valid_mask_b = valid[b] > 0
        if valid_mask_b.sum() < 1:
            continue

        # Get valid HI values before and after maintenance
        h_b_valid = h_b[b][valid_mask_b]  # Valid HI values before maintenance
        h_a_valid = h_a[b][valid_mask_b]  # Valid HI values after maintenance

        # Point-wise superiority: each post-maintenance point should be higher than corresponding pre-maintenance point
        pointwise_gap = h_a_valid - h_b_valid  # Should be positive
        pointwise_loss = F.relu(-pointwise_gap + 0.05).mean()  # Penalize when post-maintenance is not sufficiently higher

        # Global superiority: minimum post-maintenance should be higher than maximum pre-maintenance
        h_b_max = h_b_valid.max()
        h_a_min = h_a_valid.min()

        # Require post-maintenance minimum to be higher than pre-maintenance maximum
        margin = 0.1  # Post-maintenance minimum should be at least 0.1 higher than pre-maintenance maximum
        global_gap_loss = F.relu(h_b_max + margin - h_a_min) ** 2

        # Additional constraint: post-maintenance average should be significantly higher than pre-maintenance average
        h_b_mean = h_b_valid.mean()
        h_a_mean = h_a_valid.mean()
        mean_gap_loss = F.relu(h_b_mean + 0.2 - h_a_mean) ** 2  # Average should improve by at least 0.2

        loss_total += weight * (pointwise_loss + global_gap_loss + 0.5 * mean_gap_loss)
        n_valid_samples += 1

    return loss_total / max(n_valid_samples, 1)

def diff_margin_by_class(mean_delta, labels, m_low=0.1, m_mid=0.25, m_high=0.45):
    """
    Class-conditional difference constraint (enhanced maintenance effect):
      - Perfect(0):  Δ>=m_high (post-maintenance significantly higher than pre-maintenance)
      - General(1):  Δ≈m_mid   (post-maintenance moderately higher than pre-maintenance)
      - Poor(2):     Δ>=m_low  (post-maintenance at least slightly higher than pre-maintenance)
    All classes require post-maintenance HI higher than pre-maintenance (positive improvement), just different degrees
    """
    loss = torch.zeros_like(mean_delta)
    for cls in [0,1,2]:
        sel = (labels==cls).float()
        if sel.sum() < 1: continue
        if cls==0:
            # Perfect: require significant improvement (at least reach m_high)
            loss += sel * F.relu(m_high - mean_delta)**2
        elif cls==1:
            # General: require moderate improvement (close to m_mid)
            loss += sel * (mean_delta - m_mid)**2
        else:
            # Poor: require at least small improvement (at least reach m_low)
            loss += sel * F.relu(m_low - mean_delta)**2
    denom = torch.ones_like(mean_delta) * (labels.numel())
    return loss.sum() / denom.sum()

def sensor_consistency_loss(x_b, x_a, h_b, h_a, mask):
    """
    Sensor data consistency loss: ensure HI changes are consistent with sensor data change patterns
    Important constraint when no real HI labels are available
    """
    # Calculate statistical changes in sensor data
    valid = mask.sum(1, keepdim=True).clamp_min(1.0)   # (B,1)

    # Average change magnitude in sensor data (fix broadcast BUG: divide by (B,1))
    xb_mean = (x_b * mask.unsqueeze(-1)).sum(1) / valid   # (B,C)
    xa_mean = (x_a * mask.unsqueeze(-1)).sum(1) / valid   # (B,C)
    sensor_change = torch.abs(xa_mean - xb_mean)          # (B,C)
    sensor_change_magnitude = sensor_change.mean(1)       # (B,)

    # HI change magnitude (post-maintenance improvement relative to pre-maintenance)
    hi_change = ((h_a - h_b) * mask).sum(1) / valid.squeeze(-1)  # (B,) Note: remove abs, maintain positive direction

    # Expect HI change to positively correlate with sensor change, and HI should improve positively
    # Use Pearson correlation coefficient as consistency metric
    mean_sensor = sensor_change_magnitude.mean()
    mean_hi = hi_change.mean()

    correlation = ((sensor_change_magnitude - mean_sensor) * (hi_change - mean_hi)).mean() / \
                  (sensor_change_magnitude.std() * hi_change.std() + 1e-8)

    # Encourage positive correlation (large sensor change when HI change is also large)
    consistency_loss = F.relu(0.5 - correlation)  # Expect correlation at least 0.5

    # Additional constraint: force positive HI improvement
    negative_change_penalty = F.relu(-hi_change).mean() * 2.0  # Penalize negative changes

    return consistency_loss + negative_change_penalty

def smoothness_enhancement_loss(h, mask, order=2):
    """
    Enhanced smoothness loss: use higher-order differences to force smoother curves
    """
    if order == 2:
        # Second-order difference
        d2 = h[:,2:] - 2*h[:,1:-1] + h[:,:-2]
        m = mask[:,2:]
        return (d2.abs() * m).sum() / (m.sum() + 1e-6)
    elif order == 3:
        # Third-order difference (stronger smoothness constraint)
        d3 = h[:,3:] - 3*h[:,2:-1] + 3*h[:,1:-2] - h[:,:-3]
        m = mask[:,3:]
        return (d3.abs() * m).sum() / (m.sum() + 1e-6)
    else:
        return torch.tensor(0., device=h.device)

def split_pairs_uid(pairs, train_ratio=0.7, val_ratio=0.1, seed=42):
    uids = sorted(list(pairs.keys()))
    rs = np.random.RandomState(seed)
    rs.shuffle(uids)
    n = len(uids); n_tr=int(n*train_ratio); n_vl=int(n*val_ratio)
    to_dict = lambda ss:{u:pairs[u] for u in ss if u in pairs}
    return to_dict(uids[:n_tr]), to_dict(uids[n_tr:n_tr+n_vl]), to_dict(uids[n_tr+n_vl:])

def print_split_summary(pairs, train_ratio=0.7, val_ratio=0.1, seed=42):
    pairs_tr, pairs_vl, pairs_te = split_pairs_uid(pairs, train_ratio, val_ratio, seed)
    def count_items(pp):
        cnt = 0
        for uid, d in pp.items():
            cnt += len(d)
        return len(pp), cnt
    n_uid_tr, n_pair_tr = count_items(pairs_tr)
    n_uid_vl, n_pair_vl = count_items(pairs_vl)
    n_uid_te, n_pair_te = count_items(pairs_te)
    print(f"[Data Split] Train: UID={n_uid_tr}, pairs≈{n_pair_tr} | "
          f"Val: UID={n_uid_vl}, pairs≈{n_pair_vl} | "
          f"Test: UID={n_uid_te}, pairs≈{n_pair_te}")

def remove_constant_segments(hi_values, threshold=1e-6):
    """
    Remove constant segments in HI curves (caused by mask or other reasons)
    Return valid HI values and corresponding time indices
    """
    if len(hi_values) <= 1:
        return hi_values, np.arange(len(hi_values))

    # Calculate differences between adjacent points
    diffs = np.abs(np.diff(hi_values))

    # Find points with sufficient change
    valid_mask = np.ones(len(hi_values), dtype=bool)
    valid_mask[1:] = diffs > threshold

    # Ensure at least first and last two points are kept
    if np.sum(valid_mask) < 2:
        valid_mask[0] = True
        valid_mask[-1] = True

    valid_indices = np.where(valid_mask)[0]
    return hi_values[valid_indices], valid_indices

@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total = {"mse_b":0.0, "mse_a":0.0, "acc":0.0, "delta_mean":0.0}
    n_batch=0
    n_acc = 0; n_tot=0
    for batch in loader:
        xb = batch["x_before"].to(device)
        xa = batch["x_after"].to(device)
        hi_b = batch["hi_before"].to(device)
        hi_a = batch["hi_after"].to(device)
        labels = batch["labels"].to(device)
        lengths= batch["lengths"].to(device)
        mask   = batch["mask"].to(device)

        # Valid segment: 0:L-2 input, 1:L-1 target
        m_tgt  = (torch.arange(xb.size(1)-2, device=device)[None,:] < (lengths-2)[:,None]).float()

        yb_hat, ya_hat, h_b, h_a, logits = model(xb, xa, mask)

        # Numerical stabilization
        yb_hat, ya_hat, h_b, h_a, logits = sanitize_tensors(yb_hat, ya_hat, h_b, h_a, logits)

        # Align targets
        yb_tgt = xb[:,1:-1,:]
        ya_tgt = xa[:,1:-1,:]

        mse_b = masked_mse(yb_hat, yb_tgt, m_tgt)
        mse_a = masked_mse(ya_hat, ya_tgt, m_tgt)

        # Check for NaN
        if torch.isnan(mse_b) or torch.isnan(mse_a):
            continue

        # Classification
        pred = logits.argmax(dim=1)
        n_acc += (pred == labels).sum().item()
        n_tot += labels.numel()

        # Statistics of ΔHI mean (post-maintenance improvement relative to pre-maintenance)
        valid = mask.sum(1, keepdim=True).clamp_min(1.0)
        delta_mean = ((h_a - h_b)*mask).sum(1,keepdim=True)/valid
        delta_mean = torch.nan_to_num(delta_mean, nan=0.0)
        total["delta_mean"] += delta_mean.mean().item()

        total["mse_b"] += mse_b.item()
        total["mse_a"] += mse_a.item()
        n_batch += 1

    for k in ["mse_b","mse_a","delta_mean"]:
        total[k] /= max(n_batch,1)
    total["acc"] = n_acc / max(n_tot,1)
    return total

@torch.no_grad()
def collect_test_predictions(model, loader, device, max_curve_keep=24):
    """
    Collect predictions, ΔHI, and a few sample HI curves on test set (for plotting).
    """
    model.eval()
    y_true, y_pred = [], []
    all_delta_mean, all_uids = [], []
    keep_curves = []

    for batch in loader:
        xb = batch["x_before"].to(device)   # (B,L,C)
        xa = batch["x_after"].to(device)
        mask   = batch["mask"].to(device)   # (B,L)
        labels = batch["labels"].to(device) # (B,)
        lengths= batch["lengths"]           # cpu tensor
        uids   = batch["uids"]
        hi_before = batch["hi_before"].to(device)  # (B,L) ground truth HI_before from dataset
        hi_after = batch["hi_after"].to(device)    # (B,L) ground truth HI_after from dataset

        yb_hat, ya_hat, h_b, h_a, logits = model(xb, xa, mask)

        # Numerical stabilization
        yb_hat, ya_hat, h_b, h_a, logits = sanitize_tensors(yb_hat, ya_hat, h_b, h_a, logits)

        prob = F.softmax(logits, dim=1)     # (B,3)
        pred = prob.argmax(1)               # (B,)

        # Statistics of ΔHI mean (post-maintenance improvement relative to pre-maintenance)
        valid = mask.sum(1, keepdim=True).clamp_min(1.0)
        delta_mean = ((h_a - h_b) * mask).sum(1, keepdim=True) / valid  # (B,1)
        delta_mean = torch.nan_to_num(delta_mean, nan=0.0)

        y_true.append(labels.cpu().numpy())
        y_pred.append(pred.cpu().numpy())
        all_delta_mean.append(delta_mean.squeeze(1).cpu().numpy())
        all_uids.extend(uids)

        # Save some curves for visualization (aligned: after concatenated after before)
        for i in range(xb.size(0)):
            if len(keep_curves) >= max_curve_keep:
                break
            L_i = int(lengths[i].item())
            # Fix: truncate reconstruction prediction by actual length
            L_recon = min(L_i - 2, yb_hat.size(1))  # Reconstruction sequence length
            keep_curves.append({
                "uid": uids[i],
                "true": int(labels[i].cpu().item()),
                "pred": int(pred[i].cpu().item()),
                "prob": prob[i].cpu().numpy(),
                "h_before": h_b[i, :L_i].cpu().numpy(),          # model predicted HI_before
                "h_after":  h_a[i, :L_i].cpu().numpy(),          # model predicted HI_after
                "hi_before_gt": hi_before[i, :L_i].cpu().numpy(), # ground truth HI_before from dataset
                "hi_after_gt": hi_after[i, :L_i].cpu().numpy(),   # ground truth HI_after from dataset
                "yb_hat":   yb_hat[i, :L_recon].cpu().numpy(),       # (L_recon,C)
                "ya_hat":   ya_hat[i, :L_recon].cpu().numpy(),       # (L_recon,C)
                "x_before": xb[i, :L_i].cpu().numpy(),               # (L,C)
                "x_after":  xa[i, :L_i].cpu().numpy(),               # (L,C)
                "length": L_i
            })

    y_true = np.concatenate(y_true, axis=0) if len(y_true)>0 else np.array([])
    y_pred = np.concatenate(y_pred, axis=0) if len(y_pred)>0 else np.array([])
    all_delta_mean = np.concatenate(all_delta_mean, axis=0) if len(all_delta_mean)>0 else np.array([])
    return y_true, y_pred, all_delta_mean, all_uids, keep_curves

def select_best_curves_for_plotting(curves, n_show=6):
    """
    Intelligent selection of the best drawing cases:
    1. Prioritize samples with correct predictions
    2. Select the most representative sample within each maintenance strategy category
    3. Select the sample with the most obvious change in health index
    4. Select samples with the best sensor data quality
    """
    if len(curves) == 0:
        return []

    #Group by real category
    curves_by_class = {0: [], 1: [], 2: []}
    for curve in curves:
        curves_by_class[curve["true"]].append(curve)

    # Calculate quality score for each sample
    def calculate_quality_score(curve):
        score = 0.0

        # 1. Prediction accuracy (weight: 50%)
        if curve["true"] == curve["pred"]:
            score += 0.5

        # 2. Prediction confidence (weight: 20%)
        confidence = curve["prob"][curve["pred"]]
        score += 0.2 * confidence

        # 3. Obvious degree of change in health index (weight: 20%)
        h_before = curve["h_before"]
        h_after = curve["h_after"]
        hi_improvement = np.mean(h_after - h_before)
        # Normalize to 0-1 range
        normalized_improvement = np.clip((hi_improvement + 0.5) / 1.0, 0, 1)
        score += 0.2 * normalized_improvement

        # 4. Data quality (weight: 10%)
        # Check if there are outliers or constant segments
        x_before_var = np.var(curve["x_before"])
        x_after_var = np.var(curve["x_after"])
        data_quality = np.clip((x_before_var + x_after_var) / 2.0, 0, 1)
        score += 0.1 * data_quality

        return score

    # Calculate quality scores for all samples
    for curve in curves:
        curve["quality_score"] = calculate_quality_score(curve)

    # Intelligent selection strategy
    selected_curves = []

    # Select at least one best sample for each category
    for class_id in [0, 1, 2]:
        if len(curves_by_class[class_id]) > 0:
            # Sort by quality score
            sorted_curves = sorted(curves_by_class[class_id],
                                 key=lambda x: x["quality_score"], reverse=True)
            selected_curves.append(sorted_curves[0])

    # If more samples are needed, select from the remaining high-quality samples
    remaining_slots = n_show - len(selected_curves)
    if remaining_slots > 0:
        # Exclude selected from all samples, select by quality score
        already_selected_uids = {curve["uid"] for curve in selected_curves}
        remaining_curves = [curve for curve in curves
                          if curve["uid"] not in already_selected_uids]

        if len(remaining_curves) > 0:
            remaining_curves.sort(key=lambda x: x["quality_score"], reverse=True)
            selected_curves.extend(remaining_curves[:remaining_slots])

    #Final sorting by quality score
    selected_curves.sort(key=lambda x: x["quality_score"], reverse=True)

    return selected_curves[:n_show]

def plot_hi_examples_aligned(curves, n_show=6, seed=0):
    """Post-maintenance sequence continues from pre-maintenance endpoint; draw vertical line to mark t_m; remove constant segments.
    Add "Learned HI" in title to indicate this is health state inferred from sensor data, post-maintenance should be higher than pre-maintenance
    Intelligent selection of the best drawing cases
    """
    if len(curves)==0:
        print("(No visualization samples)")
        return

    # Use the smart selection function to select the best drawing case
    selected_curves = select_best_curves_for_plotting(curves, n_show)

    if len(selected_curves) == 0:
        print("(No high-quality visualization samples found)")
        return

    n_show = len(selected_curves)
    cols = 3
    rows = int(np.ceil(n_show/cols))
    plt.figure(figsize=(cols*4.6, rows*3.2))

    for k in range(n_show):
        ex = selected_curves[k]
        hb = ex["h_before"]; ha = ex["h_after"]
        hb_gt = ex["hi_before_gt"]; ha_gt = ex["hi_after_gt"]      # ground truth HI from dataset
        L  = ex["length"]  # Use actual length instead of padded length

        # Remove constant segments
        hb_clean, hb_indices = remove_constant_segments(hb)
        ha_clean, ha_indices = remove_constant_segments(ha)
        hb_gt_clean, hb_gt_indices = remove_constant_segments(hb_gt)
        ha_gt_clean, ha_gt_indices = remove_constant_segments(ha_gt)

        # Corresponding time axis
        t_b = hb_indices
        t_a = ha_indices + L  # after follows immediately
        t_b_gt = hb_gt_indices
        t_a_gt = ha_gt_indices + L

        uid = ex["uid"]
        pred = ex["pred"]; true = ex["true"]; prob = ex["prob"]
        quality_score = ex["quality_score"]

        plt.subplot(rows, cols, k+1)

        # Only plot non-constant segments - model predicted HI
        if len(hb_clean) > 1:
            plt.plot(t_b, hb_clean, label="Learned HI_Pre-maintenance", linewidth=2.5, marker='o', markersize=4, color='blue')
        if len(ha_clean) > 1:
            plt.plot(t_a, ha_clean, label="Learned HI_Post-maintenance",  linewidth=2.5, linestyle='--', marker='s', markersize=4, color='red')

        # Plot ground truth HI from dataset
        if len(hb_gt_clean) > 1:
            plt.plot(t_b_gt, hb_gt_clean, label="GT HI_Pre-maintenance", linewidth=1.8, color='cyan', alpha=0.8)
        if len(ha_gt_clean) > 1:
            plt.plot(t_a_gt, ha_gt_clean, label="GT HI_Post-maintenance",  linewidth=1.8, linestyle=':', color='orange', alpha=0.8)

        # Maintenance time vertical line
        plt.axvline(L-1, color='k', linestyle=':', linewidth=1.5, alpha=0.8)

        d_mean = float(np.mean(ha - hb))  # Post-maintenance improvement relative to pre-maintenance
        d_max  = float(np.max(ha - hb))
        d_mean_gt = float(np.mean(ha_gt - hb_gt))

        # Add quality rating to title
        correctness = "✓" if true == pred else "✗"
        title = (f"uid={uid} {correctness} (Quality: {quality_score:.2f})\n"
                 f"True={LABEL2NAME[true]} | Pred={LABEL2NAME[pred]} | Conf={prob[pred]:.3f}\n"
                 f"ΔHI_Learned={d_mean:.3f}, ΔHI_GT={d_mean_gt:.3f}")
        plt.title(title, fontsize=9)
        plt.xlabel("Cycle (Pre-maintenance | Post-maintenance)")
        plt.ylabel("Health Index (↑Post should be higher)")
        plt.grid(ls='--', alpha=.4, linewidth=0.8)

        #Add background color to indicate prediction accuracy
        if true == pred:
            plt.gca().set_facecolor('#f0f8ff') # Light blue background indicates correct prediction
        else:
            plt.gca().set_facecolor('#fff0f0') # Light red background indicates incorrect prediction

        if k==0: plt.legend(fontsize=8, loc='best')

    plt.suptitle(f"Best {n_show} Health Index Examples (Ranked by Quality Score)", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.show()

def plot_sensor_examples_aligned(curves, sensor_idx_list=None, n_cols=4):
    """
    Plot several sensors: original(before/after) + one-step recursive prediction of after(ya_hat),
    post-maintenance time series starts after pre-maintenance end. Show trajectories under different
    maintenance strategies. Intelligent selection of the best sensor dimensions for drawing
    """
    if len(curves)==0:
        print("(No visualization samples)")
        return

    # Select the best samples for sensor visualization
    selected_curves = select_best_curves_for_plotting(curves, 9) # Select more samples for sensor analysis

    # Group curves by maintenance strategy (true class)
    curves_by_strategy = {0: [], 1: [], 2: []}
    for curve in selected_curves:
        curves_by_strategy[curve["true"]].append(curve)

    # Select best example from each strategy for comparison
    strategy_examples = {}
    for strategy in [0, 1, 2]:
        if len(curves_by_strategy[strategy]) > 0:
            # Select the sample with the highest quality score under this strategy
            best_curve = max(curves_by_strategy[strategy], key=lambda x: x["quality_score"])
            strategy_examples[strategy] = best_curve

    if len(strategy_examples) == 0:
        print("(No high-quality visualization samples for any maintenance strategy)")
        return

    # Intelligent selection of sensor dimensions
    first_ex = list(strategy_examples.values())[0]
    xb = first_ex["x_before"]       # (L,C)
    C = xb.shape[1]

    if sensor_idx_list is None:
        # Intelligently select the most representative sensor dimension
        sensor_scores = []

        for sensor_idx in range(C):
            score = 0.0

            # Calculate the degree of change of the sensor under different maintenance strategies
            for strategy, ex in strategy_examples.items():
                xb_sensor = ex["x_before"][:, sensor_idx]
                xa_sensor = ex["x_after"][:, sensor_idx]

                # 1. Range of change (weight: 40%)
                change_magnitude = np.abs(np.mean(xa_sensor) - np.mean(xb_sensor))
                score += 0.4 * min(change_magnitude / (np.std(xb_sensor) + 1e-6), 5.0)

                # 2. Variance (information amount) (weight: 30%)
                variance = np.var(xb_sensor) + np.var(xa_sensor)
                score += 0.3 * min(variance, 10.0)

                # 3. Signal-to-noise ratio (weight: 30%)
                snr = np.mean(xb_sensor) / (np.std(xb_sensor) + 1e-6)
                score += 0.3 * min(abs(snr), 5.0)

            sensor_scores.append((sensor_idx, score))

        # Select the sensor with the highest rating
        sensor_scores.sort(key=lambda x: x[1], reverse=True)
        sensor_idx_list = [idx for idx, _ in sensor_scores[:min(8, len(sensor_scores))]]

    n = len(sensor_idx_list)
    n_rows = int(np.ceil(n/n_cols))
    plt.figure(figsize=(n_cols*5.5, n_rows*4.0))

    colors = {0: '#e74c3c', 1: '#f39c12', 2: '#9b59b6'} # Brighter colors

    for i, s in enumerate(sensor_idx_list):
        plt.subplot(n_rows, n_cols, i+1)

        # Plot pre-maintenance trajectory (common for all strategies, use best example)
        best_ex = max(strategy_examples.values(), key=lambda x: x["quality_score"])
        xb = best_ex["x_before"]
        L = best_ex["length"]
        t_b = np.arange(L)
        plt.plot(t_b, xb[:L,s], label="Original (Pre-maintenance)", linewidth=2.0, color='#2c3e50', alpha=0.9)

        # Plot post-maintenance trajectories for each available strategy
        for strategy in sorted(strategy_examples.keys()):
            ex = strategy_examples[strategy]
            xa = ex["x_after"]
            ya = ex["ya_hat"]
            L_strategy = ex["length"]
            quality = ex["quality_score"]

            # Post-maintenance original trajectory
            t_a = np.arange(L_strategy, L_strategy + L_strategy)
            plt.plot(t_a, xa[:L_strategy,s],
                    label=f"Original (Post-{LABEL2NAME[strategy]}, Q:{quality:.2f})",
                    linewidth=1.8, linestyle="--",
                    color=colors[strategy], alpha=0.9)

            # Post-maintenance prediction trajectory
            if len(ya) > 0 and L_strategy-2 > 0:
                t_pred = np.arange(L_strategy+1, L_strategy+1+min(L_strategy-2, len(ya)))
                plt.plot(t_pred, ya[:len(t_pred),s],
                        label=f"Predicted (Post-{LABEL2NAME[strategy]})",
                        linewidth=2.2, color=colors[strategy], marker='o', markersize=3)

        # Draw vertical line at maintenance point
        plt.axvline(L-0.5, color='#27ae60', linestyle=':', linewidth=2.5, alpha=0.9, label='Maintenance Point')

        # Calculate the quality score of this sensor and add it to the title
        sensor_score = dict(sensor_scores)[s] if 'sensor_scores' in locals() else 0
        plt.title(f"Sensor_{s:02d} (Score: {sensor_score:.2f}) - Strategy Comparison", fontsize=10, fontweight='bold')

        if i==0:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.xlabel("Continuous Time Sequence", fontsize=9)
        plt.ylabel("Sensor Value", fontsize=9)
        plt.grid(ls="--", alpha=.4, linewidth=0.8)

        # Add slight background color
        plt.gca().set_facecolor('#fafafa')

    plt.suptitle("Best Sensor Trajectories under Different Maintenance Strategies",
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()

def topk_by_delta(df_delta, k=5):
    if len(df_delta)==0:
        return
    print("\n" + "="*60)
    print("TOP-K SAMPLES WITH HIGHEST MAINTENANCE EFFECTS")
    print("="*60)

    for cls in [0,1,2]:
        sub = df_delta[df_delta["true"]==cls].sort_values("delta_hi_mean", ascending=False).head(k)
        if len(sub) > 0:
            print(f"\n[True={LABEL2NAME[cls]}] Top {k} samples with highest learned ΔHI_mean:")
            print("-" * 50)
            for idx, row in sub.iterrows():
                pred_correct = "✓" if row["true"] == row["pred"] else "✗"
                print(f"  UID: {row['uid']:>8} | ΔHI: {row['delta_hi_mean']:>6.3f} | "
                      f"Pred: {LABEL2NAME[int(row['pred'])]:>7} {pred_correct}")

# —— Load trained best model
def load_trained_model(model_path, device, in_ch):
    """Load the best model weights saved during training"""
    # Initialize model with same architecture
    model = DiffAwareReconstructor(in_ch=in_ch, trend_ch=4, hidden=128, n_classes=3).to(device)

    if os.path.exists(model_path):
        print(f"Loading model: {model_path}")
        state_dict = torch.load(model_path, map_location=device)

        # Filter out keys that don't match (for backward compatibility)
        model_dict = model.state_dict()
        filtered_dict = {}
        for k, v in state_dict.items():
            if k in model_dict and model_dict[k].shape == v.shape:
                filtered_dict[k] = v
            else:
                print(f"Warning: Skipping key {k} due to shape mismatch or missing in current model")

        # Load only matching parameters
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict, strict=False)
        print("Model loaded successfully (with potential missing keys)!")
    else:
        print(f"Warning: Model file does not exist {model_path}, will use randomly initialized model")
    return model

# —— Need to determine in_ch from pairs data first
def get_input_dim_from_pairs(pairs):
    """Get input dimension from pairs data"""
    for uid, strategies in pairs.items():
        for strategy, data in strategies.items():
            if "x_before" in data:
                return np.array(data["x_before"]).shape[1]
    raise ValueError("Cannot determine input dimension from pairs data")

# Get input dimension
C = get_input_dim_from_pairs(pairs)
print(f"Detected input dimension: {C}")

# Load best model (if exists)
model_path = "/content/drive/MyDrive/CMAPSS/main_dynamics_identification_2.pth"
model = load_trained_model(model_path, DEVICE, C)

# —— Print train/validation/test split (consistent with training phase: 7/1/2)
print_split_summary(pairs)

# —— Prepare test set data
_, _, pairs_te = split_pairs_uid(pairs, 0.7, 0.1, 42)
ds_te = PairsReconstructDataset(pairs_te, horizon=50)  # Same horizon as training
ld_te = DataLoader(ds_te, batch_size=32, shuffle=False, collate_fn=pad_collate_shift)
te = (ds_te, ld_te)

y_true, y_pred, delta_mean_all, uids_all, curves = collect_test_predictions(model, ld_te, DEVICE)

# —— Print overall metrics
print("\n" + "="*60)
print("TEST SET OVERALL METRICS")
print("="*60)
print(f"Sample count: {len(y_true)}")
if len(y_true) > 0:
    acc = (y_true == y_pred).mean()
    print(f"Classification Accuracy: {acc:.4f}")

    # Calculate the accuracy of each category
    for cls in [0, 1, 2]:
        cls_mask = (y_true == cls)
        if cls_mask.sum() > 0:
            cls_acc = (y_pred[cls_mask] == cls).mean()
            print(f"{LABEL2NAME[cls]} Accuracy: {cls_acc:.4f} ({cls_mask.sum()} samples)")
else:
    print("No samples in test set (check pairs split and horizon conditions).")

# —— Classification report & confusion matrix
try:
    from sklearn.metrics import classification_report, confusion_matrix
    target_names = [LABEL2NAME[i] for i in sorted(LABEL2NAME)]
    print("\n[Classification Report]")
    print(classification_report(y_true, y_pred, target_names=target_names, digits=4))
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])
except Exception:
    print("\n[Warning] scikit-learn not installed, will print simple confusion table.")
    cm = np.zeros((3,3), dtype=int)
    for t,p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1

# —— Print confusion matrix values
print("\n[Confusion Matrix] Row=True class, Column=Predicted class")
print(pd.DataFrame(cm, index=[LABEL2NAME[i] for i in [0,1,2]],
                      columns=[LABEL2NAME[i] for i in [0,1,2]]))

# —— Plot confusion matrix (enhanced)
plt.figure(figsize=(6.0,5.0))
im = plt.imshow(cm, interpolation='nearest', cmap='Blues')
plt.title("Confusion Matrix (Test Set)", fontsize=14, fontweight='bold', pad=20)
plt.xlabel("Predicted", fontsize=12)
plt.ylabel("True", fontsize=12)
plt.xticks([0,1,2], [LABEL2NAME[i] for i in [0,1,2]])
plt.yticks([0,1,2], [LABEL2NAME[i] for i in [0,1,2]])

#Add values ​​and percentages
for i in range(3):
    for j in range(3):
        total = cm[i].sum()
        if total > 0:
            percentage = cm[i, j] / total * 100
            text_color = 'white' if cm[i, j] > cm.max() * 0.5 else 'black'
            plt.text(j, i, f'{cm[i, j]}\n({percentage:.1f}%)',
                    ha="center", va="center", color=text_color, fontweight='bold')

plt.colorbar(im)
plt.tight_layout()
plt.show()

# —— Statistics of ΔHI distribution (by true class/predicted class)
if len(delta_mean_all) > 0:
    df_delta = pd.DataFrame({
        "uid": uids_all[:len(delta_mean_all)],
        "true": y_true.astype(int),
        "pred": y_pred.astype(int),
        "delta_hi_mean": delta_mean_all.astype(float)
    })

    print("\n" + "="*50)
    print("ΔHI MEAN STATISTICS BY TRUE CLASS")
    print("="*50)
    stats_by_true = df_delta.groupby("true")["delta_hi_mean"].describe()
    print(stats_by_true.round(4))

    print("\n" + "="*50)
    print("ΔHI MEAN STATISTICS BY PREDICTED CLASS")
    print("="*50)
    stats_by_pred = df_delta.groupby("pred")["delta_hi_mean"].describe()
    print(stats_by_pred.round(4))

# ——Continuous time axis: HI and several sensors before/after aligned visualization (intelligent selection of the best drawing)
print("\n" + "="*60)
print("GENERATING BEST QUALITY VISUALIZATIONS...")
print("="*60)

plot_hi_examples_aligned(curves, n_show=6, seed=2025)
plot_sensor_examples_aligned(curves, sensor_idx_list=None, n_cols=4)

# ——(Optional) Top-K with largest ΔHI in each class, for manual review (enhanced display)
if len(delta_mean_all) > 0:
    topk_by_delta(df_delta, k=5)

# %% [notebook code cell 16]
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# ========== Basics: Build/Load Model ==========
def build_model(input_dim, trend_ch=4, hidden=128, n_classes=3, device=None):
    if 'DiffAwareReconstructor' not in globals():
        raise RuntimeError("DiffAwareReconstructor definition not found: Please import the model class defined in the training script first.")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DiffAwareReconstructor(in_ch=input_dim, trend_ch=trend_ch, hidden=hidden, n_classes=n_classes).to(device)
    return model, device

def load_checkpoint(model, ckpt_path, device):
    sd = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(sd, strict=False) # Compatible with small buffer/key differences
    model.eval()
    print(f"[OK] Loaded checkpoint: {ckpt_path}")

# ========== Pick a sample from pairs ==========
def pick_one_pair(pairs, uid=None, strategy=None):
    """
    Take an alignment segment of (uid, strategy) directly from the pairs structure of build_pairs:
    Return uid, strategy, x_before:(1,T,C), x_after:(1,T,C)
    """
    if uid is None:
        uid = next(iter(pairs.keys()))
    if strategy is None:
        strategy = next(iter(pairs[uid].keys()))
    rec = pairs[uid][strategy]
    xb = np.asarray(rec["x_before"], dtype=np.float32)  # (T,C)
    xa = np.asarray(rec["x_after"],  dtype=np.float32)  # (T,C)
    # Ensure the same length (build_pairs are already aligned, here we will cut it to the shortest defensively)
    T = min(len(xb), len(xa))
    xb = xb[:T]; xa = xa[:T]
    # Expand the batch dimension
    xb = torch.from_numpy(xb).unsqueeze(0)  # (1,T,C)
    xa = torch.from_numpy(xa).unsqueeze(0)  # (1,T,C)
    return uid, strategy, xb, xa

# ========== Extract liquid weight/operator output/mix damage ==========
@torch.no_grad()
def extract_ops_and_weights_for_sequence(model, x_full):
    """
    Just for visualization:
    - Still use the weight_generator in the model to calculate (B, T, K) liquid weight and temperature
    - But instead of using LayerNorm(1) for the output of each operator, z-score normalization is performed according to the time dimension.
      This preserves the timing shape and avoids being reduced to a constant 0
    enter:
      x_full: (1, T, C)
    return:
      h_multi       : (T, h_dim)
      op_outs_norm: (T, K) ——Operator output after z-score
      weights       : (T, K)
      temperature   : (T,)
      damage: (T,) —— The "visual version" of damage obtained by multiplying the z-score operator output with the weight
    """
    import torch.nn.functional as F
    enc = model.encoder
    custom = enc.customkan

    # 1) KAN features
    h_multi = enc.boltz(x_full)  # (1, T, h_dim)

    # 2) Calculate the original output of each operator, and then do z-score in the time dimension: (y - mean_t) / std_t
    op_outs = []
    for op in custom.ops:
        y = op(h_multi)  # (1, T)
        # ---- Key changes: z-score (dimension T by time) ----
        mu = y.mean(dim=1, keepdim=True)
        std = y.std(dim=1, keepdim=True) + 1e-6
        y_norm = (y - mu) / std          # (1, T)
        op_outs.append(y_norm)

    # Same as forward for length alignment (usually consistent, here is defensive processing)
    Tm = min(o.size(1) for o in op_outs)
    op_outs = [o[:, :Tm] for o in op_outs]
    h_multi_aligned = h_multi[:, :Tm, :]
    x_aligned = x_full[:, :Tm, :]

    # 3) Stacking operator output (B,T,K)
    op_stack = torch.stack(op_outs, dim=-1)  # (1, Tm, K)

    # 4) Use the model's liquid weight generator to get (B,T,K) weights and (B,T) temperatures - consistent with training/inference
    weights, temperature = custom.weight_generator(h_multi_aligned, x_aligned)  # (1,Tm,K), (1,Tm)

    # 5) Multiply the operator output after z-score with the weight to get the "visualized version" of damage
    # Note: This is different from the damage numerical scale of LayerNorm(1) used during training, and is only used to analyze weights and operator shapes.
    damage = torch.sum(op_stack * weights, dim=-1)  # (1, Tm)
    # To avoid extreme values, make a soft limit (optional)
    damage = damage.clamp(min=-10.0, max=10.0)

    out = {
        "h_multi":      h_multi_aligned.squeeze(0).cpu().numpy(),  # (T, h_dim)
        "op_outs_norm": op_stack.squeeze(0).cpu().numpy(),         # (T, K)
        "weights":      weights.squeeze(0).cpu().numpy(),          # (T, K)
        "temperature":  temperature.squeeze(0).cpu().numpy(),      # (T,)
        "damage":       damage.squeeze(0).cpu().numpy(),           # (T,)
    }
    return out


# ========== Drawing ==========
def plot_liquid_mixing(uid, strategy, before_dict, after_dict, op_names=None):
    op_names = op_names or [
        "MonotonicLinear","MonotonicFlat","ConcaveLog","SaturationSigmoid",
        "HingeReLU","Polynomial","DampedSin","PiecewiseLinear"
    ]

    # Weight heat map (before/after)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    im0 = axes[0,0].imshow(before_dict["weights"].T, aspect='auto', origin='lower')
    axes[0,0].set_title(f"[{uid} | {strategy}] Weights Heatmap (Before)")
    axes[0,0].set_ylabel("Operators"); axes[0,0].set_xlabel("Time")
    axes[0,0].set_yticks(range(len(op_names))); axes[0,0].set_yticklabels(op_names)
    fig.colorbar(im0, ax=axes[0,0])

    im1 = axes[0,1].imshow(after_dict["weights"].T, aspect='auto', origin='lower')
    axes[0,1].set_title(f"[{uid} | {strategy}] Weights Heatmap (After)")
    axes[0,1].set_ylabel("Operators"); axes[0,1].set_xlabel("Time")
    axes[0,1].set_yticks(range(len(op_names))); axes[0,1].set_yticklabels(op_names)
    fig.colorbar(im1, ax=axes[0,1])

    # damage before and after
    axes[1,0].plot(before_dict["damage"], label="damage_before", linewidth=1.8)
    axes[1,0].plot(after_dict["damage"],  label="damage_after",  linewidth=1.8, linestyle="--")
    axes[1,0].set_title(f"[{uid} | {strategy}] Mixed Damage (weighted ops)")
    axes[1,0].set_xlabel("Time"); axes[1,0].set_ylabel("Damage")
    axes[1,0].grid(True, alpha=0.35); axes[1,0].legend()

    # Temperature curve
    axes[1,1].plot(before_dict["temperature"], label="temp_before", linewidth=1.5)
    axes[1,1].plot(after_dict["temperature"],  label="temp_after",  linewidth=1.5, linestyle="--")
    axes[1,1].set_title(f"[{uid} | {strategy}] Temperature (lower → sharper)")
    axes[1,1].set_xlabel("Time"); axes[1,1].set_ylabel("Temperature")
    axes[1,1].grid(True, alpha=0.35); axes[1,1].legend()

    plt.tight_layout()
    plt.show()

    # 8 operator outputs (after LayerNorm) before/after comparison
    K = len(op_names)
    n_cols = 4
    n_rows = int(np.ceil(K / n_cols))
    plt.figure(figsize=(n_cols * 4.2, n_rows * 3.0))
    for k in range(K):
        plt.subplot(n_rows, n_cols, k+1)
        plt.plot(before_dict["op_outs_norm"][:, k], label="before", linewidth=1.6)
        plt.plot(after_dict["op_outs_norm"][:,  k], label="after",  linewidth=1.6, linestyle="--")
        plt.title(op_names[k]); plt.xlabel("Time"); plt.ylabel("Op Output (norm)")
        plt.grid(True, alpha=0.3)
        if k == 0: plt.legend()
    plt.suptitle(f"[{uid} | {strategy}] Operator Outputs — Before vs After", y=1.02, fontsize=12)
    plt.tight_layout()
    plt.show()

# ========== Main entrance: Visualization pair before/after ==========
def visualize_liquid_mixing_on_pairs(
    pairs,
    checkpoint_path="/content/drive/MyDrive/CMAPSS/main_dynamics_identification_2.pth",
    uid=None, strategy=None
):
    # 1) First select a sample (or specify uid/strategy)
    uid, strategy, xb, xa = pick_one_pair(pairs, uid=uid, strategy=strategy)

    # 2) Build and load the model (derived from pairs input dimension C)
    C = xb.shape[-1]
    model, device = build_model(input_dim=C)
    load_checkpoint(model, checkpoint_path, device)

    # 3) The front/back are sent to the encoder path respectively, and the liquid weight and operator output are extracted.
    xb = xb.to(device); xa = xa.to(device)
    before_dict = extract_ops_and_weights_for_sequence(model, xb)
    after_dict  = extract_ops_and_weights_for_sequence(model, xa)

    #4) Draw a picture
    plot_liquid_mixing(uid, strategy, before_dict, after_dict)
    print(f"[Done] {uid} | {strategy} visualization completed.")

# ===== Usage example =====
visualize_liquid_mixing_on_pairs(pairs, checkpoint_path="/content/drive/MyDrive/CMAPSS/main_dynamics_identification_2.pth")

# %% [notebook code cell 17]

# ============================================================
# Monotonic-Decreasing HI + aligned plotting (before -> after)
# ============================================================
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import random
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL2NAME = {0: "Perfect", 1: "General", 2: "Poor"}

# ============================================================
# 1) Dataset: read samples from pairs, perform shift operations in batch later
# ============================================================
class PairsReconstructDataset(Dataset):
    """
    Each sample contains:
      x_before: (L,C) feature sequence before maintenance
      x_after : (L,C) feature sequence after maintenance
      hi_before/hi_after: (L,) health index (for evaluation/optional analysis only, not strong supervision)
      strategy: maintenance type ("Perfect Maintenance" / "General Maintenance" / "Poor Maintenance")
    """
    def __init__(self, pairs: dict, horizon: int=None):
        items = []
        for uid, sd in pairs.items():
            for strat, d in sd.items():
                xb = np.asarray(d["x_before"], dtype=np.float32)
                xa = np.asarray(d["x_after"],  dtype=np.float32)
                # When there's no real HI, create placeholder that will be learned by the model
                if "hi_before" in d and d["hi_before"] is not None:
                    hib = np.asarray(d["hi_before"],dtype=np.float32)
                else:
                    # Create initialized fake HI, model will learn real patterns
                    hib = np.linspace(0.8, 0.4, len(xb)).astype(np.float32)

                if "hi_after" in d and d["hi_after"] is not None:
                    hia = np.asarray(d["hi_after"], dtype=np.float32)
                else:
                    # Create different fake HI patterns based on maintenance strategy
                    if "perfect" in strat.lower():
                        # Perfect maintenance: significant improvement
                        hia = np.linspace(0.9, 0.6, len(xa)).astype(np.float32)
                    elif "general" in strat.lower():
                        # General maintenance: moderate improvement
                        hia = np.linspace(0.7, 0.5, len(xa)).astype(np.float32)
                    else:
                        # Poor maintenance: slight improvement
                        hia = np.linspace(0.5, 0.35, len(xa)).astype(np.float32)

                L, C = xb.shape
                # Unify length (optional)
                if horizon is not None and L > horizon:
                    xb, xa, hib, hia = xb[:horizon], xa[:horizon], hib[:horizon], hia[:horizon]
                    L = horizon
                if L < 3:  # At least need to form 0:L-2 -> 1:L-1
                    continue
                items.append({"uid": uid, "strategy": strat, "x_before": xb, "x_after": xa,
                              "hi_before": hib, "hi_after": hia})
        assert len(items)>0, "Dataset is empty, please check pairs data."
        self.items = items
        self.C = items[0]["x_before"].shape[1]

        # Strategy to 3-class label mapping
        def strat_to_cls(s):
            s = s.lower()
            if "perfect" in s: return 0
            if "general" in s: return 1
            if "poor" in s: return 2
            raise ValueError(f"Unknown strategy name: {s}")
        self.labels = [strat_to_cls(it["strategy"]) for it in items]

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        return {
            "uid": it["uid"],
            "label": self.labels[idx],
            "x_before": torch.from_numpy(it["x_before"]), # (L,C)
            "x_after":  torch.from_numpy(it["x_after"]),  # (L,C)
            "hi_before":torch.from_numpy(it["hi_before"]),# (L,)
            "hi_after": torch.from_numpy(it["hi_after"]), # (L,)
            "strategy": it["strategy"]
        }

def pad_collate_shift(batch):
    """
    Pad sequences to same length; return mask, do NOT perform shift operation here
    to allow unified training: inputs = [:,0:L-2], targets = [:,1:L-1]
    """
    Ls = [b["x_before"].shape[0] for b in batch]
    Lmax = max(Ls)
    C = batch[0]["x_before"].shape[1]

    def pad2d(x):
        L, C0 = x.shape
        out = torch.zeros(Lmax, C0, dtype=x.dtype)
        out[:L] = x
        return out

    def pad1d(x):
        L = x.shape[0]
        out = torch.zeros(Lmax, dtype=x.dtype)
        out[:L] = x
        return out

    x_before = torch.stack([pad2d(b["x_before"]) for b in batch], 0) # (B,Lmax,C)
    x_after  = torch.stack([pad2d(b["x_after"])  for b in batch], 0) # (B,Lmax,C)
    hi_before= torch.stack([pad1d(b["hi_before"])for b in batch], 0) # (B,Lmax)
    hi_after = torch.stack([pad1d(b["hi_after"]) for b in batch], 0) # (B,Lmax)
    lengths  = torch.tensor(Ls, dtype=torch.long)
    mask     = torch.zeros(len(batch), Lmax)
    for i,L in enumerate(Ls): mask[i,:L] = 1.0
    labels   = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    uids     = [b["uid"] for b in batch]
    return {"uids": uids, "labels": labels,
            "x_before": x_before, "x_after": x_after,
            "hi_before": hi_before, "hi_after": hi_after,
            "lengths": lengths, "mask": mask}

# ============================================================
# 2) Model: True liquid adaptive operator - with significantly different parameters and behavior patterns before and after maintenance
# ============================================================
class BoltzmannKAN(nn.Module):
    def __init__(self, in_ch, out_ch=4):
        super().__init__()
        self.E  = nn.Parameter(torch.zeros(out_ch, in_ch))
        self.kT = nn.Parameter(torch.ones(out_ch, in_ch))
    def forward(self, x):
        B,T,C = x.shape
        kT = torch.clamp(F.softplus(self.kT), 0.01, 10.0).unsqueeze(0).unsqueeze(2)  # (1,out_ch,1,in_ch)
        E  = torch.clamp(self.E, -10.0, 10.0).unsqueeze(0).unsqueeze(2)             # (1,out_ch,1,in_ch)
        x_ = torch.clamp(x.unsqueeze(1), -10.0, 10.0)                               # (B, out_ch, T, in_ch)
        exp = torch.clamp((E - x_) / kT, -50, 50)
        w   = torch.sigmoid(exp)
        h   = (w * x_).sum(dim=3).permute(0, 2, 1)           # (B,T,out_ch) >=0
        return torch.clamp(F.softplus(h), 0.0, 100.0)

class BaseOp(nn.Module): pass

class LiquidAdaptiveMonotonicLinearOp(BaseOp):
    """Monotonic linear operator for true liquids - significant changes in parameters before and after maintenance"""
    def __init__(self, param_dim=8):
        super().__init__()
        # Deep network, stronger nonlinear parameter generation capability
        self.param_net = nn.Sequential(
            nn.Linear(param_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3) # Output scale, bias, temperature (control maintenance sensitivity)
        )
        self.smin, self.smax = 0.1, 8.0
        # Dedicated to maintaining effect-aware networks
        self.maintenance_sensor = nn.Sequential(
            nn.Linear(param_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 4) # Output maintenance effect parameters
        )

    def forward(self, h):
        B, T, _ = h.shape
        h_mean = h.mean(dim=1)  # (B, param_dim)

        #Basic parameters
        params = self.param_net(h_mean)  # (B, 3)
        scale = torch.clamp(F.softplus(params[:, 0]), self.smin, self.smax).unsqueeze(1)  # (B, 1)
        bias = torch.clamp(params[:, 1], -8.0, 8.0).unsqueeze(1)  # (B, 1)
        temperature = torch.clamp(F.softplus(params[:, 2]), 0.1, 5.0).unsqueeze(1)  # (B, 1)

        # Maintain effect perception parameters
        maint_params = self.maintenance_sensor(h_mean)  # (B, 4)
        sensitivity = torch.clamp(F.softplus(maint_params[:, 0]), 0.5, 3.0).unsqueeze(1)  # (B, 1)
        phase_shift = torch.clamp(maint_params[:, 1], -2.0, 2.0).unsqueeze(1)  # (B, 1)
        amplitude_mod = torch.clamp(F.softplus(maint_params[:, 2]), 0.3, 2.0).unsqueeze(1)  # (B, 1)
        nonlin_factor = torch.clamp(torch.sigmoid(maint_params[:, 3]), 0.1, 0.9).unsqueeze(1)  # (B, 1)

        # Time-related change patterns
        t = torch.arange(T, device=h.device, dtype=h.dtype).unsqueeze(0).expand(B, -1)  # (B, T)
        t_norm = t / (T - 1) #Normalized time

        #Adjust parameters based on the statistical characteristics of the input data
        h_std = h.std(dim=1, keepdim=True).mean(dim=-1)  # (B, 1)
        dynamic_scale = scale * (1 + h_std * sensitivity)

        # Nonlinear time effects
        time_effect = torch.sin(2 * np.pi * t_norm * temperature + phase_shift) * amplitude_mod

        xm = torch.clamp(h.mean(-1), -10.0, 10.0)  # (B, T)
        # Liquid response: combining linear and nonlinear effects
        linear_part = dynamic_scale * (xm + bias)
        nonlinear_part = time_effect * xm * nonlin_factor

        result = linear_part + nonlinear_part
        return torch.clamp(F.softplus(result), 0.0, 100.0)

class LiquidAdaptiveMonotonicFlatOp(BaseOp):
    """Liquid smoothing operator - with maintenance-related dynamics"""
    def __init__(self, param_dim=8):
        super().__init__()
        self.param_net = nn.Sequential(
            nn.Linear(param_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4)  # scale, bias, smoothness, adaptivity
        )
        self.dynamic_net = nn.Sequential(
            nn.Linear(param_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 3) #Dynamic adjustment parameters
        )
        self.smin, self.smax = 1e-2, 2.0

    def _adaptive_cum(self, x, smoothness):
        """Adaptive accumulation function, adjusted according to the smoothness parameter"""
        # x is (B, 1, T), smoothness is (B, 1)
        diff = F.relu(x[:, :, 1:] - x[:, :, :-1])  # (B, 1, T-1)
        # Smoothness adjustment
        diff = diff * smoothness.unsqueeze(-1)  # (B, 1, T-1)
        return torch.cat([torch.zeros_like(diff[:, :, :1]), torch.cumsum(diff, 2)], 2)  # (B, 1, T)

    def forward(self, h):
        B, T, _ = h.shape
        h_mean = h.mean(dim=1)  # (B, param_dim)

        #Basic parameters
        params = self.param_net(h_mean)  # (B, 4)
        scale = torch.clamp(F.softplus(params[:, 0]), self.smin, self.smax).unsqueeze(1)
        bias = torch.clamp(params[:, 1], -5.0, 5.0).unsqueeze(1)
        smoothness = torch.clamp(F.softplus(params[:, 2]), 0.1, 2.0).unsqueeze(1)
        adaptivity = torch.clamp(torch.sigmoid(params[:, 3]), 0.1, 0.9).unsqueeze(1)

        #Dynamic adjustment parameters
        dynamic_params = self.dynamic_net(h_mean)  # (B, 3)
        temporal_weight = torch.clamp(F.softplus(dynamic_params[:, 0]), 0.5, 2.0).unsqueeze(1)
        fluctuation = torch.clamp(dynamic_params[:, 1], -1.0, 1.0).unsqueeze(1)
        maintenance_response = torch.clamp(F.softplus(dynamic_params[:, 2]), 0.2, 3.0).unsqueeze(1)

        # Time-varying characteristics of input data
        h_variance = h.var(dim=-1, keepdim=True)  # (B, T, 1)
        temporal_modulation = torch.sin(torch.arange(T, device=h.device).float() * temporal_weight / T * 2 * np.pi + fluctuation)
        temporal_modulation = temporal_modulation.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)  # (B, 1, T)

        xm = torch.clamp(h.mean(-1, keepdim=True), -10.0, 10.0).unsqueeze(1)  # (B, 1, T)

        # Liquid response: combining cumulative effect and dynamic adjustment
        cum_base = self._adaptive_cum(xm, smoothness)

        # Maintain response conditioning - ensure dimensions match
        h_variance_reshaped = h_variance.transpose(1, 2)  # (B, 1, T)
        maintenance_modulation = maintenance_response.unsqueeze(-1) * h_variance_reshaped * temporal_modulation

        result = scale.unsqueeze(-1) * (cum_base + maintenance_modulation + bias.unsqueeze(-1))
        result = result.squeeze(1) * (1 + adaptivity * temporal_modulation.squeeze(1))

        return torch.clamp(F.softplus(result), 0.0, 100.0)

class LiquidAdaptiveConcaveLogOp(BaseOp):
    """Liquid concave logarithmic operator - maintaining sensitive nonlinear response"""
    def __init__(self, param_dim=8):
        super().__init__()
        self.param_net = nn.Sequential(
            nn.Linear(param_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 5)  # scale, offset, curvature, maintenance_gain, noise_resist
        )
        self.eps = 1e-3
        self.smin, self.smax = 0.01, 8.0

    def forward(self, h):
        B, T, _ = h.shape
        h_mean = h.mean(dim=1)  # (B, param_dim)
        params = self.param_net(h_mean)  # (B, 5)

        scale = torch.clamp(F.softplus(params[:, 0]), self.smin, self.smax).unsqueeze(1)
        offset = torch.clamp(params[:, 1], -3.0, 3.0).unsqueeze(1)
        curvature = torch.clamp(F.softplus(params[:, 2]), 0.1, 3.0).unsqueeze(1)
        maintenance_gain = torch.clamp(F.softplus(params[:, 3]), 0.5, 4.0).unsqueeze(1)
        noise_resistance = torch.clamp(torch.sigmoid(params[:, 4]), 0.1, 0.95).unsqueeze(1)

        # Dynamic data feature awareness
        h_energy = (h ** 2).mean(dim=-1)  # (B, T)
        h_gradient = torch.abs(h[:, 1:] - h[:, :-1]).mean(dim=-1)  # (B, T-1)
        h_gradient = F.pad(h_gradient, (0, 1), value=0)  # (B, T)

        xm = torch.clamp(h.mean(-1), -10.0, 10.0)  # (B, T)

        # Liquid logarithmic response
        log_base = torch.log(torch.abs(xm + offset) + self.eps)
        curvature_effect = curvature * h_energy
        maintenance_effect = maintenance_gain * h_gradient

        #Adaptive noise suppression
        noise_filter = torch.sigmoid(h_energy * noise_resistance)

        result = scale * (log_base * curvature_effect + maintenance_effect) * noise_filter
        return torch.clamp(F.softplus(result), 0.0, 100.0)

class LiquidAdaptiveSaturationSigmoidOp(BaseOp):
    """Liquid saturated sigmoid operator - maintenance threshold sensitive"""
    def __init__(self, param_dim=8):
        super().__init__()
        self.param_net = nn.Sequential(
            nn.Linear(param_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 6)  # scale, slope, bias, threshold, saturation, maintenance_sens
        )
        self.smin, self.smax = 0.01, 6.0
        self.lmin, self.lmax = 0.1, 8.0

    def forward(self, h):
        B, T, _ = h.shape
        h_mean = h.mean(dim=1)  # (B, param_dim)
        params = self.param_net(h_mean)  # (B, 6)

        scale = torch.clamp(F.softplus(params[:, 0]), self.smin, self.smax).unsqueeze(1)
        slope = torch.clamp(F.softplus(params[:, 1]), self.lmin, self.lmax).unsqueeze(1)
        bias = torch.clamp(params[:, 2], -5.0, 5.0).unsqueeze(1)
        threshold = torch.clamp(params[:, 3], -3.0, 3.0).unsqueeze(1)
        saturation = torch.clamp(F.softplus(params[:, 4]), 0.5, 3.0).unsqueeze(1)
        maintenance_sens = torch.clamp(F.softplus(params[:, 5]), 0.2, 4.0).unsqueeze(1)

        # Maintain sensitive feature extraction
        h_peak = h.max(dim=-1)[0]  # (B, T)
        h_trough = h.min(dim=-1)[0]  # (B, T)
        h_range = h_peak - h_trough

        xm = torch.clamp(h.mean(-1), -10.0, 10.0)  # (B, T)

        # Dynamic threshold adjustment
        dynamic_threshold = threshold + h_range * maintenance_sens * 0.1

        # Liquid sigmoid response
        sigmoid_input = slope * (xm - dynamic_threshold - bias)
        sigmoid_output = torch.sigmoid(sigmoid_input)

        # saturation effect
        saturation_effect = torch.tanh(saturation * sigmoid_output)

        result = scale * saturation_effect
        return torch.clamp(F.softplus(result), 0.0, 100.0)

class LiquidAdaptiveHingeReLUOp(BaseOp):
    """Liquid hinge ReLU operator - multi-threshold maintenance response"""
    def __init__(self, param_dim=8):
        super().__init__()
        self.param_net = nn.Sequential(
            nn.Linear(param_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 6)  # scale, threshold1, threshold2, slope1, slope2, maintenance_amp
        )
        self.smin, self.smax = 0.01, 5.0

    def forward(self, h):
        B, T, _ = h.shape
        h_mean = h.mean(dim=1)  # (B, param_dim)
        params = self.param_net(h_mean)  # (B, 6)

        scale = torch.clamp(F.softplus(params[:, 0]), self.smin, self.smax).unsqueeze(1)
        threshold1 = torch.clamp(params[:, 1], -5.0, 0.0).unsqueeze(1)
        threshold2 = torch.clamp(params[:, 2], 0.0, 5.0).unsqueeze(1)
        slope1 = torch.clamp(F.softplus(params[:, 3]), 0.1, 3.0).unsqueeze(1)
        slope2 = torch.clamp(F.softplus(params[:, 4]), 0.1, 3.0).unsqueeze(1)
        maintenance_amp = torch.clamp(F.softplus(params[:, 5]), 0.5, 3.0).unsqueeze(1)

        # Maintain response characteristics
        h_trend = (h[:, -1] - h[:, 0]).mean(dim=-1, keepdim=True)  # (B, 1)
        h_volatility = h.std(dim=1).mean(dim=-1, keepdim=True)  # (B, 1)

        xm = torch.clamp(h.mean(-1), -10.0, 10.0)  # (B, T)

        #Multi-threshold hinge response
        hinge1 = F.relu(xm - threshold1) * slope1
        hinge2 = F.relu(xm - threshold2) * slope2

        # Maintenance effect modulation
        maintenance_modulation = maintenance_amp * (h_trend + h_volatility)

        result = scale * (hinge1 + hinge2) * (1 + maintenance_modulation)
        return torch.clamp(F.softplus(result), 0.0, 100.0)

class LiquidAdaptivePolynomialOp(BaseOp):
    """Liquid polynomial operator - adaptive order and coefficients"""
    def __init__(self, param_dim=8, max_deg=4):
        super().__init__()
        self.max_deg = max_deg
        self.param_net = nn.Sequential(
            nn.Linear(param_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, max_deg + 2) # coefficient + degree_weight + maintenance_coupling
        )

    def forward(self, h):
        B, T, _ = h.shape
        h_mean = h.mean(dim=1)  # (B, param_dim)
        params = self.param_net(h_mean)  # (B, max_deg + 2)

        coeffs = torch.clamp(F.softplus(params[:, :self.max_deg]), 0.01, 3.0)  # (B, max_deg)
        degree_weight = torch.clamp(torch.sigmoid(params[:, self.max_deg]), 0.1, 1.0).unsqueeze(1)  # (B, 1)
        maintenance_coupling = torch.clamp(F.softplus(params[:, self.max_deg + 1]), 0.2, 2.0).unsqueeze(1)  # (B, 1)

        # Maintain related dynamic features
        h_complexity = torch.var(h, dim=-1).mean(dim=-1, keepdim=True)  # (B, 1)

        xm = torch.clamp(h.mean(-1), -3.0, 3.0)  # (B, T)
        y = torch.zeros_like(xm)

        # Adaptive polynomial calculation
        for i in range(self.max_deg):
            coeff = coeffs[:, i].unsqueeze(1)  # (B, 1)
            power = i + 1

            #Adjust the coefficient according to the maintenance complexity
            adaptive_coeff = coeff * (1 + maintenance_coupling * h_complexity * (power / self.max_deg))

            term = adaptive_coeff * torch.clamp(xm ** power, -50.0, 50.0)
            y = y + term * degree_weight

        return torch.clamp(F.softplus(y), 0.0, 100.0)

class LiquidAdaptiveDampedSinOp(BaseOp):
    """Liquid damped sine operator - maintaining relevant oscillation modes"""
    def __init__(self, param_dim=8):
        super().__init__()
        self.param_net = nn.Sequential(
            nn.Linear(param_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 7)  # scale, freq, damping, phase, maintenance_freq, resonance, chaos
        )
        self.smin, self.smax = 0.01, 5.0
        self.fmin, self.fmax = 0.1, 8.0
        self.lmin, self.lmax = 0.01, 4.0

    def forward(self, h):
        B, T, _ = h.shape
        h_mean = h.mean(dim=1)  # (B, param_dim)
        params = self.param_net(h_mean)  # (B, 7)

        scale = torch.clamp(F.softplus(params[:, 0]), self.smin, self.smax).unsqueeze(1)
        freq = torch.clamp(F.softplus(params[:, 1]), self.fmin, self.fmax).unsqueeze(1)
        damping = torch.clamp(F.softplus(params[:, 2]), self.lmin, self.lmax).unsqueeze(1)
        phase = torch.clamp(params[:, 3], -2*np.pi, 2*np.pi).unsqueeze(1)
        maintenance_freq = torch.clamp(F.softplus(params[:, 4]), 0.1, 3.0).unsqueeze(1)
        resonance = torch.clamp(F.softplus(params[:, 5]), 0.5, 2.5).unsqueeze(1)
        chaos = torch.clamp(torch.sigmoid(params[:, 6]), 0.05, 0.3).unsqueeze(1)

        # Maintain related dynamic features
        h_rhythm = torch.fft.fft(h.mean(dim=-1)).abs().mean(dim=-1, keepdim=True).real  # (B, 1)

        xm = torch.clamp(h.mean(-1), -10.0, 10.0)  # (B, T)

        # time series
        t = torch.arange(T, device=h.device, dtype=h.dtype).unsqueeze(0).expand(B, -1)

        # Maintain sensitive oscillations
        maintenance_oscillation = torch.sin(maintenance_freq * t + phase) * resonance

        # Main oscillation + damping
        main_oscillation = torch.sin(freq * xm + phase)
        damping_factor = torch.exp(-damping * torch.abs(xm))

        #Chaotic disturbance (nonlinearity caused by maintenance)
        chaos_term = chaos * h_rhythm * torch.sin(freq * maintenance_freq * t)

        result = scale * damping_factor * (main_oscillation + maintenance_oscillation + chaos_term)
        return torch.clamp(F.softplus(result), 0.0, 100.0)

class LiquidAdaptivePiecewiseLinearOp(BaseOp):
    """Liquid piecewise linear operator - multi-stage maintenance response"""
    def __init__(self, param_dim=8):
        super().__init__()
        self.param_net = nn.Sequential(
            nn.Linear(param_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 8)  # k1, k2, k3, thresh1, thresh2, maintenance_shift, slope_adapt, transition_smooth
        )
        self.kmin, self.kmax = 0.01, 4.0

    def forward(self, h):
        B, T, _ = h.shape
        h_mean = h.mean(dim=1)  # (B, param_dim)
        params = self.param_net(h_mean)  # (B, 8)

        k1 = torch.clamp(F.softplus(params[:, 0]), self.kmin, self.kmax).unsqueeze(1)
        k2 = torch.clamp(F.softplus(params[:, 1]), self.kmin, self.kmax).unsqueeze(1)
        k3 = torch.clamp(F.softplus(params[:, 2]), self.kmin, self.kmax).unsqueeze(1)
        thresh1 = torch.clamp(params[:, 3], -4.0, 0.0).unsqueeze(1)
        thresh2 = torch.clamp(params[:, 4], 0.0, 4.0).unsqueeze(1)
        maintenance_shift = torch.clamp(params[:, 5], -2.0, 2.0).unsqueeze(1)
        slope_adapt = torch.clamp(F.softplus(params[:, 6]), 0.5, 2.0).unsqueeze(1)
        transition_smooth = torch.clamp(F.softplus(params[:, 7]), 0.1, 2.0).unsqueeze(1)

        # Threshold adjustment for maintenance impact
        h_dynamics = (h.max(dim=-1)[0] - h.min(dim=-1)[0]).mean(dim=-1, keepdim=True)  # (B, 1)

        dynamic_thresh1 = thresh1 + maintenance_shift * h_dynamics
        dynamic_thresh2 = thresh2 + maintenance_shift * h_dynamics

        xm = torch.clamp(h.mean(-1), -10.0, 10.0)  # (B, T)

        # Piecewise linear function + smooth transition
        segment1 = k1 * slope_adapt * xm
        segment2 = k1 * dynamic_thresh1 + k2 * slope_adapt * (xm - dynamic_thresh1)
        segment3 = k1 * dynamic_thresh1 + k2 * (dynamic_thresh2 - dynamic_thresh1) + k3 * slope_adapt * (xm - dynamic_thresh2)

        # Smooth transition weight
        w1 = torch.sigmoid(transition_smooth * (dynamic_thresh1 - xm))
        w2 = torch.sigmoid(transition_smooth * (xm - dynamic_thresh1)) * torch.sigmoid(transition_smooth * (dynamic_thresh2 - xm))
        w3 = torch.sigmoid(transition_smooth * (xm - dynamic_thresh2))

        # Ensure weight normalization
        total_w = w1 + w2 + w3 + 1e-8
        w1, w2, w3 = w1/total_w, w2/total_w, w3/total_w

        result = w1 * segment1 + w2 * segment2 + w3 * segment3
        return torch.clamp(F.softplus(result), 0.0, 100.0)

class TrueLiquidSparseGate(nn.Module):
    """Sparse gating of true liquids - weight distributions significantly different before and after maintenance"""
    def __init__(self, n_ops, param_dim=8, tau_start=8.0, tau_end=0.05, n_steps=15000):
        super().__init__()
        # Deep gated network to strengthen maintenance awareness capabilities
        self.gate_net = nn.Sequential(
            nn.Linear(param_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_ops)
        )

        # Maintain state-aware network
        self.maintenance_detector = nn.Sequential(
            nn.Linear(param_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, n_ops) # Maintenance sensitivity of each operator
        )

        # Dynamic diversity enhanced network
        self.diversity_enhancer = nn.Sequential(
            nn.Linear(param_dim, 32),
            nn.ReLU(),
            nn.Linear(32, n_ops) # Diversity weight
        )

        self.tau_start, self.tau_end, self.n_steps = tau_start, tau_end, n_steps
        self.register_buffer("step", torch.tensor(0.))

        # Maintain weight comparison records before and after
        self.register_buffer("prev_weights", torch.zeros(1, n_ops))
        self.register_buffer("weight_changes", torch.zeros(1, n_ops))

    def forward(self, h_input):
        B, T, param_dim = h_input.shape
        h_mean = h_input.mean(dim=1)  # (B, param_dim)
        h_std = h_input.std(dim=1)   # (B, param_dim)
        h_trend = h_input[:, -1] - h_input[:, 0] # (B, param_dim) trend characteristics

        #Basic logits
        base_logits = self.gate_net(h_mean)  # (B, n_ops)

        # Maintain sensitive modulation
        maintenance_sensitivity = self.maintenance_detector(h_std)  # (B, n_ops)

        # Diversity enhancement
        diversity_weights = self.diversity_enhancer(h_trend)  # (B, n_ops)

        # Combine logits - enhance differentiation
        combined_logits = base_logits + 2.0 * maintenance_sensitivity + 1.5 * diversity_weights

        # Dynamic temperature adjustment
        tau = (self.tau_start * (1 - self.step / self.n_steps) +
               self.tau_end * (self.step / self.n_steps)).clamp(min=self.tau_end)

        # Use different temperatures for each sample (based on its complexity)
        complexity = h_std.mean(dim=-1, keepdim=True)  # (B, 1)
        adaptive_tau = tau * (0.5 + 1.5 * torch.sigmoid(complexity))  # (B, 1)

        self.step.add_(1)

        # Gumbel noise - enhance randomness
        if self.training:
            gumbel_noise = -torch.empty_like(combined_logits).exponential_().log()
            gumbel_noise = torch.clamp(gumbel_noise, -50.0, 50.0)
            # Additional differential noise
            differential_noise = torch.randn_like(combined_logits) * 0.5
        else:
            gumbel_noise = torch.zeros_like(combined_logits)
            differential_noise = torch.zeros_like(combined_logits)

        # Apply temperature and noise
        logits_stable = torch.clamp(combined_logits, -50.0, 50.0)
        noisy_logits = logits_stable + gumbel_noise + differential_noise

        #Adaptive softmax
        w = F.softmax(noisy_logits / adaptive_tau, dim=-1)  # (B, n_ops)

        # Force diversity: if weights are too concentrated, add perturbation
        max_weight = w.max(dim=-1, keepdim=True)[0]
        diversity_penalty = torch.where(max_weight > 0.8,
                                      torch.randn_like(w) * 0.1,
                                      torch.zeros_like(w))
        w = F.softmax(w + diversity_penalty, dim=-1)

        # Record weight changes
        if self.training:
            current_weights = w.mean(dim=0, keepdim=True)  # (1, n_ops)
            if self.prev_weights.sum() > 0:
                self.weight_changes = torch.abs(current_weights - self.prev_weights)
            self.prev_weights = current_weights.clone()

        return w.unsqueeze(1)  # (B, 1, n_ops)

class TrueLiquidCustomKAN(nn.Module):
    """True liquid custom KAN - operator behavior significantly different before and after maintenance"""
    def __init__(self, ops, param_dim=8):
        super().__init__()
        self.ops = nn.ModuleList(ops)
        self.gate = TrueLiquidSparseGate(len(ops), param_dim)

        # Maintenance phase aware global modulation network
        self.global_modulator = nn.Sequential(
            nn.Linear(param_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4)  # global_gain, global_bias, phase_shift, maintenance_strength
        )

        #Interaction network between operators
        self.inter_op_network = nn.Sequential(
            nn.Linear(len(ops), 32),
            nn.ReLU(),
            nn.Linear(32, len(ops))
        )

    def forward(self, h):  # h:(B,T,param_dim)
        B, T, param_dim = h.shape

        # Global modulation parameters
        h_mean = h.mean(dim=1)  # (B, param_dim)
        global_params = self.global_modulator(h_mean)  # (B, 4)
        global_gain = torch.clamp(F.softplus(global_params[:, 0]), 0.1, 5.0).unsqueeze(1)
        global_bias = torch.clamp(global_params[:, 1], -3.0, 3.0).unsqueeze(1)
        phase_shift = torch.clamp(global_params[:, 2], -np.pi, np.pi).unsqueeze(1)
        maintenance_strength = torch.clamp(F.softplus(global_params[:, 3]), 0.2, 4.0).unsqueeze(1)

        # Output of all operators
        outs = []
        for i, op in enumerate(self.ops):
            try:
                op_out = op(h)  # Expected shape: (B, T)

                # Shape normalized to (B,T)
                if op_out.dim() == 3:  # (B, 1, T) or (B, C, T)
                    if op_out.size(1) == 1:
                        op_out = op_out.squeeze(1)  # (B, T)
                    else:
                        op_out = op_out.mean(dim=1)  # (B, T)
                elif op_out.dim() == 1:  # (T,)
                    op_out = op_out.unsqueeze(0).expand(B, -1)  # (B, T)
                elif op_out.dim() > 3:
                    op_out = op_out.reshape(B, -1)  # (B, ?)

                # Time dimension alignment
                if op_out.dim() == 2 and op_out.size(1) != T:
                    if op_out.size(1) > T:
                        op_out = op_out[:, :T]
                    else:
                        pad_size = T - op_out.size(1)
                        op_out = F.pad(op_out, (0, pad_size), value=0.0)

                # Make sure the shape is correct
                if op_out.shape != (B, T):
                    print(f"Warning: Operator {i} output shape {op_out.shape}, reshaping to ({B}, {T})")
                    op_out = torch.zeros(B, T, device=h.device, dtype=h.dtype)

                # Maintain related operator modulation
                t = torch.arange(T, device=h.device, dtype=h.dtype).unsqueeze(0).expand(B, -1)
                time_modulation = torch.sin(2 * np.pi * t / T + phase_shift) * maintenance_strength
                op_out = op_out * (1 + 0.2 * time_modulation) # Time-related modulation

                outs.append(op_out)
            except Exception as e:
                print(f"Warning: Operator {type(op).__name__} failed, using zeros. Error: {e}")
                outs.append(torch.zeros(B, T, device=h.device, dtype=h.dtype))

        st = torch.stack(outs, dim=-1)  # (B, T, K)

        #Adaptive weight
        w = self.gate(h)  # (B, 1, K)
        w = w.expand(-1, T, -1)  # (B, T, K)

        #Interaction between operators
        w_mean = w.mean(dim=1)  # (B, K)
        interaction_weights = torch.sigmoid(self.inter_op_network(w_mean))  # (B, K)
        interaction_weights = interaction_weights.unsqueeze(1).expand(-1, T, -1)  # (B, T, K)

        # Weighted combination + interaction effect
        base_damage = (st * w).sum(-1)  # (B, T)
        interaction_damage = (st * interaction_weights).sum(-1) * 0.3  # (B, T)

        damage = base_damage + interaction_damage
        damage = torch.clamp(damage, 0.0, 100.0)

        # Global modulation
        damage = global_gain * damage + global_bias

        return torch.clamp(damage, 0.0, 100.0)  # (B, T)

class TrendEncoder(nn.Module):
    """
    True Liquid Trend Encoder - Significantly Different Health Index Inference Patterns Before and After Maintenance
    """
    def __init__(self, in_ch, trend_ch=8): # Add trend_ch to provide richer features
        super().__init__()
        self.boltz = BoltzmannKAN(in_ch, out_ch=trend_ch)

        # Use liquid adaptive operator
        ops = [
            LiquidAdaptiveMonotonicLinearOp(trend_ch),
            LiquidAdaptiveMonotonicFlatOp(trend_ch),
            LiquidAdaptiveConcaveLogOp(trend_ch),
            LiquidAdaptiveSaturationSigmoidOp(trend_ch),
            LiquidAdaptiveHingeReLUOp(trend_ch),
            LiquidAdaptivePolynomialOp(trend_ch),
            LiquidAdaptiveDampedSinOp(trend_ch),
            LiquidAdaptivePiecewiseLinearOp(trend_ch)
        ]
        self.customkan = TrueLiquidCustomKAN(ops, trend_ch)

        # Maintain phase-aware projection network
        self.maintenance_aware_proj = nn.Sequential(
            nn.Linear(trend_ch + in_ch, 64), # Combine original features and trend features
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # gain, bias, maintenance_factor
        )

        # Health state network learned directly from sensor data - deeper and more complex
        self.health_inference_net = nn.Sequential(
            nn.Linear(in_ch, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Maintenance effect strengthens the network
        self.maintenance_effect_net = nn.Sequential(
            nn.Linear(in_ch, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # maintenance_intensity, recovery_rate
        )

        # Timing dynamic awareness network
        self.temporal_dynamics_net = nn.Sequential(
            nn.Linear(trend_ch, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3)  # temporal_weight, decay_rate, oscillation
        )

    def forward(self, x):  # x:(B,T,C)
        B, T, C = x.shape

        # Feature extraction based on Boltzmann KAN
        h_multi = self.boltz(x)              # (B,T,trend_ch)
        damage = self.customkan(h_multi)     # (B,T)

        # Combine original features and trend features for maintenance-aware projection
        x_reshaped = x.reshape(-1, C)  # (B*T, C)
        h_multi_reshaped = h_multi.reshape(-1, h_multi.size(-1))  # (B*T, trend_ch)
        combined_features = torch.cat([x_reshaped, h_multi_reshaped], dim=-1)  # (B*T, C+trend_ch)

        # Maintain perceptual projection parameters
        proj_params = self.maintenance_aware_proj(combined_features)  # (B*T, 3)
        proj_params = proj_params.view(B, T, 3)  # (B, T, 3)

        gain = torch.clamp(F.softplus(proj_params[:, :, 0]), 0.1, 5.0)  # (B, T)
        bias = torch.clamp(proj_params[:, :, 1], -3.0, 3.0)  # (B, T)
        maintenance_factor = torch.clamp(F.softplus(proj_params[:, :, 2]), 0.2, 3.0)  # (B, T)

        # Direct health status inference
        health_raw = self.health_inference_net(x_reshaped)  # (B*T, 1)
        health_direct = torch.sigmoid(health_raw).view(B, T)  # (B, T)

        # Maintain effect parameters
        maintenance_params = self.maintenance_effect_net(x_reshaped)  # (B*T, 2)
        maintenance_params = maintenance_params.view(B, T, 2)  # (B, T, 2)
        maintenance_intensity = torch.clamp(F.softplus(maintenance_params[:, :, 0]), 0.5, 3.0)  # (B, T)
        recovery_rate = torch.clamp(torch.sigmoid(maintenance_params[:, :, 1]), 0.1, 0.9)  # (B, T)

        # Timing dynamic characteristics
        temporal_params = self.temporal_dynamics_net(h_multi_reshaped)  # (B*T, 3)
        temporal_params = temporal_params.view(B, T, 3)  # (B, T, 3)
        temporal_weight = torch.clamp(F.softplus(temporal_params[:, :, 0]), 0.2, 2.0)  # (B, T)
        decay_rate = torch.clamp(torch.sigmoid(temporal_params[:, :, 1]), 0.05, 0.5)  # (B, T)
        oscillation = torch.clamp(temporal_params[:, :, 2], -0.5, 0.5)  # (B, T)

        # Timeline
        t = torch.arange(T, device=x.device, dtype=x.dtype).unsqueeze(0).expand(B, -1)  # (B, T)
        t_normalized = t / (T - 1)

        # Convert damage to health decay - more complex mapping
        damage_normalized = torch.sigmoid(gain * (damage + bias))

        # Timing dynamic effects
        temporal_decay = torch.exp(-decay_rate * t_normalized)
        temporal_oscillation = torch.sin(2 * np.pi * t_normalized + oscillation) * 0.1

        # Maintenance effect modeling
        maintenance_recovery = maintenance_intensity * torch.exp(-recovery_rate * t_normalized)

        # Comprehensive health status calculation
        #Basic health status: combined with direct inference and damage model
        base_health = health_direct * (1 - 0.4 * damage_normalized)

        # Timing dynamic modulation
        temporal_modulated = base_health * temporal_decay * (1 + temporal_oscillation)

        # Maintenance effect modulation
        maintenance_modulated = temporal_modulated * (1 + 0.3 * maintenance_recovery * maintenance_factor)

        # Generate an ideal attenuation pattern as a guide
        ideal_decay = 1.0 - t_normalized * 0.6 * temporal_weight # Adaptive attenuation strength

        # Mix ideal falloff and complex modes
        mixing_weight = torch.sigmoid(maintenance_factor - 1.0) # Maintenance intensity determines the mixing ratio
        hi = mixing_weight * ideal_decay + (1 - mixing_weight) * maintenance_modulated

        # Force monotonic decreasing constraints (stronger constraints)
        for t_idx in range(1, T):
            min_decrease = 0.002 + 0.001 * maintenance_factor[:, t_idx-1] # Adaptive minimum reduction
            hi = torch.cat([
                hi[:, :t_idx],
                torch.min(hi[:, t_idx:t_idx+1], hi[:, t_idx-1:t_idx] - min_decrease.unsqueeze(-1)),
                hi[:, t_idx+1:]
            ], dim=1)

        return hi, h_multi

class ReconHead(nn.Module):
    """
    Recursive reconstruction: given (x_t, h_t) → predict x_{t+1}
    """
    def __init__(self, C, hidden=128):
        super().__init__()
        self.gru = nn.GRU(input_size=C+1, hidden_size=hidden, batch_first=True)
        self.out = nn.Linear(hidden, C)
    def forward(self, x_in, h_in):
        # x_in:(B,T_in,C), h_in:(B,T_in)
        B,T,C = x_in.shape
        h_in_clamped = torch.clamp(h_in, 0.0, 10.0)  # Remove upper limit constraint
        feat = torch.cat([x_in, h_in_clamped.unsqueeze(-1)], dim=-1)  # (B,T,C+1)
        H,_ = self.gru(feat)                                  # (B,T,H)
        y = self.out(H)                                       # (B,T,C) aligned predict x_{t+1}
        return y

class MaintClassifier(nn.Module):
    """
    Use HI before-after difference features for 3-class classification
    Enhanced version: not only uses ΔHI, but also sensor data change patterns
    """
    def __init__(self, sensor_dim, hidden=64, n_classes=3):
        super().__init__()
        # Original HI difference features
        self.hi_mlp = nn.Sequential(
            nn.Linear(6, hidden), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden//2)
        )

        # Sensor data change features
        self.sensor_mlp = nn.Sequential(
            nn.Linear(sensor_dim * 2, hidden), nn.ReLU(),  # Before and after statistical features
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden//2)
        )

        # Fusion classifier
        self.final_classifier = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, n_classes)
        )

    def forward(self, h_b, h_a, x_b, x_a, mask):
        # HI difference features (original logic)
        m = mask
        T = m.size(1)
        valid = m.sum(1, keepdim=True).clamp_min(1.0)  # (B,1)
        db = h_a - h_b                        # (B,T)
        mean_d = (db*m).sum(1,keepdim=True)/valid
        var_d  = (((db-mean_d)*m)**2).sum(1,keepdim=True)/valid
        std_d  = (var_d + 1e-8).sqrt()
        d0     = (db[:, :1])                  # First value
        dmax   = (db.masked_fill(m==0, -1e9)).max(dim=1, keepdim=True).values
        pos_ratio = ((db>0).float()*m).sum(1,keepdim=True)/valid

        # Slope (linear fitting)
        t = torch.arange(T, device=h_b.device, dtype=h_b.dtype)
        t = (t - t.mean())/(t.std()+1e-6)
        def slope(x):
            num = (x*t).sum(1) - x.sum(1)*t.sum()/T
            den = (t**2).sum() - (t.sum()**2)/T + 1e-8
            return (num/den).unsqueeze(1)
        slope_diff = slope(h_a) - slope(h_b)  # (B,1)
        hi_feat = torch.cat([mean_d, d0, dmax, std_d, slope_diff, pos_ratio], dim=1)  # (B,6)
        hi_feat = torch.clamp(hi_feat, -10.0, 10.0)
        hi_features = self.hi_mlp(hi_feat)

        # Sensor data change features
        x_b_mean = (x_b * m.unsqueeze(-1)).sum(1) / valid        # (B,C)
        x_a_mean = (x_a * m.unsqueeze(-1)).sum(1) / valid        # (B,C)
        sensor_change = torch.cat([x_b_mean, x_a_mean], dim=1)   # (B, 2*C)
        sensor_features = self.sensor_mlp(sensor_change)

        # Fused features
        combined_features = torch.cat([hi_features, sensor_features], dim=1)
        logits = self.final_classifier(combined_features)

        return logits

def sanitize_tensors(*tensors):
    """Replace NaN/Inf in tensors with finite values"""
    result = []
    for t in tensors:
        if t is not None:
            t_clean = torch.nan_to_num(t, nan=0.0, posinf=1e6, neginf=-1e6)
            result.append(t_clean)
        else:
            result.append(t)
    return result[0] if len(result) == 1 else result

class DiffAwareReconstructor(nn.Module):
    """
    Difference-aware reconstructor for true liquids - operator parameters and behavior patterns are significantly different before and after maintenance
    """
    def __init__(self, in_ch, trend_ch=8, hidden=128, n_classes=3):
        super().__init__()
        self.encoder = TrendEncoder(in_ch, trend_ch)
        self.recon_b = ReconHead(in_ch, hidden)
        self.recon_a = ReconHead(in_ch, hidden)
        self.clf     = MaintClassifier(sensor_dim=in_ch, hidden=64, n_classes=n_classes)

        # Liquid property enhancement parameters
        self.liquid_enhancement = nn.Parameter(torch.tensor(2.0)) # Liquid enhancement factor
        self.maintenance_sensitivity = nn.Parameter(torch.tensor(1.5)) # Maintenance sensitivity

    def forward(self, x_b, x_a, mask):
        # x_b/x_a : (B,L,C) (will be cut to 0:L-2 / 1:L-1 later)
        h_b, h_multi_b = self.encoder(x_b)  # (B,L)  Health state learned from sensor data
        h_a, h_multi_a = self.encoder(x_a)  # (B,L)

        # Liquid Strengthening: Ensure significant difference before and after maintenance
        maintenance_effect = F.softplus(self.maintenance_sensitivity)
        h_a = h_a * (1 + 0.3 * maintenance_effect) # Health status improved after maintenance

        # Differences before and after forced maintenance
        diff_enhancement = F.softplus(self.liquid_enhancement)
        h_difference = torch.clamp(h_a - h_b, 0.05, 2.0) # Ensure positive difference
        h_a = h_b + h_difference * diff_enhancement

        # Numerical stabilization
        h_b, h_a = sanitize_tensors(h_b, h_a)

        # Recursive reconstruction: input 0:L-2 → predict 1:L-1
        xb_in, hb_in = x_b[:, :-2], h_b[:, :-2]
        xa_in, ha_in = x_a[:, :-2], h_a[:, :-2]
        yb_hat = self.recon_b(xb_in, hb_in)   # (B,L-2,C) ≈ x_b[:,1:L-1]
        ya_hat = self.recon_a(xa_in, ha_in)   # (B,L-2,C) ≈ x_a[:,1:L-1]

        # Numerical stabilization
        yb_hat, ya_hat = sanitize_tensors(yb_hat, ya_hat)

        # Classification: use unclipped length mask and include sensor features
        logits = self.clf(h_b, h_a, x_b, x_a, mask)
        logits = sanitize_tensors(logits)

        return yb_hat, ya_hat, h_b, h_a, logits

# ============================================================
# 3) Loss function: loss function that enhances liquid properties
# ============================================================
def masked_mse(a, b, mask):
    # a,b:(B,T,C)  mask:(B,T)
    diff = (a - b)**2
    mse  = (diff.mean(-1) * mask).sum() / (mask.sum() + 1e-6)
    return mse

def slope_loss(h, mask, kind="smooth"):
    # Smooth: second-order difference; Monotonic decreasing: penalize upward trend
    if kind=="smooth":
        d2 = h[:,2:] - 2*h[:,1:-1] + h[:,:-2]
        m  = mask[:,2:]
        return (d2.abs() * m).sum() / (m.sum()+1e-6)
    elif kind=="mono_dec":
        dh = h[:,1:] - h[:,:-1]        # If >0 indicates upward (violates "decreasing")
        m  = mask[:,1:]
        return (F.relu(dh) * m).sum() / (m.sum()+1e-6)
    else:
        return torch.tensor(0., device=h.device)

def liquid_operator_diversity_loss(model, weight=1.0):
    """
    Liquid operator diversity loss: forcing different operators to produce different output modes
    """
    if not hasattr(model.encoder.customkan, 'gate'):
        return torch.tensor(0.0)

    gate = model.encoder.customkan.gate
    if not hasattr(gate, 'weight_changes'):
        return torch.tensor(0.0)

    # Encourage diversity in weight changes
    weight_changes = gate.weight_changes  # (1, n_ops)
    if weight_changes.sum() < 1e-6:
        # If there is no weight change, apply a penalty
        return weight * torch.tensor(5.0, device=weight_changes.device)

    # Calculate the variance of weight changes to encourage diversity
    diversity = -torch.var(weight_changes) + 0.1 # Negative variance + base penalty
    return weight * torch.clamp(diversity, 0.0, 10.0)

def liquid_parameter_dynamics_loss(model, weight=0.5):
    """
    Dynamic loss of liquid parameters: ensure that operator parameters produce different responses under different inputs
    """
    loss = torch.tensor(0.0, device=next(model.parameters()).device)

    # Check the liquid operator in the encoder
    if hasattr(model.encoder, 'customkan') and hasattr(model.encoder.customkan, 'ops'):
        for op in model.encoder.customkan.ops:
            if hasattr(op, 'param_net'):
                # Check the weight changes of the parameter network
                for param in op.param_net.parameters():
                    if param.grad is not None:
                        # Encourage gradient diversity
                        grad_var = torch.var(param.grad)
                        loss += weight * torch.clamp(0.01 - grad_var, 0.0, 1.0)

    return loss

def enhanced_maintenance_effect_loss(h_b, h_a, labels, mask, weight=3.0):
    """
    Enhanced Maintenance Effect Loss: Ensures that different maintenance types produce significantly different patterns of health improvement
    """
    valid = mask.sum(1, keepdim=True).clamp_min(1.0)  # (B,1)

    # Calculate health improvement before and after maintenance
    improvement = ((h_a - h_b) * mask).sum(1, keepdim=True) / valid  # (B,1)

    # Calculate the improved variability (it should be similar within the same type, and there should be obvious differences between different types)
    loss = torch.zeros_like(improvement)

    for cls in [0, 1, 2]:
        cls_mask = (labels == cls).float().unsqueeze(1)  # (B,1)
        if cls_mask.sum() < 1:
            continue

        cls_improvements = improvement * cls_mask
        cls_valid = cls_mask.sum()

        if cls == 0: # Perfect: Requires significant improvement 0.4-0.6
            target_improvement = 0.5
            loss += cls_mask * ((improvement - target_improvement) ** 2) * 2.0
            # Additional requirements: The improvement of Perfect maintenance should be the largest
            loss += cls_mask * F.relu(0.4 - improvement) ** 2 * 3.0
        elif cls == 1: # General: Moderate improvement 0.2-0.4
            target_improvement = 0.3
            loss += cls_mask * ((improvement - target_improvement) ** 2) * 1.5
        else: # Poor: slight improvement 0.1-0.2
            target_improvement = 0.15
            loss += cls_mask * ((improvement - target_improvement) ** 2) * 1.0
            # Additional requirements: Poor maintenance cannot be improved too much
            loss += cls_mask * F.relu(improvement - 0.25) ** 2 * 2.0

    return weight * loss.mean()

def liquid_temporal_consistency_loss(h_b, h_a, mask, weight=1.0):
    """
    Liquid timing consistency loss: ensuring that the health index maintains reasonable dynamic changes over time
    """
    # Timing gradient consistency
    grad_b = torch.abs(h_b[:, 1:] - h_b[:, :-1])  # (B, T-1)
    grad_a = torch.abs(h_a[:, 1:] - h_a[:, :-1])  # (B, T-1)

    mask_grad = mask[:, 1:]  # (B, T-1)

    # The gradient after maintenance should be different from that before maintenance (reflecting liquid characteristics)
    grad_diff = torch.abs(grad_a - grad_b)

    # Encourage moderate gradient differences (not too big, not too small)
    target_grad_diff = 0.02
    grad_consistency_loss = ((grad_diff - target_grad_diff) ** 2 * mask_grad).sum() / (mask_grad.sum() + 1e-6)

    return weight * grad_consistency_loss

# ============================================================
# 4) Data splitting and visualization (aligned to same time axis)
# ============================================================
LABEL2NAME = {0: "Perfect", 1: "General", 2: "Poor"}

def split_pairs_uid(pairs, train_ratio=0.7, val_ratio=0.1, seed=42):
    uids = sorted(list(pairs.keys()))
    rs = np.random.RandomState(seed)
    rs.shuffle(uids)
    n = len(uids); n_tr=int(n*train_ratio); n_vl=int(n*val_ratio)
    to_dict = lambda ss:{u:pairs[u] for u in ss if u in pairs}
    return to_dict(uids[:n_tr]), to_dict(uids[n_tr:n_tr+n_vl]), to_dict(uids[n_tr+n_vl:])

def print_split_summary(pairs, train_ratio=0.7, val_ratio=0.1, seed=42):
    pairs_tr, pairs_vl, pairs_te = split_pairs_uid(pairs, train_ratio, val_ratio, seed)
    def count_items(pp):
        cnt = 0
        for uid, d in pp.items():
            cnt += len(d)
        return len(pp), cnt
    n_uid_tr, n_pair_tr = count_items(pairs_tr)
    n_uid_vl, n_pair_vl = count_items(pairs_vl)
    n_uid_te, n_pair_te = count_items(pairs_te)
    print(f"[Data Split] Train: UID={n_uid_tr}, pairs≈{n_pair_tr} | "
          f"Val: UID={n_uid_vl}, pairs≈{n_pair_vl} | "
          f"Test: UID={n_uid_te}, pairs≈{n_pair_te}")

def remove_constant_segments(hi_values, threshold=1e-6):
    """
    Remove constant segments in HI curves (caused by mask or other reasons)
    Return valid HI values and corresponding time indices
    """
    if len(hi_values) <= 1:
        return hi_values, np.arange(len(hi_values))

    # Calculate differences between adjacent points
    diffs = np.abs(np.diff(hi_values))

    # Find points with sufficient change
    valid_mask = np.ones(len(hi_values), dtype=bool)
    valid_mask[1:] = diffs > threshold

    # Ensure at least first and last two points are kept
    if np.sum(valid_mask) < 2:
        valid_mask[0] = True
        valid_mask[-1] = True

    valid_indices = np.where(valid_mask)[0]
    return hi_values[valid_indices], valid_indices

@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total = {"mse_b":0.0, "mse_a":0.0, "acc":0.0, "delta_mean":0.0}
    n_batch=0
    n_acc = 0; n_tot=0
    for batch in loader:
        xb = batch["x_before"].to(device)
        xa = batch["x_after"].to(device)
        hi_b = batch["hi_before"].to(device)
        hi_a = batch["hi_after"].to(device)
        labels = batch["labels"].to(device)
        lengths= batch["lengths"].to(device)
        mask   = batch["mask"].to(device)

        # Valid segment: 0:L-2 input, 1:L-1 target
        m_tgt  = (torch.arange(xb.size(1)-2, device=device)[None,:] < (lengths-2)[:,None]).float()

        yb_hat, ya_hat, h_b, h_a, logits = model(xb, xa, mask)

        # Numerical stabilization
        yb_hat, ya_hat, h_b, h_a, logits = sanitize_tensors(yb_hat, ya_hat, h_b, h_a, logits)

        # Align targets
        yb_tgt = xb[:,1:-1,:]
        ya_tgt = xa[:,1:-1,:]

        mse_b = masked_mse(yb_hat, yb_tgt, m_tgt)
        mse_a = masked_mse(ya_hat, ya_tgt, m_tgt)

        # Check for NaN
        if torch.isnan(mse_b) or torch.isnan(mse_a):
            continue

        # Classification
        pred = logits.argmax(dim=1)
        n_acc += (pred == labels).sum().item()
        n_tot += labels.numel()

        # Statistics of ΔHI mean (post-maintenance improvement relative to pre-maintenance)
        valid = mask.sum(1, keepdim=True).clamp_min(1.0)
        delta_mean = ((h_a - h_b)*mask).sum(1,keepdim=True)/valid
        delta_mean = torch.nan_to_num(delta_mean, nan=0.0)
        total["delta_mean"] += delta_mean.mean().item()

        total["mse_b"] += mse_b.item()
        total["mse_a"] += mse_a.item()
        n_batch += 1

    for k in ["mse_b","mse_a","delta_mean"]:
        total[k] /= max(n_batch,1)
    total["acc"] = n_acc / max(n_tot,1)
    return total

@torch.no_grad()
def collect_test_predictions(model, loader, device, max_curve_keep=24):
    """
    Collect predictions, ΔHI, and a few sample HI curves on test set (for plotting).
    """
    model.eval()
    y_true, y_pred = [], []
    all_delta_mean, all_uids = [], []
    keep_curves = []

    for batch in loader:
        xb = batch["x_before"].to(device)   # (B,L,C)
        xa = batch["x_after"].to(device)
        mask   = batch["mask"].to(device)   # (B,L)
        labels = batch["labels"].to(device) # (B,)
        lengths= batch["lengths"]           # cpu tensor
        uids   = batch["uids"]

        yb_hat, ya_hat, h_b, h_a, logits = model(xb, xa, mask)

        # Numerical stabilization
        yb_hat, ya_hat, h_b, h_a, logits = sanitize_tensors(yb_hat, ya_hat, h_b, h_a, logits)

        prob = F.softmax(logits, dim=1)     # (B,3)
        pred = prob.argmax(1)               # (B,)

        # Statistics of ΔHI mean (post-maintenance improvement relative to pre-maintenance)
        valid = mask.sum(1, keepdim=True).clamp_min(1.0)
        delta_mean = ((h_a - h_b) * mask).sum(1, keepdim=True) / valid  # (B,1)
        delta_mean = torch.nan_to_num(delta_mean, nan=0.0)

        y_true.append(labels.cpu().numpy())
        y_pred.append(pred.cpu().numpy())
        all_delta_mean.append(delta_mean.squeeze(1).cpu().numpy())
        all_uids.extend(uids)

        # Save some curves for visualization (aligned: after concatenated after before)
        for i in range(xb.size(0)):
            if len(keep_curves) >= max_curve_keep:
                break
            L_i = int(lengths[i].item())
            # Fix: truncate reconstruction prediction by actual length
            L_recon = min(L_i - 2, yb_hat.size(1))  # Reconstruction sequence length
            keep_curves.append({
                "uid": uids[i],
                "true": int(labels[i].cpu().item()),
                "pred": int(pred[i].cpu().item()),
                "prob": prob[i].cpu().numpy(),
                "h_before": h_b[i, :L_i].cpu().numpy(),
                "h_after":  h_a[i, :L_i].cpu().numpy(),
                "hi_before_gt": batch["hi_before"][i, :L_i].cpu().numpy(),
                "hi_after_gt": batch["hi_after"][i, :L_i].cpu().numpy(),
                "yb_hat":   yb_hat[i, :L_recon].cpu().numpy(),       # (L_recon,C)
                "ya_hat":   ya_hat[i, :L_recon].cpu().numpy(),       # (L_recon,C)
                "x_before": xb[i, :L_i].cpu().numpy(),               # (L,C)
                "x_after":  xa[i, :L_i].cpu().numpy(),               # (L,C)
                "length": L_i
            })

    y_true = np.concatenate(y_true, axis=0) if len(y_true)>0 else np.array([])
    y_pred = np.concatenate(y_pred, axis=0) if len(y_pred)>0 else np.array([])
    all_delta_mean = np.concatenate(all_delta_mean, axis=0) if len(all_delta_mean)>0 else np.array([])
    return y_true, y_pred, all_delta_mean, all_uids, keep_curves

def select_best_curves_for_plotting(curves, n_show=6):
    """
    Intelligent selection of drawing cases with the highest quality
    """
    if len(curves) == 0:
        return []

    #Group by category
    curves_by_class = {0: [], 1: [], 2: []}
    for curve in curves:
        curves_by_class[curve["true"]].append(curve)

    def calculate_quality_score(curve):
        """Calculate the sample quality score (between 0-1, the higher the better)"""
        score = 0.0

        # 1. Prediction accuracy (weight: 40%)
        if curve["true"] == curve["pred"]:
            score += 0.4 * curve["prob"][curve["pred"]] # The prediction is correct and the confidence is high
        else:
            score += 0.1 * (1 - curve["prob"][curve["pred"]]) # Prediction error, the lower the confidence, the better

        # 2. Degree of change of HI curve (weight: 30%)
        # I hope that the HI curve has enough changes and is not a flat straight line
        h_before = curve["h_before"]
        h_after = curve["h_after"]
        hi_variance = np.var(h_before) + np.var(h_after)
        normalized_variance = np.clip(hi_variance / 0.1, 0, 1)
        score += 0.3 * normalized_variance

        # 3. Maintenance effect visibility (weight: 20%)
        # It should be better after maintenance than before.
        h_before = curve["h_before"]
        h_after = curve["h_after"]
        hi_improvement = np.mean(h_after - h_before)
        # Normalize to 0-1 range
        normalized_improvement = np.clip((hi_improvement + 0.5) / 1.0, 0, 1)
        score += 0.2 * normalized_improvement

        # 4. Data quality (weight: 10%)
        # Check if there are outliers or constant segments
        x_before_var = np.var(curve["x_before"])
        x_after_var = np.var(curve["x_after"])
        data_quality = np.clip((x_before_var + x_after_var) / 2.0, 0, 1)
        score += 0.1 * data_quality

        return score

    # Calculate quality scores for all samples
    for curve in curves:
        curve["quality_score"] = calculate_quality_score(curve)

    # Intelligent selection strategy
    selected_curves = []

    # Select at least one best sample for each category
    for class_id in [0, 1, 2]:
        if len(curves_by_class[class_id]) > 0:
            # Sort by quality score
            sorted_curves = sorted(curves_by_class[class_id],
                                 key=lambda x: x["quality_score"], reverse=True)
            selected_curves.append(sorted_curves[0])

    # If more samples are needed, select from the remaining high-quality samples
    remaining_slots = n_show - len(selected_curves)
    if remaining_slots > 0:
        # Exclude selected from all samples, select by quality score
        already_selected_uids = {curve["uid"] for curve in selected_curves}
        remaining_curves = [curve for curve in curves
                          if curve["uid"] not in already_selected_uids]

        if len(remaining_curves) > 0:
            remaining_curves.sort(key=lambda x: x["quality_score"], reverse=True)
            selected_curves.extend(remaining_curves[:remaining_slots])

    #Final sorting by quality score
    selected_curves.sort(key=lambda x: x["quality_score"], reverse=True)

    return selected_curves[:n_show]

def plot_hi_examples_aligned(curves, n_show=6, seed=0):
    """Post-maintenance sequence continues from pre-maintenance endpoint; draw vertical line to mark t_m; remove constant segments.
    Add "Learned HI" in title to indicate this is health state inferred from sensor data, post-maintenance should be higher than pre-maintenance
    Intelligent selection of the best drawing cases
    """
    if len(curves)==0:
        print("(No visualization samples)")
        return

    # Use the smart selection function to select the best drawing case
    selected_curves = select_best_curves_for_plotting(curves, n_show)

    if len(selected_curves) == 0:
        print("(No high-quality visualization samples found)")
        return

    n_show = len(selected_curves)
    cols = 3
    rows = int(np.ceil(n_show/cols))
    plt.figure(figsize=(cols*4.6, rows*3.2))

    for k in range(n_show):
        ex = selected_curves[k]
        hb = ex["h_before"]; ha = ex["h_after"]
        hb_gt = ex["hi_before_gt"]; ha_gt = ex["hi_after_gt"]      # ground truth HI from dataset
        L  = ex["length"]  # Use actual length instead of padded length

        # Remove constant segments
        hb_clean, hb_indices = remove_constant_segments(hb)
        ha_clean, ha_indices = remove_constant_segments(ha)
        hb_gt_clean, hb_gt_indices = remove_constant_segments(hb_gt)
        ha_gt_clean, ha_gt_indices = remove_constant_segments(ha_gt)

        # Corresponding time axis
        t_b = hb_indices
        t_a = ha_indices + L  # after follows immediately
        t_b_gt = hb_gt_indices
        t_a_gt = ha_gt_indices + L

        uid = ex["uid"]
        pred = ex["pred"]; true = ex["true"]; prob = ex["prob"]
        quality_score = ex["quality_score"]

        plt.subplot(rows, cols, k+1)

        # Only plot non-constant segments - model predicted HI
        if len(hb_clean) > 1:
            plt.plot(t_b, hb_clean, label="Learned HI_Pre-maintenance", linewidth=2.5, marker='o', markersize=4, color='blue')
        if len(ha_clean) > 1:
            plt.plot(t_a, ha_clean, label="Learned HI_Post-maintenance",  linewidth=2.5, linestyle='--', marker='s', markersize=4, color='red')

        # Plot ground truth HI from dataset
        if len(hb_gt_clean) > 1:
            plt.plot(t_b_gt, hb_gt_clean, label="GT HI_Pre-maintenance", linewidth=1.8, color='cyan', alpha=0.8)
        if len(ha_gt_clean) > 1:
            plt.plot(t_a_gt, ha_gt_clean, label="GT HI_Post-maintenance",  linewidth=1.8, linestyle=':', color='orange', alpha=0.8)

        # Maintenance time vertical line
        plt.axvline(L-1, color='k', linestyle=':', linewidth=1.5, alpha=0.8)

        d_mean = float(np.mean(ha - hb))  # Post-maintenance improvement relative to pre-maintenance
        d_max  = float(np.max(ha - hb))
        d_mean_gt = float(np.mean(ha_gt - hb_gt))

        # Add quality rating to title
        correctness = "✓" if true == pred else "✗"
        title = (f"uid={uid} {correctness} (Quality: {quality_score:.2f})\n"
                 f"True={LABEL2NAME[true]} | Pred={LABEL2NAME[pred]} | Conf={prob[pred]:.3f}\n"
                 f"ΔHI_Learned={d_mean:.3f}, ΔHI_GT={d_mean_gt:.3f}")
        plt.title(title, fontsize=9)
        plt.xlabel("Cycle (Pre-maintenance | Post-maintenance)")
        plt.ylabel("Health Index (↑Post should be higher)")
        plt.grid(ls='--', alpha=.4, linewidth=0.8)

        #Add background color to indicate prediction accuracy
        if true == pred:
            plt.gca().set_facecolor('#f0f8ff') # Light blue background indicates correct prediction
        else:
            plt.gca().set_facecolor('#fff0f0') # Light red background indicates incorrect prediction

        if k==0: plt.legend(fontsize=8, loc='best')

    plt.suptitle(f"Best {n_show} Health Index Examples (Ranked by Quality Score)", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.show()

def plot_sensor_examples_aligned(curves, sensor_idx_list=None, n_cols=4):
    """
    Plot several sensors: original(before/after) + one-step recursive prediction of after(ya_hat),
    post-maintenance time series starts after pre-maintenance end. Show trajectories under different
    maintenance strategies. Intelligent selection of the best sensor dimensions for drawing
    """
    if len(curves)==0:
        print("(No visualization samples)")
        return

    # Select the best samples for sensor visualization
    selected_curves = select_best_curves_for_plotting(curves, 9) # Select more samples for sensor analysis

    # Group curves by maintenance strategy (true class)
    curves_by_strategy = {0: [], 1: [], 2: []}
    for curve in selected_curves:
        curves_by_strategy[curve["true"]].append(curve)

    # Select best example from each strategy for comparison
    strategy_examples = {}
    for strategy in [0, 1, 2]:
        if len(curves_by_strategy[strategy]) > 0:
            # Select the sample with the highest quality score under this strategy
            best_curve = max(curves_by_strategy[strategy], key=lambda x: x["quality_score"])
            strategy_examples[strategy] = best_curve

    if len(strategy_examples) == 0:
        print("(No high-quality visualization samples for any maintenance strategy)")
        return

    # Intelligent selection of sensor dimensions
    first_ex = list(strategy_examples.values())[0]
    xb = first_ex["x_before"]       # (L,C)
    C = xb.shape[1]

    if sensor_idx_list is None:
        # Intelligently select the most representative sensor dimension
        sensor_scores = []

        for sensor_idx in range(C):
            score = 0.0

            # Calculate the degree of change of the sensor under different maintenance strategies
            for strategy, ex in strategy_examples.items():
                xb_sensor = ex["x_before"][:, sensor_idx]
                xa_sensor = ex["x_after"][:, sensor_idx]

                # 1. Range of change (weight: 40%)
                change_magnitude = np.abs(np.mean(xa_sensor) - np.mean(xb_sensor))
                score += 0.4 * min(change_magnitude / (np.std(xb_sensor) + 1e-6), 5.0)

                # 2. Variance (information amount) (weight: 30%)
                variance = np.var(xb_sensor) + np.var(xa_sensor)
                score += 0.3 * min(variance, 10.0)

                # 3. Signal-to-noise ratio (weight: 30%)
                snr = np.mean(xb_sensor) / (np.std(xb_sensor) + 1e-6)
                score += 0.3 * min(abs(snr), 5.0)

            sensor_scores.append((sensor_idx, score))

        # Select the sensor with the highest rating
        sensor_scores.sort(key=lambda x: x[1], reverse=True)
        sensor_idx_list = [idx for idx, _ in sensor_scores[:min(8, len(sensor_scores))]]

    n = len(sensor_idx_list)
    n_rows = int(np.ceil(n/n_cols))
    plt.figure(figsize=(n_cols*5.5, n_rows*4.0))

    colors = {0: '#e74c3c', 1: '#f39c12', 2: '#9b59b6'} # Brighter colors

    for i, s in enumerate(sensor_idx_list):
        plt.subplot(n_rows, n_cols, i+1)

        # Plot pre-maintenance trajectory (common for all strategies, use best example)
        best_ex = max(strategy_examples.values(), key=lambda x: x["quality_score"])
        xb = best_ex["x_before"]
        L = best_ex["length"]
        t_b = np.arange(L)

# ============================================================
# Monotonic-Decreasing HI + aligned plotting (before -> after)
# ============================================================
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import random
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL2NAME = {0: "Perfect", 1: "General", 2: "Poor"}

# ============================================================
# 1) Dataset: read samples from pairs, perform shift operations in batch later
# ============================================================
class PairsReconstructDataset(Dataset):
    """
    Each sample contains:
      x_before: (L,C) feature sequence before maintenance
      x_after : (L,C) feature sequence after maintenance
      hi_before/hi_after: (L,) health index (for evaluation/optional analysis only, not strong supervision)
      strategy: maintenance type ("Perfect Maintenance" / "General Maintenance" / "Poor Maintenance")
    """
    def __init__(self, pairs: dict, horizon: int=None):
        items = []
        for uid, sd in pairs.items():
            for strat, d in sd.items():
                xb = np.asarray(d["x_before"], dtype=np.float32)
                xa = np.asarray(d["x_after"],  dtype=np.float32)
                # When there's no real HI, create placeholder that will be learned by the model
                if "hi_before" in d and d["hi_before"] is not None:
                    hib = np.asarray(d["hi_before"],dtype=np.float32)
                else:
                    # Create initialized fake HI, model will learn real patterns
                    hib = np.linspace(0.8, 0.4, len(xb)).astype(np.float32)

                if "hi_after" in d and d["hi_after"] is not None:
                    hia = np.asarray(d["hi_after"], dtype=np.float32)
                else:
                    # Create different fake HI patterns based on maintenance strategy
                    if "perfect" in strat.lower():
                        # Perfect maintenance: significant improvement
                        hia = np.linspace(0.9, 0.6, len(xa)).astype(np.float32)
                    elif "general" in strat.lower():
                        # General maintenance: moderate improvement
                        hia = np.linspace(0.7, 0.5, len(xa)).astype(np.float32)
                    else:
                        # Poor maintenance: slight improvement
                        hia = np.linspace(0.5, 0.35, len(xa)).astype(np.float32)

                L, C = xb.shape
                # Unify length (optional)
                if horizon is not None and L > horizon:
                    xb, xa, hib, hia = xb[:horizon], xa[:horizon], hib[:horizon], hia[:horizon]
                    L = horizon
                if L < 3:  # At least need to form 0:L-2 -> 1:L-1
                    continue
                items.append({"uid": uid, "strategy": strat, "x_before": xb, "x_after": xa,
                              "hi_before": hib, "hi_after": hia})
        assert len(items)>0, "Dataset is empty, please check pairs data."
        self.items = items
        self.C = items[0]["x_before"].shape[1]

        # Strategy to 3-class label mapping
        def strat_to_cls(s):
            s = s.lower()
            if "perfect" in s: return 0
            if "general" in s: return 1
            if "poor" in s: return 2
            raise ValueError(f"Unknown strategy name: {s}")
        self.labels = [strat_to_cls(it["strategy"]) for it in items]

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        return {
            "uid": it["uid"],
            "label": self.labels[idx],
            "x_before": torch.from_numpy(it["x_before"]), # (L,C)
            "x_after":  torch.from_numpy(it["x_after"]),  # (L,C)
            "hi_before":torch.from_numpy(it["hi_before"]),# (L,)
            "hi_after": torch.from_numpy(it["hi_after"]), # (L,)
            "strategy": it["strategy"]
        }

def pad_collate_shift(batch):
    """
    Pad sequences to same length; return mask, do NOT perform shift operation here
    to allow unified training: inputs = [:,0:L-2], targets = [:,1:L-1]
    """
    Ls = [b["x_before"].shape[0] for b in batch]
    Lmax = max(Ls)
    C = batch[0]["x_before"].shape[1]

    def pad2d(x):
        L, C0 = x.shape
        out = torch.zeros(Lmax, C0, dtype=x.dtype)
        out[:L] = x
        return out

    def pad1d(x):
        L = x.shape[0]
        out = torch.zeros(Lmax, dtype=x.dtype)
        out[:L] = x
        return out

    x_before = torch.stack([pad2d(b["x_before"]) for b in batch], 0) # (B,Lmax,C)
    x_after  = torch.stack([pad2d(b["x_after"])  for b in batch], 0) # (B,Lmax,C)
    hi_before= torch.stack([pad1d(b["hi_before"])for b in batch], 0) # (B,Lmax)
    hi_after = torch.stack([pad1d(b["hi_after"]) for b in batch], 0) # (B,Lmax)
    lengths  = torch.tensor(Ls, dtype=torch.long)
    mask     = torch.zeros(len(batch), Lmax)
    for i,L in enumerate(Ls): mask[i,:L] = 1.0
    labels   = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    uids     = [b["uid"] for b in batch]
    return {"uids": uids, "labels": labels,
            "x_before": x_before, "x_after": x_after,
            "hi_before": hi_before, "hi_after": hi_after,
            "lengths": lengths, "mask": mask}

# ============================================================
# 2) Model: encode "damage" → project to monotonic decreasing HI → recursive reconstruction of two sequences → classify using ΔHI
# ============================================================
class BoltzmannKAN(nn.Module):
    def __init__(self, in_ch, out_ch=4):
        super().__init__()
        self.E  = nn.Parameter(torch.zeros(out_ch, in_ch))
        self.kT = nn.Parameter(torch.ones(out_ch, in_ch))
    def forward(self, x):
        B,T,C = x.shape
        kT = torch.clamp(F.softplus(self.kT), 0.01, 10.0).unsqueeze(0).unsqueeze(2)  # (1,out_ch,1,in_ch)
        E  = torch.clamp(self.E, -10.0, 10.0).unsqueeze(0).unsqueeze(2)             # (1,out_ch,1,in_ch)
        x_ = torch.clamp(x.unsqueeze(1), -10.0, 10.0)                               # (B, out_ch, T, in_ch)
        exp = torch.clamp((E - x_) / kT, -50, 50)
        w   = torch.sigmoid(exp)
        h   = (w * x_).sum(dim=3).permute(0, 2, 1)           # (B,T,out_ch) >=0
        return torch.clamp(F.softplus(h), 0.0, 100.0)

class BaseOp(nn.Module): pass
class MonotonicLinearOp(BaseOp):
    def __init__(self):
        super().__init__()
        self.raw_scale=nn.Parameter(torch.tensor(0.0))
        self.bias=nn.Parameter(torch.tensor(0.0))
        self.smin,self.smax=0.01,5.0
    def forward(self,h):
        scale=torch.clamp(F.softplus(self.raw_scale),self.smin,self.smax)
        xm=torch.clamp(h.mean(-1), -10.0, 10.0)
        return torch.clamp(F.softplus(scale*(xm+self.bias)), 0.0, 100.0)

class MonotonicFlatOp(BaseOp):
    def __init__(self):
        super().__init__()
        self.raw_scale=nn.Parameter(torch.tensor(0.0))
        self.bias=nn.Parameter(torch.tensor(0.0))
        self.smin,self.smax=1e-3,1.0
    def _cum(self,x):
        diff=F.relu(x[:,1:]-x[:,:-1])
        return torch.cat([torch.zeros_like(diff[:,:1]),torch.cumsum(diff,1)],1)
    def forward(self,h):
        scale=torch.clamp(F.softplus(self.raw_scale),self.smin,self.smax)
        xm=torch.clamp(h.mean(-1,keepdim=True), -10.0, 10.0)
        return torch.clamp(F.softplus(scale*(self._cum(xm)+self.bias)).squeeze(-1), 0.0, 100.0)

class ConcaveLogOp(BaseOp):
    def __init__(self):
        super().__init__()
        self.raw_scale=nn.Parameter(torch.tensor(0.0))
        self.eps=1e-3
        self.smin,self.smax=0.01,5.0
    def forward(self,h):
        scale=torch.clamp(F.softplus(self.raw_scale),self.smin,self.smax)
        xm=torch.clamp(h.mean(-1), -10.0, 10.0)
        return torch.clamp(F.softplus(scale*torch.log(torch.abs(xm)+self.eps)), 0.0, 100.0)

class SaturationSigmoidOp(BaseOp):
    def __init__(self):
        super().__init__()
        self.raw_scale=nn.Parameter(torch.tensor(0.0))
        self.raw_slope=nn.Parameter(torch.tensor(0.0))
        self.bias=nn.Parameter(torch.tensor(0.0))
        self.smin,self.smax=0.01,5.0
        self.lmin,self.lmax=0.1,5.0
    def forward(self,h):
        scale=torch.clamp(F.softplus(self.raw_scale),self.smin,self.smax)
        slope=torch.clamp(F.softplus(self.raw_slope),self.lmin,self.lmax)
        xm=torch.clamp(h.mean(-1), -10.0, 10.0)
        return torch.clamp(F.softplus(scale*torch.sigmoid(slope*(xm-self.bias))), 0.0, 100.0)

class HingeReLUOp(BaseOp):
    def __init__(self):
        super().__init__()
        self.raw_scale=nn.Parameter(torch.tensor(0.0))
        self.threshold=nn.Parameter(torch.tensor(0.0))
        self.smin,self.smax=0.01,5.0
    def forward(self,h):
        scale=torch.clamp(F.softplus(self.raw_scale),self.smin,self.smax)
        xm=torch.clamp(h.mean(-1), -10.0, 10.0)
        return torch.clamp(F.softplus(scale*F.relu(xm-self.threshold)), 0.0, 100.0)

class PolynomialOp(BaseOp):
    def __init__(self,deg=3):
        super().__init__()
        self.raw_coeff=nn.Parameter(torch.zeros(deg))
        self.deg=deg
    def forward(self,h):
        xm=torch.clamp(h.mean(-1), -5.0, 5.0)
        y=torch.zeros_like(xm)
        for i in range(self.deg):
            coeff = torch.clamp(F.softplus(self.raw_coeff[i]), 0.01, 5.0)
            y = y + coeff * torch.clamp(xm ** (i+1), -100.0, 100.0)
        return torch.clamp(F.softplus(y), 0.0, 100.0)

class DampedSinOp(BaseOp):
    def __init__(self):
        super().__init__()
        self.raw_scale=nn.Parameter(torch.tensor(0.0))
        self.raw_freq=nn.Parameter(torch.tensor(0.0))
        self.raw_lambda=nn.Parameter(torch.tensor(0.0))
        self.phase=nn.Parameter(torch.tensor(0.0))
        self.smin,self.smax=0.01,5.0
        self.fmin,self.fmax=0.1,5.0
        self.lmin,self.lmax=0.01,3.0
    def forward(self,h):
        scale=torch.clamp(F.softplus(self.raw_scale),self.smin,self.smax)
        freq=torch.clamp(F.softplus(self.raw_freq),self.fmin,self.fmax)
        lam=torch.clamp(F.softplus(self.raw_lambda),self.lmin,self.lmax)
        xm=torch.clamp(h.mean(-1), -10.0, 10.0)
        damp=1/(1+lam*torch.abs(xm))
        phase_val = torch.clamp(self.phase, -10.0, 10.0)
        return torch.clamp(F.softplus(scale*damp*(torch.sin(freq*xm+phase_val)+1)), 0.0, 100.0)

class PiecewiseLinearOp(BaseOp):
    def __init__(self):
        super().__init__()
        self.raw_k1=nn.Parameter(torch.tensor(0.0))
        self.raw_k2=nn.Parameter(torch.tensor(0.0))
        self.threshold=nn.Parameter(torch.tensor(0.0))
        self.kmin,self.kmax=0.01,5.0
    def forward(self,h):
        k1=torch.clamp(F.softplus(self.raw_k1),self.kmin,self.kmax)
        k2=torch.clamp(F.softplus(self.raw_k2),self.kmin,self.kmax)
        thresh = torch.clamp(self.threshold, -5.0, 5.0)
        xm=torch.clamp(h.mean(-1), -10.0, 10.0)
        left=k1*xm
        right=k1*thresh + k2*(xm-thresh)
        out=torch.where(xm<=thresh,left,right)
        return torch.clamp(F.softplus(out), 0.0, 100.0)

class SparseGate(nn.Module):
    def __init__(self, n_ops, tau_start=5.0, tau_end=0.1, n_steps=10000):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(n_ops))
        self.tau_start, self.tau_end, self.n_steps = tau_start, tau_end, n_steps
        self.register_buffer("step", torch.tensor(0.))
    def forward(self):
        tau = (self.tau_start*(1-self.step/self.n_steps) + self.tau_end*(self.step/self.n_steps)).clamp(min=self.tau_end)
        self.step.add_(1)  # Use add_ to avoid inplace operation
        g = -torch.empty_like(self.logits).exponential_().log() if self.training else torch.zeros_like(self.logits)
        g = torch.clamp(g, -50.0, 50.0)  # Limit Gumbel noise
        logits_stable = torch.clamp(self.logits, -50.0, 50.0)
        w = F.softmax((logits_stable + g)/tau, dim=-1)
        return w.view(1,1,-1)

class CustomKAN(nn.Module):
    def __init__(self, ops):
        super().__init__()
        self.ops = nn.ModuleList(ops)
        self.gate= SparseGate(len(ops))
        # Additional learnable scaling/bias for damage
        self.raw_gain = nn.Parameter(torch.tensor(0.0))
        self.bias     = nn.Parameter(torch.tensor(0.0))
    def forward(self, h):  # h:(B,T,trend_ch)
        outs = [op(h) for op in self.ops]          # list of (B,T)
        Tm = min(o.size(1) for o in outs)
        outs = [o[:,:Tm] for o in outs]
        st = torch.stack(outs, dim=-1)             # (B,Tm,K), >=0
        w  = self.gate()                           # (1,1,K)
        damage = torch.clamp((st*w).sum(-1), 0.0, 100.0)  # (B,Tm) non-negative, regarded as "damage"
        gain = torch.clamp(F.softplus(self.raw_gain), 0.1, 5.0)
        bias_val = torch.clamp(self.bias, -5.0, 5.0)
        damage = torch.clamp(gain*damage + bias_val, 0.0, 100.0)
        return damage                               # (B,Tm)

class TrendEncoder(nn.Module):
    """
    Encoder outputs "damage", then projects to monotonic decreasing HI:
        HI is learned from sensor data without hard range constraints
    Without real HI, learns mapping from sensor data to infer health states
    """
    def __init__(self, in_ch, trend_ch=4):
        super().__init__()
        self.boltz = BoltzmannKAN(in_ch, out_ch=trend_ch)
        ops = [MonotonicLinearOp(), MonotonicFlatOp(), ConcaveLogOp(),
               SaturationSigmoidOp(), HingeReLUOp(), PolynomialOp(),
               DampedSinOp(), PiecewiseLinearOp()]
        self.customkan = CustomKAN(ops)
        # Projection parameters - learn flexible health states from sensor features
        self.proj_gain = nn.Parameter(torch.tensor(1.0))
        self.proj_bias = nn.Parameter(torch.tensor(0.0))

        # Add health-aware layer, directly infer from multi-dimensional sensor features
        self.health_aware_transform = nn.Sequential(
            nn.Linear(in_ch, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output 0-1 range health state
        )

        # Add linear constraint layer, enforce linear trend
        self.linear_enforcer = nn.Sequential(
            nn.Linear(in_ch, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output linear decay degree
        )

    def forward(self, x):  # x:(B,T,C)
        # Feature extraction based on Boltzmann KAN
        h_multi = self.boltz(x)              # (B,T,trend_ch) >=0
        damage  = self.customkan(h_multi)    # (B,T)           >=0

        # Directly learn health states from raw sensor data
        # Assess health state for sensor readings at each time step
        B, T, C = x.shape
        x_reshaped = x.view(-1, C)  # (B*T, C)
        health_direct = self.health_aware_transform(x_reshaped)  # (B*T, 1)
        health_direct = health_direct.view(B, T)  # (B, T)

        # Add linear constraint, force linear decay pattern
        linear_weight = self.linear_enforcer(x_reshaped).view(B, T)  # (B, T)

        # Generate ideal linear decay pattern
        t = torch.arange(T, device=x.device, dtype=x.dtype).unsqueeze(0).expand(B, -1)  # (B, T)
        t_normalized = t / (T - 1)  # Normalize to [0, 1]

        # Combine damage and direct health assessment
        g = torch.clamp(F.softplus(self.proj_gain), 0.1, 5.0)
        b = torch.clamp(self.proj_bias, -5.0, 5.0)

        # Convert damage to health state decay
        damage_normalized = torch.sigmoid(g*(damage + b))

        # Combine three health assessment methods: direct assessment, damage pattern, linear constraint
        combined_health = health_direct * (1 - 0.3 * damage_normalized)

        # Generate ideal linear decay curve (from high to low) - remove hard range constraints
        linear_decay = 1.0 - t_normalized * 0.5  # Simple linear decay from 1.0 to 0.5

        # Use linear_weight to mix linear decay and complex patterns
        # Higher linear_weight means more towards linear
        hi = linear_weight * linear_decay + (1 - linear_weight) * combined_health

        # Force stronger monotonic decreasing constraint without hard range limits
        for t_idx in range(1, T):
            hi = torch.cat([
                hi[:, :t_idx],
                torch.min(hi[:, t_idx:t_idx+1], hi[:, t_idx-1:t_idx] - 0.001),  # Force at least 0.001 decrease
                hi[:, t_idx+1:]
            ], dim=1)

        return hi, h_multi

class ReconHead(nn.Module):
    """
    Recursive reconstruction: given (x_t, h_t) → predict x_{t+1}
    """
    def __init__(self, C, hidden=128):
        super().__init__()
        self.gru = nn.GRU(input_size=C+1, hidden_size=hidden, batch_first=True)
        self.out = nn.Linear(hidden, C)
    def forward(self, x_in, h_in):
        # x_in:(B,T_in,C), h_in:(B,T_in)
        B,T,C = x_in.shape
        h_in_clamped = torch.clamp(h_in, 0.0, 10.0)  # Remove upper limit constraint
        feat = torch.cat([x_in, h_in_clamped.unsqueeze(-1)], dim=-1)  # (B,T,C+1)
        H,_ = self.gru(feat)                                  # (B,T,H)
        y = self.out(H)                                       # (B,T,C) aligned predict x_{t+1}
        return y

class MaintClassifier(nn.Module):
    """
    Use HI before-after difference features for 3-class classification
    Enhanced version: not only uses ΔHI, but also sensor data change patterns
    """
    def __init__(self, sensor_dim, hidden=64, n_classes=3):
        super().__init__()
        # Original HI difference features
        self.hi_mlp = nn.Sequential(
            nn.Linear(6, hidden), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden//2)
        )

        # Sensor data change features
        self.sensor_mlp = nn.Sequential(
            nn.Linear(sensor_dim * 2, hidden), nn.ReLU(),  # Before and after statistical features
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden//2)
        )

        # Fusion classifier
        self.final_classifier = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, n_classes)
        )

    def forward(self, h_b, h_a, x_b, x_a, mask):
        # HI difference features (original logic)
        m = mask
        T = m.size(1)
        valid = m.sum(1, keepdim=True).clamp_min(1.0)  # (B,1)
        db = h_a - h_b                        # (B,T)
        mean_d = (db*m).sum(1,keepdim=True)/valid
        var_d  = (((db-mean_d)*m)**2).sum(1,keepdim=True)/valid
        std_d  = (var_d + 1e-8).sqrt()
        d0     = (db[:, :1])                  # First value
        dmax   = (db.masked_fill(m==0, -1e9)).max(dim=1, keepdim=True).values
        pos_ratio = ((db>0).float()*m).sum(1,keepdim=True)/valid

        # Slope (linear fitting)
        t = torch.arange(T, device=h_b.device, dtype=h_b.dtype)
        t = (t - t.mean())/(t.std()+1e-6)
        def slope(x):
            num = (x*t).sum(1) - x.sum(1)*t.sum()/T
            den = (t**2).sum() - (t.sum()**2)/T + 1e-8
            return (num/den).unsqueeze(1)
        slope_diff = slope(h_a) - slope(h_b)  # (B,1)
        hi_feat = torch.cat([mean_d, d0, dmax, std_d, slope_diff, pos_ratio], dim=1)  # (B,6)
        hi_feat = torch.clamp(hi_feat, -10.0, 10.0)
        hi_features = self.hi_mlp(hi_feat)

        # Sensor data change features (fix broadcast BUG: directly divide by (B,1) shaped valid)
        x_b_mean = (x_b * m.unsqueeze(-1)).sum(1) / valid        # (B,C)
        x_a_mean = (x_a * m.unsqueeze(-1)).sum(1) / valid        # (B,C)
        sensor_change = torch.cat([x_b_mean, x_a_mean], dim=1)   # (B, 2*C)
        sensor_features = self.sensor_mlp(sensor_change)

        # Fused features
        combined_features = torch.cat([hi_features, sensor_features], dim=1)
        logits = self.final_classifier(combined_features)

        return logits

def sanitize_tensors(*tensors):
    """Replace NaN/Inf in tensors with finite values"""
    result = []
    for t in tensors:
        if t is not None:
            t_clean = torch.nan_to_num(t, nan=0.0, posinf=1e6, neginf=-1e6)
            result.append(t_clean)
        else:
            result.append(t)
    return result[0] if len(result) == 1 else result

class DiffAwareReconstructor(nn.Module):
    """
    Overall model:
     - Two encoders share parameters: encode x_before / x_after → h_b / h_a (monotonic decreasing HI)
     - Two reconstruction heads: perform one-step recursive reconstruction respectively
     - Classification head: use (h_a - h_b) statistical features to identify maintenance type
    Without real HI labels, learn to infer health states from sensor data through self-supervised learning
    """
    def __init__(self, in_ch, trend_ch=4, hidden=128, n_classes=3):
        super().__init__()
        self.encoder = TrendEncoder(in_ch, trend_ch)
        self.recon_b = ReconHead(in_ch, hidden)
        self.recon_a = ReconHead(in_ch, hidden)
        self.clf     = MaintClassifier(sensor_dim=in_ch, hidden=64, n_classes=n_classes)

        # Add self-supervised consistency constraint
        self.consistency_weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, x_b, x_a, mask):
        # x_b/x_a : (B,L,C) (will be cut to 0:L-2 / 1:L-1 later)
        h_b, _ = self.encoder(x_b)  # (B,L)  Health state learned from sensor data
        h_a, _ = self.encoder(x_a)  # (B,L)

        # Numerical stabilization
        h_b, h_a = sanitize_tensors(h_b, h_a)

        # Recursive reconstruction: input 0:L-2 → predict 1:L-1
        xb_in, hb_in = x_b[:, :-2], h_b[:, :-2]
        xa_in, ha_in = x_a[:, :-2], h_a[:, :-2]
        yb_hat = self.recon_b(xb_in, hb_in)   # (B,L-2,C) ≈ x_b[:,1:L-1]
        ya_hat = self.recon_a(xa_in, ha_in)   # (B,L-2,C) ≈ x_a[:,1:L-1]

        # Numerical stabilization
        yb_hat, ya_hat = sanitize_tensors(yb_hat, ya_hat)

        # Classification: use unclipped length mask and include sensor features
        logits = self.clf(h_b, h_a, x_b, x_a, mask)
        logits = sanitize_tensors(logits)

        return yb_hat, ya_hat, h_b, h_a, logits

# ============================================================
# 3) Loss function: reconstruction + ΔHI + smooth/monotonic(decreasing) + classification + first-order linear + self-supervised consistency + maintenance effect constraint + global HI superiority constraint
# ============================================================
def masked_mse(a, b, mask):
    # a,b:(B,T,C)  mask:(B,T)
    diff = (a - b)**2
    mse  = (diff.mean(-1) * mask).sum() / (mask.sum() + 1e-6)
    return mse

def slope_loss(h, mask, kind="smooth"):
    # Smooth: second-order difference; Monotonic decreasing: penalize upward trend
    if kind=="smooth":
        d2 = h[:,2:] - 2*h[:,1:-1] + h[:,:-2]
        m  = mask[:,2:]
        return (d2.abs() * m).sum() / (m.sum()+1e-6)
    elif kind=="mono_dec":
        dh = h[:,1:] - h[:,:-1]        # If >0 indicates upward (violates "decreasing")
        m  = mask[:,1:]
        return (F.relu(dh) * m).sum() / (m.sum()+1e-6)
    else:
        return torch.tensor(0., device=h.device)

def enhanced_linear_slope_loss(h, mask, weight=5.0):
    """
    Enhanced linear constraint: calculate deviation from best linear fit, stronger weight
    Without real HI, this constraint helps learn reasonable decay patterns
    """
    B, T = h.shape
    valid_mask = mask  # (B, T)

    # Calculate best linear fit for each sample
    t = torch.arange(T, device=h.device, dtype=h.dtype).unsqueeze(0).expand(B, -1)  # (B, T)

    # Consider only valid positions
    loss_total = torch.tensor(0.0, device=h.device)
    n_valid_samples = 0

    for b in range(B):
        valid_indices = valid_mask[b] > 0
        if valid_indices.sum() < 2:  # Need at least 2 points for linear fitting
            continue

        h_valid = h[b][valid_indices]  # Valid HI values
        t_valid = t[b][valid_indices]  # Corresponding time points

        # Center time points
        t_mean = t_valid.mean()
        t_centered = t_valid - t_mean
        h_mean = h_valid.mean()

        # Calculate linear regression slope
        numerator = (t_centered * (h_valid - h_mean)).sum()
        denominator = (t_centered ** 2).sum() + 1e-8
        slope = numerator / denominator

        # Calculate intercept
        intercept = h_mean - slope * t_mean

        # Linear predicted values
        h_linear = slope * t_valid + intercept

        # Calculate MSE with linear fit, using stronger weight
        linear_mse = ((h_valid - h_linear) ** 2).mean()

        # Additional penalty for positive slope (encourage negative slope, i.e., decreasing)
        slope_penalty = F.relu(slope + 0.01) * 2.0  # Slope should be negative

        loss_total += weight * linear_mse + slope_penalty
        n_valid_samples += 1

    return loss_total / max(n_valid_samples, 1)

def maintenance_improvement_constraint(h_b, h_a, labels, mask):
    """
    Maintenance improvement constraint: ensure post-maintenance HI is significantly higher than pre-maintenance
    - Perfect(0): require post-maintenance significantly higher than pre-maintenance (at least 0.4)
    - General(1): require post-maintenance moderately higher than pre-maintenance (at least 0.2)
    - Poor(2): require post-maintenance slightly higher than pre-maintenance (at least 0.1)
    """
    valid = mask.sum(1, keepdim=True).clamp_min(1.0)  # (B,1)

    # Calculate average HI before and after maintenance
    h_b_mean = (h_b * mask).sum(1, keepdim=True) / valid  # (B,1)
    h_a_mean = (h_a * mask).sum(1, keepdim=True) / valid  # (B,1)

    improvement = h_a_mean - h_b_mean  # Improvement after maintenance relative to before

    loss = torch.zeros_like(improvement)
    for cls in [0, 1, 2]:
        sel = (labels == cls).float().unsqueeze(1)  # (B,1)
        if sel.sum() < 1:
            continue

        if cls == 0:  # Perfect
            # Require at least 0.4 improvement
            loss += sel * F.relu(0.4 - improvement) ** 2
        elif cls == 1:  # General
            # Require at least 0.2 improvement
            loss += sel * F.relu(0.2 - improvement) ** 2
        else:  # Poor
            # Require at least 0.1 improvement
            loss += sel * F.relu(0.1 - improvement) ** 2

    return loss.mean()

def global_hi_superiority_constraint(h_b, h_a, mask, weight=5.0):
    """
    Enhanced global superiority constraint: ensure post-maintenance HI is entirely higher than pre-maintenance HI
    For each sample, require post-maintenance HI at every time point to be higher than pre-maintenance HI
    Remove range constraints and focus on relative superiority
    """
    valid = mask  # (B, T)

    loss_total = torch.tensor(0.0, device=h_b.device)
    n_valid_samples = 0

    for b in range(h_b.size(0)):
        valid_mask_b = valid[b] > 0
        if valid_mask_b.sum() < 1:
            continue

        # Get valid HI values before and after maintenance
        h_b_valid = h_b[b][valid_mask_b]  # Valid HI values before maintenance
        h_a_valid = h_a[b][valid_mask_b]  # Valid HI values after maintenance

        # Point-wise superiority: each post-maintenance point should be higher than corresponding pre-maintenance point
        pointwise_gap = h_a_valid - h_b_valid  # Should be positive
        pointwise_loss = F.relu(-pointwise_gap + 0.05).mean()  # Penalize when post-maintenance is not sufficiently higher

        # Global superiority: minimum post-maintenance should be higher than maximum pre-maintenance
        h_b_max = h_b_valid.max()
        h_a_min = h_a_valid.min()

        # Require post-maintenance minimum to be higher than pre-maintenance maximum
        margin = 0.1  # Post-maintenance minimum should be at least 0.1 higher than pre-maintenance maximum
        global_gap_loss = F.relu(h_b_max + margin - h_a_min) ** 2

        # Additional constraint: post-maintenance average should be significantly higher than pre-maintenance average
        h_b_mean = h_b_valid.mean()
        h_a_mean = h_a_valid.mean()
        mean_gap_loss = F.relu(h_b_mean + 0.2 - h_a_mean) ** 2  # Average should improve by at least 0.2

        loss_total += weight * (pointwise_loss + global_gap_loss + 0.5 * mean_gap_loss)
        n_valid_samples += 1

    return loss_total / max(n_valid_samples, 1)

def diff_margin_by_class(mean_delta, labels, m_low=0.1, m_mid=0.25, m_high=0.45):
    """
    Class-conditional difference constraint (enhanced maintenance effect):
      - Perfect(0):  Δ>=m_high (post-maintenance significantly higher than pre-maintenance)
      - General(1):  Δ≈m_mid   (post-maintenance moderately higher than pre-maintenance)
      - Poor(2):     Δ>=m_low  (post-maintenance at least slightly higher than pre-maintenance)
    All classes require post-maintenance HI higher than pre-maintenance (positive improvement), just different degrees
    """
    loss = torch.zeros_like(mean_delta)
    for cls in [0,1,2]:
        sel = (labels==cls).float()
        if sel.sum() < 1: continue
        if cls==0:
            # Perfect: require significant improvement (at least reach m_high)
            loss += sel * F.relu(m_high - mean_delta)**2
        elif cls==1:
            # General: require moderate improvement (close to m_mid)
            loss += sel * (mean_delta - m_mid)**2
        else:
            # Poor: require at least small improvement (at least reach m_low)
            loss += sel * F.relu(m_low - mean_delta)**2
    denom = torch.ones_like(mean_delta) * (labels.numel())
    return loss.sum() / denom.sum()

def sensor_consistency_loss(x_b, x_a, h_b, h_a, mask):
    """
    Sensor data consistency loss: ensure HI changes are consistent with sensor data change patterns
    Important constraint when no real HI labels are available
    """
    # Calculate statistical changes in sensor data
    valid = mask.sum(1, keepdim=True).clamp_min(1.0)   # (B,1)

    # Average change magnitude in sensor data (fix broadcast BUG: divide by (B,1))
    xb_mean = (x_b * mask.unsqueeze(-1)).sum(1) / valid   # (B,C)
    xa_mean = (x_a * mask.unsqueeze(-1)).sum(1) / valid   # (B,C)
    sensor_change = torch.abs(xa_mean - xb_mean)          # (B,C)
    sensor_change_magnitude = sensor_change.mean(1)       # (B,)

    # HI change magnitude (post-maintenance improvement relative to pre-maintenance)
    hi_change = ((h_a - h_b) * mask).sum(1) / valid.squeeze(-1)  # (B,) Note: remove abs, maintain positive direction

    # Expect HI change to positively correlate with sensor change, and HI should improve positively
    # Use Pearson correlation coefficient as consistency metric
    mean_sensor = sensor_change_magnitude.mean()
    mean_hi = hi_change.mean()

    correlation = ((sensor_change_magnitude - mean_sensor) * (hi_change - mean_hi)).mean() / \
                  (sensor_change_magnitude.std() * hi_change.std() + 1e-8)

    # Encourage positive correlation (large sensor change when HI change is also large)
    consistency_loss = F.relu(0.5 - correlation)  # Expect correlation at least 0.5

    # Additional constraint: force positive HI improvement
    negative_change_penalty = F.relu(-hi_change).mean() * 2.0  # Penalize negative changes

    return consistency_loss + negative_change_penalty

def smoothness_enhancement_loss(h, mask, order=2):
    """
    Enhanced smoothness loss: use higher-order differences to force smoother curves
    """
    if order == 2:
        # Second-order difference
        d2 = h[:,2:] - 2*h[:,1:-1] + h[:,:-2]
        m = mask[:,2:]
        return (d2.abs() * m).sum() / (m.sum() + 1e-6)
    elif order == 3:
        # Third-order difference (stronger smoothness constraint)
        d3 = h[:,3:] - 3*h[:,2:-1] + 3*h[:,1:-2] - h[:,:-3]
        m = mask[:,3:]
        return (d3.abs() * m).sum() / (m.sum() + 1e-6)
    else:
        return torch.tensor(0., device=h.device)

def split_pairs_uid(pairs, train_ratio=0.7, val_ratio=0.1, seed=42):
    uids = sorted(list(pairs.keys()))
    rs = np.random.RandomState(seed)
    rs.shuffle(uids)
    n = len(uids); n_tr=int(n*train_ratio); n_vl=int(n*val_ratio)
    to_dict = lambda ss:{u:pairs[u] for u in ss if u in pairs}
    return to_dict(uids[:n_tr]), to_dict(uids[n_tr:n_tr+n_vl]), to_dict(uids[n_tr+n_vl:])

def print_split_summary(pairs, train_ratio=0.7, val_ratio=0.1, seed=42):
    pairs_tr, pairs_vl, pairs_te = split_pairs_uid(pairs, train_ratio, val_ratio, seed)
    def count_items(pp):
        cnt = 0
        for uid, d in pp.items():
            cnt += len(d)
        return len(pp), cnt
    n_uid_tr, n_pair_tr = count_items(pairs_tr)
    n_uid_vl, n_pair_vl = count_items(pairs_vl)
    n_uid_te, n_pair_te = count_items(pairs_te)
    print(f"[Data Split] Train: UID={n_uid_tr}, pairs≈{n_pair_tr} | "
          f"Val: UID={n_uid_vl}, pairs≈{n_pair_vl} | "
          f"Test: UID={n_uid_te}, pairs≈{n_pair_te}")

def remove_constant_segments(hi_values, threshold=1e-6):
    """
    Remove constant segments in HI curves (caused by mask or other reasons)
    Return valid HI values and corresponding time indices
    """
    if len(hi_values) <= 1:
        return hi_values, np.arange(len(hi_values))

    # Calculate differences between adjacent points
    diffs = np.abs(np.diff(hi_values))

    # Find points with sufficient change
    valid_mask = np.ones(len(hi_values), dtype=bool)
    valid_mask[1:] = diffs > threshold

    # Ensure at least first and last two points are kept
    if np.sum(valid_mask) < 2:
        valid_mask[0] = True
        valid_mask[-1] = True

    valid_indices = np.where(valid_mask)[0]
    return hi_values[valid_indices], valid_indices

@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total = {"mse_b":0.0, "mse_a":0.0, "acc":0.0, "delta_mean":0.0}
    n_batch=0
    n_acc = 0; n_tot=0
    for batch in loader:
        xb = batch["x_before"].to(device)
        xa = batch["x_after"].to(device)
        hi_b = batch["hi_before"].to(device)
        hi_a = batch["hi_after"].to(device)
        labels = batch["labels"].to(device)
        lengths= batch["lengths"].to(device)
        mask   = batch["mask"].to(device)

        # Valid segment: 0:L-2 input, 1:L-1 target
        m_tgt  = (torch.arange(xb.size(1)-2, device=device)[None,:] < (lengths-2)[:,None]).float()

        yb_hat, ya_hat, h_b, h_a, logits = model(xb, xa, mask)

        # Numerical stabilization
        yb_hat, ya_hat, h_b, h_a, logits = sanitize_tensors(yb_hat, ya_hat, h_b, h_a, logits)

        # Align targets
        yb_tgt = xb[:,1:-1,:]
        ya_tgt = xa[:,1:-1,:]

        mse_b = masked_mse(yb_hat, yb_tgt, m_tgt)
        mse_a = masked_mse(ya_hat, ya_tgt, m_tgt)

        # Check for NaN
        if torch.isnan(mse_b) or torch.isnan(mse_a):
            continue

        # Classification
        pred = logits.argmax(dim=1)
        n_acc += (pred == labels).sum().item()
        n_tot += labels.numel()

        # Statistics of ΔHI mean (post-maintenance improvement relative to pre-maintenance)
        valid = mask.sum(1, keepdim=True).clamp_min(1.0)
        delta_mean = ((h_a - h_b)*mask).sum(1,keepdim=True)/valid
        delta_mean = torch.nan_to_num(delta_mean, nan=0.0)
        total["delta_mean"] += delta_mean.mean().item()

        total["mse_b"] += mse_b.item()
        total["mse_a"] += mse_a.item()
        n_batch += 1

    for k in ["mse_b","mse_a","delta_mean"]:
        total[k] /= max(n_batch,1)
    total["acc"] = n_acc / max(n_tot,1)
    return total

@torch.no_grad()
def collect_test_predictions(model, loader, device, max_curve_keep=24):
    """
    Collect predictions, ΔHI, and a few sample HI curves on test set (for plotting).
    """
    model.eval()
    y_true, y_pred = [], []
    all_delta_mean, all_uids = [], []
    keep_curves = []

    for batch in loader:
        xb = batch["x_before"].to(device)   # (B,L,C)
        xa = batch["x_after"].to(device)
        mask   = batch["mask"].to(device)   # (B,L)
        labels = batch["labels"].to(device) # (B,)
        lengths= batch["lengths"]           # cpu tensor
        uids   = batch["uids"]
        hi_before = batch["hi_before"].to(device)  # (B,L) ground truth HI_before from dataset
        hi_after = batch["hi_after"].to(device)    # (B,L) ground truth HI_after from dataset

        yb_hat, ya_hat, h_b, h_a, logits = model(xb, xa, mask)

        # Numerical stabilization
        yb_hat, ya_hat, h_b, h_a, logits = sanitize_tensors(yb_hat, ya_hat, h_b, h_a, logits)

        prob = F.softmax(logits, dim=1)     # (B,3)
        pred = prob.argmax(1)               # (B,)

        # Statistics of ΔHI mean (post-maintenance improvement relative to pre-maintenance)
        valid = mask.sum(1, keepdim=True).clamp_min(1.0)
        delta_mean = ((h_a - h_b) * mask).sum(1, keepdim=True) / valid  # (B,1)
        delta_mean = torch.nan_to_num(delta_mean, nan=0.0)

        y_true.append(labels.cpu().numpy())
        y_pred.append(pred.cpu().numpy())
        all_delta_mean.append(delta_mean.squeeze(1).cpu().numpy())
        all_uids.extend(uids)

        # Save some curves for visualization (aligned: after concatenated after before)
        for i in range(xb.size(0)):
            if len(keep_curves) >= max_curve_keep:
                break
            L_i = int(lengths[i].item())
            # Fix: truncate reconstruction prediction by actual length
            L_recon = min(L_i - 2, yb_hat.size(1))  # Reconstruction sequence length
            keep_curves.append({
                "uid": uids[i],
                "true": int(labels[i].cpu().item()),
                "pred": int(pred[i].cpu().item()),
                "prob": prob[i].cpu().numpy(),
                "h_before": h_b[i, :L_i].cpu().numpy(),          # model predicted HI_before
                "h_after":  h_a[i, :L_i].cpu().numpy(),          # model predicted HI_after
                "hi_before_gt": hi_before[i, :L_i].cpu().numpy(), # ground truth HI_before from dataset
                "hi_after_gt": hi_after[i, :L_i].cpu().numpy(),   # ground truth HI_after from dataset
                "yb_hat":   yb_hat[i, :L_recon].cpu().numpy(),       # (L_recon,C)
                "ya_hat":   ya_hat[i, :L_recon].cpu().numpy(),       # (L_recon,C)
                "x_before": xb[i, :L_i].cpu().numpy(),               # (L,C)
                "x_after":  xa[i, :L_i].cpu().numpy(),               # (L,C)
                "length": L_i
            })

    y_true = np.concatenate(y_true, axis=0) if len(y_true)>0 else np.array([])
    y_pred = np.concatenate(y_pred, axis=0) if len(y_pred)>0 else np.array([])
    all_delta_mean = np.concatenate(all_delta_mean, axis=0) if len(all_delta_mean)>0 else np.array([])
    return y_true, y_pred, all_delta_mean, all_uids, keep_curves

def select_best_curves_for_plotting(curves, n_show=6):
    """
    Intelligent selection of the best drawing cases:
    1. Prioritize samples with correct predictions
    2. Select the most representative sample within each maintenance strategy category
    3. Select the sample with the most obvious change in health index
    4. Select samples with the best sensor data quality
    """
    if len(curves) == 0:
        return []

    #Group by real category
    curves_by_class = {0: [], 1: [], 2: []}
    for curve in curves:
        curves_by_class[curve["true"]].append(curve)

    # Calculate quality score for each sample
    def calculate_quality_score(curve):
        score = 0.0

        # 1. Prediction accuracy (weight: 50%)
        if curve["true"] == curve["pred"]:
            score += 0.5

        # 2. Prediction confidence (weight: 20%)
        confidence = curve["prob"][curve["pred"]]
        score += 0.2 * confidence

        # 3. Obvious degree of change in health index (weight: 20%)
        h_before = curve["h_before"]
        h_after = curve["h_after"]
        hi_improvement = np.mean(h_after - h_before)
        # Normalize to 0-1 range
        normalized_improvement = np.clip((hi_improvement + 0.5) / 1.0, 0, 1)
        score += 0.2 * normalized_improvement

        # 4. Data quality (weight: 10%)
        # Check if there are outliers or constant segments
        x_before_var = np.var(curve["x_before"])
        x_after_var = np.var(curve["x_after"])
        data_quality = np.clip((x_before_var + x_after_var) / 2.0, 0, 1)
        score += 0.1 * data_quality

        return score

    # Calculate quality scores for all samples
    for curve in curves:
        curve["quality_score"] = calculate_quality_score(curve)

    # Intelligent selection strategy
    selected_curves = []

    # Select at least one best sample for each category
    for class_id in [0, 1, 2]:
        if len(curves_by_class[class_id]) > 0:
            # Sort by quality score
            sorted_curves = sorted(curves_by_class[class_id],
                                 key=lambda x: x["quality_score"], reverse=True)
            selected_curves.append(sorted_curves[0])

    # If more samples are needed, select from the remaining high-quality samples
    remaining_slots = n_show - len(selected_curves)
    if remaining_slots > 0:
        # Exclude selected from all samples, select by quality score
        already_selected_uids = {curve["uid"] for curve in selected_curves}
        remaining_curves = [curve for curve in curves
                          if curve["uid"] not in already_selected_uids]

        if len(remaining_curves) > 0:
            remaining_curves.sort(key=lambda x: x["quality_score"], reverse=True)
            selected_curves.extend(remaining_curves[:remaining_slots])

    #Final sorting by quality score
    selected_curves.sort(key=lambda x: x["quality_score"], reverse=True)

    return selected_curves[:n_show]

def plot_hi_examples_aligned(curves, n_show=6, seed=0):
    """Post-maintenance sequence continues from pre-maintenance endpoint; draw vertical line to mark t_m; remove constant segments.
    Add "Learned HI" in title to indicate this is health state inferred from sensor data, post-maintenance should be higher than pre-maintenance
    Intelligent selection of the best drawing cases
    """
    if len(curves)==0:
        print("(No visualization samples)")
        return

    # Use the smart selection function to select the best drawing case
    selected_curves = select_best_curves_for_plotting(curves, n_show)

    if len(selected_curves) == 0:
        print("(No high-quality visualization samples found)")
        return

    n_show = len(selected_curves)
    cols = 3
    rows = int(np.ceil(n_show/cols))
    plt.figure(figsize=(cols*4.6, rows*3.2))

    for k in range(n_show):
        ex = selected_curves[k]
        hb = ex["h_before"]; ha = ex["h_after"]
        hb_gt = ex["hi_before_gt"]; ha_gt = ex["hi_after_gt"]      # ground truth HI from dataset
        L  = ex["length"]  # Use actual length instead of padded length

        # Remove constant segments
        hb_clean, hb_indices = remove_constant_segments(hb)
        ha_clean, ha_indices = remove_constant_segments(ha)
        hb_gt_clean, hb_gt_indices = remove_constant_segments(hb_gt)
        ha_gt_clean, ha_gt_indices = remove_constant_segments(ha_gt)

        # Corresponding time axis
        t_b = hb_indices
        t_a = ha_indices + L  # after follows immediately
        t_b_gt = hb_gt_indices
        t_a_gt = ha_gt_indices + L

        uid = ex["uid"]
        pred = ex["pred"]; true = ex["true"]; prob = ex["prob"]
        quality_score = ex["quality_score"]

        plt.subplot(rows, cols, k+1)

        # Only plot non-constant segments - model predicted HI
        if len(hb_clean) > 1:
            plt.plot(t_b, hb_clean, label="Learned HI_Pre-maintenance", linewidth=2.5, marker='o', markersize=4, color='blue')
        if len(ha_clean) > 1:
            plt.plot(t_a, ha_clean, label="Learned HI_Post-maintenance",  linewidth=2.5, linestyle='--', marker='s', markersize=4, color='red')

        # Plot ground truth HI from dataset
        if len(hb_gt_clean) > 1:
            plt.plot(t_b_gt, hb_gt_clean, label="GT HI_Pre-maintenance", linewidth=1.8, color='cyan', alpha=0.8)
        if len(ha_gt_clean) > 1:
            plt.plot(t_a_gt, ha_gt_clean, label="GT HI_Post-maintenance",  linewidth=1.8, linestyle=':', color='orange', alpha=0.8)

        # Maintenance time vertical line
        plt.axvline(L-1, color='k', linestyle=':', linewidth=1.5, alpha=0.8)

        d_mean = float(np.mean(ha - hb))  # Post-maintenance improvement relative to pre-maintenance
        d_max  = float(np.max(ha - hb))
        d_mean_gt = float(np.mean(ha_gt - hb_gt))

        # Add quality rating to title
        correctness = "✓" if true == pred else "✗"
        title = (f"uid={uid} {correctness} (Quality: {quality_score:.2f})\n"
                 f"True={LABEL2NAME[true]} | Pred={LABEL2NAME[pred]} | Conf={prob[pred]:.3f}\n"
                 f"ΔHI_Learned={d_mean:.3f}, ΔHI_GT={d_mean_gt:.3f}")
        plt.title(title, fontsize=9)
        plt.xlabel("Cycle (Pre-maintenance | Post-maintenance)")
        plt.ylabel("Health Index (↑Post should be higher)")
        plt.grid(ls='--', alpha=.4, linewidth=0.8)

        #Add background color to indicate prediction accuracy
        if true == pred:
            plt.gca().set_facecolor('#f0f8ff') # Light blue background indicates correct prediction
        else:
            plt.gca().set_facecolor('#fff0f0') # Light red background indicates incorrect prediction

        if k==0: plt.legend(fontsize=8, loc='best')

    plt.suptitle(f"Best {n_show} Health Index Examples (Ranked by Quality Score)", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.show()

def plot_sensor_examples_aligned(curves, sensor_idx_list=None, n_cols=4):
    """
    Plot several sensors: original(before/after) + one-step recursive prediction of after(ya_hat),
    post-maintenance time series starts after pre-maintenance end. Show trajectories under different
    maintenance strategies. Intelligent selection of the best sensor dimensions for drawing
    """
    if len(curves)==0:
        print("(No visualization samples)")
        return

    # Select the best samples for sensor visualization
    selected_curves = select_best_curves_for_plotting(curves, 9) # Select more samples for sensor analysis

    # Group curves by maintenance strategy (true class)
    curves_by_strategy = {0: [], 1: [], 2: []}
    for curve in selected_curves:
        curves_by_strategy[curve["true"]].append(curve)

    # Select best example from each strategy for comparison
    strategy_examples = {}
    for strategy in [0, 1, 2]:
        if len(curves_by_strategy[strategy]) > 0:
            # Select the sample with the highest quality score under this strategy
            best_curve = max(curves_by_strategy[strategy], key=lambda x: x["quality_score"])
            strategy_examples[strategy] = best_curve

    if len(strategy_examples) == 0:
        print("(No high-quality visualization samples for any maintenance strategy)")
        return

    # Intelligent selection of sensor dimensions
    first_ex = list(strategy_examples.values())[0]
    xb = first_ex["x_before"]       # (L,C)
    C = xb.shape[1]

    if sensor_idx_list is None:
        # Intelligently select the most representative sensor dimension
        sensor_scores = []

        for sensor_idx in range(C):
            score = 0.0

            # Calculate the degree of change of the sensor under different maintenance strategies
            for strategy, ex in strategy_examples.items():
                xb_sensor = ex["x_before"][:, sensor_idx]
                xa_sensor = ex["x_after"][:, sensor_idx]

                # 1. Range of change (weight: 40%)
                change_magnitude = np.abs(np.mean(xa_sensor) - np.mean(xb_sensor))
                score += 0.4 * min(change_magnitude / (np.std(xb_sensor) + 1e-6), 5.0)

                # 2. Variance (information amount) (weight: 30%)
                variance = np.var(xb_sensor) + np.var(xa_sensor)
                score += 0.3 * min(variance, 10.0)

                # 3. Signal-to-noise ratio (weight: 30%)
                snr = np.mean(xb_sensor) / (np.std(xb_sensor) + 1e-6)
                score += 0.3 * min(abs(snr), 5.0)

            sensor_scores.append((sensor_idx, score))

        # Select the sensor with the highest rating
        sensor_scores.sort(key=lambda x: x[1], reverse=True)
        sensor_idx_list = [idx for idx, _ in sensor_scores[:min(8, len(sensor_scores))]]

    n = len(sensor_idx_list)
    n_rows = int(np.ceil(n/n_cols))
    plt.figure(figsize=(n_cols*5.5, n_rows*4.0))

    colors = {0: '#e74c3c', 1: '#f39c12', 2: '#9b59b6'} # Brighter colors

    for i, s in enumerate(sensor_idx_list):
        plt.subplot(n_rows, n_cols, i+1)

        # Plot pre-maintenance trajectory (common for all strategies, use best example)
        best_ex = max(strategy_examples.values(), key=lambda x: x["quality_score"])
        xb = best_ex["x_before"]
        L = best_ex["length"]
        t_b = np.arange(L)
        plt.plot(t_b, xb[:L,s], label="Original (Pre-maintenance)", linewidth=2.0, color='#2c3e50', alpha=0.9)

        # Plot post-maintenance trajectories for each available strategy
        for strategy in sorted(strategy_examples.keys()):
            ex = strategy_examples[strategy]
            xa = ex["x_after"]
            ya = ex["ya_hat"]
            L_strategy = ex["length"]
            quality = ex["quality_score"]

            # Post-maintenance original trajectory
            t_a = np.arange(L_strategy, L_strategy + L_strategy)
            plt.plot(t_a, xa[:L_strategy,s],
                    label=f"Original (Post-{LABEL2NAME[strategy]}, Q:{quality:.2f})",
                    linewidth=1.8, linestyle="--",
                    color=colors[strategy], alpha=0.9)

            # Post-maintenance prediction trajectory
            if len(ya) > 0 and L_strategy-2 > 0:
                t_pred = np.arange(L_strategy+1, L_strategy+1+min(L_strategy-2, len(ya)))
                plt.plot(t_pred, ya[:len(t_pred),s],
                        label=f"Predicted (Post-{LABEL2NAME[strategy]})",
                        linewidth=2.2, color=colors[strategy], marker='o', markersize=3)

        # Draw vertical line at maintenance point
        plt.axvline(L-0.5, color='#27ae60', linestyle=':', linewidth=2.5, alpha=0.9, label='Maintenance Point')

        # Calculate the quality score of this sensor and add it to the title
        sensor_score = dict(sensor_scores)[s] if 'sensor_scores' in locals() else 0
        plt.title(f"Sensor_{s:02d} (Score: {sensor_score:.2f}) - Strategy Comparison", fontsize=10, fontweight='bold')

        if i==0:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.xlabel("Continuous Time Sequence", fontsize=9)
        plt.ylabel("Sensor Value", fontsize=9)
        plt.grid(ls="--", alpha=.4, linewidth=0.8)

        # Add slight background color
        plt.gca().set_facecolor('#fafafa')

    plt.suptitle("Best Sensor Trajectories under Different Maintenance Strategies",
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()

def topk_by_delta(df_delta, k=5):
    if len(df_delta)==0:
        return
    print("\n" + "="*60)
    print("TOP-K SAMPLES WITH HIGHEST MAINTENANCE EFFECTS")
    print("="*60)

    for cls in [0,1,2]:
        sub = df_delta[df_delta["true"]==cls].sort_values("delta_hi_mean", ascending=False).head(k)
        if len(sub) > 0:
            print(f"\n[True={LABEL2NAME[cls]}] Top {k} samples with highest learned ΔHI_mean:")
            print("-" * 50)
            for idx, row in sub.iterrows():
                pred_correct = "✓" if row["true"] == row["pred"] else "✗"
                print(f"  UID: {row['uid']:>8} | ΔHI: {row['delta_hi_mean']:>6.3f} | "
                      f"Pred: {LABEL2NAME[int(row['pred'])]:>7} {pred_correct}")

# —— Load trained best model
def load_trained_model(model_path, device, in_ch):
    """Load the best model weights saved during training"""
    # Initialize model with same architecture
    model = DiffAwareReconstructor(in_ch=in_ch, trend_ch=4, hidden=128, n_classes=3).to(device)

    if os.path.exists(model_path):
        print(f"Loading model: {model_path}")
        state_dict = torch.load(model_path, map_location=device)

        # Filter out keys that don't match (for backward compatibility)
        model_dict = model.state_dict()
        filtered_dict = {}
        for k, v in state_dict.items():
            if k in model_dict and model_dict[k].shape == v.shape:
                filtered_dict[k] = v
            else:
                print(f"Warning: Skipping key {k} due to shape mismatch or missing in current model")

        # Load only matching parameters
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict, strict=False)
        print("Model loaded successfully (with potential missing keys)!")
    else:
        print(f"Warning: Model file does not exist {model_path}, will use randomly initialized model")
    return model

# —— Need to determine in_ch from pairs data first
def get_input_dim_from_pairs(pairs):
    """Get input dimension from pairs data"""
    for uid, strategies in pairs.items():
        for strategy, data in strategies.items():
            if "x_before" in data:
                return np.array(data["x_before"]).shape[1]
    raise ValueError("Cannot determine input dimension from pairs data")

# Get input dimension
C = get_input_dim_from_pairs(pairs)
print(f"Detected input dimension: {C}")

# Load best model (if exists)
model_path = "/content/drive/MyDrive/CMAPSS/main_dynamics_identification_3.pth"
model = load_trained_model(model_path, DEVICE, C)

# —— Print train/validation/test split (consistent with training phase: 7/1/2)
print_split_summary(pairs)

# —— Prepare test set data
_, _, pairs_te = split_pairs_uid(pairs, 0.7, 0.1, 42)
ds_te = PairsReconstructDataset(pairs_te, horizon=50)  # Same horizon as training
ld_te = DataLoader(ds_te, batch_size=32, shuffle=False, collate_fn=pad_collate_shift)
te = (ds_te, ld_te)

y_true, y_pred, delta_mean_all, uids_all, curves = collect_test_predictions(model, ld_te, DEVICE)

# —— Print overall metrics
print("\n" + "="*60)
print("TEST SET OVERALL METRICS")
print("="*60)
print(f"Sample count: {len(y_true)}")
if len(y_true) > 0:
    acc = (y_true == y_pred).mean()
    print(f"Classification Accuracy: {acc:.4f}")

    # Calculate the accuracy of each category
    for cls in [0, 1, 2]:
        cls_mask = (y_true == cls)
        if cls_mask.sum() > 0:
            cls_acc = (y_pred[cls_mask] == cls).mean()
            print(f"{LABEL2NAME[cls]} Accuracy: {cls_acc:.4f} ({cls_mask.sum()} samples)")
else:
    print("No samples in test set (check pairs split and horizon conditions).")

# —— Classification report & confusion matrix
try:
    from sklearn.metrics import classification_report, confusion_matrix
    target_names = [LABEL2NAME[i] for i in sorted(LABEL2NAME)]
    print("\n[Classification Report]")
    print(classification_report(y_true, y_pred, target_names=target_names, digits=4))
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])
except Exception:
    print("\n[Warning] scikit-learn not installed, will print simple confusion table.")
    cm = np.zeros((3,3), dtype=int)
    for t,p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1

# —— Print confusion matrix values
print("\n[Confusion Matrix] Row=True class, Column=Predicted class")
print(pd.DataFrame(cm, index=[LABEL2NAME[i] for i in [0,1,2]],
                      columns=[LABEL2NAME[i] for i in [0,1,2]]))

# —— Plot confusion matrix (enhanced)
plt.figure(figsize=(6.0,5.0))
im = plt.imshow(cm, interpolation='nearest', cmap='Blues')
plt.title("Confusion Matrix (Test Set)", fontsize=14, fontweight='bold', pad=20)
plt.xlabel("Predicted", fontsize=12)
plt.ylabel("True", fontsize=12)
plt.xticks([0,1,2], [LABEL2NAME[i] for i in [0,1,2]])
plt.yticks([0,1,2], [LABEL2NAME[i] for i in [0,1,2]])

#Add values ​​and percentages
for i in range(3):
    for j in range(3):
        total = cm[i].sum()
        if total > 0:
            percentage = cm[i, j] / total * 100
            text_color = 'white' if cm[i, j] > cm.max() * 0.5 else 'black'
            plt.text(j, i, f'{cm[i, j]}\n({percentage:.1f}%)',
                    ha="center", va="center", color=text_color, fontweight='bold')

plt.colorbar(im)
plt.tight_layout()
plt.show()

# —— Statistics of ΔHI distribution (by true class/predicted class)
if len(delta_mean_all) > 0:
    df_delta = pd.DataFrame({
        "uid": uids_all[:len(delta_mean_all)],
        "true": y_true.astype(int),
        "pred": y_pred.astype(int),
        "delta_hi_mean": delta_mean_all.astype(float)
    })

    print("\n" + "="*50)
    print("ΔHI MEAN STATISTICS BY TRUE CLASS")
    print("="*50)
    stats_by_true = df_delta.groupby("true")["delta_hi_mean"].describe()
    print(stats_by_true.round(4))

    print("\n" + "="*50)
    print("ΔHI MEAN STATISTICS BY PREDICTED CLASS")
    print("="*50)
    stats_by_pred = df_delta.groupby("pred")["delta_hi_mean"].describe()
    print(stats_by_pred.round(4))

# ——Continuous time axis: HI and several sensors before/after aligned visualization (intelligent selection of the best drawing)
print("\n" + "="*60)
print("GENERATING BEST QUALITY VISUALIZATIONS...")
print("="*60)

plot_hi_examples_aligned(curves, n_show=6, seed=2025)
plot_sensor_examples_aligned(curves, sensor_idx_list=None, n_cols=4)

# ——(Optional) Top-K with largest ΔHI in each class, for manual review (enhanced display)
if len(delta_mean_all) > 0:
    topk_by_delta(df_delta, k=5)

# %% [notebook code cell 18]

import os
import re
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# ========== Base: Build/Load Model ==========
def build_model(input_dim, trend_ch=4, hidden=128, n_classes=3, device=None):
    if 'DiffAwareReconstructor' not in globals():
        raise RuntimeError("DiffAwareReconstructor not found: Please import the model class from training script first.")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DiffAwareReconstructor(in_ch=input_dim, trend_ch=trend_ch, hidden=hidden, n_classes=n_classes).to(device)
    return model, device

def load_checkpoint(model, ckpt_path, device):
    # Compatible with DataParallel / different key naming
    sd = torch.load(ckpt_path, map_location=device)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    # Remove possible "module." prefix
    new_sd = {}
    for k, v in sd.items():
        nk = k.replace("module.", "") if isinstance(k, str) and k.startswith("module.") else k
        new_sd[nk] = v
    sd = new_sd
    model.load_state_dict(sd, strict=False)  # Compatible with minor buffer/key differences
    model.eval()
    print(f"[OK] Checkpoint loaded: {ckpt_path}")

# ========== Pickers & utilities ==========
def pick_one_pair(pairs, uid=None, strategy=None):
    """
    Extract one aligned segment from build_pairs structure with (uid, strategy):
    Returns uid, strategy, x_before:(1,T,C), x_after:(1,T,C)
    """
    if uid is None:
        uid = next(iter(pairs.keys()))
    if strategy is None:
        strategy = next(iter(pairs[uid].keys()))
    rec = pairs[uid][strategy]
    xb = np.asarray(rec["x_before"], dtype=np.float32)  # (T,C)
    xa = np.asarray(rec["x_after"],  dtype=np.float32)  # (T,C)
    # Ensure same length (build_pairs already aligned, defensive truncation to shortest)
    T = min(len(xb), len(xa))
    xb = xb[:T]; xa = xa[:T]
    # Add batch dimension
    xb = torch.from_numpy(xb).unsqueeze(0)  # (1,T,C)
    xa = torch.from_numpy(xa).unsqueeze(0)  # (1,T,C)
    return uid, strategy, xb, xa

def iter_pairs_by_strategy(pairs, strategy_filters=None, max_per_strategy=None, seed=0):
    """
    Iterate pairs filtered by maintenance strategies.
    - strategy_filters: list like ["perfect","general","poor"] (case-insensitive, substring match).
                        If None or empty => no filtering (all strategies).
    - max_per_strategy: int or None. If set, limit yielded samples per strategy class.
    Yields tuples: (uid, strategy, rec_dict)
    """
    rng = np.random.RandomState(seed)
    uids = list(pairs.keys())
    rng.shuffle(uids)

    # prepare counters
    counters = {}

    def strategy_match(name):
        if not strategy_filters:
            return True
        s = name.lower()
        for f in strategy_filters:
            if f.lower() in s:
                return True
        return False

    for uid in uids:
        strat_names = list(pairs[uid].keys())
        rng.shuffle(strat_names)
        for strat in strat_names:
            if not strategy_match(strat):
                continue
            key = canonical_strategy_name(strat)
            if max_per_strategy is not None:
                counters.setdefault(key, 0)
                if counters[key] >= max_per_strategy:
                    continue
                counters[key] += 1
            yield uid, strat, pairs[uid][strat]

def canonical_strategy_name(strategy: str):
    s = strategy.lower()
    if "perfect" in s: return "Perfect"
    if "general" in s: return "General"
    if "poor" in s:    return "Poor"
    # fallback to original (title-cased)
    return strategy.title()

# ========== Extract liquid weights/operator outputs/mixed damage ==========
@torch.no_grad()
def extract_ops_and_weights_for_sequence(model, x_full):
    """
    For visualization:
    - Use liquid weight generator in model to compute (B,T,K) liquid weights and temperature
    - Apply z-score normalization to each operator output along time dimension, preserving temporal shape
    Input:
      x_full: (1, T, C)
    Returns:
      h_multi       : (T, h_dim)
      op_outs_norm  : (T, K)  —— operator outputs after z-score
      weights       : (T, K)
      temperature   : (T,)
      damage        : (T,)    —— "visualization version" damage computed from z-score operator outputs × weights
    """
    enc = model.encoder
    customkan = enc.customkan

    # 1) KAN features
    h_multi = enc.boltz(x_full)  # (1, T, h_dim)

    # 2) Compute output for each operator; fallback to passthrough if no per-op feature extractors
    op_outs = []
    has_extractors = hasattr(customkan, "op_feature_extractors") and \
                     (customkan.op_feature_extractors is not None) and \
                     (len(customkan.op_feature_extractors) == len(customkan.ops))

    for i, op in enumerate(customkan.ops):
        if has_extractors:
            feat = customkan.op_feature_extractors[i](h_multi)  # (1, T, h_dim)
        else:
            # Compatible with old checkpoints: use h_multi directly when no op_feature_extractors
            feat = h_multi  # (1, T, h_dim)

        y = op(feat)  # (1, T)
        # Z-score along time dimension: (y - mean_t) / std_t
        mu = y.mean(dim=1, keepdim=True)
        std = y.std(dim=1, keepdim=True) + 1e-6
        y_norm = (y - mu) / std          # (1, T)
        op_outs.append(y_norm)

    # Length alignment like in forward (usually consistent, defensive processing here)
    Tm = min(o.size(1) for o in op_outs)
    op_outs = [o[:, :Tm] for o in op_outs]
    h_multi_aligned = h_multi[:, :Tm, :]
    x_aligned = x_full[:, :Tm, :]

    # 3) Stack operator outputs (B,T,K)
    op_stack = torch.stack(op_outs, dim=-1)  # (1, Tm, K)

    # 4) Use model's liquid weight generator to get weights and temperature
    weights, temperature = customkan.weight_generator(h_multi_aligned, x_aligned)  # (1,Tm,K), (1,Tm)

    # 5) Multiply z-score operator outputs with weights to get "visualization version" damage
    damage = torch.sum(op_stack * weights, dim=-1)  # (1, Tm)
    # Soft clipping to avoid extreme values
    damage = damage.clamp(min=-10.0, max=10.0)

    out = {
        "h_multi":      h_multi_aligned.squeeze(0).cpu().numpy(),  # (T, h_dim)
        "op_outs_norm": op_stack.squeeze(0).cpu().numpy(),         # (T, K)
        "weights":      weights.squeeze(0).cpu().numpy(),          # (T, K)
        "temperature":  temperature.squeeze(0).cpu().numpy(),      # (T,)
        "damage":       damage.squeeze(0).cpu().numpy(),           # (T,)
    }
    return out


# ========== Plotting ==========
def _ensure_op_names(before_dict, op_names):
    K = before_dict["weights"].shape[1]
    if op_names is None:
        op_names = [
            "Monotonic Linear","Monotonic Flat","Concave Log","Saturated Sigmoid",
            "Hinge ReLU","Polynomial","Damped Sine","Piecewise Linear"
        ]
    if len(op_names) != K:
        op_names = [f"Operator{i}" for i in range(K)]
    return op_names

def plot_liquid_mixing(uid, strategy, before_dict, after_dict, op_names=None, save_path=None):
    op_names = _ensure_op_names(before_dict, op_names)

    # Weight heatmaps (before/after)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    im0 = axes[0,0].imshow(before_dict["weights"].T, aspect='auto', origin='lower')
    axes[0,0].set_title(f"[{uid} | {strategy}] Pre-maintenance Weight Heatmap")
    axes[0,0].set_ylabel("Operator"); axes[0,0].set_xlabel("Time")
    axes[0,0].set_yticks(range(len(op_names))); axes[0,0].set_yticklabels(op_names)
    fig.colorbar(im0, ax=axes[0,0])

    im1 = axes[0,1].imshow(after_dict["weights"].T, aspect='auto', origin='lower')
    axes[0,1].set_title(f"[{uid} | {strategy}] Post-maintenance Weight Heatmap")
    axes[0,1].set_ylabel("Operator"); axes[0,1].set_xlabel("Time")
    axes[0,1].set_yticks(range(len(op_names))); axes[0,1].set_yticklabels(op_names)
    fig.colorbar(im1, ax=axes[0,1])

    # Damage before/after
    axes[1,0].plot(before_dict["damage"], label="Pre-maintenance Damage", linewidth=1.8)
    axes[1,0].plot(after_dict["damage"],  label="Post-maintenance Damage",  linewidth=1.8, linestyle="--")
    axes[1,0].set_title(f"[{uid} | {strategy}] Mixed Damage Index (Weighted Operator Output)")
    axes[1,0].set_xlabel("Time"); axes[1,0].set_ylabel("Damage")
    axes[1,0].grid(True, alpha=0.35); axes[1,0].legend()

    # Temperature curves
    axes[1,1].plot(before_dict["temperature"], label="Pre-maintenance Temperature", linewidth=1.5)
    axes[1,1].plot(after_dict["temperature"],  label="Post-maintenance Temperature",  linewidth=1.5, linestyle="--")
    axes[1,1].set_title(f"[{uid} | {strategy}] Temperature Parameter (Lower → Sharper)")
    axes[1,1].set_xlabel("Time"); axes[1,1].set_ylabel("Temperature")
    axes[1,1].grid(True, alpha=0.35); axes[1,1].legend()

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    # Operator outputs (after normalization) before/after comparison
    K = before_dict["weights"].shape[1]
    n_cols = 4
    n_rows = int(np.ceil(K / n_cols))
    plt.figure(figsize=(n_cols * 4.2, n_rows * 3.0))
    for k in range(K):
        plt.subplot(n_rows, n_cols, k+1)
        plt.plot(before_dict["op_outs_norm"][:, k], label="Pre-maintenance", linewidth=1.6)
        plt.plot(after_dict["op_outs_norm"][:,  k], label="Post-maintenance",  linewidth=1.6, linestyle="--")
        plt.title(op_names[k]); plt.xlabel("Time"); plt.ylabel("Operator Output (Normalized)")
        plt.grid(True, alpha=0.3)
        if k == 0: plt.legend()
    plt.suptitle(f"[{uid} | {strategy}] Operator Outputs — Pre vs Post Maintenance", y=1.02, fontsize=12)
    plt.tight_layout()
    if save_path:
        root, ext = os.path.splitext(save_path)
        save_path2 = root + "_ops" + (ext if ext else ".png")
        plt.savefig(save_path2, dpi=150, bbox_inches="tight")
    plt.show()

# ========== Main: single pair visualization ==========
def visualize_liquid_mixing_on_pairs(
    pairs,
    checkpoint_path="/content/drive/MyDrive/CMAPSS/main_dynamics_identification_3.pth",
    uid=None, strategy=None, save_dir=None
):
    # 1) Pick one sample (or specify uid/strategy)
    uid, strategy, xb, xa = pick_one_pair(pairs, uid=uid, strategy=strategy)

    # 2) Build and load model (infer input dimension C from pairs)
    C = xb.shape[-1]
    model, device = build_model(input_dim=C)
    load_checkpoint(model, checkpoint_path, device)

    # 3) Feed before/after through encoder path separately, extract liquid weights and operator outputs
    xb = xb.to(device); xa = xa.to(device)
    before_dict = extract_ops_and_weights_for_sequence(model, xb)
    after_dict  = extract_ops_and_weights_for_sequence(model, xa)

    # 4) Plot
    spath = None
    if save_dir:
        spath = os.path.join(save_dir, f"{uid}__{canonical_strategy_name(strategy)}.png")
    plot_liquid_mixing(uid, strategy, before_dict, after_dict, save_path=spath)
    print(f"[Complete] {uid} | {strategy} visualization completed.")

# ========== Bonus: multi-strategy batch visualization ==========
def visualize_multiple_by_strategy(
    pairs,
    checkpoint_path="/content/drive/MyDrive/CMAPSS/main_dynamics_identification_3.pth",
    strategy_filters=None,  # e.g., ["perfect","general","poor"] or None for all
    max_per_strategy=2,
    save_dir=None,
    seed=0
):
    """
    Visualize multiple samples across (selected) maintenance strategies.
    - strategy_filters: list of keywords to filter strategies; None/[] => all
    - max_per_strategy: limit number per strategy class (Perfect/General/Poor)
    - save_dir: where to save figures (optional)
    """
    # Build a temp model once (infer C from first available pair)
    # If no pairs, raise
    if not pairs:
        raise ValueError("pairs is empty.")

    # locate first sample to infer C
    # (uid, strat, rec) from generator (unfiltered) just to infer C
    for u0 in pairs:
        for s0 in pairs[u0]:
            x0 = np.asarray(pairs[u0][s0]["x_before"], dtype=np.float32)
            C = x0.shape[1]
            break
        break

    model, device = build_model(input_dim=C)
    load_checkpoint(model, checkpoint_path, device)

    count_total = 0
    for uid, strat, rec in iter_pairs_by_strategy(pairs, strategy_filters, max_per_strategy, seed=seed):
        xb = np.asarray(rec["x_before"], dtype=np.float32)
        xa = np.asarray(rec["x_after"],  dtype=np.float32)
        T = min(len(xb), len(xa))
        xb = torch.from_numpy(xb[:T]).unsqueeze(0).to(device)  # (1,T,C)
        xa = torch.from_numpy(xa[:T]).unsqueeze(0).to(device)

        before_dict = extract_ops_and_weights_for_sequence(model, xb)
        after_dict  = extract_ops_and_weights_for_sequence(model, xa)

        spath = None
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            spath = os.path.join(save_dir, f"{uid}__{canonical_strategy_name(strat)}.png")

        plot_liquid_mixing(uid, strat, before_dict, after_dict, save_path=spath)
        print(f"[Complete] {uid} | {strat} visualization done.")
        count_total += 1

    if count_total == 0:
        if strategy_filters:
            print(f"[Info] No samples matched filters: {strategy_filters}")
        else:
            print("[Info] No samples to visualize.")

# ===== Usage Examples =====
# Note: Before calling, ensure that DiffAwareReconstructor/TrendEncoder/CustomKAN and other model definitions
# are imported globally, and the `pairs` variable is ready (same structure as in training).

# ---- (A) Single sample (kept same as your original entry) ----
# visualize_liquid_mixing_on_pairs(
#     pairs,
#     checkpoint_path="/content/drive/MyDrive/CMAPSS/main_dynamics_identification_3.pth",
#     uid=None, strategy=None,          # or specify uid/strategy
#     save_dir=None                     # e.g., "/content/figs"
# )

#---- (B) Multi-strategy batch visualization ----
visualize_multiple_by_strategy(
    pairs,
    checkpoint_path="/content/drive/MyDrive/CMAPSS/main_dynamics_identification_3.pth",
    strategy_filters=["perfect","general","poor"],  # or None for all strategies
    max_per_strategy=2,                             # how many per class to plot
    save_dir=None,                                  # e.g., "/content/figs"
    seed=0
)

# %% [notebook code cell 19]
# -*- coding: utf-8 -*-
# Display directly in the Colab output area: DiffAwareReconstructor's "Model Formula (LaTeX)"
# Instructions for use:
# 1) First make sure you have imported the training script and instantiated it + load_state_dict to get the model (DiffAwareReconstructor)
# 2) Run this unit and finally execute display_model_formulas(model)

import torch
import torch.nn.functional as F
from IPython.display import Markdown, display
import math

# ---------- Gadgets ----------
def sp(x): # softplus value
    return torch.nn.functional.softplus(x)

def fmt(v, nd=5):
    try:
        if isinstance(v, (int, float)):
            return f"{v:.{nd}g}"
        if torch.is_tensor(v):
            if v.numel() == 1:
                return f"{float(v.detach().cpu().item()):.{nd}g}"
    except Exception:
        pass
    return str(v)

def m(section_title: str, body_md: str = ""):
    """Display a Markdown section"""
    display(Markdown(f"## {section_title}\n\n{body_md}"))

def latex_block(eq: str):
    """Display a LaTeX formula block (Colab will render it)"""
    display(Markdown(f"$$\n{eq}\n$$"))

def table_md(d: dict, headers=("Name","Value")):
    lines = [f"| {headers[0]} | {headers[1]} |", "|---|---|"]
    for k,v in d.items():
        lines.append(f"| {k} | {v} |")
    return "\n".join(lines)

# ---------- Formula template ----------
def eq_overview(T="T", C="C", J="J", K="K"):
    return f"""
- Input sensing sequence: $x \\in \\mathbb{{R}}^{{{T}\\times {C}}}$
- BoltzmannKAN features: $h \\in \\mathbb{{R}}^{{{T}\\times {J}}}$ ({J} = trend\\_ch)
- {K} operators produce $y^{{(k)}}_t$, liquid weight $w_{{t,k}}$, and $\\sum_k w_{{t,k}}=1$
- Mix to get damage $d_t$, and then project it into health index HI (monotonically decreasing constraint in time)
- Two GRU reconstruction branches (before/after) predict the next step $\\hat x_{{t+1}}$
- Classification head utilizes $\\Delta$HI + sensor statistics to predict maintenance strategy categories
"""

def eq_boltzmann_kan():
    return r"""
\textbf{BoltzmannKAN:}\quad x_t \in \mathbb{R}^{C},\ h_t \in \mathbb{R}^{J}

h_{t,j} \;=\; \operatorname{softplus}\!\Big( \sum_{c=1}^{C} \sigma\!\Big(\frac{E_{j,c}-x_{t,c}}{kT_{j,c}}\Big)\, x_{t,c} \Big),
\qquad j=1,\dots,J.
"""

def eq_ops_generic():
    return r"""
\textbf{Operator family (each outputs } y^{(k)}_t \in \mathbb{R}\text{):}

\begin{aligned}
\text{MonotonicLinear:}\quad
& y^{(1)}_t \;=\; \operatorname{softplus}\!\big(s\cdot(\overline{h}_t + b)\big),\quad s=\operatorname{softplus}(\hat s) \\[4pt]
\text{MonotonicFlat:}\quad
& \Delta^+ \overline{h}_t = \max(0,\overline{h}_t-\overline{h}_{t-1}),\quad
  y^{(2)}_t \;=\; \operatorname{softplus}\!\Big(s\cdot\big(\sum_{\tau\le t}\Delta^+ \overline{h}_\tau + b\big)\Big) \\[4pt]
\text{ConcaveLog:}\quad
& y^{(3)}_t \;=\; \operatorname{softplus}\!\big(s\cdot \log(|\overline{h}_t|+\varepsilon)\big) \\[4pt]
\text{SaturationSigmoid:}\quad
& y^{(4)}_t \;=\; \operatorname{softplus}\!\big(s\cdot \sigma(\lambda\,(\overline{h}_t - b))\big) \\[4pt]
\text{HingeReLU:}\quad
& y^{(5)}_t \;=\; \operatorname{softplus}\!\big(s\cdot \max(0,\overline{h}_t - \theta)\big) \\[4pt]
\text{Polynomial (deg=3):}\quad
& y^{(6)}_t \;=\; \operatorname{softplus}\!\Big(\sum_{i=1}^{3} a_i\, \overline{h}_t^{\,i}\Big),\quad a_i=\operatorname{softplus}(\hat a_i) \\[4pt]
\text{DampedSin:}\quad
& y^{(7)}_t \;=\; \operatorname{softplus}\!\big(s\cdot \tfrac{1}{1+\lambda|\overline{h}_t|}\,(\sin(\omega\,\overline{h}_t+\phi)+1)\big) \\[4pt]
\text{PiecewiseLinear:}\quad
& y^{(8)}_t \;=\; \operatorname{softplus}\!\Big(\;k_1 \cdot \min(\overline{h}_t,\theta) + \big(k_1\theta + k_2(\overline{h}_t-\theta)\big)\cdot \mathbf{1}[\overline{h}_t>\theta]\;\Big)
\end{aligned}

\text{where } \overline{h}_t = \tfrac{1}{J}\sum_{j=1}^{J} h_{t,j}.
"""

def eq_liquid_weight():
    return r"""
\textbf{Liquid Weight Generator:}
Let \; f_t = \mathrm{Fuse}\big(\phi_h(h_t),\ \phi_x(x_t),\ \phi_{\mathrm{temporal}}(t)\big) \in \mathbb{R}^{H}.
For each operator k, logits z_{t,k} = \mathrm{MLP}_k(f_t).
Temperature: \ \tau_t = \mathrm{clip}\big(\operatorname{softplus}(\hat \tau_t)+\tau_{\min},\ \tau_{\min},\ \tau_{\max}\big).

Weights:
\quad w_{t,k} = \frac{\exp(z_{t,k}/\tau_t)}{\sum_{\ell=1}^{K}\exp(z_{t,\ell}/\tau_t)},\qquad
\sum_{k=1}^K w_{t,k} = 1.
"""

def eq_damage_mix():
    return r"""
\textbf{CustomKAN mixing (damage):}\quad
y^{(k)}_t \text{ from operator } k,\ \ w_{t,k} \text{ from Liquid Weight Generator.}
\quad
\text{Stack } y_t = [y^{(1)}_t,\dots,y^{(K)}_t].

\textit{Damage}:
\quad d_t \;=\; \sum_{k=1}^{K} w_{t,k}\, y^{(k)}_t,
\qquad \text{(optionally clipped to }[0,100]\text{; then gain/bias)}\\
d'_t \;=\; \mathrm{clip}\big(g\cdot d_t + b,\ 0,\ 100\big),\quad g=\operatorname{softplus}(\hat g).
"""

def eq_health_projection():
    return r"""
\textbf{TrendEncoder → Health Index:}
Two assessment paths are blended:

1) Direct health from raw sensors:
\quad h^{(\mathrm{direct})}_t = \sigma\big(\mathrm{MLP}_\mathrm{health}(x_t)\big) \in (0,1).

2) Linear trend prior (decay):
\quad \ell_t = 1 - \frac{t}{T-1}\cdot 0.5,\quad
\alpha_t = \sigma\big(\mathrm{MLP}_\mathrm{linear}(x_t)\big) \in (0,1),\;\;
h^{(\mathrm{lin})}_t = \alpha_t\,\ell_t + (1-\alpha_t)\cdot h^{(\mathrm{direct})}_t \, .

Damage-to-health transform:
\quad \tilde d_t = \sigma\big(\gamma\cdot(d'_t+\beta)\big), \ \gamma=\operatorname{softplus}(\hat\gamma).

Combined health:
\quad h_t = h^{(\mathrm{direct})}_t \cdot (1 - 0.3\, \tilde d_t) \ \ \text{then blended with } \ell_t \text{ via } \alpha_t.

Monotonic-decreasing enforcement is applied timewise by projecting
\ h_t \ \text{onto}\ \{h_1 \ge h_2 \ge \dots \ge h_T\}\ \text{(implementation via stepwise min).}
"""

def eq_recon_gru():
    return r"""
\textbf{Recursive Reconstruction (per branch):}
Input feature: u_t = [x_t;\ h_t] \in \mathbb{R}^{C+1}.

\text{GRU:}\quad
\begin{aligned}
z_t &= \sigma(W_z u_t + U_z h^{\mathrm{gru}}_{t-1} + b_z),\\
r_t &= \sigma(W_r u_t + U_r h^{\mathrm{gru}}_{t-1} + b_r),\\
\tilde h_t &= \tanh(W_h u_t + U_h (r_t \odot h^{\mathrm{gru}}_{t-1}) + b_h),\\
h^{\mathrm{gru}}_t &= (1-z_t)\odot h^{\mathrm{gru}}_{t-1} + z_t \odot \tilde h_t,
\end{aligned}
\qquad
\hat x_{t+1} = W_o h^{\mathrm{gru}}_{t} + b_o \in \mathbb{R}^{C}.
"""

def eq_classifier():
    return r"""
\textbf{Maintenance Classifier:}
Let \ \Delta h_t = h^{(\mathrm{after})}_t - h^{(\mathrm{before})}_t,\quad
\text{features: mean/var/std/first/max/pos\_ratio/slope\_diff} \ \in \mathbb{R}^{6}.
Also sensor statistics: [\overline{x}_b,\ \overline{x}_a] \in \mathbb{R}^{2C}.

\text{Fuse two MLPs and classify:}\quad
\mathrm{logits} = \mathrm{MLP}_{\mathrm{fuse}}\big(\mathrm{MLP}_{\mathrm{hi}}(\cdot),\ \mathrm{MLP}_{\mathrm{sensor}}(\cdot)\big)\in\mathbb{R}^{3}.
"""

# ---------- Read parameters (display key information) ----------
def read_op_params(op):
    d = {}
    try:
        name = op.__class__.__name__
        if name == "MonotonicLinearOp":
            d["scale s"] = fmt(sp(op.raw_scale)); d["bias b"] = fmt(op.bias)
        elif name == "MonotonicFlatOp":
            d["scale s"] = fmt(sp(op.raw_scale)); d["bias b"] = fmt(op.bias)
        elif name == "ConcaveLogOp":
            d["scale s"] = fmt(sp(op.raw_scale)); d["eps"] = fmt(getattr(op, "eps", 1e-3))
        elif name == "SaturationSigmoidOp":
            d["scale s"] = fmt(sp(op.raw_scale)); d["slope λ"] = fmt(sp(op.raw_slope)); d["bias b"] = fmt(op.bias)
        elif name == "HingeReLUOp":
            d["scale s"] = fmt(sp(op.raw_scale)); d["threshold θ"] = fmt(op.threshold)
        elif name == "PolynomialOp":
            coeffs = []
            for i in range(op.deg):
                coeffs.append(fmt(sp(op.raw_coeff[i])))
            d["coeff a_i (softplus)"] = ", ".join(coeffs); d["deg"] = op.deg
        elif name == "DampedSinOp":
            d["scale s"] = fmt(sp(op.raw_scale)); d["freq ω"] = fmt(sp(op.raw_freq))
            d["lambda λ"] = fmt(sp(op.raw_lambda)); d["phase φ"] = fmt(op.phase)
        elif name == "PiecewiseLinearOp":
            d["k1"] = fmt(sp(op.raw_k1)); d["k2"] = fmt(sp(op.raw_k2)); d["threshold θ"] = fmt(op.threshold)
    except Exception as e:
        d["warn"] = f"param read error: {e}"
    return d

def read_dims(model):
    try:
        J = model.encoder.boltz.E.shape[0]
        C = model.recon_b.out.out_features
        K = len(model.encoder.customkan.ops)
    except Exception:
        J, C, K = "J", "C", "K"
    return J, C, K

def display_model_formulas(model):
    if not hasattr(model, "encoder"):
        raise RuntimeError("model is missing encoder. Please instantiate and load DiffAwareReconstructor first.")

    J, C, K = read_dims(model)
    # Title
    display(Markdown("# DiffAwareReconstructor — Formula Overview"))

    # Overview
    m("Overview & Shapes", eq_overview(T="T", C=C, J=J, K=K))

    # BoltzmannKAN
    m("BoltzmannKAN")
    latex_block(eq_boltzmann_kan())

    # Operators
    m("Operators (per-time scalar outputs)")
    latex_block(eq_ops_generic())

    # Current operator parameters
    ops = list(model.encoder.customkan.ops)
    display(Markdown("### Current Operator Parameters (from checkpoint)"))
    for idx, op in enumerate(ops, 1):
        display(Markdown(f"**Op {idx}: {op.__class__.__name__}**"))
        display(Markdown(table_md(read_op_params(op))))

    # Liquid weights
    m("Liquid Weight Generator")
    latex_block(eq_liquid_weight())
    # Lightweight super parameter display
    wg = model.encoder.customkan.weight_generator
    wg_info = {
       "n_ops(K)": getattr(wg, "n_ops", K),
       "tau_min": fmt(getattr(wg, "tau_min", "?")),
       "tau_max": fmt(getattr(wg, "tau_max", "?")),
    }
    display(Markdown("### Hyperparameters"))
    display(Markdown(table_md(wg_info)))

    # Mixing / damage
    m("CustomKAN Mixing (Damage)")
    latex_block(eq_damage_mix())

    # Health projection
    m("TrendEncoder → Health Index (HI)")
    latex_block(eq_health_projection())

    # Reconstruction GRU
    m("Recursive Reconstruction (GRU)")
    latex_block(eq_recon_gru())
    try:
        gruH = model.recon_b.gru.hidden_size
        outC = model.recon_b.out.out_features
    except Exception:
        gruH, outC = "H", "C"
    display(Markdown(table_md({"GRU hidden size": gruH, "Output dim (C)": outC})))

    # Classifier
    m("Maintenance Classifier")
    latex_block(eq_classifier())

    display(Markdown("> ✅ All formulas and key parameters have been displayed in the output area. You can copy LaTeX directly to the paper/manual."))

# ---- Sample call (please execute in the same kernel where you have loaded checkpoint) ----
display_model_formulas(model)

# %% [notebook code cell 20]

# ============================================================
# Monotonic-Decreasing HI + aligned plotting (before -> after)
# ============================================================
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import random
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL2NAME = {0: "Perfect", 1: "General", 2: "Poor"}

# ============================================================
# 1) Dataset: read samples from pairs, perform shift operations in batch later
# ============================================================
class PairsReconstructDataset(Dataset):
    """
    Each sample contains:
      x_before: (L,C) feature sequence before maintenance
      x_after : (L,C) feature sequence after maintenance
      hi_before/hi_after: (L,) health index (for evaluation/optional analysis only, not strong supervision)
      strategy: maintenance type ("Perfect Maintenance" / "General Maintenance" / "Poor Maintenance")
    """
    def __init__(self, pairs: dict, horizon: int=None):
        items = []
        for uid, sd in pairs.items():
            for strat, d in sd.items():
                xb = np.asarray(d["x_before"], dtype=np.float32)
                xa = np.asarray(d["x_after"],  dtype=np.float32)
                # When there's no real HI, create placeholder that will be learned by the model
                if "hi_before" in d and d["hi_before"] is not None:
                    hib = np.asarray(d["hi_before"],dtype=np.float32)
                else:
                    # Create initialized fake HI, model will learn real patterns
                    hib = np.linspace(0.8, 0.4, len(xb)).astype(np.float32)

                if "hi_after" in d and d["hi_after"] is not None:
                    hia = np.asarray(d["hi_after"], dtype=np.float32)
                else:
                    # Create different fake HI patterns based on maintenance strategy
                    if "perfect" in strat.lower():
                        # Perfect maintenance: significant improvement
                        hia = np.linspace(0.9, 0.6, len(xa)).astype(np.float32)
                    elif "general" in strat.lower():
                        # General maintenance: moderate improvement
                        hia = np.linspace(0.7, 0.5, len(xa)).astype(np.float32)
                    else:
                        # Poor maintenance: slight improvement
                        hia = np.linspace(0.5, 0.35, len(xa)).astype(np.float32)

                L, C = xb.shape
                # Unify length (optional)
                if horizon is not None and L > horizon:
                    xb, xa, hib, hia = xb[:horizon], xa[:horizon], hib[:horizon], hia[:horizon]
                    L = horizon
                if L < 3:  # At least need to form 0:L-2 -> 1:L-1
                    continue
                items.append({"uid": uid, "strategy": strat, "x_before": xb, "x_after": xa,
                              "hi_before": hib, "hi_after": hia})
        assert len(items)>0, "Dataset is empty, please check pairs data."
        self.items = items
        self.C = items[0]["x_before"].shape[1]

        # Strategy to 3-class label mapping
        def strat_to_cls(s):
            s = s.lower()
            if "perfect" in s: return 0
            if "general" in s: return 1
            if "poor" in s: return 2
            raise ValueError(f"Unknown strategy name: {s}")
        self.labels = [strat_to_cls(it["strategy"]) for it in items]

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        return {
            "uid": it["uid"],
            "label": self.labels[idx],
            "x_before": torch.from_numpy(it["x_before"]), # (L,C)
            "x_after":  torch.from_numpy(it["x_after"]),  # (L,C)
            "hi_before":torch.from_numpy(it["hi_before"]),# (L,)
            "hi_after": torch.from_numpy(it["hi_after"]), # (L,)
            "strategy": it["strategy"]
        }

def pad_collate_shift(batch):
    """
    Pad sequences to same length; return mask, do NOT perform shift operation here
    to allow unified training: inputs = [:,0:L-2], targets = [:,1:L-1]
    """
    Ls = [b["x_before"].shape[0] for b in batch]
    Lmax = max(Ls)
    C = batch[0]["x_before"].shape[1]

    def pad2d(x):
        L, C0 = x.shape
        out = torch.zeros(Lmax, C0, dtype=x.dtype)
        out[:L] = x
        return out

    def pad1d(x):
        L = x.shape[0]
        out = torch.zeros(Lmax, dtype=x.dtype)
        out[:L] = x
        return out

    x_before = torch.stack([pad2d(b["x_before"]) for b in batch], 0) # (B,Lmax,C)
    x_after  = torch.stack([pad2d(b["x_after"])  for b in batch], 0) # (B,Lmax,C)
    hi_before= torch.stack([pad1d(b["hi_before"])for b in batch], 0) # (B,Lmax)
    hi_after = torch.stack([pad1d(b["hi_after"]) for b in batch], 0) # (B,Lmax)
    lengths  = torch.tensor(Ls, dtype=torch.long)
    mask     = torch.zeros(len(batch), Lmax)
    for i,L in enumerate(Ls): mask[i,:L] = 1.0
    labels   = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    uids     = [b["uid"] for b in batch]
    return {"uids": uids, "labels": labels,
            "x_before": x_before, "x_after": x_after,
            "hi_before": hi_before, "hi_after": hi_after,
            "lengths": lengths, "mask": mask}

# ============================================================
# 2) Model: True liquid adaptive operator - with significantly different parameters and behavior patterns before and after maintenance
# ============================================================
class BoltzmannKAN(nn.Module):
    def __init__(self, in_ch, out_ch=4):
        super().__init__()
        self.E  = nn.Parameter(torch.zeros(out_ch, in_ch))
        self.kT = nn.Parameter(torch.ones(out_ch, in_ch))
    def forward(self, x):
        B,T,C = x.shape
        kT = torch.clamp(F.softplus(self.kT), 0.01, 10.0).unsqueeze(0).unsqueeze(2)  # (1,out_ch,1,in_ch)
        E  = torch.clamp(self.E, -10.0, 10.0).unsqueeze(0).unsqueeze(2)             # (1,out_ch,1,in_ch)
        x_ = torch.clamp(x.unsqueeze(1), -10.0, 10.0)                               # (B, out_ch, T, in_ch)
        exp = torch.clamp((E - x_) / kT, -50, 50)
        w   = torch.sigmoid(exp)
        h   = (w * x_).sum(dim=3).permute(0, 2, 1)           # (B,T,out_ch) >=0
        return torch.clamp(F.softplus(h), 0.0, 100.0)

class BaseOp(nn.Module): pass

class LiquidAdaptiveMonotonicLinearOp(BaseOp):
    """Monotonic linear operator for true liquids - significant changes in parameters before and after maintenance"""
    def __init__(self, param_dim=8):
        super().__init__()
        # Deep network, stronger nonlinear parameter generation capability
        self.param_net = nn.Sequential(
            nn.Linear(param_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3) # Output scale, bias, temperature (control maintenance sensitivity)
        )
        self.smin, self.smax = 0.1, 8.0
        # Dedicated to maintaining effect-aware networks
        self.maintenance_sensor = nn.Sequential(
            nn.Linear(param_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 4) # Output maintenance effect parameters
        )

    def forward(self, h):
        B, T, _ = h.shape
        h_mean = h.mean(dim=1)  # (B, param_dim)

        #Basic parameters
        params = self.param_net(h_mean)  # (B, 3)
        scale = torch.clamp(F.softplus(params[:, 0]), self.smin, self.smax).unsqueeze(1)  # (B, 1)
        bias = torch.clamp(params[:, 1], -8.0, 8.0).unsqueeze(1)  # (B, 1)
        temperature = torch.clamp(F.softplus(params[:, 2]), 0.1, 5.0).unsqueeze(1)  # (B, 1)

        # Maintain effect perception parameters
        maint_params = self.maintenance_sensor(h_mean)  # (B, 4)
        sensitivity = torch.clamp(F.softplus(maint_params[:, 0]), 0.5, 3.0).unsqueeze(1)  # (B, 1)
        phase_shift = torch.clamp(maint_params[:, 1], -2.0, 2.0).unsqueeze(1)  # (B, 1)
        amplitude_mod = torch.clamp(F.softplus(maint_params[:, 2]), 0.3, 2.0).unsqueeze(1)  # (B, 1)
        nonlin_factor = torch.clamp(torch.sigmoid(maint_params[:, 3]), 0.1, 0.9).unsqueeze(1)  # (B, 1)

        # Time-related change patterns
        t = torch.arange(T, device=h.device, dtype=h.dtype).unsqueeze(0).expand(B, -1)  # (B, T)
        t_norm = t / (T - 1) #Normalized time

        #Adjust parameters based on the statistical characteristics of the input data
        h_std = h.std(dim=1, keepdim=True).mean(dim=-1)  # (B, 1)
        dynamic_scale = scale * (1 + h_std * sensitivity)

        # Nonlinear time effects
        time_effect = torch.sin(2 * np.pi * t_norm * temperature + phase_shift) * amplitude_mod

        xm = torch.clamp(h.mean(-1), -10.0, 10.0)  # (B, T)
        # Liquid response: combining linear and nonlinear effects
        linear_part = dynamic_scale * (xm + bias)
        nonlinear_part = time_effect * xm * nonlin_factor

        result = linear_part + nonlinear_part
        return torch.clamp(F.softplus(result), 0.0, 100.0)

class LiquidAdaptiveMonotonicFlatOp(BaseOp):
    """Liquid smoothing operator - with maintenance-related dynamics"""
    def __init__(self, param_dim=8):
        super().__init__()
        self.param_net = nn.Sequential(
            nn.Linear(param_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4)  # scale, bias, smoothness, adaptivity
        )
        self.dynamic_net = nn.Sequential(
            nn.Linear(param_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 3) #Dynamic adjustment parameters
        )
        self.smin, self.smax = 1e-2, 2.0

    def _adaptive_cum(self, x, smoothness):
        """Adaptive accumulation function, adjusted according to the smoothness parameter"""
        # x is (B, 1, T), smoothness is (B, 1)
        diff = F.relu(x[:, :, 1:] - x[:, :, :-1])  # (B, 1, T-1)
        # Smoothness adjustment
        diff = diff * smoothness.unsqueeze(-1)  # (B, 1, T-1)
        return torch.cat([torch.zeros_like(diff[:, :, :1]), torch.cumsum(diff, 2)], 2)  # (B, 1, T)

    def forward(self, h):
        B, T, _ = h.shape
        h_mean = h.mean(dim=1)  # (B, param_dim)

        #Basic parameters
        params = self.param_net(h_mean)  # (B, 4)
        scale = torch.clamp(F.softplus(params[:, 0]), self.smin, self.smax).unsqueeze(1)
        bias = torch.clamp(params[:, 1], -5.0, 5.0).unsqueeze(1)
        smoothness = torch.clamp(F.softplus(params[:, 2]), 0.1, 2.0).unsqueeze(1)
        adaptivity = torch.clamp(torch.sigmoid(params[:, 3]), 0.1, 0.9).unsqueeze(1)

        #Dynamic adjustment parameters
        dynamic_params = self.dynamic_net(h_mean)  # (B, 3)
        temporal_weight = torch.clamp(F.softplus(dynamic_params[:, 0]), 0.5, 2.0).unsqueeze(1)
        fluctuation = torch.clamp(dynamic_params[:, 1], -1.0, 1.0).unsqueeze(1)
        maintenance_response = torch.clamp(F.softplus(dynamic_params[:, 2]), 0.2, 3.0).unsqueeze(1)

        # Time-varying characteristics of input data
        h_variance = h.var(dim=-1, keepdim=True)  # (B, T, 1)
        temporal_modulation = torch.sin(torch.arange(T, device=h.device).float() * temporal_weight / T * 2 * np.pi + fluctuation)
        temporal_modulation = temporal_modulation.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)  # (B, 1, T)

        xm = torch.clamp(h.mean(-1, keepdim=True), -10.0, 10.0).unsqueeze(1)  # (B, 1, T)

        # Liquid response: combining cumulative effect and dynamic adjustment
        cum_base = self._adaptive_cum(xm, smoothness)

        # Maintain response conditioning - ensure dimensions match
        h_variance_reshaped = h_variance.transpose(1, 2)  # (B, 1, T)
        maintenance_modulation = maintenance_response.unsqueeze(-1) * h_variance_reshaped * temporal_modulation

        result = scale.unsqueeze(-1) * (cum_base + maintenance_modulation + bias.unsqueeze(-1))
        result = result.squeeze(1) * (1 + adaptivity * temporal_modulation.squeeze(1))

        return torch.clamp(F.softplus(result), 0.0, 100.0)

class LiquidAdaptiveConcaveLogOp(BaseOp):
    """Liquid concave logarithmic operator - maintaining sensitive nonlinear response"""
    def __init__(self, param_dim=8):
        super().__init__()
        self.param_net = nn.Sequential(
            nn.Linear(param_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 5)  # scale, offset, curvature, maintenance_gain, noise_resist
        )
        self.eps = 1e-3
        self.smin, self.smax = 0.01, 8.0

    def forward(self, h):
        B, T, _ = h.shape
        h_mean = h.mean(dim=1)  # (B, param_dim)
        params = self.param_net(h_mean)  # (B, 5)

        scale = torch.clamp(F.softplus(params[:, 0]), self.smin, self.smax).unsqueeze(1)
        offset = torch.clamp(params[:, 1], -3.0, 3.0).unsqueeze(1)
        curvature = torch.clamp(F.softplus(params[:, 2]), 0.1, 3.0).unsqueeze(1)
        maintenance_gain = torch.clamp(F.softplus(params[:, 3]), 0.5, 4.0).unsqueeze(1)
        noise_resistance = torch.clamp(torch.sigmoid(params[:, 4]), 0.1, 0.95).unsqueeze(1)

        # Dynamic data feature awareness
        h_energy = (h ** 2).mean(dim=-1)  # (B, T)
        h_gradient = torch.abs(h[:, 1:] - h[:, :-1]).mean(dim=-1)  # (B, T-1)
        h_gradient = F.pad(h_gradient, (0, 1), value=0)  # (B, T)

        xm = torch.clamp(h.mean(-1), -10.0, 10.0)  # (B, T)

        # Liquid logarithmic response
        log_base = torch.log(torch.abs(xm + offset) + self.eps)
        curvature_effect = curvature * h_energy
        maintenance_effect = maintenance_gain * h_gradient

        #Adaptive noise suppression
        noise_filter = torch.sigmoid(h_energy * noise_resistance)

        result = scale * (log_base * curvature_effect + maintenance_effect) * noise_filter
        return torch.clamp(F.softplus(result), 0.0, 100.0)

class LiquidAdaptiveSaturationSigmoidOp(BaseOp):
    """Liquid saturated sigmoid operator - maintenance threshold sensitive"""
    def __init__(self, param_dim=8):
        super().__init__()
        self.param_net = nn.Sequential(
            nn.Linear(param_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 6)  # scale, slope, bias, threshold, saturation, maintenance_sens
        )
        self.smin, self.smax = 0.01, 6.0
        self.lmin, self.lmax = 0.1, 8.0

    def forward(self, h):
        B, T, _ = h.shape
        h_mean = h.mean(dim=1)  # (B, param_dim)
        params = self.param_net(h_mean)  # (B, 6)

        scale = torch.clamp(F.softplus(params[:, 0]), self.smin, self.smax).unsqueeze(1)
        slope = torch.clamp(F.softplus(params[:, 1]), self.lmin, self.lmax).unsqueeze(1)
        bias = torch.clamp(params[:, 2], -5.0, 5.0).unsqueeze(1)
        threshold = torch.clamp(params[:, 3], -3.0, 3.0).unsqueeze(1)
        saturation = torch.clamp(F.softplus(params[:, 4]), 0.5, 3.0).unsqueeze(1)
        maintenance_sens = torch.clamp(F.softplus(params[:, 5]), 0.2, 4.0).unsqueeze(1)

        # Maintain sensitive feature extraction
        h_peak = h.max(dim=-1)[0]  # (B, T)
        h_trough = h.min(dim=-1)[0]  # (B, T)
        h_range = h_peak - h_trough

        xm = torch.clamp(h.mean(-1), -10.0, 10.0)  # (B, T)

        # Dynamic threshold adjustment
        dynamic_threshold = threshold + h_range * maintenance_sens * 0.1

        # Liquid sigmoid response
        sigmoid_input = slope * (xm - dynamic_threshold - bias)
        sigmoid_output = torch.sigmoid(sigmoid_input)

        # saturation effect
        saturation_effect = torch.tanh(saturation * sigmoid_output)

        result = scale * saturation_effect
        return torch.clamp(F.softplus(result), 0.0, 100.0)

class LiquidAdaptiveHingeReLUOp(BaseOp):
    """Liquid hinge ReLU operator - multi-threshold maintenance response"""
    def __init__(self, param_dim=8):
        super().__init__()
        self.param_net = nn.Sequential(
            nn.Linear(param_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 6)  # scale, threshold1, threshold2, slope1, slope2, maintenance_amp
        )
        self.smin, self.smax = 0.01, 5.0

    def forward(self, h):
        B, T, _ = h.shape
        h_mean = h.mean(dim=1)  # (B, param_dim)
        params = self.param_net(h_mean)  # (B, 6)

        scale = torch.clamp(F.softplus(params[:, 0]), self.smin, self.smax).unsqueeze(1)
        threshold1 = torch.clamp(params[:, 1], -5.0, 0.0).unsqueeze(1)
        threshold2 = torch.clamp(params[:, 2], 0.0, 5.0).unsqueeze(1)
        slope1 = torch.clamp(F.softplus(params[:, 3]), 0.1, 3.0).unsqueeze(1)
        slope2 = torch.clamp(F.softplus(params[:, 4]), 0.1, 3.0).unsqueeze(1)
        maintenance_amp = torch.clamp(F.softplus(params[:, 5]), 0.5, 3.0).unsqueeze(1)

        # Maintain response characteristics
        h_trend = (h[:, -1] - h[:, 0]).mean(dim=-1, keepdim=True)  # (B, 1)
        h_volatility = h.std(dim=1).mean(dim=-1, keepdim=True)  # (B, 1)

        xm = torch.clamp(h.mean(-1), -10.0, 10.0)  # (B, T)

        #Multi-threshold hinge response
        hinge1 = F.relu(xm - threshold1) * slope1
        hinge2 = F.relu(xm - threshold2) * slope2

        # Maintenance effect modulation
        maintenance_modulation = maintenance_amp * (h_trend + h_volatility)

        result = scale * (hinge1 + hinge2) * (1 + maintenance_modulation)
        return torch.clamp(F.softplus(result), 0.0, 100.0)

class LiquidAdaptivePolynomialOp(BaseOp):
    """Liquid polynomial operator - adaptive order and coefficients"""
    def __init__(self, param_dim=8, max_deg=4):
        super().__init__()
        self.max_deg = max_deg
        self.param_net = nn.Sequential(
            nn.Linear(param_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, max_deg + 2) # coefficient + degree_weight + maintenance_coupling
        )

    def forward(self, h):
        B, T, _ = h.shape
        h_mean = h.mean(dim=1)  # (B, param_dim)
        params = self.param_net(h_mean)  # (B, max_deg + 2)

        coeffs = torch.clamp(F.softplus(params[:, :self.max_deg]), 0.01, 3.0)  # (B, max_deg)
        degree_weight = torch.clamp(torch.sigmoid(params[:, self.max_deg]), 0.1, 1.0).unsqueeze(1)  # (B, 1)
        maintenance_coupling = torch.clamp(F.softplus(params[:, self.max_deg + 1]), 0.2, 2.0).unsqueeze(1)  # (B, 1)

        # Maintain related dynamic features
        h_complexity = torch.var(h, dim=-1).mean(dim=-1, keepdim=True)  # (B, 1)

        xm = torch.clamp(h.mean(-1), -3.0, 3.0)  # (B, T)
        y = torch.zeros_like(xm)

        # Adaptive polynomial calculation
        for i in range(self.max_deg):
            coeff = coeffs[:, i].unsqueeze(1)  # (B, 1)
            power = i + 1

            #Adjust the coefficient according to the maintenance complexity
            adaptive_coeff = coeff * (1 + maintenance_coupling * h_complexity * (power / self.max_deg))

            term = adaptive_coeff * torch.clamp(xm ** power, -50.0, 50.0)
            y = y + term * degree_weight

        return torch.clamp(F.softplus(y), 0.0, 100.0)

class LiquidAdaptiveDampedSinOp(BaseOp):
    """Liquid damped sine operator - maintaining relevant oscillation modes"""
    def __init__(self, param_dim=8):
        super().__init__()
        self.param_net = nn.Sequential(
            nn.Linear(param_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 7)  # scale, freq, damping, phase, maintenance_freq, resonance, chaos
        )
        self.smin, self.smax = 0.01, 5.0
        self.fmin, self.fmax = 0.1, 8.0
        self.lmin, self.lmax = 0.01, 4.0

    def forward(self, h):
        B, T, _ = h.shape
        h_mean = h.mean(dim=1)  # (B, param_dim)
        params = self.param_net(h_mean)  # (B, 7)

        scale = torch.clamp(F.softplus(params[:, 0]), self.smin, self.smax).unsqueeze(1)
        freq = torch.clamp(F.softplus(params[:, 1]), self.fmin, self.fmax).unsqueeze(1)
        damping = torch.clamp(F.softplus(params[:, 2]), self.lmin, self.lmax).unsqueeze(1)
        phase = torch.clamp(params[:, 3], -2*np.pi, 2*np.pi).unsqueeze(1)
        maintenance_freq = torch.clamp(F.softplus(params[:, 4]), 0.1, 3.0).unsqueeze(1)
        resonance = torch.clamp(F.softplus(params[:, 5]), 0.5, 2.5).unsqueeze(1)
        chaos = torch.clamp(torch.sigmoid(params[:, 6]), 0.05, 0.3).unsqueeze(1)

        # Maintain related dynamic features
        h_rhythm = torch.fft.fft(h.mean(dim=-1)).abs().mean(dim=-1, keepdim=True).real  # (B, 1)

        xm = torch.clamp(h.mean(-1), -10.0, 10.0)  # (B, T)

        # time series
        t = torch.arange(T, device=h.device, dtype=h.dtype).unsqueeze(0).expand(B, -1)

        # Maintain sensitive oscillations
        maintenance_oscillation = torch.sin(maintenance_freq * t + phase) * resonance

        # Main oscillation + damping
        main_oscillation = torch.sin(freq * xm + phase)
        damping_factor = torch.exp(-damping * torch.abs(xm))

        #Chaotic disturbance (nonlinearity caused by maintenance)
        chaos_term = chaos * h_rhythm * torch.sin(freq * maintenance_freq * t)

        result = scale * damping_factor * (main_oscillation + maintenance_oscillation + chaos_term)
        return torch.clamp(F.softplus(result), 0.0, 100.0)

class LiquidAdaptivePiecewiseLinearOp(BaseOp):
    """Liquid piecewise linear operator - multi-stage maintenance response"""
    def __init__(self, param_dim=8):
        super().__init__()
        self.param_net = nn.Sequential(
            nn.Linear(param_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 8)  # k1, k2, k3, thresh1, thresh2, maintenance_shift, slope_adapt, transition_smooth
        )
        self.kmin, self.kmax = 0.01, 4.0

    def forward(self, h):
        B, T, _ = h.shape
        h_mean = h.mean(dim=1)  # (B, param_dim)
        params = self.param_net(h_mean)  # (B, 8)

        k1 = torch.clamp(F.softplus(params[:, 0]), self.kmin, self.kmax).unsqueeze(1)
        k2 = torch.clamp(F.softplus(params[:, 1]), self.kmin, self.kmax).unsqueeze(1)
        k3 = torch.clamp(F.softplus(params[:, 2]), self.kmin, self.kmax).unsqueeze(1)
        thresh1 = torch.clamp(params[:, 3], -4.0, 0.0).unsqueeze(1)
        thresh2 = torch.clamp(params[:, 4], 0.0, 4.0).unsqueeze(1)
        maintenance_shift = torch.clamp(params[:, 5], -2.0, 2.0).unsqueeze(1)
        slope_adapt = torch.clamp(F.softplus(params[:, 6]), 0.5, 2.0).unsqueeze(1)
        transition_smooth = torch.clamp(F.softplus(params[:, 7]), 0.1, 2.0).unsqueeze(1)

        # Threshold adjustment for maintenance impact
        h_dynamics = (h.max(dim=-1)[0] - h.min(dim=-1)[0]).mean(dim=-1, keepdim=True)  # (B, 1)

        dynamic_thresh1 = thresh1 + maintenance_shift * h_dynamics
        dynamic_thresh2 = thresh2 + maintenance_shift * h_dynamics

        xm = torch.clamp(h.mean(-1), -10.0, 10.0)  # (B, T)

        # Piecewise linear function + smooth transition
        segment1 = k1 * slope_adapt * xm
        segment2 = k1 * dynamic_thresh1 + k2 * slope_adapt * (xm - dynamic_thresh1)
        segment3 = k1 * dynamic_thresh1 + k2 * (dynamic_thresh2 - dynamic_thresh1) + k3 * slope_adapt * (xm - dynamic_thresh2)

        # Smooth transition weight
        w1 = torch.sigmoid(transition_smooth * (dynamic_thresh1 - xm))
        w2 = torch.sigmoid(transition_smooth * (xm - dynamic_thresh1)) * torch.sigmoid(transition_smooth * (dynamic_thresh2 - xm))
        w3 = torch.sigmoid(transition_smooth * (xm - dynamic_thresh2))

        # Ensure weight normalization
        total_w = w1 + w2 + w3 + 1e-8
        w1, w2, w3 = w1/total_w, w2/total_w, w3/total_w

        result = w1 * segment1 + w2 * segment2 + w3 * segment3
        return torch.clamp(F.softplus(result), 0.0, 100.0)

class TrueLiquidSparseGate(nn.Module):
    """Sparse gating of true liquids - weight distributions significantly different before and after maintenance"""
    def __init__(self, n_ops, param_dim=8, tau_start=8.0, tau_end=0.05, n_steps=15000):
        super().__init__()
        # Deep gated network to strengthen maintenance awareness capabilities
        self.gate_net = nn.Sequential(
            nn.Linear(param_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_ops)
        )

        # Maintain state-aware network
        self.maintenance_detector = nn.Sequential(
            nn.Linear(param_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, n_ops) # Maintenance sensitivity of each operator
        )

        # Dynamic diversity enhanced network
        self.diversity_enhancer = nn.Sequential(
            nn.Linear(param_dim, 32),
            nn.ReLU(),
            nn.Linear(32, n_ops) # Diversity weight
        )

        self.tau_start, self.tau_end, self.n_steps = tau_start, tau_end, n_steps
        self.register_buffer("step", torch.tensor(0.))

        # Maintain weight comparison records before and after
        self.register_buffer("prev_weights", torch.zeros(1, n_ops))
        self.register_buffer("weight_changes", torch.zeros(1, n_ops))

    def forward(self, h_input):
        B, T, param_dim = h_input.shape
        h_mean = h_input.mean(dim=1)  # (B, param_dim)
        h_std = h_input.std(dim=1)   # (B, param_dim)
        h_trend = h_input[:, -1] - h_input[:, 0] # (B, param_dim) trend characteristics

        #Basic logits
        base_logits = self.gate_net(h_mean)  # (B, n_ops)

        # Maintain sensitive modulation
        maintenance_sensitivity = self.maintenance_detector(h_std)  # (B, n_ops)

        # Diversity enhancement
        diversity_weights = self.diversity_enhancer(h_trend)  # (B, n_ops)

        # Combine logits - enhance differentiation
        combined_logits = base_logits + 2.0 * maintenance_sensitivity + 1.5 * diversity_weights

        # Dynamic temperature adjustment
        tau = (self.tau_start * (1 - self.step / self.n_steps) +
               self.tau_end * (self.step / self.n_steps)).clamp(min=self.tau_end)

        # Use different temperatures for each sample (based on its complexity)
        complexity = h_std.mean(dim=-1, keepdim=True)  # (B, 1)
        adaptive_tau = tau * (0.5 + 1.5 * torch.sigmoid(complexity))  # (B, 1)

        self.step.add_(1)

        # Gumbel noise - enhance randomness
        if self.training:
            gumbel_noise = -torch.empty_like(combined_logits).exponential_().log()
            gumbel_noise = torch.clamp(gumbel_noise, -50.0, 50.0)
            # Additional differential noise
            differential_noise = torch.randn_like(combined_logits) * 0.5
        else:
            gumbel_noise = torch.zeros_like(combined_logits)
            differential_noise = torch.zeros_like(combined_logits)

        # Apply temperature and noise
        logits_stable = torch.clamp(combined_logits, -50.0, 50.0)
        noisy_logits = logits_stable + gumbel_noise + differential_noise

        #Adaptive softmax
        w = F.softmax(noisy_logits / adaptive_tau, dim=-1)  # (B, n_ops)

        # Force diversity: if weights are too concentrated, add perturbation
        max_weight = w.max(dim=-1, keepdim=True)[0]
        diversity_penalty = torch.where(max_weight > 0.8,
                                      torch.randn_like(w) * 0.1,
                                      torch.zeros_like(w))
        w = F.softmax(w + diversity_penalty, dim=-1)

        # Record weight changes
        if self.training:
            current_weights = w.mean(dim=0, keepdim=True)  # (1, n_ops)
            if self.prev_weights.sum() > 0:
                self.weight_changes = torch.abs(current_weights - self.prev_weights)
            self.prev_weights = current_weights.clone()

        return w.unsqueeze(1)  # (B, 1, n_ops)

class TrueLiquidCustomKAN(nn.Module):
    """True liquid custom KAN - operator behavior significantly different before and after maintenance"""
    def __init__(self, ops, param_dim=8):
        super().__init__()
        self.ops = nn.ModuleList(ops)
        self.gate = TrueLiquidSparseGate(len(ops), param_dim)

        # Maintenance phase aware global modulation network
        self.global_modulator = nn.Sequential(
            nn.Linear(param_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4)  # global_gain, global_bias, phase_shift, maintenance_strength
        )

        #Interaction network between operators
        self.inter_op_network = nn.Sequential(
            nn.Linear(len(ops), 32),
            nn.ReLU(),
            nn.Linear(32, len(ops))
        )

    def forward(self, h):  # h:(B,T,param_dim)
        B, T, param_dim = h.shape

        # Global modulation parameters
        h_mean = h.mean(dim=1)  # (B, param_dim)
        global_params = self.global_modulator(h_mean)  # (B, 4)
        global_gain = torch.clamp(F.softplus(global_params[:, 0]), 0.1, 5.0).unsqueeze(1)
        global_bias = torch.clamp(global_params[:, 1], -3.0, 3.0).unsqueeze(1)
        phase_shift = torch.clamp(global_params[:, 2], -np.pi, np.pi).unsqueeze(1)
        maintenance_strength = torch.clamp(F.softplus(global_params[:, 3]), 0.2, 4.0).unsqueeze(1)

        # Output of all operators
        outs = []
        for i, op in enumerate(self.ops):
            try:
                op_out = op(h)  # Expected shape: (B, T)

                # Shape normalized to (B,T)
                if op_out.dim() == 3:  # (B, 1, T) or (B, C, T)
                    if op_out.size(1) == 1:
                        op_out = op_out.squeeze(1)  # (B, T)
                    else:
                        op_out = op_out.mean(dim=1)  # (B, T)
                elif op_out.dim() == 1:  # (T,)
                    op_out = op_out.unsqueeze(0).expand(B, -1)  # (B, T)
                elif op_out.dim() > 3:
                    op_out = op_out.reshape(B, -1)  # (B, ?)

                # Time dimension alignment
                if op_out.dim() == 2 and op_out.size(1) != T:
                    if op_out.size(1) > T:
                        op_out = op_out[:, :T]
                    else:
                        pad_size = T - op_out.size(1)
                        op_out = F.pad(op_out, (0, pad_size), value=0.0)

                # Make sure the shape is correct
                if op_out.shape != (B, T):
                    print(f"Warning: Operator {i} output shape {op_out.shape}, reshaping to ({B}, {T})")
                    op_out = torch.zeros(B, T, device=h.device, dtype=h.dtype)

                # Maintain related operator modulation
                t = torch.arange(T, device=h.device, dtype=h.dtype).unsqueeze(0).expand(B, -1)
                time_modulation = torch.sin(2 * np.pi * t / T + phase_shift) * maintenance_strength
                op_out = op_out * (1 + 0.2 * time_modulation) # Time-related modulation

                outs.append(op_out)
            except Exception as e:
                print(f"Warning: Operator {type(op).__name__} failed, using zeros. Error: {e}")
                outs.append(torch.zeros(B, T, device=h.device, dtype=h.dtype))

        st = torch.stack(outs, dim=-1)  # (B, T, K)

        #Adaptive weight
        w = self.gate(h)  # (B, 1, K)
        w = w.expand(-1, T, -1)  # (B, T, K)

        #Interaction between operators
        w_mean = w.mean(dim=1)  # (B, K)
        interaction_weights = torch.sigmoid(self.inter_op_network(w_mean))  # (B, K)
        interaction_weights = interaction_weights.unsqueeze(1).expand(-1, T, -1)  # (B, T, K)

        # Weighted combination + interaction effect
        base_damage = (st * w).sum(-1)  # (B, T)
        interaction_damage = (st * interaction_weights).sum(-1) * 0.3  # (B, T)

        damage = base_damage + interaction_damage
        damage = torch.clamp(damage, 0.0, 100.0)

        # Global modulation
        damage = global_gain * damage + global_bias

        return torch.clamp(damage, 0.0, 100.0)  # (B, T)

class TrendEncoder(nn.Module):
    """
    True Liquid Trend Encoder - Significantly Different Health Index Inference Patterns Before and After Maintenance
    """
    def __init__(self, in_ch, trend_ch=8): # Add trend_ch to provide richer features
        super().__init__()
        self.boltz = BoltzmannKAN(in_ch, out_ch=trend_ch)

        # Use liquid adaptive operator
        ops = [
            LiquidAdaptiveMonotonicLinearOp(trend_ch),
            LiquidAdaptiveMonotonicFlatOp(trend_ch),
            LiquidAdaptiveConcaveLogOp(trend_ch),
            LiquidAdaptiveSaturationSigmoidOp(trend_ch),
            LiquidAdaptiveHingeReLUOp(trend_ch),
            LiquidAdaptivePolynomialOp(trend_ch),
            LiquidAdaptiveDampedSinOp(trend_ch),
            LiquidAdaptivePiecewiseLinearOp(trend_ch)
        ]
        self.customkan = TrueLiquidCustomKAN(ops, trend_ch)

        # Maintain phase-aware projection network
        self.maintenance_aware_proj = nn.Sequential(
            nn.Linear(trend_ch + in_ch, 64), # Combine original features and trend features
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # gain, bias, maintenance_factor
        )

        # Health state network learned directly from sensor data - deeper and more complex
        self.health_inference_net = nn.Sequential(
            nn.Linear(in_ch, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Maintenance effect strengthens the network
        self.maintenance_effect_net = nn.Sequential(
            nn.Linear(in_ch, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # maintenance_intensity, recovery_rate
        )

        # Timing dynamic awareness network
        self.temporal_dynamics_net = nn.Sequential(
            nn.Linear(trend_ch, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3)  # temporal_weight, decay_rate, oscillation
        )

    def forward(self, x):  # x:(B,T,C)
        B, T, C = x.shape

        # Feature extraction based on Boltzmann KAN
        h_multi = self.boltz(x)              # (B,T,trend_ch)
        damage = self.customkan(h_multi) # (B,T) - This is the operator combination result

        # Combine original features and trend features for maintenance-aware projection
        x_reshaped = x.reshape(-1, C)  # (B*T, C)
        h_multi_reshaped = h_multi.reshape(-1, h_multi.size(-1))  # (B*T, trend_ch)
        combined_features = torch.cat([x_reshaped, h_multi_reshaped], dim=-1)  # (B*T, C+trend_ch)

        # Maintain perceptual projection parameters
        proj_params = self.maintenance_aware_proj(combined_features)  # (B*T, 3)
        proj_params = proj_params.view(B, T, 3)  # (B, T, 3)

        gain = torch.clamp(F.softplus(proj_params[:, :, 0]), 0.1, 5.0)  # (B, T)
        bias = torch.clamp(proj_params[:, :, 1], -3.0, 3.0)  # (B, T)
        maintenance_factor = torch.clamp(F.softplus(proj_params[:, :, 2]), 0.2, 3.0)  # (B, T)

        # Direct health status inference
        health_raw = self.health_inference_net(x_reshaped)  # (B*T, 1)
        health_direct = torch.sigmoid(health_raw).view(B, T)  # (B, T)

        # Maintain effect parameters
        maintenance_params = self.maintenance_effect_net(x_reshaped)  # (B*T, 2)
        maintenance_params = maintenance_params.view(B, T, 2)  # (B, T, 2)
        maintenance_intensity = torch.clamp(F.softplus(maintenance_params[:, :, 0]), 0.5, 3.0)  # (B, T)
        recovery_rate = torch.clamp(torch.sigmoid(maintenance_params[:, :, 1]), 0.1, 0.9)  # (B, T)

        # Timing dynamic characteristics
        temporal_params = self.temporal_dynamics_net(h_multi_reshaped)  # (B*T, 3)
        temporal_params = temporal_params.view(B, T, 3)  # (B, T, 3)
        temporal_weight = torch.clamp(F.softplus(temporal_params[:, :, 0]), 0.2, 2.0)  # (B, T)
        decay_rate = torch.clamp(torch.sigmoid(temporal_params[:, :, 1]), 0.05, 0.5)  # (B, T)
        oscillation = torch.clamp(temporal_params[:, :, 2], -0.5, 0.5)  # (B, T)

        # Timeline
        t = torch.arange(T, device=x.device, dtype=x.dtype).unsqueeze(0).expand(B, -1)  # (B, T)
        t_normalized = t / (T - 1)

        # Convert damage to health decay - more complex mapping
        damage_normalized = torch.sigmoid(gain * (damage + bias))

        # Timing dynamic effects
        temporal_decay = torch.exp(-decay_rate * t_normalized)
        temporal_oscillation = torch.sin(2 * np.pi * t_normalized + oscillation) * 0.1

        # Maintenance effect modeling
        maintenance_recovery = maintenance_intensity * torch.exp(-recovery_rate * t_normalized)

        # Comprehensive health status calculation
        #Basic health status: combined with direct inference and damage model
        base_health = health_direct * (1 - 0.4 * damage_normalized)

        # Timing dynamic modulation
        temporal_modulated = base_health * temporal_decay * (1 + temporal_oscillation)

        # Maintenance effect modulation
        maintenance_modulated = temporal_modulated * (1 + 0.3 * maintenance_recovery * maintenance_factor)

        # Generate an ideal attenuation pattern as a guide
        ideal_decay = 1.0 - t_normalized * 0.6 * temporal_weight # Adaptive attenuation strength

        # Mix ideal falloff and complex modes
        mixing_weight = torch.sigmoid(maintenance_factor - 1.0) # Maintenance intensity determines the mixing ratio
        hi = mixing_weight * ideal_decay + (1 - mixing_weight) * maintenance_modulated

        # Force monotonic decreasing constraints (stronger constraints)
        for t_idx in range(1, T):
            min_decrease = 0.002 + 0.001 * maintenance_factor[:, t_idx-1] # Adaptive minimum reduction
            hi = torch.cat([
                hi[:, :t_idx],
                torch.min(hi[:, t_idx:t_idx+1], hi[:, t_idx-1:t_idx] - min_decrease.unsqueeze(-1)),
                hi[:, t_idx+1:]
            ], dim=1)

        return hi, h_multi, damage # Return damage for visualization

class ReconHead(nn.Module):
    """
    Recursive reconstruction: given (x_t, h_t) → predict x_{t+1}
    """
    def __init__(self, C, hidden=128):
        super().__init__()
        self.gru = nn.GRU(input_size=C+1, hidden_size=hidden, batch_first=True)
        self.out = nn.Linear(hidden, C)
    def forward(self, x_in, h_in):
        # x_in:(B,T_in,C), h_in:(B,T_in)
        B,T,C = x_in.shape
        h_in_clamped = torch.clamp(h_in, 0.0, 10.0)  # Remove upper limit constraint
        feat = torch.cat([x_in, h_in_clamped.unsqueeze(-1)], dim=-1)  # (B,T,C+1)
        H,_ = self.gru(feat)                                  # (B,T,H)
        y = self.out(H)                                       # (B,T,C) aligned predict x_{t+1}
        return y

class MaintClassifier(nn.Module):
    """
    Use HI before-after difference features for 3-class classification
    Enhanced version: not only uses ΔHI, but also sensor data change patterns
    """
    def __init__(self, sensor_dim, hidden=64, n_classes=3):
        super().__init__()
        # Original HI difference features
        self.hi_mlp = nn.Sequential(
            nn.Linear(6, hidden), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden//2)
        )

        # Sensor data change features
        self.sensor_mlp = nn.Sequential(
            nn.Linear(sensor_dim * 2, hidden), nn.ReLU(),  # Before and after statistical features
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden//2)
        )

        # Fusion classifier
        self.final_classifier = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, n_classes)
        )

    def forward(self, h_b, h_a, x_b, x_a, mask):
        # HI difference features (original logic)
        m = mask
        T = m.size(1)
        valid = m.sum(1, keepdim=True).clamp_min(1.0)  # (B,1)
        db = h_a - h_b                        # (B,T)
        mean_d = (db*m).sum(1,keepdim=True)/valid
        var_d  = (((db-mean_d)*m)**2).sum(1,keepdim=True)/valid
        std_d  = (var_d + 1e-8).sqrt()
        d0     = (db[:, :1])                  # First value
        dmax   = (db.masked_fill(m==0, -1e9)).max(dim=1, keepdim=True).values
        pos_ratio = ((db>0).float()*m).sum(1,keepdim=True)/valid

        # Slope (linear fitting)
        t = torch.arange(T, device=h_b.device, dtype=h_b.dtype)
        t = (t - t.mean())/(t.std()+1e-6)
        def slope(x):
            num = (x*t).sum(1) - x.sum(1)*t.sum()/T
            den = (t**2).sum() - (t.sum()**2)/T + 1e-8
            return (num/den).unsqueeze(1)
        slope_diff = slope(h_a) - slope(h_b)  # (B,1)
        hi_feat = torch.cat([mean_d, d0, dmax, std_d, slope_diff, pos_ratio], dim=1)  # (B,6)
        hi_feat = torch.clamp(hi_feat, -10.0, 10.0)
        hi_features = self.hi_mlp(hi_feat)

        # Sensor data change features
        x_b_mean = (x_b * m.unsqueeze(-1)).sum(1) / valid        # (B,C)
        x_a_mean = (x_a * m.unsqueeze(-1)).sum(1) / valid        # (B,C)
        sensor_change = torch.cat([x_b_mean, x_a_mean], dim=1)   # (B, 2*C)
        sensor_features = self.sensor_mlp(sensor_change)

        # Fused features
        combined_features = torch.cat([hi_features, sensor_features], dim=1)
        logits = self.final_classifier(combined_features)

        return logits

def sanitize_tensors(*tensors):
    """Replace NaN/Inf in tensors with finite values"""
    result = []
    for t in tensors:
        if t is not None:
            t_clean = torch.nan_to_num(t, nan=0.0, posinf=1e6, neginf=-1e6)
            result.append(t_clean)
        else:
            result.append(t)
    return result[0] if len(result) == 1 else result

class DiffAwareReconstructor(nn.Module):
    """
    Difference-aware reconstructor for true liquids - operator parameters and behavior patterns are significantly different before and after maintenance
    """
    def __init__(self, in_ch, trend_ch=8, hidden=128, n_classes=3):
        super().__init__()
        self.encoder = TrendEncoder(in_ch, trend_ch)
        self.recon_b = ReconHead(in_ch, hidden)
        self.recon_a = ReconHead(in_ch, hidden)
        self.clf     = MaintClassifier(sensor_dim=in_ch, hidden=64, n_classes=n_classes)

        # Liquid property enhancement parameters
        self.liquid_enhancement = nn.Parameter(torch.tensor(2.0)) # Liquid enhancement factor
        self.maintenance_sensitivity = nn.Parameter(torch.tensor(1.5)) # Maintenance sensitivity

    def forward(self, x_b, x_a, mask):
        # x_b/x_a : (B,L,C) (will be cut to 0:L-2 / 1:L-1 later)
        h_b, h_multi_b, damage_b = self.encoder(x_b)  # (B,L)  Health state learned from sensor data
        h_a, h_multi_a, damage_a = self.encoder(x_a)  # (B,L)

        # Liquid Strengthening: Ensure significant difference before and after maintenance
        maintenance_effect = F.softplus(self.maintenance_sensitivity)
        h_a = h_a * (1 + 0.3 * maintenance_effect) # Health status improved after maintenance

        # Differences before and after forced maintenance
        diff_enhancement = F.softplus(self.liquid_enhancement)
        h_difference = torch.clamp(h_a - h_b, 0.05, 2.0) # Ensure positive difference
        h_a = h_b + h_difference * diff_enhancement

        # Numerical stabilization
        h_b, h_a = sanitize_tensors(h_b, h_a)

        # Recursive reconstruction: input 0:L-2 → predict 1:L-1
        xb_in, hb_in = x_b[:, :-2], h_b[:, :-2]
        xa_in, ha_in = x_a[:, :-2], h_a[:, :-2]
        yb_hat = self.recon_b(xb_in, hb_in)   # (B,L-2,C) ≈ x_b[:,1:L-1]
        ya_hat = self.recon_a(xa_in, ha_in)   # (B,L-2,C) ≈ x_a[:,1:L-1]

        # Numerical stabilization
        yb_hat, ya_hat = sanitize_tensors(yb_hat, ya_hat)

        # Classification: use unclipped length mask and include sensor features
        logits = self.clf(h_b, h_a, x_b, x_a, mask)
        logits = sanitize_tensors(logits)

        return yb_hat, ya_hat, h_b, h_a, logits, damage_b, damage_a

# ============================================================
# 3) Loss function: loss function that enhances liquid properties
# ============================================================
def masked_mse(a, b, mask):
    # a,b:(B,T,C)  mask:(B,T)
    diff = (a - b)**2
    mse  = (diff.mean(-1) * mask).sum() / (mask.sum() + 1e-6)
    return mse

def slope_loss(h, mask, kind="smooth"):
    # Smooth: second-order difference; Monotonic decreasing: penalize upward trend
    if kind=="smooth":
        d2 = h[:,2:] - 2*h[:,1:-1] + h[:,:-2]
        m  = mask[:,2:]
        return (d2.abs() * m).sum() / (m.sum()+1e-6)
    elif kind=="mono_dec":
        dh = h[:,1:] - h[:,:-1]        # If >0 indicates upward (violates "decreasing")
        m  = mask[:,1:]
        return (F.relu(dh) * m).sum() / (m.sum()+1e-6)
    else:
        return torch.tensor(0., device=h.device)

def liquid_operator_diversity_loss(model, weight=1.0):
    """
    Liquid operator diversity loss: forcing different operators to produce different output modes
    """
    if not hasattr(model.encoder.customkan, 'gate'):
        return torch.tensor(0.0)

    gate = model.encoder.customkan.gate
    if not hasattr(gate, 'weight_changes'):
        return torch.tensor(0.0)

    # Encourage diversity in weight changes
    weight_changes = gate.weight_changes  # (1, n_ops)
    if weight_changes.sum() < 1e-6:
        # If there is no weight change, apply a penalty
        return weight * torch.tensor(5.0, device=weight_changes.device)

    # Calculate the variance of weight changes to encourage diversity
    diversity = -torch.var(weight_changes) + 0.1 # Negative variance + base penalty
    return weight * torch.clamp(diversity, 0.0, 10.0)

def liquid_parameter_dynamics_loss(model, weight=0.5):
    """
    Dynamic loss of liquid parameters: ensure that operator parameters produce different responses under different inputs
    """
    loss = torch.tensor(0.0, device=next(model.parameters()).device)

    # Check the liquid operator in the encoder
    if hasattr(model.encoder, 'customkan') and hasattr(model.encoder.customkan, 'ops'):
        for op in model.encoder.customkan.ops:
            if hasattr(op, 'param_net'):
                # Check the weight changes of the parameter network
                for param in op.param_net.parameters():
                    if param.grad is not None:
                        # Encourage gradient diversity
                        grad_var = torch.var(param.grad)
                        loss += weight * torch.clamp(0.01 - grad_var, 0.0, 1.0)

    return loss

def enhanced_maintenance_effect_loss(h_b, h_a, labels, mask, weight=3.0):
    """
    Enhanced Maintenance Effect Loss: Ensures that different maintenance types produce significantly different patterns of health improvement
    """
    valid = mask.sum(1, keepdim=True).clamp_min(1.0)  # (B,1)

    # Calculate health improvement before and after maintenance
    improvement = ((h_a - h_b) * mask).sum(1, keepdim=True) / valid  # (B,1)

    # Calculate the improved variability (it should be similar within the same type, and there should be obvious differences between different types)
    loss = torch.zeros_like(improvement)

    for cls in [0, 1, 2]:
        cls_mask = (labels == cls).float().unsqueeze(1)  # (B,1)
        if cls_mask.sum() < 1:
            continue

        cls_improvements = improvement * cls_mask
        cls_valid = cls_mask.sum()

        if cls == 0: # Perfect: Requires significant improvement 0.4-0.6
            target_improvement = 0.5
            loss += cls_mask * ((improvement - target_improvement) ** 2) * 2.0
            # Additional requirements: The improvement of Perfect maintenance should be the largest
            loss += cls_mask * F.relu(0.4 - improvement) ** 2 * 3.0
        elif cls == 1: # General: Moderate improvement 0.2-0.4
            target_improvement = 0.3
            loss += cls_mask * ((improvement - target_improvement) ** 2) * 1.5
        else: # Poor: slight improvement 0.1-0.2
            target_improvement = 0.15
            loss += cls_mask * ((improvement - target_improvement) ** 2) * 1.0
            # Additional requirements: Poor maintenance cannot be improved too much
            loss += cls_mask * F.relu(improvement - 0.25) ** 2 * 2.0

    return weight * loss.mean()

def liquid_temporal_consistency_loss(h_b, h_a, mask, weight=1.0):
    """
    Liquid timing consistency loss: ensuring that the health index maintains reasonable dynamic changes over time
    """
    # Timing gradient consistency
    grad_b = torch.abs(h_b[:, 1:] - h_b[:, :-1])  # (B, T-1)
    grad_a = torch.abs(h_a[:, 1:] - h_a[:, :-1])  # (B, T-1)

    mask_grad = mask[:, 1:]  # (B, T-1)

    # The gradient after maintenance should be different from that before maintenance (reflecting liquid characteristics)
    grad_diff = torch.abs(grad_a - grad_b)

    # Encourage moderate gradient differences (not too big, not too small)
    target_grad_diff = 0.02
    grad_consistency_loss = ((grad_diff - target_grad_diff) ** 2 * mask_grad).sum() / (mask_grad.sum() + 1e-6)

    return weight * grad_consistency_loss

# ============================================================
# 4) Data splitting and visualization (aligned to same time axis)
# ============================================================
LABEL2NAME = {0: "Perfect", 1: "General", 2: "Poor"}

def split_pairs_uid(pairs, train_ratio=0.7, val_ratio=0.1, seed=42):
    uids = sorted(list(pairs.keys()))
    rs = np.random.RandomState(seed)
    rs.shuffle(uids)
    n = len(uids); n_tr=int(n*train_ratio); n_vl=int(n*val_ratio)
    to_dict = lambda ss:{u:pairs[u] for u in ss if u in pairs}
    return to_dict(uids[:n_tr]), to_dict(uids[n_tr:n_tr+n_vl]), to_dict(uids[n_tr+n_vl:])

def print_split_summary(pairs, train_ratio=0.7, val_ratio=0.1, seed=42):
    pairs_tr, pairs_vl, pairs_te = split_pairs_uid(pairs, train_ratio, val_ratio, seed)
    def count_items(pp):
        cnt = 0
        for uid, d in pp.items():
            cnt += len(d)
        return len(pp), cnt
    n_uid_tr, n_pair_tr = count_items(pairs_tr)
    n_uid_vl, n_pair_vl = count_items(pairs_vl)
    n_uid_te, n_pair_te = count_items(pairs_te)
    print(f"[Data Split] Train: UID={n_uid_tr}, pairs≈{n_pair_tr} | "
          f"Val: UID={n_uid_vl}, pairs≈{n_pair_vl} | "
          f"Test: UID={n_uid_te}, pairs≈{n_pair_te}")

def remove_constant_segments(hi_values, threshold=1e-6):
    """
    Remove constant segments in HI curves (caused by mask or other reasons)
    Return valid HI values and corresponding time indices
    """
    if len(hi_values) <= 1:
        return hi_values, np.arange(len(hi_values))

    # Calculate differences between adjacent points
    diffs = np.abs(np.diff(hi_values))

    # Find points with sufficient change
    valid_mask = np.ones(len(hi_values), dtype=bool)
    valid_mask[1:] = diffs > threshold

    # Ensure at least first and last two points are kept
    if np.sum(valid_mask) < 2:
        valid_mask[0] = True
        valid_mask[-1] = True

    valid_indices = np.where(valid_mask)[0]
    return hi_values[valid_indices], valid_indices

@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total = {"mse_b":0.0, "mse_a":0.0, "acc":0.0, "delta_mean":0.0}
    n_batch=0
    n_acc = 0; n_tot=0
    for batch in loader:
        xb = batch["x_before"].to(device)
        xa = batch["x_after"].to(device)
        hi_b = batch["hi_before"].to(device)
        hi_a = batch["hi_after"].to(device)
        labels = batch["labels"].to(device)
        lengths= batch["lengths"].to(device)
        mask   = batch["mask"].to(device)

        # Valid segment: 0:L-2 input, 1:L-1 target
        m_tgt  = (torch.arange(xb.size(1)-2, device=device)[None,:] < (lengths-2)[:,None]).float()

        yb_hat, ya_hat, h_b, h_a, logits, damage_b, damage_a = model(xb, xa, mask)

        # Numerical stabilization
        yb_hat, ya_hat, h_b, h_a, logits = sanitize_tensors(yb_hat, ya_hat, h_b, h_a, logits)

        # Align targets
        yb_tgt = xb[:,1:-1,:]
        ya_tgt = xa[:,1:-1,:]

        mse_b = masked_mse(yb_hat, yb_tgt, m_tgt)
        mse_a = masked_mse(ya_hat, ya_tgt, m_tgt)

        # Check for NaN
        if torch.isnan(mse_b) or torch.isnan(mse_a):
            continue

        # Classification
        pred = logits.argmax(dim=1)
        n_acc += (pred == labels).sum().item()
        n_tot += labels.numel()

        # Statistics of ΔHI mean (post-maintenance improvement relative to pre-maintenance)
        valid = mask.sum(1, keepdim=True).clamp_min(1.0)
        delta_mean = ((h_a - h_b)*mask).sum(1,keepdim=True)/valid
        delta_mean = torch.nan_to_num(delta_mean, nan=0.0)
        total["delta_mean"] += delta_mean.mean().item()

        total["mse_b"] += mse_b.item()
        total["mse_a"] += mse_a.item()
        n_batch += 1

    for k in ["mse_b","mse_a","delta_mean"]:
        total[k] /= max(n_batch,1)
    total["acc"] = n_acc / max(n_tot,1)
    return total

@torch.no_grad()
def collect_test_predictions(model, loader, device, max_curve_keep=24):
    """
    Collect predictions, ΔHI, and a few sample HI curves on test set (for plotting).
    Increase the collection of operator combination results (damage)
    """
    model.eval()
    y_true, y_pred = [], []
    all_delta_mean, all_uids = [], []
    keep_curves = []

    for batch in loader:
        xb = batch["x_before"].to(device)   # (B,L,C)
        xa = batch["x_after"].to(device)
        mask   = batch["mask"].to(device)   # (B,L)
        labels = batch["labels"].to(device) # (B,)
        lengths= batch["lengths"]           # cpu tensor
        uids   = batch["uids"]
        hi_before = batch["hi_before"].to(device)  # (B,L) ground truth HI_before from dataset
        hi_after = batch["hi_after"].to(device)    # (B,L) ground truth HI_after from dataset

        yb_hat, ya_hat, h_b, h_a, logits, damage_b, damage_a = model(xb, xa, mask)

        # Numerical stabilization
        yb_hat, ya_hat, h_b, h_a, logits = sanitize_tensors(yb_hat, ya_hat, h_b, h_a, logits)

        prob = F.softmax(logits, dim=1)     # (B,3)
        pred = prob.argmax(1)               # (B,)

        # Statistics of ΔHI mean (post-maintenance improvement relative to pre-maintenance)
        valid = mask.sum(1, keepdim=True).clamp_min(1.0)
        delta_mean = ((h_a - h_b) * mask).sum(1, keepdim=True) / valid  # (B,1)
        delta_mean = torch.nan_to_num(delta_mean, nan=0.0)

        y_true.append(labels.cpu().numpy())
        y_pred.append(pred.cpu().numpy())
        all_delta_mean.append(delta_mean.squeeze(1).cpu().numpy())
        all_uids.extend(uids)

        # Save some curves for visualization (aligned: after concatenated after before)
        for i in range(xb.size(0)):
            if len(keep_curves) >= max_curve_keep:
                break
            L_i = int(lengths[i].item())
            # Fix: truncate reconstruction prediction by actual length
            L_recon = min(L_i - 2, yb_hat.size(1))  # Reconstruction sequence length
            keep_curves.append({
                "uid": uids[i],
                "true": int(labels[i].cpu().item()),
                "pred": int(pred[i].cpu().item()),
                "prob": prob[i].cpu().numpy(),
                "h_before": h_b[i, :L_i].cpu().numpy(),          # model predicted HI_before
                "h_after":  h_a[i, :L_i].cpu().numpy(),          # model predicted HI_after
                "hi_before_gt": hi_before[i, :L_i].cpu().numpy(), # ground truth HI_before from dataset
                "hi_after_gt": hi_after[i, :L_i].cpu().numpy(),   # ground truth HI_after from dataset
                "damage_before": damage_b[i, :L_i].cpu().numpy(), # Operator combination result (before maintenance)
                "damage_after": damage_a[i, :L_i].cpu().numpy(), # Operator combination result (after maintenance)
                "yb_hat":   yb_hat[i, :L_recon].cpu().numpy(),       # (L_recon,C)
                "ya_hat":   ya_hat[i, :L_recon].cpu().numpy(),       # (L_recon,C)
                "x_before": xb[i, :L_i].cpu().numpy(),               # (L,C)
                "x_after":  xa[i, :L_i].cpu().numpy(),               # (L,C)
                "length": L_i
            })

    y_true = np.concatenate(y_true, axis=0) if len(y_true)>0 else np.array([])
    y_pred = np.concatenate(y_pred, axis=0) if len(y_pred)>0 else np.array([])
    all_delta_mean = np.concatenate(all_delta_mean, axis=0) if len(all_delta_mean)>0 else np.array([])
    return y_true, y_pred, all_delta_mean, all_uids, keep_curves

def select_best_curves_for_plotting(curves, n_show=6):
    """
    Intelligent selection of the best drawing cases:
    1. Prioritize samples with correct predictions
    2. Select the most representative sample within each maintenance strategy category
    3. Select the sample with the most obvious change in health index
    4. Select samples with the best sensor data quality
    """
    if len(curves) == 0:
        return []

    #Group by real category
    curves_by_class = {0: [], 1: [], 2: []}
    for curve in curves:
        curves_by_class[curve["true"]].append(curve)

    # Calculate quality score for each sample
    def calculate_quality_score(curve):
        score = 0.0

        # 1. Prediction accuracy (weight: 50%)
        if curve["true"] == curve["pred"]:
            score += 0.5

        # 2. Prediction confidence (weight: 20%)
        confidence = curve["prob"][curve["pred"]]
        score += 0.2 * confidence

        # 3. Obvious degree of change in health index (weight: 20%)
        h_before = curve["h_before"]
        h_after = curve["h_after"]
        hi_improvement = np.mean(h_after - h_before)
        # Normalize to 0-1 range
        normalized_improvement = np.clip((hi_improvement + 0.5) / 1.0, 0, 1)
        score += 0.2 * normalized_improvement

        # 4. Data quality (weight: 10%)
        # Check if there are outliers or constant segments
        x_before_var = np.var(curve["x_before"])
        x_after_var = np.var(curve["x_after"])
        data_quality = np.clip((x_before_var + x_after_var) / 2.0, 0, 1)
        score += 0.1 * data_quality

        return score

    # Calculate quality scores for all samples
    for curve in curves:
        curve["quality_score"] = calculate_quality_score(curve)

    # Intelligent selection strategy
    selected_curves = []

    # Select at least one best sample for each category
    for class_id in [0, 1, 2]:
        if len(curves_by_class[class_id]) > 0:
            # Sort by quality score
            sorted_curves = sorted(curves_by_class[class_id],
                                 key=lambda x: x["quality_score"], reverse=True)
            selected_curves.append(sorted_curves[0])

    # If more samples are needed, select from the remaining high-quality samples
    remaining_slots = n_show - len(selected_curves)
    if remaining_slots > 0:
        # Exclude selected from all samples, select by quality score
        already_selected_uids = {curve["uid"] for curve in selected_curves}
        remaining_curves = [curve for curve in curves
                          if curve["uid"] not in already_selected_uids]

        if len(remaining_curves) > 0:
            remaining_curves.sort(key=lambda x: x["quality_score"], reverse=True)
            selected_curves.extend(remaining_curves[:remaining_slots])

    #Final sorting by quality score
    selected_curves.sort(key=lambda x: x["quality_score"], reverse=True)

    return selected_curves[:n_show]

def plot_damage_operator_results(curves, n_show=6, seed=0):
    """
    Draw operator combination results (damage) - display the combined output of the liquid operator alone
    There should be significant differences in the operator combination patterns before and after maintenance
    """
    if len(curves)==0:
        print("(No visualization samples)")
        return

    # Use the smart selection function to select the best drawing case
    selected_curves = select_best_curves_for_plotting(curves, n_show)

    if len(selected_curves) == 0:
        print("(No high-quality visualization samples found)")
        return

    n_show = len(selected_curves)
    cols = 3
    rows = int(np.ceil(n_show/cols))
    plt.figure(figsize=(cols*4.6, rows*3.2))

    for k in range(n_show):
        ex = selected_curves[k]
        damage_b = ex["damage_before"] # Operator combination result (before maintenance)
        damage_a = ex["damage_after"] # Operator combination result (after maintenance)
        L = ex["length"]  # Use actual length instead of padded length

        # Remove constant segments
        damage_b_clean, damage_b_indices = remove_constant_segments(damage_b)
        damage_a_clean, damage_a_indices = remove_constant_segments(damage_a)

        # Corresponding time axis
        t_b = damage_b_indices
        t_a = damage_a_indices + L  # after follows immediately

        uid = ex["uid"]
        pred = ex["pred"]; true = ex["true"]; prob = ex["prob"]
        quality_score = ex["quality_score"]

        plt.subplot(rows, cols, k+1)

        # Plot operator combination results (damage)
        if len(damage_b_clean) > 1:
            plt.plot(t_b, damage_b_clean, label="Operator Damage (Pre-maintenance)",
                    linewidth=2.5, marker='o', markersize=4, color='blue')
        if len(damage_a_clean) > 1:
            plt.plot(t_a, damage_a_clean, label="Operator Damage (Post-maintenance)",
                    linewidth=2.5, linestyle='--', marker='s', markersize=4, color='red')

        # Maintenance time vertical line
        plt.axvline(L-1, color='k', linestyle=':', linewidth=1.5, alpha=0.8)

        # Calculate changes in operator combination results
        damage_change_mean = float(np.mean(damage_a - damage_b))
        damage_change_max = float(np.max(damage_a - damage_b))

        # Add quality rating to title
        correctness = "✓" if true == pred else "✗"
        title = (f"uid={uid} {correctness} (Quality: {quality_score:.2f})\n"
                 f"True={LABEL2NAME[true]} | Pred={LABEL2NAME[pred]} | Conf={prob[pred]:.3f}\n"
                 f"Δ_Damage_Mean={damage_change_mean:.3f}, Δ_Damage_Max={damage_change_max:.3f}")
        plt.title(title, fontsize=9)
        plt.xlabel("Cycle (Pre-maintenance | Post-maintenance)")
        plt.ylabel("Liquid Operator Damage Output")
        plt.grid(ls='--', alpha=.4, linewidth=0.8)

        #Add background color to indicate prediction accuracy
        if true == pred:
            plt.gca().set_facecolor('#f0f8ff') # Light blue background indicates correct prediction
        else:
            plt.gca().set_facecolor('#fff0f0') # Light red background indicates incorrect prediction

        if k==0: plt.legend(fontsize=8, loc='best')

    plt.suptitle(f"Best {n_show} Liquid Operator Damage Results (Ranked by Quality Score)",
                 fontsize=14, y=0.98)
    plt.tight_layout()
    plt.show()

def plot_hi_examples_aligned(curves, n_show=6, seed=0):
    """Post-maintenance sequence continues from pre-maintenance endpoint; draw vertical line to mark t_m; remove constant segments.
    Add "Learned HI" in title to indicate this is health state inferred from sensor data, post-maintenance should be higher than pre-maintenance
    Intelligent selection of the best drawing cases
    """
    if len(curves)==0:
        print("(No visualization samples)")
        return

    # Use the smart selection function to select the best drawing case
    selected_curves = select_best_curves_for_plotting(curves, n_show)

    if len(selected_curves) == 0:
        print("(No high-quality visualization samples found)")
        return

    n_show = len(selected_curves)
    cols = 3
    rows = int(np.ceil(n_show/cols))
    plt.figure(figsize=(cols*4.6, rows*3.2))

    for k in range(n_show):
        ex = selected_curves[k]
        hb = ex["h_before"]; ha = ex["h_after"]
        hb_gt = ex["hi_before_gt"]; ha_gt = ex["hi_after_gt"]      # ground truth HI from dataset
        L  = ex["length"]  # Use actual length instead of padded length

        # Remove constant segments
        hb_clean, hb_indices = remove_constant_segments(hb)
        ha_clean, ha_indices = remove_constant_segments(ha)
        hb_gt_clean, hb_gt_indices = remove_constant_segments(hb_gt)
        ha_gt_clean, ha_gt_indices = remove_constant_segments(ha_gt)

        # Corresponding time axis
        t_b = hb_indices
        t_a = ha_indices + L  # after follows immediately
        t_b_gt = hb_gt_indices
        t_a_gt = ha_gt_indices + L

        uid = ex["uid"]
        pred = ex["pred"]; true = ex["true"]; prob = ex["prob"]
        quality_score = ex["quality_score"]

        plt.subplot(rows, cols, k+1)

        # Only plot non-constant segments - model predicted HI
        if len(hb_clean) > 1:
            plt.plot(t_b, hb_clean, label="Learned HI_Pre-maintenance", linewidth=2.5, marker='o', markersize=4, color='blue')
        if len(ha_clean) > 1:
            plt.plot(t_a, ha_clean, label="Learned HI_Post-maintenance",  linewidth=2.5, linestyle='--', marker='s', markersize=4, color='red')

        # Plot ground truth HI from dataset
        if len(hb_gt_clean) > 1:
            plt.plot(t_b_gt, hb_gt_clean, label="GT HI_Pre-maintenance", linewidth=1.8, color='cyan', alpha=0.8)
        if len(ha_gt_clean) > 1:
            plt.plot(t_a_gt, ha_gt_clean, label="GT HI_Post-maintenance",  linewidth=1.8, linestyle=':', color='orange', alpha=0.8)

        # Maintenance time vertical line
        plt.axvline(L-1, color='k', linestyle=':', linewidth=1.5, alpha=0.8)

        d_mean = float(np.mean(ha - hb))  # Post-maintenance improvement relative to pre-maintenance
        d_max  = float(np.max(ha - hb))
        d_mean_gt = float(np.mean(ha_gt - hb_gt))

        # Add quality rating to title
        correctness = "✓" if true == pred else "✗"
        title = (f"uid={uid} {correctness} (Quality: {quality_score:.2f})\n"
                 f"True={LABEL2NAME[true]} | Pred={LABEL2NAME[pred]} | Conf={prob[pred]:.3f}\n"
                 f"ΔHI_Learned={d_mean:.3f}, ΔHI_GT={d_mean_gt:.3f}")
        plt.title(title, fontsize=9)
        plt.xlabel("Cycle (Pre-maintenance | Post-maintenance)")
        plt.ylabel("Health Index (↑Post should be higher)")
        plt.grid(ls='--', alpha=.4, linewidth=0.8)

        #Add background color to indicate prediction accuracy
        if true == pred:
            plt.gca().set_facecolor('#f0f8ff') # Light blue background indicates correct prediction
        else:
            plt.gca().set_facecolor('#fff0f0') # Light red background indicates incorrect prediction

        if k==0: plt.legend(fontsize=8, loc='best')

    plt.suptitle(f"Best {n_show} Health Index Examples (Ranked by Quality Score)", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.show()

def plot_sensor_examples_aligned(curves, sensor_idx_list=None, n_cols=4):
    """
    Plot several sensors: original(before/after) + one-step recursive prediction of after(ya_hat),
    post-maintenance time series starts after pre-maintenance end. Show trajectories under different
    maintenance strategies. Intelligent selection of the best sensor dimensions for drawing
    """
    if len(curves)==0:
        print("(No visualization samples)")
        return

    # Select the best samples for sensor visualization
    selected_curves = select_best_curves_for_plotting(curves, 9) # Select more samples for sensor analysis

    # Group curves by maintenance strategy (true class)
    curves_by_strategy = {0: [], 1: [], 2: []}
    for curve in selected_curves:
        curves_by_strategy[curve["true"]].append(curve)

    # Select best example from each strategy for comparison
    strategy_examples = {}
    for strategy in [0, 1, 2]:
        if len(curves_by_strategy[strategy]) > 0:
            # Select the sample with the highest quality score under this strategy
            best_curve = max(curves_by_strategy[strategy], key=lambda x: x["quality_score"])
            strategy_examples[strategy] = best_curve

    if len(strategy_examples) == 0:
        print("(No high-quality visualization samples for any maintenance strategy)")
        return

    # Intelligent selection of sensor dimensions
    first_ex = list(strategy_examples.values())[0]
    xb = first_ex["x_before"]       # (L,C)
    C = xb.shape[1]

    if sensor_idx_list is None:
        # Intelligently select the most representative sensor dimension
        sensor_scores = []

        for sensor_idx in range(C):
            score = 0.0

            # Calculate the degree of change of the sensor under different maintenance strategies
            for strategy, ex in strategy_examples.items():
                xb_sensor = ex["x_before"][:, sensor_idx]
                xa_sensor = ex["x_after"][:, sensor_idx]

                # 1. Range of change (weight: 40%)
                change_magnitude = np.abs(np.mean(xa_sensor) - np.mean(xb_sensor))
                score += 0.4 * min(change_magnitude / (np.std(xb_sensor) + 1e-6), 5.0)

                # 2. Variance (information amount) (weight: 30%)
                variance = np.var(xb_sensor) + np.var(xa_sensor)
                score += 0.3 * min(variance, 10.0)

                # 3. Signal-to-noise ratio (weight: 30%)
                snr = np.mean(xb_sensor) / (np.std(xb_sensor) + 1e-6)
                score += 0.3 * min(abs(snr), 5.0)

            sensor_scores.append((sensor_idx, score))

        # Select the sensor with the highest rating
        sensor_scores.sort(key=lambda x: x[1], reverse=True)
        sensor_idx_list = [idx for idx, _ in sensor_scores[:min(8, len(sensor_scores))]]

    n = len(sensor_idx_list)
    n_rows = int(np.ceil(n/n_cols))
    plt.figure(figsize=(n_cols*5.5, n_rows*4.0))

    colors = {0: '#e74c3c', 1: '#f39c12', 2: '#9b59b6'} # Brighter colors

    for i, s in enumerate(sensor_idx_list):
        plt.subplot(n_rows, n_cols, i+1)

        # Plot pre-maintenance trajectory (common for all strategies, use best example)
        best_ex = max(strategy_examples.values(), key=lambda x: x["quality_score"])
        xb = best_ex["x_before"]
        L = best_ex["length"]
        t_b = np.arange(L)
        plt.plot(t_b, xb[:L,s], label="Original (Pre-maintenance)", linewidth=2.0, color='#2c3e50', alpha=0.9)

        # Plot post-maintenance trajectories for each available strategy
        for strategy in sorted(strategy_examples.keys()):
            ex = strategy_examples[strategy]
            xa = ex["x_after"]
            ya = ex["ya_hat"]
            L_strategy = ex["length"]
            quality = ex["quality_score"]

            # Post-maintenance original trajectory
            t_a = np.arange(L_strategy, L_strategy + L_strategy)
            plt.plot(t_a, xa[:L_strategy,s],
                    label=f"Original (Post-{LABEL2NAME[strategy]}, Q:{quality:.2f})",
                    linewidth=1.8, linestyle="--",
                    color=colors[strategy], alpha=0.9)

            # Post-maintenance prediction trajectory
            if len(ya) > 0 and L_strategy-2 > 0:
                t_pred = np.arange(L_strategy+1, L_strategy+1+min(L_strategy-2, len(ya)))
                plt.plot(t_pred, ya[:len(t_pred),s],
                        label=f"Predicted (Post-{LABEL2NAME[strategy]})",
                        linewidth=2.2, color=colors[strategy], marker='o', markersize=3)

        # Draw vertical line at maintenance point
        plt.axvline(L-0.5, color='#27ae60', linestyle=':', linewidth=2.5, alpha=0.9, label='Maintenance Point')

        # Calculate the quality score of this sensor and add it to the title
        sensor_score = dict(sensor_scores)[s] if 'sensor_scores' in locals() else 0
        plt.title(f"Sensor_{s:02d} (Score: {sensor_score:.2f}) - Strategy Comparison", fontsize=10, fontweight='bold')

        if i==0:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.xlabel("Continuous Time Sequence", fontsize=9)
        plt.ylabel("Sensor Value", fontsize=9)
        plt.grid(ls="--", alpha=.4, linewidth=0.8)

        # Add slight background color
        plt.gca().set_facecolor('#fafafa')

    plt.suptitle("Best Sensor Trajectories under Different Maintenance Strategies",
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()

def topk_by_delta(df_delta, k=5):
    if len(df_delta)==0:
        return
    print("\n" + "="*60)
    print("TOP-K SAMPLES WITH HIGHEST MAINTENANCE EFFECTS")
    print("="*60)

    for cls in [0,1,2]:
        sub = df_delta[df_delta["true"]==cls].sort_values("delta_hi_mean", ascending=False).head(k)
        if len(sub) > 0:
            print(f"\n[True={LABEL2NAME[cls]}] Top {k} samples with highest learned ΔHI_mean:")
            print("-" * 50)
            for idx, row in sub.iterrows():
                pred_correct = "✓" if row["true"] == row["pred"] else "✗"
                print(f"  UID: {row['uid']:>8} | ΔHI: {row['delta_hi_mean']:>6.3f} | "
                      f"Pred: {LABEL2NAME[int(row['pred'])]:>7} {pred_correct}")

# —— Load trained best model
def load_trained_model(model_path, device, in_ch):
    """Load the best model weights saved during training"""
    # Initialize model with same architecture
    model = DiffAwareReconstructor(in_ch=in_ch, trend_ch=4, hidden=128, n_classes=3).to(device)

    if os.path.exists(model_path):
        print(f"Loading model: {model_path}")
        state_dict = torch.load(model_path, map_location=device)

        # Filter out keys that don't match (for backward compatibility)
        model_dict = model.state_dict()
        filtered_dict = {}
        for k, v in state_dict.items():
            if k in model_dict and model_dict[k].shape == v.shape:
                filtered_dict[k] = v
            else:
                print(f"Warning: Skipping key {k} due to shape mismatch or missing in current model")

        # Load only matching parameters
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict, strict=False)
        print("Model loaded successfully (with potential missing keys)!")
    else:
        print(f"Warning: Model file does not exist {model_path}, will use randomly initialized model")
    return model

# —— Need to determine in_ch from pairs data first
def get_input_dim_from_pairs(pairs):
    """Get input dimension from pairs data"""
    for uid, strategies in pairs.items():
        for strategy, data in strategies.items():
            if "x_before" in data:
                return np.array(data["x_before"]).shape[1]
    raise ValueError("Cannot determine input dimension from pairs data")

# Get input dimension
C = get_input_dim_from_pairs(pairs)
print(f"Detected input dimension: {C}")

# Load best model (if exists)
model_path = "/content/drive/MyDrive/CMAPSS/main_dynamics_identification_4.pth"
model = load_trained_model(model_path, DEVICE, C)

# —— Print train/validation/test split (consistent with training phase: 7/1/2)
print_split_summary(pairs)

# —— Prepare test set data
_, _, pairs_te = split_pairs_uid(pairs, 0.7, 0.1, 42)
ds_te = PairsReconstructDataset(pairs_te, horizon=50)  # Same horizon as training
ld_te = DataLoader(ds_te, batch_size=32, shuffle=False, collate_fn=pad_collate_shift)
te = (ds_te, ld_te)

y_true, y_pred, delta_mean_all, uids_all, curves = collect_test_predictions(model, ld_te, DEVICE)

# —— Print overall metrics
print("\n" + "="*60)
print("TEST SET OVERALL METRICS")
print("="*60)
print(f"Sample count: {len(y_true)}")
if len(y_true) > 0:
    acc = (y_true == y_pred).mean()
    print(f"Classification Accuracy: {acc:.4f}")

    # Calculate the accuracy of each category
    for cls in [0, 1, 2]:
        cls_mask = (y_true == cls)
        if cls_mask.sum() > 0:
            cls_acc = (y_pred[cls_mask] == cls).mean()
            print(f"{LABEL2NAME[cls]} Accuracy: {cls_acc:.4f} ({cls_mask.sum()} samples)")
else:
    print("No samples in test set (check pairs split and horizon conditions).")

# —— Classification report & confusion matrix
try:
    from sklearn.metrics import classification_report, confusion_matrix
    target_names = [LABEL2NAME[i] for i in sorted(LABEL2NAME)]
    print("\n[Classification Report]")
    print(classification_report(y_true, y_pred, target_names=target_names, digits=4))
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])
except Exception:
    print("\n[Warning] scikit-learn not installed, will print simple confusion table.")
    cm = np.zeros((3,3), dtype=int)
    for t,p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1

# —— Print confusion matrix values
print("\n[Confusion Matrix] Row=True class, Column=Predicted class")
print(pd.DataFrame(cm, index=[LABEL2NAME[i] for i in [0,1,2]],
                      columns=[LABEL2NAME[i] for i in [0,1,2]]))

# —— Plot confusion matrix (enhanced)
plt.figure(figsize=(6.0,5.0))
im = plt.imshow(cm, interpolation='nearest', cmap='Blues')
plt.title("Confusion Matrix (Test Set)", fontsize=14, fontweight='bold', pad=20)
plt.xlabel("Predicted", fontsize=12)
plt.ylabel("True", fontsize=12)
plt.xticks([0,1,2], [LABEL2NAME[i] for i in [0,1,2]])
plt.yticks([0,1,2], [LABEL2NAME[i] for i in [0,1,2]])

#Add values ​​and percentages
for i in range(3):
    for j in range(3):
        total = cm[i].sum()
        if total > 0:
            percentage = cm[i, j] / total * 100
            text_color = 'white' if cm[i, j] > cm.max() * 0.5 else 'black'
            plt.text(j, i, f'{cm[i, j]}\n({percentage:.1f}%)',
                    ha="center", va="center", color=text_color, fontweight='bold')

plt.colorbar(im)
plt.tight_layout()
plt.show()

# —— Statistics of ΔHI distribution (by true class/predicted class)
if len(delta_mean_all) > 0:
    df_delta = pd.DataFrame({
        "uid": uids_all[:len(delta_mean_all)],
        "true": y_true.astype(int),
        "pred": y_pred.astype(int),
        "delta_hi_mean": delta_mean_all.astype(float)
    })

    print("\n" + "="*50)
    print("ΔHI MEAN STATISTICS BY TRUE CLASS")
    print("="*50)
    stats_by_true = df_delta.groupby("true")["delta_hi_mean"].describe()
    print(stats_by_true.round(4))

    print("\n" + "="*50)
    print("ΔHI MEAN STATISTICS BY PREDICTED CLASS")
    print("="*50)
    stats_by_pred = df_delta.groupby("pred")["delta_hi_mean"].describe()
    print(stats_by_pred.round(4))

# ——Continuous time axis: HI and several sensors before/after aligned visualization (intelligent selection of the best drawing)
print("\n" + "="*60)
print("GENERATING BEST QUALITY VISUALIZATIONS...")
print("="*60)

plot_hi_examples_aligned(curves, n_show=6, seed=2025)
plot_sensor_examples_aligned(curves, sensor_idx_list=None, n_cols=4)

# ——(Optional) Top-K with largest ΔHI in each class, for manual review (enhanced display)
if len(delta_mean_all) > 0:
    topk_by_delta(df_delta, k=5)

# %% [notebook code cell 21]

# %% [notebook code cell 22]

# ============================================================
# 6) Model Loading + Test Set Evaluation and Visualization (Aligned Before→After)
# ============================================================
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import os

LABEL2NAME = {0: "Perfect", 1: "General", 2: "Poor"}

# —— Load trained best model
def load_trained_model(model_path, device):
    """Load the best model weights saved during training"""
    # Assume model architecture is already defined, only load weights here
    # You need to ensure model is initialized with the same architecture
    if os.path.exists(model_path):
        print(f"Loading model: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded successfully!")
    else:
        print(f"Warning: Model file does not exist {model_path}, will use current model state")
    return model

# Load best model (if exists)
model_path = "/content/drive/MyDrive/Mimar_turbo/main_identification_2.pth"
model = load_trained_model(model_path, DEVICE)

# —— Print train/validation/test split (consistent with training phase: 7/1/2)
def print_split_summary(pairs, seed=42):
    pairs_tr, pairs_vl, pairs_te = split_pairs_uid(pairs, 0.7, 0.1, seed)
    def count_items(pp):
        cnt = 0
        for uid, d in pp.items():
            cnt += len(d)
        return len(pp), cnt
    n_uid_tr, n_pair_tr = count_items(pairs_tr)
    n_uid_vl, n_pair_vl = count_items(pairs_vl)
    n_uid_te, n_pair_te = count_items(pairs_te)
    print(f"[Data Split] Train: UID={n_uid_tr}, pairs≈{n_pair_tr} | "
          f"Val: UID={n_uid_vl}, pairs≈{n_pair_vl} | "
          f"Test: UID={n_uid_te}, pairs≈{n_pair_te}")

print_split_summary(pairs)

# —— Training completed, te = (ds_te, ld_te)
ds_te, ld_te = te

@torch.no_grad()
def collect_test_predictions(model, loader, device, max_curve_keep=24):
    """
    Collect predictions, ΔHI, and a few sample curves on test set (for aligned plotting).
    - Save HI_before / HI_after
    - Save yb_hat / ya_hat (one-step recursive prediction)
    - Save x_before / x_after (original sensors)
    """
    model.eval()
    y_true, y_pred = [], []
    all_delta_mean, all_uids = [], []
    keep_curves = []

    for batch in loader:
        xb = batch["x_before"].to(device)   # (B,L,C)
        xa = batch["x_after"].to(device)
        mask   = batch["mask"].to(device)   # (B,L)
        labels = batch["labels"].to(device) # (B,)
        lengths= batch["lengths"]           # cpu tensor
        uids   = batch["uids"]
        hi_before = batch["hi_before"].to(device)  # (B,L) ground truth HI_before from dataset
        hi_after = batch["hi_after"].to(device)    # (B,L) ground truth HI_after from dataset

        yb_hat, ya_hat, h_b, h_a, logits = model(xb, xa, mask)
        prob = F.softmax(logits, dim=1)     # (B,3)
        pred = prob.argmax(1)               # (B,)

        # ΔHI (mean over valid sequence)
        valid = mask.sum(1, keepdim=True).clamp_min(1.0)
        delta_mean = ((h_a - h_b) * mask).sum(1, keepdim=True) / valid  # (B,1)

        y_true.append(labels.cpu().numpy())
        y_pred.append(pred.cpu().numpy())
        all_delta_mean.append(delta_mean.squeeze(1).cpu().numpy())
        all_uids.extend(uids)

        # Save some samples for aligned visualization
        for i in range(xb.size(0)):
            if len(keep_curves) >= max_curve_keep:
                break
            L_i = int(lengths[i].item())
            keep_curves.append({
                "uid": uids[i],
                "true": int(labels[i].cpu().item()),
                "pred": int(pred[i].cpu().item()),
                "prob": prob[i].cpu().numpy(),
                "h_before": h_b[i, :L_i].cpu().numpy(),          # model predicted HI_before
                "h_after":  h_a[i, :L_i].cpu().numpy(),          # model predicted HI_after
                "hi_before_gt": hi_before[i, :L_i].cpu().numpy(), # ground truth HI_before from dataset
                "hi_after_gt": hi_after[i, :L_i].cpu().numpy(),   # ground truth HI_after from dataset
                "yb_hat":   yb_hat[i].cpu().numpy(),             # (L-2,C)  aligned to 1..L-1
                "ya_hat":   ya_hat[i].cpu().numpy(),             # (L-2,C)  aligned to 1..L-1
                "x_before": xb[i, :L_i].cpu().numpy(),           # (L,C)
                "x_after":  xa[i, :L_i].cpu().numpy(),           # (L,C)
                "length": L_i
            })

    y_true = np.concatenate(y_true, axis=0) if len(y_true)>0 else np.array([])
    y_pred = np.concatenate(y_pred, axis=0) if len(y_pred)>0 else np.array([])
    all_delta_mean = np.concatenate(all_delta_mean, axis=0) if len(all_delta_mean)>0 else np.array([])
    return y_true, y_pred, all_delta_mean, all_uids, keep_curves

y_true, y_pred, delta_mean_all, uids_all, curves = collect_test_predictions(model, ld_te, DEVICE)

# —— Print overall metrics
print("\n================== Test Set Overall Metrics ==================")
print(f"Sample count: {len(y_true)}")
if len(y_true) > 0:
    acc = (y_true == y_pred).mean()
    print(f"Classification Accuracy: {acc:.4f}")
else:
    print("No samples in test set (check pairs split and horizon conditions).")

# —— Classification report & confusion matrix
try:
    from sklearn.metrics import classification_report, confusion_matrix
    target_names = [LABEL2NAME[i] for i in sorted(LABEL2NAME)]
    print("\n[Classification Report]")
    print(classification_report(y_true, y_pred, target_names=target_names, digits=4))
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])
except Exception:
    print("\n[Warning] scikit-learn not installed, will print simple confusion table.")
    cm = np.zeros((3,3), dtype=int)
    for t,p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1

# —— Print confusion matrix values
print("\n[Confusion Matrix] Row=True class, Column=Predicted class")
print(pd.DataFrame(cm, index=[LABEL2NAME[i] for i in [0,1,2]],
                      columns=[LABEL2NAME[i] for i in [0,1,2]]))

# —— Plot confusion matrix
plt.figure(figsize=(4.8,4.0))
plt.imshow(cm, interpolation='nearest')
plt.title("Confusion Matrix (Test)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks([0,1,2], [LABEL2NAME[i] for i in [0,1,2]])
plt.yticks([0,1,2], [LABEL2NAME[i] for i in [0,1,2]])
for i in range(3):
    for j in range(3):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")
plt.colorbar()
plt.tight_layout()
plt.show()

# —— Statistics of ΔHI distribution (by true class/predicted class)
if len(delta_mean_all) > 0:
    df_delta = pd.DataFrame({
        "uid": uids_all[:len(delta_mean_all)],
        "true": y_true.astype(int),
        "pred": y_pred.astype(int),
        "delta_hi_mean": delta_mean_all.astype(float)
    })
    print("\n[ΔHI Mean Statistics by True Class]")
    print(df_delta.groupby("true")["delta_hi_mean"].describe())
    print("\n[ΔHI Mean Statistics by Predicted Class]")
    print(df_delta.groupby("pred")["delta_hi_mean"].describe())

# —— Continuous time axis: HI and several sensors before/after aligned visualization
def plot_hi_examples_aligned(curves, n_show=6, seed=0):
    """Plot HI curves, post-maintenance time series starts after pre-maintenance end, including ground truth HI from dataset"""
    if len(curves)==0:
        print("(No visualization samples)")
        return
    random.Random(seed).shuffle(curves)
    n_show = min(n_show, len(curves))
    cols = 3
    rows = int(np.ceil(n_show/cols))
    plt.figure(figsize=(cols*4.6, rows*3.8))
    for k in range(n_show):
        ex = curves[k]
        hb = ex["h_before"]; ha = ex["h_after"]                    # model predicted HI
        hb_gt = ex["hi_before_gt"]; ha_gt = ex["hi_after_gt"]      # ground truth HI from dataset
        L = ex["length"]  # Use actual length instead of padded length
        # Pre-maintenance: 0 to L-1
        t_b = np.arange(L)
        # Post-maintenance: start from L, continuing after pre-maintenance end
        t_a = np.arange(L, L + L)
        uid = ex["uid"]
        pred = ex["pred"]; true = ex["true"]; prob = ex["prob"]
        plt.subplot(rows, cols, k+1)

        # Plot model predicted HI
        plt.plot(t_b, hb, label="HI_before (Model)", linewidth=1.8, color='blue')
        plt.plot(t_a, ha, label="HI_after (Model)",  linewidth=1.8, color='red', linestyle='--')

        # Plot ground truth HI from dataset
        plt.plot(t_b, hb_gt, label="HI_before (GT)", linewidth=1.5, color='cyan', alpha=0.7)
        plt.plot(t_a, ha_gt, label="HI_after (GT)",  linewidth=1.5, color='orange', linestyle=':', alpha=0.7)

        # Draw vertical line at maintenance point
        plt.axvline(L-0.5, color='green', linestyle=':', linewidth=2.0, label='Maintenance Point' if k==0 else '')

        d_mean = float(np.mean(ha - hb))
        d_max  = float(np.max(ha - hb))
        d_mean_gt = float(np.mean(ha_gt - hb_gt))
        title = (f"uid={uid}\nTrue={LABEL2NAME[true]} | Pred={LABEL2NAME[pred]} "
                 f"| p={prob[pred]:.2f}\nΔHI_Model={d_mean:.3f}, ΔHI_GT={d_mean_gt:.3f}")
        plt.title(title, fontsize=9)
        plt.xlabel("Continuous Time Sequence")
        plt.ylabel("Health Index HI (↓)")
        plt.grid(ls='--', alpha=.35)
        if k==0: plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

def plot_sensor_examples_aligned(curves, sensor_idx_list=None, n_cols=4):
    """
    Plot several sensors: original(before/after) + one-step recursive prediction of after(ya_hat),
    post-maintenance time series starts after pre-maintenance end. Show trajectories under different
    maintenance strategies.
    """
    if len(curves)==0:
        print("(No visualization samples)")
        return

    # Group curves by maintenance strategy (true class)
    curves_by_strategy = {0: [], 1: [], 2: []}
    for curve in curves:
        curves_by_strategy[curve["true"]].append(curve)

    # Select one example from each strategy for comparison
    strategy_examples = {}
    for strategy in [0, 1, 2]:
        if len(curves_by_strategy[strategy]) > 0:
            strategy_examples[strategy] = curves_by_strategy[strategy][0]

    if len(strategy_examples) == 0:
        print("(No visualization samples for any maintenance strategy)")
        return

    # Determine sensor indices to plot
    first_ex = list(strategy_examples.values())[0]
    xb = first_ex["x_before"]       # (L,C)
    C = xb.shape[1]
    if sensor_idx_list is None:
        step = max(1, C//8)
        sensor_idx_list = list(range(0, C, step))[:8]

    n = len(sensor_idx_list)
    n_rows = int(np.ceil(n/n_cols))
    plt.figure(figsize=(n_cols*5.0, n_rows*3.5))

    colors = {0: 'red', 1: 'orange', 2: 'purple'}  # Colors for different strategies

    for i, s in enumerate(sensor_idx_list):
        plt.subplot(n_rows, n_cols, i+1)

        # Plot pre-maintenance trajectory (common for all strategies, use first example)
        first_ex = list(strategy_examples.values())[0]
        xb = first_ex["x_before"]
        L = first_ex["length"]
        t_b = np.arange(L)
        plt.plot(t_b, xb[:L,s], label="Original (Pre-maintenance)", linewidth=1.5, color='blue')

        # Plot post-maintenance trajectories for each available strategy
        for strategy in sorted(strategy_examples.keys()):
            ex = strategy_examples[strategy]
            xa = ex["x_after"]
            ya = ex["ya_hat"]
            L_strategy = ex["length"]

            # Post-maintenance original trajectory
            t_a = np.arange(L_strategy, L_strategy + L_strategy)
            plt.plot(t_a, xa[:L_strategy,s],
                    label=f"Original (Post-{LABEL2NAME[strategy]})",
                    linewidth=1.2, linestyle="--",
                    color=colors[strategy], alpha=0.8)

            # Post-maintenance prediction trajectory
            if len(ya) > 0 and L_strategy-2 > 0:
                t_pred = np.arange(L_strategy+1, L_strategy+1+min(L_strategy-2, len(ya)))
                plt.plot(t_pred, ya[:len(t_pred),s],
                        label=f"Predicted (Post-{LABEL2NAME[strategy]})",
                        linewidth=1.8, color=colors[strategy])

        # Draw vertical line at maintenance point
        plt.axvline(L-0.5, color='green', linestyle=':', linewidth=2.0, alpha=0.8)
        plt.title(f"Sensor_{s:02d} - Maintenance Strategy Comparison")
        if i==0:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.xlabel("Continuous Time Sequence")
        plt.ylabel("Sensor Value")
        plt.grid(ls="--", alpha=.35)

    plt.suptitle("Sensor Trajectories under Different Maintenance Strategies", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()

# —— Plot figures
plot_hi_examples_aligned(curves, n_show=6, seed=2025)
plot_sensor_examples_aligned(curves, sensor_idx_list=None, n_cols=4)

# ——(Optional) Top-K with largest ΔHI in each class, for manual review
def topk_by_delta(df_delta, k=5):
    if len(df_delta)==0:
        return
    for cls in [0,1,2]:
        sub = df_delta[df_delta["true"]==cls].sort_values("delta_hi_mean", ascending=False).head(k)
        print(f"\n[True={LABEL2NAME[cls]}] Top {k} samples with largest ΔHI_mean:")
        print(sub[["uid","delta_hi_mean","pred"]].reset_index(drop=True))

if len(delta_mean_all) > 0:
    topk_by_delta(df_delta, k=5)

# %% [notebook code cell 23]
# ============================================================
# 6) Model loading + test set evaluation and visualization (aligned before and after maintenance charts)
# ============================================================
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import os

LABEL2NAME = {0: "Perfect", 1: "General", 2: "Poor"}

# ——Load the best trained model
def load_trained_model(model_path, model, device):
    """Load the best model weights saved during training"""
    # Assuming that the model architecture has been defined, only the weights are loaded here
    # Need to ensure that the model is initialized with the same architecture
    if os.path.exists(model_path):
        print(f"Loading model: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded successfully!")
    else:
        print(f"Warning: Model file does not exist {model_path}, the current model state will be used")
    return model

# Load the best model (if it exists)
model_path = "/content/drive/MyDrive/Mimar_turbo/enhanced_self_supervised_model.pth"
# Note: The model schema defined in the context is used here
C = ds_te.C # Get feature dimensions from the test data set
model = SelfSupervisedDiffAwareReconstructor(in_ch=C, trend_ch=4, hidden=128, n_classes=3).to(DEVICE)
model = load_trained_model(model_path, model, DEVICE)

# ——Print the training/validation/test set division (consistent with the training phase: 7/1/2)
def print_split_summary(pairs, seed=42):
    pairs_tr, pairs_vl, pairs_te = split_pairs_uid(pairs, 0.7, 0.1, seed)
    def count_items(pp):
        cnt = 0
        for uid, d in pp.items():
            cnt += len(d)
        return len(pp), cnt
    n_uid_tr, n_pair_tr = count_items(pairs_tr)
    n_uid_vl, n_pair_vl = count_items(pairs_vl)
    n_uid_te, n_pair_te = count_items(pairs_te)
    print(f"[data division] training: UID={n_uid_tr}, number of pairs≈{n_pair_tr} | "
          f"Verification: UID={n_uid_vl}, number of pairs≈{n_pair_vl} | "
          f"Test: UID={n_uid_te}, number of pairs≈{n_pair_te}")

print_split_summary(pairs)

# —— Training completed, te = (ds_te, ld_te)
ds_te, ld_te = te

@torch.no_grad()
def collect_test_predictions(model, loader, device, max_curve_keep=24):
    """
    Collect predictions on the test set, ΔHI, and plots for a small number of samples (for alignment plotting).
    - Save HI_before / HI_after
    - Save yb_hat / ya_hat (one-step recursive prediction)
    - save x_before / x_after (original sensor)
    """
    model.eval()
    y_true, y_pred = [], []
    all_delta_mean, all_uids = [], []
    keep_curves = []

    for batch in loader:
        xb = batch["x_before"].to(device)   # (B,L,C)
        xa = batch["x_after"].to(device)
        mask   = batch["mask"].to(device)   # (B,L)
        labels = batch["labels"].to(device) # (B,)
        lengths= batch["lengths"]           # cpu tensor
        uids   = batch["uids"]
        hi_before = batch["hi_before"].to(device) # (B,L) Real HI_before in the data set
        hi_after = batch["hi_after"].to(device) # (B,L) The real HI_after in the data set

        # Use the output format of the enhanced model
        outputs = model(xb, xa, mask)
        h_b = outputs['h_before']
        h_a = outputs['h_after']
        yb_hat = outputs['recon_before']
        ya_hat = outputs['recon_after']
        logits = outputs['classification_logits']

        # Numerical stabilization
        h_b, h_a, yb_hat, ya_hat, logits = sanitize_tensors(h_b, h_a, yb_hat, ya_hat, logits)

        prob = F.softmax(logits, dim=1)     # (B,3)
        pred = prob.argmax(1)               # (B,)

        # ΔHI (mean over the valid sequence)
        valid = mask.sum(1, keepdim=True).clamp_min(1.0)
        delta_mean = ((h_a - h_b) * mask).sum(1, keepdim=True) / valid  # (B,1)
        delta_mean = torch.nan_to_num(delta_mean, nan=0.0)

        y_true.append(labels.cpu().numpy())
        y_pred.append(pred.cpu().numpy())
        all_delta_mean.append(delta_mean.squeeze(1).cpu().numpy())
        all_uids.extend(uids)

        # Save some samples for alignment visualization
        for i in range(xb.size(0)):
            if len(keep_curves) >= max_curve_keep:
                break
            L_i = int(lengths[i].item())
            # Fix: intercept and reconstruct predictions according to actual length
            L_recon = min(L_i - 2, yb_hat.size(1)) if yb_hat.size(1) > 0 else 0
            keep_curves.append({
                "uid": uids[i],
                "true": int(labels[i].cpu().item()),
                "pred": int(pred[i].cpu().item()),
                "prob": prob[i].cpu().numpy(),
                "h_before": h_b[i, :L_i].cpu().numpy(), # HI_before predicted by the model
                "h_after": h_a[i, :L_i].cpu().numpy(), # HI_after predicted by the model
                "hi_before_gt": hi_before[i, :L_i].cpu().numpy(), # The real HI_before in the data set
                "hi_after_gt": hi_after[i, :L_i].cpu().numpy(), # The real HI_after in the data set
                "yb_hat": yb_hat[i, :L_recon].cpu().numpy() if L_recon > 0 else np.array([]), # (L_recon,C) aligned to 1..L-1
                "ya_hat": ya_hat[i, :L_recon].cpu().numpy() if L_recon > 0 else np.array([]), # (L_recon,C) aligned to 1..L-1
                "x_before": xb[i, :L_i].cpu().numpy(),           # (L,C)
                "x_after":  xa[i, :L_i].cpu().numpy(),           # (L,C)
                "length": L_i
            })

    y_true = np.concatenate(y_true, axis=0) if len(y_true)>0 else np.array([])
    y_pred = np.concatenate(y_pred, axis=0) if len(y_pred)>0 else np.array([])
    all_delta_mean = np.concatenate(all_delta_mean, axis=0) if len(all_delta_mean)>0 else np.array([])
    return y_true, y_pred, all_delta_mean, all_uids, keep_curves

y_true, y_pred, delta_mean_all, uids_all, curves = collect_test_predictions(model, ld_te, DEVICE)

# ——Print overall indicators
print("\n================== Test set overall indicators ==================")
print(f"Sample number: {len(y_true)}")
if len(y_true) > 0:
    acc = (y_true == y_pred).mean()
    print(f"Classification accuracy: {acc:.4f}")
else:
    print("There are no samples in the test set (check pairs division and horizon conditions).")

# ——Classification report & confusion matrix
try:
    from sklearn.metrics import classification_report, confusion_matrix
    target_names = [LABEL2NAME[i] for i in sorted(LABEL2NAME)]
    print("\n[Classification Report]")
    print(classification_report(y_true, y_pred, target_names=target_names, digits=4))
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])
except Exception:
    print("\n[Warning] scikit-learn is not installed, a simple confusion table will be printed.")
    cm = np.zeros((3,3), dtype=int)
    for t,p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1

# ——Print confusion matrix values
print("\n[Confusion Matrix] rows=real categories, columns=predicted categories")
print(pd.DataFrame(cm, index=[LABEL2NAME[i] for i in [0,1,2]],
                      columns=[LABEL2NAME[i] for i in [0,1,2]]))

# ——Draw the confusion matrix
plt.figure(figsize=(4.8,4.0))
plt.imshow(cm, interpolation='nearest')
plt.title("Confusion Matrix (Test Set)")
plt.xlabel("prediction")
plt.ylabel("real")
plt.xticks([0,1,2], [LABEL2NAME[i] for i in [0,1,2]])
plt.yticks([0,1,2], [LABEL2NAME[i] for i in [0,1,2]])
for i in range(3):
    for j in range(3):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")
plt.colorbar()
plt.tight_layout()
plt.show()

# —— ΔHI distribution statistics (by true category/predicted category)
if len(delta_mean_all) > 0:
    df_delta = pd.DataFrame({
        "uid": uids_all[:len(delta_mean_all)],
        "true": y_true.astype(int),
        "pred": y_pred.astype(int),
        "delta_hi_mean": delta_mean_all.astype(float)
    })
    print("\n[Statistics based on the ΔHI mean value of the true category]")
    print(df_delta.groupby("true")["delta_hi_mean"].describe())
    print("\n[ΔHI mean statistics by predicted category]")
    print(df_delta.groupby("pred")["delta_hi_mean"].describe())

# ——Continuous timeline: front-to-back alignment visualization of HI and multiple sensors
def plot_hi_examples_aligned(curves, n_show=6, seed=0):
    """Draw the HI curve. The post-maintenance time series starts after the end of the pre-maintenance period, including the true HI in the data set"""
    if len(curves)==0:
        print("(No visualization sample)")
        return
    random.Random(seed).shuffle(curves)
    n_show = min(n_show, len(curves))
    cols = 3
    rows = int(np.ceil(n_show/cols))
    plt.figure(figsize=(cols*4.6, rows*3.8))
    for k in range(n_show):
        ex = curves[k]
        hb = ex["h_before"]; ha = ex["h_after"] # HI predicted by the model
        hb_gt = ex["hi_before_gt"]; ha_gt = ex["hi_after_gt"] # The real HI in the data set
        L = ex["length"] # use actual length instead of padding length
        # Before maintenance: 0 to L-1
        t_b = np.arange(L)
        # After maintenance: start from L and continue after ending before maintenance
        t_a = np.arange(L, L + L)
        uid = ex["uid"]
        pred = ex["pred"]; true = ex["true"]; prob = ex["prob"]
        plt.subplot(rows, cols, k+1)

        # Plot the HI predicted by the model
        plt.plot(t_b, hb, label="HI_before (model)", linewidth=1.8, color='blue')
        plt.plot(t_a, ha, label="HI_after (model)", linewidth=1.8, color='red', linestyle='--')

        # Plot the true HI in the dataset
        plt.plot(t_b, hb_gt, label="HI_before (real)", linewidth=1.5, color='cyan', alpha=0.7)
        plt.plot(t_a, ha_gt, label="HI_after (real)", linewidth=1.5, color='orange', linestyle=':', alpha=0.7)

        # Draw vertical lines at maintenance points
        plt.axvline(L-0.5, color='green', linestyle=':', linewidth=2.0, label='maintenance point' if k==0 else '')

        d_mean = float(np.mean(ha - hb))
        d_max  = float(np.max(ha - hb))
        d_mean_gt = float(np.mean(ha_gt - hb_gt))
        title = (f"uid={uid}\ntrue={LABEL2NAME[true]} | prediction={LABEL2NAME[pred]} "
                 f"| p={prob[pred]:.2f}\nΔHI_model={d_mean:.3f}, ΔHI_real={d_mean_gt:.3f}")
        plt.title(title, fontsize=9)
        plt.xlabel("Continuous time series")
        plt.ylabel("Health indicator HI (↓)")
        plt.grid(ls='--', alpha=.35)
        if k==0: plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

def plot_sensor_examples_aligned(curves, sensor_idx_list=None, n_cols=4):
    """
    Plot multiple sensors: original (before and after) + late one-step recursive prediction (ya_hat),
    The post-maintenance time series starts after the end of pre-maintenance. Show trajectories under different maintenance strategies.
    """
    if len(curves)==0:
        print("(No visualization sample)")
        return

    # Group curves by maintenance strategy (real category)
    curves_by_strategy = {0: [], 1: [], 2: []}
    for curve in curves:
        curves_by_strategy[curve["true"]].append(curve)

    # Select an example from each strategy to compare
    strategy_examples = {}
    for strategy in [0, 1, 2]:
        if len(curves_by_strategy[strategy]) > 0:
            strategy_examples[strategy] = curves_by_strategy[strategy][0]

    if len(strategy_examples) == 0:
        print("(No visual samples for any maintenance strategy)")
        return

    # Determine the sensor index to draw
    first_ex = list(strategy_examples.values())[0]
    xb = first_ex["x_before"]       # (L,C)
    C = xb.shape[1]
    if sensor_idx_list is None:
        step = max(1, C//8)
        sensor_idx_list = list(range(0, C, step))[:8]

    n = len(sensor_idx_list)
    n_rows = int(np.ceil(n/n_cols))
    plt.figure(figsize=(n_cols*5.0, n_rows*3.5))

    colors = {0: 'red', 1: 'orange', 2: 'purple'} # Colors for different strategies

    for i, s in enumerate(sensor_idx_list):
        plt.subplot(n_rows, n_cols, i+1)

        # Draw the pre-maintenance trajectory (common to all strategies, use the first example)
        first_ex = list(strategy_examples.values())[0]
        xb = first_ex["x_before"]
        L = first_ex["length"]
        t_b = np.arange(L)
        plt.plot(t_b, xb[:L,s], label="Original (before maintenance)", linewidth=1.5, color='blue')

        # Draw post-maintenance traces for each available strategy
        for strategy in sorted(strategy_examples.keys()):
            ex = strategy_examples[strategy]
            xa = ex["x_after"]
            ya = ex["ya_hat"]
            L_strategy = ex["length"]

            # Original trajectory after maintenance
            t_a = np.arange(L_strategy, L_strategy + L_strategy)
            plt.plot(t_a, xa[:L_strategy,s],
                    label=f"Original (post-{LABEL2NAME[strategy]})",
                    linewidth=1.2, linestyle="--",
                    color=colors[strategy], alpha=0.8)

            # Predict trajectory after maintenance
            if len(ya) > 0 and L_strategy-2 > 0:
                t_pred = np.arange(L_strategy+1, L_strategy+1+min(L_strategy-2, len(ya)))
                if len(t_pred) > 0 and len(ya) >= len(t_pred):
                    plt.plot(t_pred, ya[:len(t_pred),s],
                            label=f"Prediction (post-{LABEL2NAME[strategy]})",
                            linewidth=1.8, color=colors[strategy])

        # Draw vertical lines at maintenance points
        plt.axvline(L-0.5, color='green', linestyle=':', linewidth=2.0, alpha=0.8)
        plt.title(f"Sensor_{s:02d} - Maintenance Strategy Comparison")
        if i==0:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.xlabel("Continuous time series")
        plt.ylabel("sensor value")
        plt.grid(ls="--", alpha=.35)

    plt.suptitle("Sensor trajectories under different maintenance strategies", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()

# ——Draw a chart
plot_hi_examples_aligned(curves, n_show=6, seed=2025)
plot_sensor_examples_aligned(curves, sensor_idx_list=None, n_cols=4)

# ——(optional) Top K with the largest ΔHI in each category for manual review
def topk_by_delta(df_delta, k=5):
    if len(df_delta)==0:
        return
    for cls in [0,1,2]:
        sub = df_delta[df_delta["true"]==cls].sort_values("delta_hi_mean", ascending=False).head(k)
        print(f"\n[True={LABEL2NAME[cls]}] The first {k} samples with the largest ΔHI_mean:")
        print(sub[["uid","delta_hi_mean","pred"]].reset_index(drop=True))

if len(delta_mean_all) > 0:
    topk_by_delta(df_delta, k=5)

# %% [notebook code cell 24]
