# ============================================================
# COMPARISON WITH BASELINES (All Models) — JIAXING + PALO ALTO
# Hybrid/Probabilistic Multi-horizon Quantile Forecasting
# ============================================================

# ============================================================
# GLOBAL IMPORTS & CONFIG
# ============================================================

import os, random, numpy as np, pandas as pd, torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE =", DEVICE)

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(SEED)

Cu, Co = 2.0, 1.0
QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
N_QUANTILES = len(QUANTILES)
quant_t = torch.tensor(QUANTILES, device=DEVICE).view(1, 1, -1)

TARGET_COL = "Energy"

EVAL_BATCH_SIZE = 32
N_EPOCHS = 50
PATIENCE = 5
BATCH_SIZE = 32

# ============================================================
# UTILITIES
# ============================================================

def build_multihorizon_sequences(X, y, lookback, horizons):
    Xs, Ys = [], []
    T = len(X)
    max_h = max(horizons)
    for i in range(T - lookback - max_h + 1):
        Xs.append(X[i:i+lookback])
        Ys.append([y[i+lookback+h-1] for h in horizons])
    return np.array(Xs, np.float32), np.array(Ys, np.float32)

def quantile_loss(pred, target):
    target = target.unsqueeze(-1)
    err = target - pred
    return torch.max(quant_t * err, (quant_t - 1) * err).mean()

def compute_cost_metrics(y_true, schedule):
    y_true = np.asarray(y_true); schedule = np.asarray(schedule)
    under = np.maximum(0, y_true - schedule)
    over  = np.maximum(0, schedule - y_true)
    cost = Cu * under + Co * over
    return {
        "avg_cost": float(cost.mean()),
        "avg_under": float(under.mean()),
        "avg_over": float(over.mean())
    }

def compute_crps(y, q10, q50, q90):
    return float((q90 - q10) + 2 * np.abs(y - q50))

def compute_wis(y, q10, q50, q90):
    return float(
        0.5 * (q90 - q10) +
        np.maximum(q10 - y, 0) +
        np.maximum(y - q90, 0)
    )

# ============================================================
# TRAINING + EVALUATION PIPELINE
# ============================================================

def train_one_model(model_name, model, X_train, y_train, X_val, y_val,
                    LOOKBACK, HORIZONS):

    print(f"\n================ TRAINING: {model_name} ================")

    Xtr, Ytr = build_multihorizon_sequences(X_train, y_train, LOOKBACK, HORIZONS)
    Xv,  Yv  = build_multihorizon_sequences(X_val,   y_val,   LOOKBACK, HORIZONS)

    Xtr = torch.tensor(Xtr).float().to(DEVICE)
    Ytr = torch.tensor(Ytr).float().to(DEVICE)
    Xv  = torch.tensor(Xv ).float().to(DEVICE)
    Yv  = torch.tensor(Yv ).float().to(DEVICE)

    tr_loader = DataLoader(TensorDataset(Xtr, Ytr),
                           batch_size=BATCH_SIZE, shuffle=True)
    v_loader  = DataLoader(TensorDataset(Xv, Yv),
                           batch_size=BATCH_SIZE)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    best = float("inf"); patience = 0
    trL = []; valL = []
    best_state = None

    for ep in range(N_EPOCHS):
        # TRAIN
        model.train(); run = 0.0
        for xb, yb in tr_loader:
            opt.zero_grad()
            pred = model(xb)
            loss = quantile_loss(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run += loss.item()
        trL.append(run / len(tr_loader))

        # VALIDATION
        model.eval(); run = 0.0
        with torch.no_grad():
            for xb, yb in v_loader:
                run += quantile_loss(model(xb), yb).item()
        val_loss = run / len(v_loader)
        valL.append(val_loss)

        print(f"Epoch {ep+1}/{N_EPOCHS} Train={trL[-1]:.4f} Val={val_loss:.4f}")

        # EARLY STOPPING
        if val_loss < best - 1e-4:
            best = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, Xv, Yv, trL, valL


def evaluate_model(model_name, model, Xv, Yv, trL, valL, y_scaler, HORIZONS):

    print(f"\n================ EVALUATING: {model_name} ================")

    # ---------------- Loss curves ----------------
    plt.figure(figsize=(7, 3))
    plt.plot(trL, label="Train")
    plt.plot(valL, label="Val")
    plt.title(f"{model_name} Loss")
    plt.grid()
    plt.legend()
    plt.show()

    # ---------------- Predictions ----------------
    preds = []
    loader = DataLoader(TensorDataset(Xv), batch_size=EVAL_BATCH_SIZE)

    with torch.no_grad():
        for (xb,) in loader:
            preds.append(model(xb).detach().cpu().numpy())

    preds = np.concatenate(preds, axis=0)  # [N,H,Q]
    Ynp = Yv.detach().cpu().numpy()        # [N,H]

    # ---------------------------------------------------------
    # Quantile preparation (sorted for integration consistency)
    # ---------------------------------------------------------
    qs = np.asarray(QUANTILES, dtype=np.float32)
    qsort = np.argsort(qs)
    qs_sorted = qs[qsort]

    # ---------------- Pinball ----------------
    def pinball_avg(y, qhat_sorted):
        y2 = y[:, None]
        err = y2 - qhat_sorted
        losses = np.maximum(qs_sorted[None, :] * err,
                            (qs_sorted[None, :] - 1.0) * err)
        return float(np.mean(losses))

    # ---------------- CRPS_Q ----------------
    def crps_q(y, qhat_sorted):
        y2 = y[:, None]
        err = y2 - qhat_sorted
        rho = np.maximum(qs_sorted[None, :] * err,
                         (qs_sorted[None, :] - 1.0) * err)
        integral = np.trapz(rho, qs_sorted, axis=1)
        return float(2.0 * np.mean(integral))

    # ---------------- Interval Score ----------------
    def interval_score(y, lo, hi, alpha):
        width = hi - lo
        below = np.maximum(lo - y, 0.0)
        above = np.maximum(y - hi, 0.0)
        return width + (2.0 / alpha) * (below + above)

    # ---------------- WIS_multi ----------------
    def wis_multi(y, qhat):

        idx = {t: QUANTILES.index(t) for t in QUANTILES}

        q05 = qhat[:, idx[0.05]]
        q10 = qhat[:, idx[0.10]]
        q25 = qhat[:, idx[0.25]]
        q50 = qhat[:, idx[0.50]]
        q75 = qhat[:, idx[0.75]]
        q90 = qhat[:, idx[0.90]]
        q95 = qhat[:, idx[0.95]]

        intervals = [
            (q25, q75, 0.50),  # 50%
            (q10, q90, 0.20),  # 80%
            (q05, q95, 0.10)   # 90%
        ]

        K = len(intervals)

        total = 0.5 * np.abs(y - q50)

        for lo, hi, alpha in intervals:
            total += (alpha / 2.0) * interval_score(y, lo, hi, alpha)

        return float(np.mean(total / (K + 0.5)))

    # ---------------- Indices for 10–90 interval ----------------
    idx_q10 = QUANTILES.index(0.10)
    idx_q50 = QUANTILES.index(0.50)
    idx_q90 = QUANTILES.index(0.90)

    results_orig = []
    results_norm = []

    for i, h in enumerate(HORIZONS):

        y = Ynp[:, i]
        pred_q = preds[:, i, :]
        pred_q_sorted = pred_q[:, qsort]

        # Extract required quantiles
        q10 = pred_q[:, idx_q10]
        q50 = pred_q[:, idx_q50]
        q90 = pred_q[:, idx_q90]

        # ---------------- Point metrics (median) ----------------
        mse  = mean_squared_error(y, q50)
        rmse = np.sqrt(mse)
        mae  = mean_absolute_error(y, q50)
        r2   = r2_score(y, q50)
        mape = np.mean(np.abs((y - q50) / np.where(y == 0, 1, y))) * 100

        # Normalized
        y_norm = y_scaler.transform(y.reshape(-1, 1)).flatten()
        q50_norm = y_scaler.transform(q50.reshape(-1, 1)).flatten()

        mse_norm  = mean_squared_error(y_norm, q50_norm)
        rmse_norm = np.sqrt(mse_norm)
        mae_norm  = mean_absolute_error(y_norm, q50_norm)

        # ---------------- Probabilistic metrics ----------------
        PICP = float(((y >= q10) & (y <= q90)).mean())
        MIW  = float((q90 - q10).mean())

        PINBALL = pinball_avg(y, pred_q_sorted)
        CRPS_Q  = crps_q(y, pred_q_sorted)
        WIS_M   = wis_multi(y, pred_q)

        # ---------------- Cost ----------------
        cost = compute_cost_metrics(y, q50)

        print(f"\n---- {model_name} | {h}h ----")
        print(f"MSE={mse:.3f} RMSE={rmse:.3f} MAE={mae:.3f} R2={r2:.3f}")
        print(f"PICP(10-90)={PICP:.3f} Pinball={PINBALL:.3f} CRPS_Q={CRPS_Q:.3f} WIS_multi={WIS_M:.3f}")

        results_orig.append([
            model_name, h, mse, rmse, mae, r2, mape,
            PICP, MIW, PINBALL, CRPS_Q, WIS_M,
            cost["avg_cost"], cost["avg_under"], cost["avg_over"]
        ])

        results_norm.append([
            model_name, h, mse_norm, rmse_norm, mae_norm
        ])

    return results_orig, results_norm

# ============================================================
# LOADERS
# ============================================================

def setup_jiaxing():
    df = pd.read_csv("Jiaxing_hourly.csv")
    df["time_hour"] = pd.to_datetime(df["time_hour"])
    df = df.sort_values("time_hour").reset_index(drop=True)

    num_cols = [
        "ChargingTime_hours","Fee","Hour",
        "Temp","Humidity","Precip","WindSpeed","CondCode"
    ]
    cat_cols = [
        "Location Information","District Name","end_cause",
        "DayOfWeek","DayType","Season","Holiday"
    ]
    for c in cat_cols:
        df[c] = LabelEncoder().fit_transform(df[c].astype(str))

    split = int(0.80 * len(df))
    train_df = df.iloc[:split].copy()
    val_df   = df.iloc[split:].copy()

    scaler = StandardScaler()
    train_scaled = train_df.copy()
    val_scaled   = val_df.copy()
    train_scaled[num_cols] = scaler.fit_transform(train_df[num_cols])
    val_scaled[num_cols]   = scaler.transform(val_df[num_cols])

    all_features = num_cols + cat_cols

    X_train = train_scaled[all_features].values.astype(np.float32)
    y_train = train_df[TARGET_COL].values.astype(np.float32)
    X_val   = val_scaled[all_features].values.astype(np.float32)
    y_val   = val_df[TARGET_COL].values.astype(np.float32)

    y_scaler = StandardScaler()
    y_scaler.fit(y_train.reshape(-1, 1))

    return X_train, y_train, X_val, y_val, y_scaler, all_features

def setup_paloalto():
    df = pd.read_csv("processed_palo_alto.csv")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    num_cols = [
        'Charging Time','Total Duration','Fee','Hour',
        'Humidity','Precipitation','Pressure','Area ID'
    ]
    cat_cols = [
        'Station Name','MAC Address','Address','Ended By',
        'Day of Week','Day Type','Season','Holiday'
    ]
    for c in cat_cols:
        df[c] = LabelEncoder().fit_transform(df[c].astype(str))

    split = int(0.80 * len(df))
    train_df = df.iloc[:split].copy()
    val_df   = df.iloc[split:].copy()

    scaler = StandardScaler()
    train_scaled = train_df.copy()
    val_scaled   = val_df.copy()
    train_scaled[num_cols] = scaler.fit_transform(train_df[num_cols])
    val_scaled[num_cols]   = scaler.transform(val_df[num_cols])

    all_features = num_cols + cat_cols

    X_train = train_scaled[all_features].values.astype(np.float32)
    y_train = train_df[TARGET_COL].values.astype(np.float32)
    X_val   = val_scaled[all_features].values.astype(np.float32)
    y_val   = val_df[TARGET_COL].values.astype(np.float32)

    y_scaler = StandardScaler()
    y_scaler.fit(y_train.reshape(-1, 1))

    return X_train, y_train, X_val, y_val, y_scaler, all_features

# ============================================================
# MODELS
# ============================================================

class MLP_Prob(nn.Module):
    def __init__(self, input_dim, lookback, horizons, quantiles, hidden=128):
        super().__init__()
        self.H = len(horizons)
        self.Q = len(quantiles)
        self.fc = nn.Sequential(
            nn.Linear(input_dim * lookback, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.H * self.Q)
        )

    def forward(self, x):
        B = x.size(0)
        x = x.reshape(B, -1)
        out = self.fc(x)
        return out.view(B, self.H, self.Q)

class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size,
                              padding=pad, dilation=dilation)
        self.relu = nn.ReLU()
        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        out = self.conv(x)
        out = out[:, :, :-self.conv.padding[0]]
        res = x if self.down is None else self.down(x)
        return self.relu(out + res)

class TCN_Prob(nn.Module):
    def __init__(self, input_dim, lookback, horizons, quantiles, channels=64, levels=4):
        super().__init__()
        self.H = len(horizons)
        self.Q = len(quantiles)

        layers = []
        in_ch = input_dim
        for i in range(levels):
            layers.append(TCNBlock(in_ch, channels, 3, dilation=2**i))
            in_ch = channels
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(channels, self.H * self.Q)

    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.tcn(x)
        last = out[:, :, -1]
        last = self.fc(last)
        return last.view(-1, self.H, self.Q)

class BiLSTM_Prob(nn.Module):
    def __init__(self, input_dim, horizons, quantiles, hidden=64):
        super().__init__()
        self.H = len(horizons)
        self.Q = len(quantiles)
        self.lstm = nn.LSTM(input_dim, hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, self.H * self.Q)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        last = self.fc(last)
        return last.view(-1, self.H, self.Q)

class BiGRU_Prob(nn.Module):
    def __init__(self, input_dim, horizons, quantiles, hidden=64):
        super().__init__()
        self.H = len(horizons)
        self.Q = len(quantiles)
        self.gru = nn.GRU(input_dim, hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, self.H * self.Q)

    def forward(self, x):
        out, _ = self.gru(x)
        last = out[:, -1, :]
        return self.fc(last).view(-1, self.H, self.Q)

class GatedResidualNetwork(nn.Module):
    def __init__(self, inp, hidden):
        super().__init__()
        self.fc1 = nn.Linear(inp, hidden)
        self.fc2 = nn.Linear(hidden, inp)
        self.elu = nn.ELU()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        r = x
        x = self.elu(self.fc1(x))
        x = self.fc2(x)
        g = self.sig(x)
        return g * x + (1 - g) * r

class InterpretableMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    def forward(self, x):
        att, _ = self.attn(x, x, x)
        return att

class TFT_Prob(nn.Module):
    def __init__(self, input_dim, horizons, quantiles, d_model=64, hidden=128):
        super().__init__()
        self.H = len(horizons)
        self.Q = len(quantiles)

        self.proj = nn.Linear(input_dim, d_model)
        self.vsn = GatedResidualNetwork(d_model, hidden)

        self.enc_lstm = nn.LSTM(d_model, d_model, batch_first=True)
        self.dec_lstm = nn.LSTM(d_model, d_model, batch_first=True)

        self.attn = InterpretableMultiHeadAttention(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(),
            nn.Linear(hidden, d_model)
        )
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, self.H * self.Q)

    def forward(self, x):
        x = self.proj(x)
        x = self.vsn(x)

        enc, _ = self.enc_lstm(x)
        dec, _ = self.dec_lstm(enc)

        att = self.attn(dec)
        x = self.norm(att + dec)
        x = self.norm(self.ff(x) + x)

        last = x[:, -1, :]
        return self.fc(last).view(-1, self.H, self.Q)

class PatchEmbedding(nn.Module):
    def __init__(self, patch_len, input_dim, d_model):
        super().__init__()
        self.patch_len = patch_len
        self.proj = nn.Linear(patch_len * input_dim, d_model)

    def forward(self, x):
        B, L, C = x.shape
        num_patches = L // self.patch_len
        L_eff = num_patches * self.patch_len
        x = x[:, -L_eff:, :]
        x = x.view(B, num_patches, self.patch_len * C)
        return self.proj(x)

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads=4, ff_dim=128):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, d_model)
        )
        self.n1 = nn.LayerNorm(d_model)
        self.n2 = nn.LayerNorm(d_model)

    def forward(self, x):
        a, _ = self.attn(x, x, x)
        x = self.n1(x + a)
        f = self.ff(x)
        return self.n2(x + f)

class PatchTST_Prob(nn.Module):
    def __init__(self, input_dim, lookback, horizons, quantiles,
                 patch_len=16, d_model=64, layers=3):
        super().__init__()
        self.H = len(horizons)
        self.Q = len(quantiles)

        self.embed = PatchEmbedding(patch_len, input_dim, d_model)
        self.encs = nn.ModuleList([TransformerEncoder(d_model) for _ in range(layers)])
        self.fc = nn.Linear(d_model, self.H * self.Q)

    def forward(self, x):
        x = self.embed(x)
        for enc in self.encs:
            x = enc(x)
        last = x[:, -1, :]
        return self.fc(last).view(-1, self.H, self.Q)

class CrossformerBlock(nn.Module):
    def __init__(self, d_model, n_heads=4, ff_dim=128):
        super().__init__()
        self.attn_local = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.attn_global = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, d_model)
        )

    def forward(self, x):
        la, _ = self.attn_local(x, x, x)
        ga, _ = self.attn_global(x, x, x)
        x = self.norm(x + la + ga)
        f = self.ff(x)
        return self.norm(x + f)

class Crossformer_Prob(nn.Module):
    def __init__(self, input_dim, horizons, quantiles, d_model=64, layers=3):
        super().__init__()
        self.H = len(horizons)
        self.Q = len(quantiles)
        self.proj = nn.Linear(input_dim, d_model)
        self.blocks = nn.ModuleList([CrossformerBlock(d_model) for _ in range(layers)])
        self.fc = nn.Linear(d_model, self.H * self.Q)

    def forward(self, x):
        x = self.proj(x)
        for b in self.blocks:
            x = b(x)
        last = x[:, -1, :]
        return self.fc(last).view(-1, self.H, self.Q)

class InceptionBlock1D(nn.Module):
    """
    Multi-kernel 1D conv block used by TimesNet-style temporal modeling.
    """
    def __init__(self, in_ch, out_ch, kernels=(1, 3, 5, 7), dropout=0.1):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k//2),
                nn.GELU()
            )
            for k in kernels
        ])
        self.proj = nn.Conv1d(out_ch * len(kernels), out_ch, kernel_size=1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        outs = [b(x) for b in self.branches]     # each: (B, out_ch, L)
        x = torch.cat(outs, dim=1)               # (B, out_ch*nb, L)
        x = self.proj(x)                         # (B, out_ch, L)
        return self.drop(x)

class TimesBlock(nn.Module):
    """
    Lightweight TimesNet-inspired block:
    - Multi-kernel conv (Inception-style)
    - Residual + LayerNorm
    """
    def __init__(self, d_model=64, ff_dim=128, dropout=0.1, kernels=(1, 3, 5, 7)):
        super().__init__()
        self.incep = InceptionBlock1D(d_model, d_model, kernels=kernels, dropout=dropout)
        self.ff = nn.Sequential(
            nn.Conv1d(d_model, ff_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(ff_dim, d_model, kernel_size=1),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, L, D)
        r = x
        y = x.transpose(1, 2)        # (B, D, L)
        y = self.incep(y)            # (B, D, L)
        y = y.transpose(1, 2)        # (B, L, D)
        x = self.norm1(r + y)

        r = x
        y = x.transpose(1, 2)        # (B, D, L)
        y = self.ff(y)               # (B, D, L)
        y = y.transpose(1, 2)        # (B, L, D)
        x = self.norm2(r + y)
        return x

class TimesNet_Prob(nn.Module):
    """
    TimesNet-style baseline for multi-horizon quantile forecasts.
    Output shape: (B, H, Q)
    """
    def __init__(self, input_dim, horizons, quantiles,
                 d_model=64, layers=3, ff_dim=128, dropout=0.1, kernels=(1, 3, 5, 7)):
        super().__init__()
        self.H = len(horizons)
        self.Q = len(quantiles)

        self.proj_in = nn.Linear(input_dim, d_model)
        self.blocks = nn.ModuleList([
            TimesBlock(d_model=d_model, ff_dim=ff_dim, dropout=dropout, kernels=kernels)
            for _ in range(layers)
        ])
        self.proj_out = nn.Linear(d_model, self.H * self.Q)

    def forward(self, x):
        x = self.proj_in(x)          # (B, L, D)
        for b in self.blocks:
            x = b(x)                 # (B, L, D)
        last = x[:, -1, :]           # (B, D)
        out = self.proj_out(last)    # (B, H*Q)
        return out.view(-1, self.H, self.Q)

# ============================================================
# RUN ALL MODELS FOR ONE DATASET
# ============================================================

def run_all_models(dataset_name, X_train, y_train, X_val, y_val,
                   y_scaler, all_features, LOOKBACK, HORIZONS):

    MODELS = [
        ("MLP",         lambda: MLP_Prob(len(all_features), LOOKBACK, HORIZONS, QUANTILES)),
        ("TCN",         lambda: TCN_Prob(len(all_features), LOOKBACK, HORIZONS, QUANTILES)),
        ("BiLSTM",      lambda: BiLSTM_Prob(len(all_features), HORIZONS, QUANTILES)),
        ("BiGRU",       lambda: BiGRU_Prob(len(all_features), HORIZONS, QUANTILES)),
        ("TFT",         lambda: TFT_Prob(len(all_features), HORIZONS, QUANTILES)),
        ("PatchTST",    lambda: PatchTST_Prob(len(all_features), LOOKBACK, HORIZONS, QUANTILES)),
        ("Crossformer", lambda: Crossformer_Prob(len(all_features), HORIZONS, QUANTILES)),
        ("TimesNet",    lambda: TimesNet_Prob(len(all_features), HORIZONS, QUANTILES,
                                              d_model=64, layers=3, ff_dim=128, dropout=0.1))
    ]

    all_results_orig = []
    all_results_norm = []

    for name, fn in MODELS:
        print("\n==============================================")
        print(f"Running {name} on {dataset_name}")
        print("==============================================")

        model = fn().to(DEVICE)

        model, Xv, Yv, trL, valL = train_one_model(
            f"{name}-{dataset_name}",
            model,
            X_train, y_train,
            X_val, y_val,
            LOOKBACK,
            HORIZONS
        )

        results_orig, results_norm = evaluate_model(
            f"{name}-{dataset_name}",
            model,
            Xv, Yv,
            trL, valL,
            y_scaler,
            HORIZONS
        )

        all_results_orig.extend(results_orig)
        all_results_norm.extend(results_norm)

    df_orig = pd.DataFrame(all_results_orig, columns=[
        "Model", "Horizon", "MSE", "RMSE", "MAE", "R2", "MAPE",
        "PICP", "MIW", "Pinball", "CRPS_Q", "WIS_multi",
        "avg_cost", "avg_under", "avg_over"
    ])
    df_norm = pd.DataFrame(all_results_norm, columns=[
        "Model", "Horizon", "MSE_norm", "RMSE_norm", "MAE_norm"
    ])

    df_orig.to_csv(f"{dataset_name}_all_models_orig.csv", index=False)
    df_norm.to_csv(f"{dataset_name}_all_models_norm.csv", index=False)

    print(f"\n✔ Saved combined CSVs for {dataset_name}:")
    print(f"    {dataset_name}_all_models_orig.csv")
    print(f"    {dataset_name}_all_models_norm.csv")

# ============================================================
# MAIN: RUN FOR BOTH DATASETS
# ============================================================

if __name__ == "__main__":

    LOOKBACK = 48
    HORIZONS = [1, 3, 6, 24]

    # ---------------- JIAXING ----------------
    X_train_jx, y_train_jx, X_val_jx, y_val_jx, y_scaler_jx, all_features_jx = setup_jiaxing()

    run_all_models(
        "jiaxing",
        X_train_jx, y_train_jx,
        X_val_jx,   y_val_jx,
        y_scaler_jx,
        all_features_jx,
        LOOKBACK,
        HORIZONS
    )

    # ---------------- PALO ALTO ----------------
    X_train_pa, y_train_pa, X_val_pa, y_val_pa, y_scaler_pa, all_features_pa = setup_paloalto()

    run_all_models(
        "paloalto",
        X_train_pa, y_train_pa,
        X_val_pa,   y_val_pa,
        y_scaler_pa,
        all_features_pa,
        LOOKBACK,
        HORIZONS
    )

    print("\n================ ALL DATASETS & ALL MODELS COMPLETE ================")
