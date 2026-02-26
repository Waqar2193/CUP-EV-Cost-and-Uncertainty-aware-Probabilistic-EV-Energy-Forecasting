# ============================================================
# STATISTICAL ANALYSIS — SEEDS × HORIZONS × MODELS
# (Friedman + Nemenyi + Wilcoxon-Holm) for Jiaxing & Palo Alto
# NOTE: Converted from Jupyter notebook cells (Part 1/3, 2/3, 3/3)
# ============================================================

import os, random, numpy as np, pandas as pd, torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

from scipy.stats import friedmanchisquare, wilcoxon, rankdata

# Optional (for Nemenyi). If not installed, code will still run and will compute CD + plot ranks.
try:
    import scikit_posthocs as sp
    HAS_SPH = True
except Exception:
    HAS_SPH = False

# ============================================================
# GLOBAL CONFIG
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE =", DEVICE)

# ---- seeds requested
SEEDS = [1, 21, 42, 786, 2023]

Cu, Co = 2.0, 1.0

# ---------------- UPDATED QUANTILE GRID (same as your proposed code) ----------------
QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
N_QUANTILES = len(QUANTILES)

TARGET_COL = "Energy"

# defaults (baselines use these)
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 32
N_EPOCHS = 50
PATIENCE = 5
LR_DEFAULT = 1e-3

quant_t = torch.tensor(QUANTILES, device=DEVICE).view(1, 1, -1)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(seed)


# ============================================================
# UTILITIES (EXACT AS YOURS, BUT UPDATED PROB METRICS LIKE PROPOSED CODE)
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
    """
    Pinball loss averaged across all quantiles and horizons (training objective).
    pred: [B,H,Q], target: [B,H]
    """
    target = target.unsqueeze(-1)
    err = target - pred
    return torch.max(quant_t * err, (quant_t - 1) * err).mean()


def compute_cost_metrics(y_true, schedule):
    y_true = np.asarray(y_true); schedule = np.asarray(schedule)
    under = np.maximum(0, y_true - schedule)
    over  = np.maximum(0, schedule - y_true)
    cost = Cu*under + Co*over
    return {
        "avg_cost": float(cost.mean()),
        "avg_under": float(under.mean()),
        "avg_over":  float(over.mean())
    }


# ---------------- UPDATED: vectorized pinball / CRPS / WIS (as in your proposed code) ----------------

def pinball_loss_np(y, q_pred, taus):
    """
    y: [N,]
    q_pred: [N,Q] aligned with taus
    returns: scalar mean pinball loss averaged over N and Q
    """
    y = np.asarray(y).reshape(-1, 1)               # [N,1]
    q_pred = np.asarray(q_pred)                    # [N,Q]
    taus = np.asarray(taus).reshape(1, -1)         # [1,Q]
    err = y - q_pred                               # [N,Q]
    loss = np.maximum(taus*err, (taus-1.0)*err)    # [N,Q]
    return float(loss.mean())


def crps_from_quantiles(y, q_pred, taus):
    """
    Quantile-based CRPS approximation:
    CRPS(y,F) = 2 * ∫_0^1 ρ_tau(y - q_tau) d tau
    Approximate integral over tau-grid using trapezoidal rule.
    """
    y = np.asarray(y).reshape(-1, 1)     # [N,1]
    q_pred = np.asarray(q_pred)          # [N,Q]
    taus = np.asarray(taus)              # [Q,]

    err = y - q_pred
    rho = np.maximum(taus.reshape(1,-1)*err, (taus.reshape(1,-1)-1.0)*err)  # [N,Q]

    dt = np.diff(taus)  # [Q-1]
    integral = (0.5*(rho[:, :-1] + rho[:, 1:]) * dt.reshape(1,-1)).sum(axis=1)  # [N,]
    crps = 2.0 * integral
    return float(crps.mean())


def wis_multi_alpha(y, q_pred, taus, alpha_levels=None):
    """
    Weighted Interval Score using multiple central intervals.
    Standard WIS:
      WIS = (1/(K+0.5)) * ( 0.5*|y-m| + Σ_{k=1..K} (α_k/2)*IS_{α_k} )
    """
    y = np.asarray(y).reshape(-1)      # [N]
    q_pred = np.asarray(q_pred)        # [N,Q]
    taus = np.asarray(taus)            # [Q]

    if alpha_levels is None:
        alpha_levels = [0.10, 0.20, 0.50]

    if 0.50 not in taus:
        raise ValueError("0.50 must be in QUANTILES for WIS median term.")
    tau_list = list(taus)
    m = q_pred[:, tau_list.index(0.50)]  # [N]

    K = len(alpha_levels)
    wis = 0.5*np.abs(y - m)

    for a in alpha_levels:
        lo_tau = a/2.0
        hi_tau = 1.0 - a/2.0
        if (lo_tau not in taus) or (hi_tau not in taus):
            continue

        lo = q_pred[:, tau_list.index(lo_tau)]
        hi = q_pred[:, tau_list.index(hi_tau)]
        width = (hi - lo)
        below = np.maximum(0.0, lo - y)
        above = np.maximum(0.0, y - hi)
        interval_score = width + (2.0/a)*below + (2.0/a)*above
        wis += (a/2.0)*interval_score

    denom = (K + 0.5)
    return float(np.mean(wis / denom))


# ============================================================
# DATA LOADERS (EXACT AS YOURS)
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
    y_scaler.fit(y_train.reshape(-1,1))
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
    y_scaler.fit(y_train.reshape(-1,1))
    return X_train, y_train, X_val, y_val, y_scaler, all_features


# ============================================================
# TRAIN + EVAL (compatible with both your scripts)
# ============================================================

def train_one_model(
    model_name, model, X_train, y_train, X_val, y_val,
    LOOKBACK, HORIZONS,
    batch_size=None, eval_batch_size=None, lr=None
):
    batch_size = BATCH_SIZE if batch_size is None else batch_size
    eval_batch_size = EVAL_BATCH_SIZE if eval_batch_size is None else eval_batch_size
    lr = LR_DEFAULT if lr is None else lr

    print(f"\n================ TRAINING: {model_name} ================")

    Xtr, Ytr = build_multihorizon_sequences(X_train, y_train, LOOKBACK, HORIZONS)
    Xv,  Yv  = build_multihorizon_sequences(X_val,   y_val,   LOOKBACK, HORIZONS)

    Xtr = torch.tensor(Xtr).float().to(DEVICE)
    Ytr = torch.tensor(Ytr).float().to(DEVICE)
    Xv  = torch.tensor(Xv ).float().to(DEVICE)
    Yv  = torch.tensor(Yv ).float().to(DEVICE)

    tr_loader = DataLoader(TensorDataset(Xtr, Ytr), batch_size=batch_size, shuffle=True)
    v_loader  = DataLoader(TensorDataset(Xv,  Yv ), batch_size=eval_batch_size, shuffle=False)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best = float("inf"); patience = 0
    trL, valL = [], []
    best_state = None

    for ep in range(N_EPOCHS):
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

        model.eval(); run = 0.0
        with torch.no_grad():
            for xb, yb in v_loader:
                run += quantile_loss(model(xb), yb).item()
        val_loss = run / len(v_loader)
        valL.append(val_loss)

        print(f"Epoch {ep+1}/{N_EPOCHS} Train={trL[-1]:.4f} Val={val_loss:.4f}")

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


def evaluate_model(
    model_name, model, Xv, Yv, trL, valL,
    y_scaler, HORIZONS, eval_batch_size=None,
    show_plots=False
):
    eval_batch_size = EVAL_BATCH_SIZE if eval_batch_size is None else eval_batch_size
    print(f"\n================ EVALUATING: {model_name} ================")

    if show_plots:
        plt.figure(figsize=(7,3))
        plt.plot(trL, label="Train"); plt.plot(valL, label="Val")
        plt.title(f"{model_name} Loss"); plt.grid(False); plt.legend()
        plt.show()

    preds = []
    loader = DataLoader(TensorDataset(Xv), batch_size=eval_batch_size, shuffle=False)
    model.eval()
    with torch.no_grad():
        for (xb,) in loader:
            preds.append(model(xb).detach().cpu().numpy())
    preds = np.concatenate(preds, axis=0)  # [N,H,Q]

    Ynp = Yv.detach().cpu().numpy()        # [N,H]

    results_orig = []
    results_norm = []

    # for legacy 10–90 interval + median
    idx_q10 = QUANTILES.index(0.10)
    idx_q50 = QUANTILES.index(0.50)
    idx_q90 = QUANTILES.index(0.90)

    for i, h in enumerate(HORIZONS):
        y = Ynp[:, i]
        pred_q = preds[:, i, :]  # [N,Q]

        q10 = pred_q[:, idx_q10]
        q50 = pred_q[:, idx_q50]
        q90 = pred_q[:, idx_q90]

        mse  = mean_squared_error(y, q50)
        rmse = np.sqrt(mse)
        mae  = mean_absolute_error(y, q50)
        r2   = r2_score(y, q50)
        mape = np.mean(np.abs((y - q50) / np.where(y == 0, 1, y))) * 100

        y_norm   = y_scaler.transform(y.reshape(-1,1)).flatten()
        q50_norm = y_scaler.transform(q50.reshape(-1,1)).flatten()
        mse_norm  = mean_squared_error(y_norm, q50_norm)
        rmse_norm = np.sqrt(mse_norm)
        mae_norm  = mean_absolute_error(y_norm, q50_norm)

        # UPDATED probabilistic metrics (match proposed code)
        PICP = ((y >= q10) & (y <= q90)).mean()
        MIW  = (q90 - q10).mean()
        PINBALL = pinball_loss_np(y, pred_q, QUANTILES)
        CRPS_Q  = crps_from_quantiles(y, pred_q, QUANTILES)
        WIS_M   = wis_multi_alpha(y, pred_q, QUANTILES, alpha_levels=[0.10, 0.20, 0.50])

        cost = compute_cost_metrics(y, q50)

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
# MODELS (YOUR PROPOSED + YOUR BASELINES)
# ============================================================

class DLinearResidualProb(nn.Module):
    """
    Final simple, best-performing baseline model:
    DLinear + Residual MLP + Learnable Mixing
    """
    def __init__(self, input_dim, lookback, horizons, quantiles,
                 hidden_dim=128, alpha_init=0.5):
        super().__init__()
        self.H = len(horizons)
        self.Q = len(quantiles)
        self.in_dim = lookback * input_dim

        self.linear = nn.Linear(self.in_dim, self.H * self.Q)

        self.residual = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.H * self.Q)
        )

        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self, x):
        B = x.size(0)
        flat = x.reshape(B, -1)

        out_lin = self.linear(flat)
        out_res = self.residual(flat)

        a = torch.sigmoid(self.alpha)
        out = a*out_lin + (1-a)*out_res
        return out.view(B, self.H, self.Q)


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
        x = x.transpose(1,2)
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
        self.fc = nn.Linear(hidden*2, self.H * self.Q)

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
        self.fc = nn.Linear(hidden*2, self.H * self.Q)

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
    def __init__(self, input_dim, horizons, quantiles,
                 d_model=64, hidden=128):
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
        x = x.view(B, num_patches, self.patch_len*C)
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
        self.fc = nn.Linear(d_model, self.H*self.Q)

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
        self.fc = nn.Linear(d_model, self.H*self.Q)

    def forward(self, x):
        x = self.proj(x)
        for b in self.blocks:
            x = b(x)
        last = x[:, -1, :]
        return self.fc(last).view(-1, self.H, self.Q)


def baseline_factories(all_features, LOOKBACK, HORIZONS):
    return [
        ("MLP",         lambda: MLP_Prob(len(all_features), LOOKBACK, HORIZONS, QUANTILES)),
        ("TCN",         lambda: TCN_Prob(len(all_features), LOOKBACK, HORIZONS, QUANTILES)),
        ("BiLSTM",      lambda: BiLSTM_Prob(len(all_features), HORIZONS, QUANTILES)),
        ("BiGRU",       lambda: BiGRU_Prob(len(all_features), HORIZONS, QUANTILES)),
        ("TFT",         lambda: TFT_Prob(len(all_features), HORIZONS, QUANTILES)),
        ("PatchTST",    lambda: PatchTST_Prob(len(all_features), LOOKBACK, HORIZONS, QUANTILES)),
        ("Crossformer", lambda: Crossformer_Prob(len(all_features), HORIZONS, QUANTILES))
    ]


# ============================================================
# ORCHESTRATION: RUN ALL SEEDS + SAVE CSVs + STATS TESTS + PLOTS
# ============================================================

HORIZONS = [1, 3, 6, 24]

PROPOSED_HP = {
    "jiaxing":  dict(LOOKBACK=96, hidden_dim=107, alpha_init=0.3379962828893909, lr=0.002089303905987213,
                    batch_size=64, eval_batch_size=16),
    "paloalto": dict(LOOKBACK=84, hidden_dim=175, alpha_init=0.4090457449472471, lr=0.0025585941124396,
                    batch_size=64, eval_batch_size=32),
}

BASELINE_LOOKBACK = 48


def normalize_model_name(model_name: str) -> str:
    for suf in ["-jiaxing", "-paloalto"]:
        if model_name.endswith(suf):
            return model_name[:-len(suf)]
    return model_name


def run_one_seed_one_dataset(dataset_name: str, seed: int, show_plots=False):
    set_seed(seed)

    if dataset_name == "jiaxing":
        X_train, y_train, X_val, y_val, y_scaler, all_features = setup_jiaxing()
    elif dataset_name == "paloalto":
        X_train, y_train, X_val, y_val, y_scaler, all_features = setup_paloalto()
    else:
        raise ValueError("dataset_name must be 'jiaxing' or 'paloalto'")

    all_results_orig = []
    all_results_norm = []

    # 1) PROPOSED
    hp = PROPOSED_HP[dataset_name]
    model = DLinearResidualProb(
        input_dim=len(all_features),
        lookback=hp["LOOKBACK"],
        horizons=HORIZONS,
        quantiles=QUANTILES,
        hidden_dim=hp["hidden_dim"],
        alpha_init=hp["alpha_init"]
    ).to(DEVICE)

    model, Xv, Yv, trL, valL = train_one_model(
        f"DLinear-Res-{dataset_name}",
        model,
        X_train, y_train,
        X_val, y_val,
        hp["LOOKBACK"], HORIZONS,
        batch_size=hp["batch_size"],
        eval_batch_size=hp["eval_batch_size"],
        lr=hp["lr"]
    )

    res_orig, res_norm = evaluate_model(
        f"DLinear-Res-{dataset_name}",
        model,
        Xv, Yv,
        trL, valL,
        y_scaler,
        HORIZONS,
        eval_batch_size=hp["eval_batch_size"],
        show_plots=show_plots
    )
    all_results_orig.extend(res_orig)
    all_results_norm.extend(res_norm)

    # 2) BASELINES
    for name, fn in baseline_factories(all_features, BASELINE_LOOKBACK, HORIZONS):
        model = fn().to(DEVICE)
        model, Xv, Yv, trL, valL = train_one_model(
            f"{name}-{dataset_name}",
            model,
            X_train, y_train,
            X_val, y_val,
            BASELINE_LOOKBACK, HORIZONS
        )
        res_orig, res_norm = evaluate_model(
            f"{name}-{dataset_name}",
            model,
            Xv, Yv,
            trL, valL,
            y_scaler,
            HORIZONS,
            show_plots=show_plots
        )
        all_results_orig.extend(res_orig)
        all_results_norm.extend(res_norm)

    df_orig = pd.DataFrame(all_results_orig, columns=[
        "Model","Horizon","MSE","RMSE","MAE","R2","MAPE",
        "PICP","MIW","Pinball","CRPS_Q","WIS_multi",
        "avg_cost","avg_under","avg_over"
    ])
    df_norm = pd.DataFrame(all_results_norm, columns=[
        "Model","Horizon","MSE_norm","RMSE_norm","MAE_norm"
    ])

    df_orig["Model"] = df_orig["Model"].apply(normalize_model_name)
    df_norm["Model"] = df_norm["Model"].apply(normalize_model_name)

    df_orig["Model"] = df_orig["Model"].replace({"DLinear-Res": "DLinear-Res-Prob"})
    df_norm["Model"] = df_norm["Model"].replace({"DLinear-Res": "DLinear-Res-Prob"})

    df_orig.insert(0, "Seed", seed)
    df_norm.insert(0, "Seed", seed)

    return df_orig, df_norm


def run_all_seeds_and_save():
    os.makedirs("stats_outputs", exist_ok=True)

    all_dataset_dfs = {}
    for dataset in ["jiaxing", "paloalto"]:
        dfs_o, dfs_n = [], []
        for seed in SEEDS:
            df_o, df_n = run_one_seed_one_dataset(dataset, seed, show_plots=False)
            dfs_o.append(df_o); dfs_n.append(df_n)

        dfO = pd.concat(dfs_o, ignore_index=True)
        dfN = pd.concat(dfs_n, ignore_index=True)

        dfO.to_csv(f"stats_outputs/{dataset}_results_orig_all_seeds.csv", index=False)
        dfN.to_csv(f"stats_outputs/{dataset}_results_norm_all_seeds.csv", index=False)

        print(f"Saved: stats_outputs/{dataset}_results_orig_all_seeds.csv")
        print(f"Saved: stats_outputs/{dataset}_results_norm_all_seeds.csv")

        all_dataset_dfs[dataset] = (dfO, dfN)

    return all_dataset_dfs


# -------------------------
# STATISTICAL TESTS
# -------------------------

def holm_adjust(pvals, alpha=0.05):
    pvals = np.asarray(pvals, float)
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, float)

    prev = 0.0
    for k, idx in enumerate(order):
        adj_val = (m - k) * pvals[idx]
        adj_val = max(adj_val, prev)
        prev = adj_val
        adj[idx] = min(adj_val, 1.0)
    return adj


def make_blocks_matrix(df, metric, models_order=None):
    if models_order is None:
        models_order = sorted(df["Model"].unique().tolist())

    blocks = df[["Seed","Horizon"]].drop_duplicates().sort_values(["Seed","Horizon"])
    rows = []
    for seed, h in blocks.itertuples(index=False):
        row = []
        for m in models_order:
            val = df[(df.Seed==seed) & (df.Horizon==h) & (df.Model==m)][metric].values
            if len(val) != 1:
                raise RuntimeError(f"Missing/duplicate value for block (seed={seed},h={h},model={m}) metric={metric}")
            row.append(val[0])
        rows.append(row)
    X = np.array(rows, float)
    return X, blocks, models_order


def average_ranks_lower_is_better(X):
    ranks = np.apply_along_axis(lambda row: rankdata(row, method="average"), 1, X)
    return ranks.mean(axis=0), ranks


def friedman_nemenyi_wilcoxon(df, dataset_label, metric, proposed_name_contains="DLinear-Res"):
    X, blocks, models = make_blocks_matrix(df, metric)

    args = [X[:, j] for j in range(X.shape[1])]
    stat, p = friedmanchisquare(*args)

    avg_rank, _ = average_ranks_lower_is_better(X)

    df_fried = pd.DataFrame([{
        "Dataset": dataset_label,
        "Metric": metric,
        "NumBlocks": X.shape[0],
        "NumModels": X.shape[1],
        "Friedman_Chi2": float(stat),
        "Friedman_p": float(p)
    }])

    df_ranks = pd.DataFrame({
        "Dataset": dataset_label,
        "Metric": metric,
        "Model": models,
        "AvgRank": avg_rank
    }).sort_values("AvgRank")

    if HAS_SPH:
        nemenyi = sp.posthoc_nemenyi_friedman(X)
        nemenyi.index = models
        nemenyi.columns = models
        df_nem = (
            nemenyi.stack()
            .reset_index()
            .rename(columns={"level_0":"ModelA","level_1":"ModelB", 0:"p_value"})
        )
        df_nem.insert(0, "Metric", metric)
        df_nem.insert(0, "Dataset", dataset_label)
    else:
        df_nem = pd.DataFrame([{
            "Dataset": dataset_label,
            "Metric": metric,
            "Note": "Install scikit-posthocs for Nemenyi p-values. (Fallback: ranks plotted only)"
        }])

    proposed_candidates = [m for m in models if proposed_name_contains in m]
    if len(proposed_candidates) == 0:
        raise RuntimeError(f"Could not find proposed model containing '{proposed_name_contains}'. Models={models}")
    proposed = proposed_candidates[0]

    j_prop = models.index(proposed)
    pvals = []
    pairs = []
    for m in models:
        if m == proposed:
            continue
        j = models.index(m)
        w_stat, w_p = wilcoxon(X[:, j_prop], X[:, j], zero_method="wilcox",
                               alternative="two-sided", mode="auto")
        pvals.append(w_p)
        pairs.append((proposed, m, float(w_stat), float(w_p)))

    p_adj = holm_adjust(pvals)
    df_w = pd.DataFrame([{
        "Dataset": dataset_label,
        "Metric": metric,
        "ModelA": a,
        "ModelB": b,
        "Wilcoxon_stat": s,
        "p_value": pv,
        "p_holm": float(p_adj[i])
    } for i, (a,b,s,pv) in enumerate(pairs)]).sort_values("p_holm")

    return df_fried, df_ranks, df_nem, df_w


# -------------------------
# PLOTS
# -------------------------

def plot_metric_box(df, metric, title):
    plt.figure(figsize=(11,4))
    models = sorted(df["Model"].unique())
    data = [df[df.Model==m][metric].values for m in models]
    plt.boxplot(data, labels=models, showfliers=False)
    plt.xticks(rotation=30, ha="right")
    plt.title(title)
    plt.ylabel(metric)
    plt.grid(False)
    plt.tight_layout()
    plt.show()


def plot_avg_ranks(df_ranks, title):
    plt.figure(figsize=(8,4))
    tmp = df_ranks.sort_values("AvgRank")
    plt.bar(tmp["Model"], tmp["AvgRank"])
    plt.xticks(rotation=30, ha="right")
    plt.title(title)
    plt.ylabel("Average Rank (lower is better)")
    plt.grid(False)
    plt.tight_layout()
    plt.show()


def plot_pvalue_heatmap(df_nem, models, title):
    if not HAS_SPH:
        return
    mat = df_nem.pivot(index="ModelA", columns="ModelB", values="p_value").loc[models, models].values
    plt.figure(figsize=(6,5))
    plt.imshow(mat, aspect="auto")
    plt.colorbar(label="p-value")
    plt.xticks(range(len(models)), models, rotation=45, ha="right")
    plt.yticks(range(len(models)), models)
    plt.title(title)
    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN ORCHESTRATION
# ============================================================

def run_everything():
    all_dfs = run_all_seeds_and_save()

    METRICS_TO_TEST = ["MAE", "WIS_multi", "avg_cost"]

    fried_all = []
    ranks_all = []
    nem_all = []
    wilc_all = []

    for dataset, (dfO, _) in all_dfs.items():
        for metric in METRICS_TO_TEST:
            df_f, df_r, df_n, df_w = friedman_nemenyi_wilcoxon(dfO, dataset, metric)
            fried_all.append(df_f); ranks_all.append(df_r); nem_all.append(df_n); wilc_all.append(df_w)

            plot_metric_box(dfO, metric, f"{dataset.upper()} — {metric} distribution (blocks=seed×horizon)")
            plot_avg_ranks(df_r, f"{dataset.upper()} — {metric} average ranks")

            model_order = df_r.sort_values("AvgRank")["Model"].tolist()
            if HAS_SPH and isinstance(df_n, pd.DataFrame) and "p_value" in df_n.columns:
                plot_pvalue_heatmap(df_n, model_order, f"{dataset.upper()} — {metric} Nemenyi p-values")

    df_combined = pd.concat([
        all_dfs["jiaxing"][0].assign(Dataset="jiaxing"),
        all_dfs["paloalto"][0].assign(Dataset="paloalto")
    ], ignore_index=True)

    df_combined["Model"] = df_combined["Model"].apply(normalize_model_name)

    def make_blocks_matrix_combined(df, metric):
        models_order = sorted(df["Model"].unique().tolist())
        blocks = df[["Dataset","Seed","Horizon"]].drop_duplicates().sort_values(["Dataset","Seed","Horizon"])
        rows = []
        for d, seed, h in blocks.itertuples(index=False):
            row = []
            for m in models_order:
                val = df[(df.Dataset==d) & (df.Seed==seed) & (df.Horizon==h) & (df.Model==m)][metric].values
                if len(val) != 1:
                    raise RuntimeError(f"Missing/duplicate (d={d},seed={seed},h={h},m={m}) metric={metric}")
                row.append(val[0])
            rows.append(row)
        return np.array(rows, float), blocks, models_order

    def friedman_nemenyi_wilcoxon_combined(df, metric):
        X, blocks, models = make_blocks_matrix_combined(df, metric)
        stat, p = friedmanchisquare(*[X[:, j] for j in range(X.shape[1])])
        avg_rank, _ = average_ranks_lower_is_better(X)

        df_fried = pd.DataFrame([{
            "Dataset": "combined",
            "Metric": metric,
            "NumBlocks": X.shape[0],
            "NumModels": X.shape[1],
            "Friedman_Chi2": float(stat),
            "Friedman_p": float(p)
        }])
        df_ranks = pd.DataFrame({
            "Dataset": "combined",
            "Metric": metric,
            "Model": models,
            "AvgRank": avg_rank
        }).sort_values("AvgRank")

        if HAS_SPH:
            nemenyi = sp.posthoc_nemenyi_friedman(X)
            nemenyi.index = models
            nemenyi.columns = models
            df_nem = (
                nemenyi.stack().reset_index()
                .rename(columns={"level_0":"ModelA","level_1":"ModelB", 0:"p_value"})
            )
            df_nem.insert(0, "Metric", metric)
            df_nem.insert(0, "Dataset", "combined")
        else:
            df_nem = pd.DataFrame([{
                "Dataset":"combined","Metric":metric,
                "Note":"Install scikit-posthocs for Nemenyi p-values."
            }])

        proposed = [m for m in models if "DLinear-Res" in m][0]
        j_prop = models.index(proposed)

        pvals, pairs = [], []
        for m in models:
            if m == proposed:
                continue
            j = models.index(m)
            w_stat, w_p = wilcoxon(X[:, j_prop], X[:, j], zero_method="wilcox",
                                   alternative="two-sided", mode="auto")
            pvals.append(w_p)
            pairs.append((proposed, m, float(w_stat), float(w_p)))
        p_adj = holm_adjust(pvals)

        df_w = pd.DataFrame([{
            "Dataset":"combined","Metric":metric,
            "ModelA":a,"ModelB":b,"Wilcoxon_stat":s,"p_value":pv,"p_holm":float(p_adj[i])
        } for i,(a,b,s,pv) in enumerate(pairs)]).sort_values("p_holm")

        return df_fried, df_ranks, df_nem, df_w

    for metric in METRICS_TO_TEST:
        df_f, df_r, df_n, df_w = friedman_nemenyi_wilcoxon_combined(df_combined, metric)
        fried_all.append(df_f); ranks_all.append(df_r); nem_all.append(df_n); wilc_all.append(df_w)

        plot_metric_box(df_combined, metric, f"COMBINED — {metric} distribution (blocks=dataset×seed×horizon)")
        plot_avg_ranks(df_r, f"COMBINED — {metric} average ranks")
        model_order = df_r.sort_values("AvgRank")["Model"].tolist()
        if HAS_SPH and "p_value" in df_n.columns:
            plot_pvalue_heatmap(df_n, model_order, f"COMBINED — {metric} Nemenyi p-values")

    df_fried_all = pd.concat(fried_all, ignore_index=True)
    df_ranks_all = pd.concat(ranks_all, ignore_index=True)
    df_nem_all   = pd.concat(nem_all,   ignore_index=True)
    df_wilc_all  = pd.concat(wilc_all,  ignore_index=True)

    df_fried_all.to_csv("stats_outputs/stat_friedman.csv", index=False)
    df_ranks_all.to_csv("stats_outputs/stat_avg_ranks.csv", index=False)
    df_nem_all.to_csv("stats_outputs/stat_nemenyi.csv", index=False)
    df_wilc_all.to_csv("stats_outputs/stat_wilcoxon_holm.csv", index=False)

    print("\nSaved statistical test results:")
    print(" - stats_outputs/stat_friedman.csv")
    print(" - stats_outputs/stat_avg_ranks.csv")
    print(" - stats_outputs/stat_nemenyi.csv")
    print(" - stats_outputs/stat_wilcoxon_holm.csv")

    return df_fried_all, df_ranks_all, df_nem_all, df_wilc_all


if __name__ == "__main__":
    df_fried, df_ranks, df_nem, df_wilc = run_everything()
