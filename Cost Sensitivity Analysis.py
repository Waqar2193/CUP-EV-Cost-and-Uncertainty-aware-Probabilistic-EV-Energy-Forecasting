# ============================================================
# COST SENSITIVITY ANALYSIS (Cu, Co) WITH τ* SCHEDULING
# HYBRID DLINEAR-RES PROBABILISTIC MODEL FOR JIAXING + PALO ALTO
# (UPDATED: richer quantile set + Pinball + multi-alpha WIS + quantile-CRPS
#          + cost-optimal τ* scheduling with quantile interpolation)
# ============================================================

import os, random, numpy as np, pandas as pd, torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ============================================================
# GLOBAL CONFIG
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE =", DEVICE)

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(SEED)

# ---------------- UPDATED QUANTILE GRID ----------------
QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
N_QUANTILES = len(QUANTILES)

TARGET_COL = "Energy"

EVAL_BATCH_SIZE = 32
N_EPOCHS = 50
PATIENCE = 5

quant_t = torch.tensor(QUANTILES, device=DEVICE).view(1, 1, -1)

# global scaler for Jiaxing target
y_scaler = StandardScaler()

# ============================================================
# COMMON UTILITIES (SEQUENCES, LOSS, METRICS)
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
    target = target.unsqueeze(-1)  # [B,H,1]
    err = target - pred
    return torch.max(quant_t * err, (quant_t - 1) * err).mean()

def compute_cost_metrics(y_true, schedule, Cu, Co):
    """
    y_true: [N,]
    schedule: [N,]
    """
    y_true = np.asarray(y_true); schedule = np.asarray(schedule)
    under = np.maximum(0, y_true - schedule)
    over  = np.maximum(0, schedule - y_true)
    cost = Cu * under + Co * over
    return {
        "avg_cost":  float(cost.mean()),
        "avg_under": float(under.mean()),
        "avg_over":  float(over.mean())
    }

# ---------------- UPDATED: vectorized pinball / CRPS / WIS ----------------

def pinball_loss_np(y, q_pred, taus):
    """
    y: [N,]
    q_pred: [N,Q] aligned with taus
    returns: scalar mean pinball loss averaged over N and Q
    """
    y = np.asarray(y).reshape(-1, 1)               # [N,1]
    q_pred = np.asarray(q_pred)                   # [N,Q]
    taus = np.asarray(taus).reshape(1, -1)        # [1,Q]
    err = y - q_pred                              # [N,Q]
    loss = np.maximum(taus * err, (taus - 1.0) * err)
    return float(loss.mean())

def crps_from_quantiles(y, q_pred, taus):
    """
    Quantile-based CRPS approximation:
    CRPS(y,F) = 2 * ∫_0^1 ρ_tau(y - q_tau) d tau
    Approximate integral over tau-grid using trapezoidal rule.
    """
    y = np.asarray(y).reshape(-1, 1)     # [N,1]
    q_pred = np.asarray(q_pred)         # [N,Q]
    taus = np.asarray(taus)             # [Q,]

    err = y - q_pred
    taus_row = taus.reshape(1, -1)
    rho = np.maximum(taus_row * err, (taus_row - 1.0) * err)  # [N,Q]

    dt = np.diff(taus)  # [Q-1]
    integral = (0.5 * (rho[:, :-1] + rho[:, 1:]) * dt.reshape(1, -1)).sum(axis=1)  # [N,]
    crps = 2.0 * integral
    return float(crps.mean())

def wis_multi_alpha(y, q_pred, taus, alpha_levels=None):
    """
    Weighted Interval Score using multiple central intervals.
    """
    y = np.asarray(y).reshape(-1)      # [N]
    q_pred = np.asarray(q_pred)        # [N,Q]
    taus = np.asarray(taus)            # [Q]

    if alpha_levels is None:
        alpha_levels = [0.10, 0.20, 0.40, 0.60, 0.80]

    if 0.50 not in taus:
        raise ValueError("0.50 must be in QUANTILES for WIS median term.")
    m = q_pred[:, list(taus).index(0.50)]  # [N]

    K = len(alpha_levels)
    wis = 0.5 * np.abs(y - m)

    for a in alpha_levels:
        lo_tau = a / 2.0
        hi_tau = 1.0 - a / 2.0
        if (lo_tau not in taus) or (hi_tau not in taus):
            continue
        lo = q_pred[:, list(taus).index(lo_tau)]
        hi = q_pred[:, list(taus).index(hi_tau)]
        width = (hi - lo)
        below = np.maximum(0.0, lo - y)
        above = np.maximum(0.0, y - hi)
        interval_score = width + (2.0 / a) * below + (2.0 / a) * above
        wis += (a / 2.0) * interval_score

    denom = (K + 0.5)
    return float(np.mean(wis / denom))

# ============================================================
# NEW: τ* SCHEDULING WITH QUANTILE INTERPOLATION
# ============================================================

def tau_star_from_costs(Cu, Co):
    # newsvendor / asymmetric loss optimal fractile
    return float(Cu / (Cu + Co))

def quantile_interpolate(pred_q, taus, tau_target):
    """
    Linear interpolation in quantile space.

    pred_q: [N,Q] predicted quantiles at taus (sorted ascending)
    taus:   [Q,]
    tau_target: scalar in (0,1)

    returns schedule: [N,]
    """
    taus = np.asarray(taus, dtype=float)
    pred_q = np.asarray(pred_q, dtype=float)
    tau = float(tau_target)

    # clamp to supported range
    if tau <= taus[0]:
        return pred_q[:, 0].copy()
    if tau >= taus[-1]:
        return pred_q[:, -1].copy()

    # exact match
    if tau in set(taus.tolist()):
        j = int(np.where(taus == tau)[0][0])
        return pred_q[:, j].copy()

    # linear interpolation between nearest taus
    j_hi = int(np.searchsorted(taus, tau, side="right"))  # 1..Q-1
    j_lo = j_hi - 1
    t0, t1 = taus[j_lo], taus[j_hi]
    w = (tau - t0) / (t1 - t0)

    return (1.0 - w) * pred_q[:, j_lo] + w * pred_q[:, j_hi]

# ============================================================
# DATASET SETUP
# ============================================================

def setup_jiaxing():
    global X_train, y_train, X_val, y_val, all_features, y_scaler

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
    train_df = df.iloc[:split]
    val_df   = df.iloc[split:]

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

    y_scaler.fit(y_train.reshape(-1,1))
    print("Loaded Jiaxing Dataset:", X_train.shape, X_val.shape)

def setup_paloalto():
    df = pd.read_csv("processed_palo_alto.csv")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    cat_cols = [
        'Station Name','MAC Address','Address','Ended By',
        'Day of Week','Day Type','Season','Holiday'
    ]
    num_cols = [
        'Charging Time','Total Duration','Fee','Hour',
        'Humidity','Precipitation','Pressure','Area ID'
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

    y_scaler_pa = StandardScaler()
    y_scaler_pa.fit(y_train.reshape(-1,1))

    print("Loaded Palo Alto Dataset:", X_train.shape, X_val.shape)
    return X_train, y_train, X_val, y_val, y_scaler_pa, all_features

# ============================================================
# MODEL
# ============================================================

class DLinearResidualProb(nn.Module):
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
        out = a * out_lin + (1 - a) * out_res
        return out.view(B, self.H, self.Q)

# ============================================================
# TRAINING
# ============================================================

def train_one_model(model_name, model, X_train, y_train, X_val, y_val,
                    LOOKBACK, HORIZONS,
                    batch_size, eval_batch_size, lr):

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

# ============================================================
# EVALUATION (BASE METRICS + COST SENSITIVITY)
# ============================================================

def predict_quantiles(model, Xv, eval_batch_size):
    preds = []
    loader = DataLoader(TensorDataset(Xv), batch_size=eval_batch_size, shuffle=False)
    model.eval()
    with torch.no_grad():
        for (xb,) in loader:
            preds.append(model(xb).detach().cpu().numpy())
    return np.concatenate(preds, axis=0)  # [N,H,Q]

def evaluate_base_metrics(model_name, preds, Yv, y_scaler, HORIZONS):
    """
    Keeps your reporting logic (pinball/CRPS/WIS + interval 10-90 + point metrics).
    Returns a dataframe per horizon.
    """
    Ynp = Yv.detach().cpu().numpy()   # [N,H]
    results = []

    idx_q10 = QUANTILES.index(0.10)
    idx_q50 = QUANTILES.index(0.50)
    idx_q90 = QUANTILES.index(0.90)

    for i, h in enumerate(HORIZONS):
        y = Ynp[:, i]
        pred_q = preds[:, i, :]  # [N,Q]
        q10, q50, q90 = pred_q[:, idx_q10], pred_q[:, idx_q50], pred_q[:, idx_q90]

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

        PICP = ((y >= q10) & (y <= q90)).mean()
        MIW  = (q90 - q10).mean()

        PINBALL = pinball_loss_np(y, pred_q, QUANTILES)
        CRPS_Q  = crps_from_quantiles(y, pred_q, QUANTILES)
        WIS_M   = wis_multi_alpha(y, pred_q, QUANTILES, alpha_levels=[0.10, 0.20, 0.50])

        results.append({
            "Model": model_name,
            "Horizon": h,
            "MSE": float(mse),
            "RMSE": float(rmse),
            "MAE": float(mae),
            "R2": float(r2),
            "MAPE": float(mape),
            "PICP_10_90": float(PICP),
            "MIW_10_90": float(MIW),
            "Pinball": float(PINBALL),
            "CRPS_Q": float(CRPS_Q),
            "WIS_multi": float(WIS_M),
            "MSE_norm": float(mse_norm),
            "RMSE_norm": float(rmse_norm),
            "MAE_norm": float(mae_norm),
        })

    return pd.DataFrame(results)

def evaluate_cost_sensitivity(dataset_name, model_name, preds, Yv, HORIZONS, cost_pairs):
    """
    Computes cost using τ* = Cu/(Cu+Co) and schedule = interpolated quantile at τ*.
    Returns long-form dataframe with AvgCost/AvgUnder/AvgOver per horizon and cost setting.
    """
    Ynp = Yv.detach().cpu().numpy()  # [N,H]
    taus = np.asarray(QUANTILES, dtype=float)

    rows = []
    for (Cu, Co) in cost_pairs:
        tau_star = tau_star_from_costs(Cu, Co)

        for i, h in enumerate(HORIZONS):
            y = Ynp[:, i]
            pred_q = preds[:, i, :]  # [N,Q]

            schedule = quantile_interpolate(pred_q, taus, tau_star)  # [N,]
            cm = compute_cost_metrics(y, schedule, Cu=Cu, Co=Co)

            rows.append({
                "Dataset": dataset_name,
                "Model": model_name,
                "Horizon": h,
                "Cu": float(Cu),
                "Co": float(Co),
                "tau_star": float(tau_star),
                "avg_cost": cm["avg_cost"],
                "avg_under": cm["avg_under"],
                "avg_over": cm["avg_over"],
            })

    return pd.DataFrame(rows)

# ============================================================
# RUN FOR DATASET (TRAIN ONCE, EVAL MANY COST SETTINGS)
# ============================================================

def run_for_dataset(
    dataset_name,
    X_train, y_train, X_val, y_val,
    y_scaler_local,
    LOOKBACK,
    HORIZONS,
    all_features,
    hidden_dim,
    alpha_init,
    batch_size,
    eval_batch_size,
    lr,
    cost_pairs
):
    print(f"\n\n======== RUNNING ON {dataset_name.upper()} DATASET ========\n")

    model = DLinearResidualProb(
        input_dim=len(all_features),
        lookback=LOOKBACK,
        horizons=HORIZONS,
        quantiles=QUANTILES,
        hidden_dim=hidden_dim,
        alpha_init=alpha_init
    ).to(DEVICE)

    model, Xv, Yv, trL, valL = train_one_model(
        f"DLinear-Res-{dataset_name}",
        model,
        X_train, y_train,
        X_val, y_val,
        LOOKBACK,
        HORIZONS,
        batch_size,
        eval_batch_size,
        lr
    )

    # (optional) loss curves
    plt.figure(figsize=(7,3))
    plt.plot(trL, label="Train"); plt.plot(valL, label="Val")
    plt.title(f"DLinear-Res-{dataset_name} Loss"); plt.grid(False); plt.legend()
    plt.show()

    preds = predict_quantiles(model, Xv, eval_batch_size)  # [N,H,Q]

    # Save base metrics (your proposed metrics table)
    df_base = evaluate_base_metrics(f"DLinear-Res-{dataset_name}", preds, Yv, y_scaler_local, HORIZONS)
    base_csv = f"{dataset_name}_dlinear_res_base_metrics_taugrid.csv"
    df_base.to_csv(base_csv, index=False)
    print(f"✔ Saved base metrics: {base_csv}")

    # Cost sensitivity with τ* scheduling
    df_cost = evaluate_cost_sensitivity(dataset_name, f"DLinear-Res-{dataset_name}", preds, Yv, HORIZONS, cost_pairs)
    return df_base, df_cost

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # ---- choose (Cu,Co) settings for sensitivity
    # balanced + moderate asymmetry + strong asymmetry (both directions)
    COST_PAIRS = [
        (1.0, 1.0),
        (2.0, 1.0),
        (5.0, 1.0),
        (10.0, 1.0),
        (1.0, 2.0),
        (1.0, 5.0),
        (1.0, 10.0),
    ]

    all_cost_dfs = []

    # ---------------- JIAXING ----------------
    setup_jiaxing()
    Xtr_jx, ytr_jx = X_train, y_train
    Xval_jx, yval_jx = X_val, y_val
    all_features_jx = all_features

    LOOKBACK_JX        = 96
    HORIZONS_JX        = [1, 3, 6, 24]
    HIDDEN_DIM_JX      = 107
    ALPHA_INIT_JX      = 0.3379962828893909
    LR_JX              = 0.002089303905987213
    BATCH_SIZE_JX      = 64
    EVAL_BATCH_SIZE_JX = 16

    _, df_cost_jx = run_for_dataset(
        "jiaxing",
        Xtr_jx, ytr_jx,
        Xval_jx, yval_jx,
        y_scaler,
        LOOKBACK_JX,
        HORIZONS_JX,
        all_features_jx,
        hidden_dim=HIDDEN_DIM_JX,
        alpha_init=ALPHA_INIT_JX,
        batch_size=BATCH_SIZE_JX,
        eval_batch_size=EVAL_BATCH_SIZE_JX,
        lr=LR_JX,
        cost_pairs=COST_PAIRS
    )
    all_cost_dfs.append(df_cost_jx)

    # ---------------- PALO ALTO ----------------
    Xtr_pa, ytr_pa, Xval_pa, yval_pa, y_scaler_pa, all_features_pa = setup_paloalto()

    LOOKBACK_PA        = 84
    HORIZONS_PA        = [1, 3, 6, 24]
    HIDDEN_DIM_PA      = 175
    ALPHA_INIT_PA      = 0.4090457449472471
    LR_PA              = 0.0025585941124396
    BATCH_SIZE_PA      = 64
    EVAL_BATCH_SIZE_PA = 32

    _, df_cost_pa = run_for_dataset(
        "paloalto",
        Xtr_pa, ytr_pa,
        Xval_pa, yval_pa,
        y_scaler_pa,
        LOOKBACK_PA,
        HORIZONS_PA,
        all_features_pa,
        hidden_dim=HIDDEN_DIM_PA,
        alpha_init=ALPHA_INIT_PA,
        batch_size=BATCH_SIZE_PA,
        eval_batch_size=EVAL_BATCH_SIZE_PA,
        lr=LR_PA,
        cost_pairs=COST_PAIRS
    )
    all_cost_dfs.append(df_cost_pa)

    # ---------------- SAVE COMBINED COST SENSITIVITY CSV ----------------
    df_all_cost = pd.concat(all_cost_dfs, ignore_index=True)
    out_csv = "all_datasets_cost_sensitivity_tau_star.csv"
    df_all_cost.to_csv(out_csv, index=False)

    print("\n✔ Saved combined cost sensitivity results:")
    print(f" - {out_csv}")

    # Optional: quick pivot summary (mean over horizons) per dataset and (Cu,Co)
    summary = (
        df_all_cost
        .groupby(["Dataset", "Cu", "Co", "tau_star"], as_index=False)[["avg_cost", "avg_under", "avg_over"]]
        .mean()
        .sort_values(["Dataset", "Cu", "Co"])
    )
    summary_csv = "all_datasets_cost_sensitivity_tau_star_summary_mean_over_horizons.csv"
    summary.to_csv(summary_csv, index=False)
    print(f" - {summary_csv}")

    print("\n============= ALL DONE =============\n")
