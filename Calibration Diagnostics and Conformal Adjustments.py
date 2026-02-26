# ============================================================
# HYBRID DLINEAR-RES PROBABILISTIC MODEL FOR JIAXING + PALO ALTO
# (REVISION PATCH: calibration curves + PIT + CQR + save to "Revision Plots")
# + Combined PIT (4 subplots) per dataset
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

Cu, Co = 2.0, 1.0        # Cost parameters

# ---------------- REVISION: more quantiles for calibration diagnostics ----------------
QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
N_QUANTILES = len(QUANTILES)

TARGET_COL = "Energy"

# default globals (we override per dataset via arguments)
EVAL_BATCH_SIZE = 32
N_EPOCHS = 50
PATIENCE = 5
BATCH_SIZE = 32

quant_t = torch.tensor(QUANTILES, device=DEVICE).view(1,1,-1)

# global scaler for Jiaxing target
y_scaler = StandardScaler()

# ---------------- REVISION: output directory ----------------
REV_DIR = "Revision Plots"
os.makedirs(REV_DIR, exist_ok=True)


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
    return np.array(Xs,np.float32), np.array(Ys,np.float32)

def quantile_loss(pred,target):
    target = target.unsqueeze(-1)  # [B,H,1]
    err = target - pred
    return torch.max(quant_t * err, (quant_t - 1) * err).mean()

def compute_cost_metrics(y_true, schedule):
    y_true=np.asarray(y_true); schedule=np.asarray(schedule)
    under = np.maximum(0, y_true-schedule)
    over  = np.maximum(0, schedule-y_true)
    cost = Cu*under + Co*over
    return {
        "avg_cost": float(cost.mean()),
        "avg_under": float(under.mean()),
        "avg_over":  float(over.mean())
    }

def compute_crps(y,q10,q50,q90):
    return float((q90-q10) + 2*np.abs(y-q50))

def compute_wis(y,q10,q50,q90):
    return float(
        0.5*(q90-q10) +
        np.maximum(q10-y,0) +
        np.maximum(y-q90,0)
    )

# ---------------- REVISION: helper functions for quantile interpolation, PIT, coverage ----------------

def _sorted_quantiles_and_preds(pred_q, quantiles):
    """
    pred_q: [N,Q] numpy
    returns quantiles sorted + preds sorted along Q
    """
    qs = np.asarray(quantiles, dtype=np.float32)
    order = np.argsort(qs)
    return qs[order], pred_q[:, order]

def get_pred_quantile(pred_q, quantiles, tau):
    """
    Piecewise-linear interpolation of predicted quantiles.
    pred_q: [N,Q], quantiles list length Q
    tau: float in (0,1)
    returns: [N,]
    """
    qs, pq = _sorted_quantiles_and_preds(pred_q, quantiles)
    tau = float(tau)
    if tau <= qs[0]:
        return pq[:,0].copy()
    if tau >= qs[-1]:
        return pq[:,-1].copy()

    j = np.searchsorted(qs, tau, side="right") - 1
    j = int(np.clip(j, 0, len(qs)-2))
    ql, qu = qs[j], qs[j+1]
    yl, yu = pq[:,j], pq[:,j+1]
    w = (tau - ql) / (qu - ql + 1e-12)
    return yl + w*(yu - yl)

def empirical_coverage(y, lo, hi):
    y = np.asarray(y); lo=np.asarray(lo); hi=np.asarray(hi)
    return float(((y >= lo) & (y <= hi)).mean())

def compute_coverage_curve(y, pred_q, quantiles, nominal_levels=(0.50,0.80,0.90)):
    """
    For central intervals at nominal level c, lower=(1-c)/2, upper=1-lower
    returns arrays: nominals, empirical_coverages
    """
    nom = []
    emp = []
    for c in nominal_levels:
        a = (1.0 - c) / 2.0
        lo_tau, hi_tau = a, 1.0 - a
        lo = get_pred_quantile(pred_q, quantiles, lo_tau)
        hi = get_pred_quantile(pred_q, quantiles, hi_tau)
        nom.append(c)
        emp.append(empirical_coverage(y, lo, hi))
    return np.array(nom), np.array(emp)

def pit_values_from_quantiles(y, pred_q, quantiles):
    """
    PIT u = Fhat(y) using piecewise-linear CDF approximation
    between predicted quantiles (tau_k, q_k).
    pred_q: [N,Q] predicted quantile values aligned with 'quantiles'
    returns u: [N,] clipped to [0,1]
    """
    y = np.asarray(y, dtype=np.float32)
    qs, pq = _sorted_quantiles_and_preds(pred_q, quantiles)  # qs: [Q], pq: [N,Q]
    N, Q = pq.shape
    u = np.zeros(N, dtype=np.float32)

    # safeguard monotonicity for CDF approximation
    pq_mono = pq.copy()
    pq_mono.sort(axis=1)

    below = y <= pq_mono[:,0]
    above = y >= pq_mono[:,-1]
    mid = ~(below | above)

    u[below] = 0.0
    u[above] = 1.0

    if mid.any():
        y_mid = y[mid]
        pq_mid = pq_mono[mid]  # [Nm,Q]

        j = np.sum(pq_mid <= y_mid[:,None], axis=1) - 1
        j = np.clip(j, 0, Q-2)

        ql = pq_mid[np.arange(len(j)), j]
        qu = pq_mid[np.arange(len(j)), j+1]
        tl = qs[j]
        tu = qs[j+1]

        w = (y_mid - ql) / (qu - ql + 1e-12)
        u_mid = tl + w*(tu - tl)
        u[mid] = u_mid.astype(np.float32)

    return np.clip(u, 0.0, 1.0)

def cqr_calibrate_intervals(y_cal, pred_q_cal, quantiles, alpha, horizon_name=""):
    """
    Split Conformal for Quantile Regression (CQR) for central (1-alpha) interval.
    Uses conformity scores s = max(q_lo - y, y - q_hi).
    Returns qhat scalar and calibrated (lo, hi) on calibration set (for diagnostics).
    """
    lo_tau = alpha/2.0
    hi_tau = 1.0 - alpha/2.0
    lo = get_pred_quantile(pred_q_cal, quantiles, lo_tau)
    hi = get_pred_quantile(pred_q_cal, quantiles, hi_tau)

    y_cal = np.asarray(y_cal)
    s = np.maximum(lo - y_cal, y_cal - hi)

    n = len(s)
    k = int(np.ceil((n + 1) * (1.0 - alpha)))  # 1-indexed order statistic
    k = min(max(k, 1), n)
    qhat = float(np.partition(s, k-1)[k-1])

    lo_c = lo - qhat
    hi_c = hi + qhat
    return qhat, lo_c, hi_c


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

    # fit global y_scaler for Jiaxing
    y_scaler.fit(y_train.reshape(-1,1))

    print("Loaded Jiaxing Dataset:", X_train.shape, X_val.shape)

# ============================================================
# HYBRID DLINEAR-RES MODEL
# ============================================================

class DLinearResidualProb(nn.Module):
    """
    Final simple, best-performing baseline model:
    DLinear + Residual MLP + Learnable Mixing
    """
    def __init__(self,input_dim,lookback,horizons,quantiles,
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

    def forward(self,x):
        B = x.size(0)
        flat = x.reshape(B, -1)

        out_lin = self.linear(flat)
        out_res = self.residual(flat)

        a = torch.sigmoid(self.alpha)
        out = a*out_lin + (1-a)*out_res

        return out.view(B, self.H, self.Q)


# ============================================================
# TRAINING + EVALUATION PIPELINE (Shared for Jiaxing & PaloAlto)
# ============================================================

def train_one_model(model_name, model, X_train, y_train, X_val, y_val,
                    LOOKBACK, HORIZONS,
                    batch_size, eval_batch_size, lr):

    print(f"\n================ TRAINING: {model_name} ================")

    Xtr, Ytr = build_multihorizon_sequences(X_train,y_train,LOOKBACK,HORIZONS)
    Xv,  Yv  = build_multihorizon_sequences(X_val,  y_val,  LOOKBACK,HORIZONS)

    Xtr = torch.tensor(Xtr).float().to(DEVICE)
    Ytr = torch.tensor(Ytr).float().to(DEVICE)
    Xv  = torch.tensor(Xv ).float().to(DEVICE)
    Yv  = torch.tensor(Yv ).float().to(DEVICE)

    tr_loader = DataLoader(TensorDataset(Xtr,Ytr),
                           batch_size=batch_size, shuffle=True)
    v_loader  = DataLoader(TensorDataset(Xv,Yv),
                           batch_size=eval_batch_size)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best=float("inf"); patience=0
    trL=[]; valL=[]

    for ep in range(N_EPOCHS):
        model.train(); run=0.0
        for xb,yb in tr_loader:
            opt.zero_grad()
            pred = model(xb)
            loss = quantile_loss(pred,yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step()
            run += loss.item()
        trL.append(run / len(tr_loader))

        model.eval(); run=0.0
        with torch.no_grad():
            for xb,yb in v_loader:
                run += quantile_loss(model(xb),yb).item()
        val_loss = run / len(v_loader)
        valL.append(val_loss)

        print(f"Epoch {ep+1}/{N_EPOCHS} Train={trL[-1]:.4f} Val={val_loss:.4f}")

        if val_loss < best - 1e-4:
            best = val_loss
            best_state = model.state_dict()
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                print("Early stopping.")
                break

    model.load_state_dict(best_state)
    return model, Xv, Yv, trL, valL


def evaluate_model(model_name, model, Xv, Yv, trL, valL, y_scaler, HORIZONS,
                   eval_batch_size, dataset_name_for_paths):

    print(f"\n================ EVALUATING: {model_name} ================")

    out_dir = os.path.join(REV_DIR, dataset_name_for_paths)
    os.makedirs(out_dir, exist_ok=True)

    # ---- Loss curves ----
    plt.figure(figsize=(7,3))
    plt.plot(trL,label="Train"); plt.plot(valL,label="Val")
    plt.title(f"{model_name} Loss"); plt.grid(False); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{model_name}_loss.png"), dpi=200)
    plt.close()

    # ---- Predictions ----
    preds=[]
    loader=DataLoader(TensorDataset(Xv),batch_size=eval_batch_size)
    with torch.no_grad():
        for (xb,) in loader:
            preds.append(model(xb).detach().cpu().numpy())
    preds = np.concatenate(preds)  # [N,H,Q]

    Ynp = Yv.detach().cpu().numpy()  # [N,H]

    results_orig = []
    results_norm = []

    calib_rows = []

    # store PIT for combined plot
    pit_store = {}  # {horizon: u_values}

    # split validation sequences for CQR (only used for Palo Alto)
    cqr_split = 0.5
    N_total = Ynp.shape[0]
    split_idx = int(cqr_split * N_total)
    split_idx = max(50, min(split_idx, N_total - 50))  # safety clamp

    idx_q10 = QUANTILES.index(0.10)
    idx_q50 = QUANTILES.index(0.50)
    idx_q90 = QUANTILES.index(0.90)

    for i,h in enumerate(HORIZONS):
        y = Ynp[:,i]
        pred_q = preds[:,i,:]  # [N,Q]

        q10,q50,q90 = pred_q[:,idx_q10], pred_q[:,idx_q50], pred_q[:,idx_q90]

        mse  = mean_squared_error(y,q50)
        rmse = np.sqrt(mse)
        mae  = mean_absolute_error(y,q50)
        r2   = r2_score(y,q50)
        mape = np.mean(np.abs((y-q50)/np.where(y==0,1,y))) * 100

        y_norm = y_scaler.transform(y.reshape(-1,1)).flatten()
        q50_norm = y_scaler.transform(q50.reshape(-1,1)).flatten()
        mse_norm  = mean_squared_error(y_norm,q50_norm)
        rmse_norm = np.sqrt(mse_norm)
        mae_norm  = mean_absolute_error(y_norm,q50_norm)

        PICP = ((y>=q10)&(y<=q90)).mean()
        MIW  = (q90-q10).mean()
        CRPS = np.mean([compute_crps(y[j],q10[j],q50[j],q90[j]) for j in range(len(y))])
        WIS  = np.mean([compute_wis(y[j],q10[j],q50[j],q90[j]) for j in range(len(y))])

        cost = compute_cost_metrics(y,q50)

        print(f"\n---- {model_name} | {h}h ----")
        print(f"ORIGINAL:   MSE={mse:.3f} RMSE={rmse:.3f} MAE={mae:.3f} R2={r2:.3f} MAPE={mape:.2f}%")
        print(f"NORMALIZED: MSE={mse_norm:.3f} RMSE={rmse_norm:.3f} MAE={mae_norm:.3f}")
        print(f"PICP={PICP:.3f} MIW={MIW:.3f} CRPS={CRPS:.3f} WIS={WIS:.3f}")
        print(f"COST={cost}")

        results_orig.append([
            model_name, h, mse, rmse, mae, r2, mape,
            float(PICP), float(MIW), float(CRPS), float(WIS),
            cost["avg_cost"], cost["avg_under"], cost["avg_over"]
        ])

        results_norm.append([
            model_name, h, mse_norm, rmse_norm, mae_norm
        ])

        # COST PLOT (save)
        plt.figure(figsize=(4,3))
        plt.bar(["avg_cost","avg_under","avg_over"],
                [cost["avg_cost"],cost["avg_under"],cost["avg_over"]])
        plt.title(f"{model_name} — {h}h Cost Breakdown"); plt.grid(axis="y")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{model_name}_{h}h_cost.png"), dpi=200)
        plt.close()

        # FORECAST PLOT (save)
        Np=min(300,len(y))
        plt.figure(figsize=(10,3))
        plt.plot(y[:Np],"k",label="Actual")
        plt.plot(q50[:Np],"b",label="Pred (q50)")
        plt.fill_between(range(Np), q10[:Np], q90[:Np], alpha=0.3)
        plt.title(f"{model_name} — {h}h Forecast")
        plt.legend(); plt.grid(False)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{model_name}_{h}h_forecast.png"), dpi=200)
        plt.close()

        # (1) Coverage vs Nominal points per horizon
        nom, emp = compute_coverage_curve(
            y=y,
            pred_q=pred_q,
            quantiles=QUANTILES,
            nominal_levels=(0.50, 0.80, 0.90)
        )
        for c, ec in zip(nom, emp):
            calib_rows.append([model_name, dataset_name_for_paths, h, float(c), float(ec), "uncalibrated"])

        # (2) PIT per horizon + store for combined plot
        u = pit_values_from_quantiles(y, pred_q, QUANTILES)
        pit_store[h] = u

        plt.figure(figsize=(4,3))
        plt.hist(u, bins=20)
        plt.title(f"{dataset_name_for_paths} PIT — {h}h")
        plt.xlabel("u"); plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{dataset_name_for_paths}_PIT_{h}h.png"), dpi=200)
        plt.close()

        # (3) CQR only for Palo Alto, out-of-sample (calibration split vs evaluation split)
        if dataset_name_for_paths.lower() == "paloalto":
            y_cal  = y[:split_idx]
            pq_cal = pred_q[:split_idx, :]

            y_eval  = y[split_idx:]
            pq_eval = pred_q[split_idx:, :]

            for alpha, label in [(0.20, "CQR_80"), (0.10, "CQR_90")]:
                qhat, _, _ = cqr_calibrate_intervals(
                    y_cal=y_cal,
                    pred_q_cal=pq_cal,
                    quantiles=QUANTILES,
                    alpha=alpha
                )

                lo_tau = alpha/2.0
                hi_tau = 1.0 - alpha/2.0
                lo_eval = get_pred_quantile(pq_eval, QUANTILES, lo_tau) - qhat
                hi_eval = get_pred_quantile(pq_eval, QUANTILES, hi_tau) + qhat

                cov_c = empirical_coverage(y_eval, lo_eval, hi_eval)
                nom_c = 1.0 - alpha
                calib_rows.append([model_name, dataset_name_for_paths, h, float(nom_c), float(cov_c), label])

    # ---------------- dataset-level coverage figure ----------------
    if len(HORIZONS) == 4:
        fig, axes = plt.subplots(2,2, figsize=(9,7))
        axes = axes.ravel()
    else:
        fig, axes = plt.subplots(1, len(HORIZONS), figsize=(4*len(HORIZONS),3))
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])

    for ax, h in zip(axes, HORIZONS):
        rows = [r for r in calib_rows if r[1]==dataset_name_for_paths and r[2]==h and r[5]=="uncalibrated"]
        rows = sorted(rows, key=lambda x: x[3])
        nom = [r[3] for r in rows]
        emp = [r[4] for r in rows]

        ax.plot(nom, emp, marker="o", label="Empirical")
        ax.plot([0,1],[0,1], linestyle="--", label="Ideal")
        ax.set_title(f"{dataset_name_for_paths} — {h}h")
        ax.set_xlabel("Nominal coverage")
        ax.set_ylabel("Empirical coverage")
        ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.grid(False)

        if dataset_name_for_paths.lower() == "paloalto":
            rows_c = [r for r in calib_rows if r[1]==dataset_name_for_paths and r[2]==h and r[5] in ("CQR_80","CQR_90")]
            for r in rows_c:
                ax.scatter([r[3]],[r[4]], marker="x", s=80, label=r[5])

    handles, labels = [], []
    for ax in axes:
        hnd, lab = ax.get_legend_handles_labels()
        for hh, ll in zip(hnd, lab):
            if ll not in labels:
                handles.append(hh); labels.append(ll)
    fig.legend(handles, labels, loc="lower center", ncol=4)
    fig.suptitle(f"Coverage vs Nominal — {dataset_name_for_paths}", y=0.98)
    plt.tight_layout(rect=[0,0.06,1,0.95])
    plt.savefig(os.path.join(out_dir, f"{dataset_name_for_paths}_coverage_vs_nominal.png"), dpi=220)
    plt.close(fig)

    # ---------------- dataset-level combined PIT figure ----------------
    if len(HORIZONS) == 4:
        fig, axes = plt.subplots(2, 2, figsize=(9, 6))
        axes = axes.ravel()
    else:
        fig, axes = plt.subplots(1, len(HORIZONS), figsize=(4*len(HORIZONS), 3))
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])

    for ax, h in zip(axes, HORIZONS):
        u = pit_store.get(h, None)
        if u is None:
            continue
        ax.hist(u, bins=20)
        ax.set_title(f"{h}h")
        ax.set_xlim(0, 1)
        ax.set_xlabel("u")
        ax.set_ylabel("Count")
        ax.grid(False)

    fig.suptitle(f"PIT Histograms — {dataset_name_for_paths}", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(out_dir, f"{dataset_name_for_paths}_PIT_combined.png"), dpi=220)
    plt.close(fig)

    # Save calibration table
    calib_df = pd.DataFrame(calib_rows, columns=[
        "Model","Dataset","Horizon","Nominal","Empirical","Type"
    ])
    calib_df.to_csv(os.path.join(out_dir, f"{dataset_name_for_paths}_calibration_table.csv"), index=False)

    return results_orig, results_norm


# ============================================================
# TRAIN & EVAL FOR BOTH DATASETS
# ============================================================

def run_for_dataset(
    dataset_name,
    X_train, y_train, X_val, y_val,
    y_scaler,
    LOOKBACK,
    HORIZONS,
    all_features,
    hidden_dim,
    alpha_init,
    batch_size,
    eval_batch_size,
    lr
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

    res_orig, res_norm = evaluate_model(
        f"DLinear-Res-{dataset_name}",
        model,
        Xv, Yv,
        trL, valL,
        y_scaler,
        HORIZONS,
        eval_batch_size,
        dataset_name_for_paths=dataset_name
    )

    df_orig = pd.DataFrame(res_orig, columns=[
        "Model","Horizon","MSE","RMSE","MAE","R2","MAPE",
        "PICP","MIW","CRPS","WIS","avg_cost","avg_under","avg_over"
    ])

    df_norm = pd.DataFrame(res_norm, columns=[
        "Model","Horizon","MSE_norm","RMSE_norm","MAE_norm"
    ])

    out_dir = os.path.join(REV_DIR, dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    df_orig.to_csv(os.path.join(out_dir, f"{dataset_name}_dlinear_res_original_TP.csv"), index=False)
    df_norm.to_csv(os.path.join(out_dir, f"{dataset_name}_dlinear_res_normalized_TP.csv"), index=False)

    print(f"\n✔ Saved results + revision plots for {dataset_name} in: {out_dir}\n")


# ============================================================
# MAIN: RUN JIAXING THEN PALO ALTO WITH TUNED HYPERPARAMETERS
# ============================================================

if __name__ == "__main__":

    # ---------------- JIAXING ----------------
    setup_jiaxing()
    Xtr_jx, ytr_jx = X_train, y_train
    Xval_jx, yval_jx = X_val, y_val
    all_features_jx = all_features

    LOOKBACK_JX        = 96
    HORIZONS_JX        = [1,3,6,24]
    HIDDEN_DIM_JX      = 107
    ALPHA_INIT_JX      = 0.3379962828893909
    LR_JX              = 0.002089303905987213
    BATCH_SIZE_JX      = 64
    EVAL_BATCH_SIZE_JX = 16

    run_for_dataset(
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
        lr=LR_JX
    )

    # ---------------- PALO ALTO ----------------
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

    split = int(0.80*len(df))
    train_df = df.iloc[:split]
    val_df   = df.iloc[split:]

    scaler = StandardScaler()
    train_scaled = train_df.copy()
    val_scaled   = val_df.copy()

    train_scaled[num_cols] = scaler.fit_transform(train_df[num_cols])
    val_scaled[num_cols]   = scaler.transform(val_df[num_cols])

    all_features_pa = num_cols + cat_cols

    X_train_pa = train_scaled[all_features_pa].values.astype(np.float32)
    y_train_pa = train_df[TARGET_COL].values.astype(np.float32)

    X_val_pa   = val_scaled[all_features_pa].values.astype(np.float32)
    y_val_pa   = val_df[TARGET_COL].values.astype(np.float32)

    y_scaler_pa = StandardScaler()
    y_scaler_pa.fit(y_train_pa.reshape(-1,1))

    LOOKBACK_PA        = 84
    HORIZONS_PA        = [1,3,6,24]
    HIDDEN_DIM_PA      = 175
    ALPHA_INIT_PA      = 0.4090457449472471
    LR_PA              = 0.0025585941124396
    BATCH_SIZE_PA      = 64
    EVAL_BATCH_SIZE_PA = 32

    run_for_dataset(
        "paloalto",
        X_train_pa, y_train_pa,
        X_val_pa, y_val_pa,
        y_scaler_pa,
        LOOKBACK_PA,
        HORIZONS_PA,
        all_features_pa,
        hidden_dim=HIDDEN_DIM_PA,
        alpha_init=ALPHA_INIT_PA,
        batch_size=BATCH_SIZE_PA,
        eval_batch_size=EVAL_BATCH_SIZE_PA,
        lr=LR_PA
    )

    print("\n============= ALL DATASETS DONE SUCCESSFULLY =============\n")
