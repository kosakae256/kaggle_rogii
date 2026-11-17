import os
import gc
import time
import warnings
import multiprocessing
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold
from scipy.signal import savgol_filter
from tqdm import tqdm

# Modular custom imports
import utils
from model import SCA_U2Net_Reg, UNetRegConfig, deep_supervision_rmse
from dataset import WellSeqDataset

warnings.filterwarnings("ignore")

# =============================================================================
# Execution Configuration
# =============================================================================

# Environment auto-detection (local-first; robust on Windows where "/kaggle/..."
# resolves to "C:\kaggle\..." and can false-positive).
try:
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:  # ノートブックセル実行時は __file__ が無い
    _PROJECT_ROOT = Path.cwd().parent
_LOCAL_DATASET = _PROJECT_ROOT / "rogii-wellbore-geology-prediction"

if _LOCAL_DATASET.exists():
    IS_KAGGLE = False
    _DATASET_PATH = _LOCAL_DATASET
    _ARTIFACTS_PATH = _PROJECT_ROOT / "artifacts_v11"  # CLAUDE.md: 成果物は artifacts_vN/
else:
    IS_KAGGLE = True
    _DATASET_PATH = Path("/kaggle/input/notebooks/santoshpatil0262/notebookb8e6f73167")
    _ARTIFACTS_PATH = Path("/kaggle/working/artifacts")

class CFG:
    dataset_path = _DATASET_PATH
    artifacts_path = _ARTIFACTS_PATH  # Output weights dir
    n_splits = 5
    cv = GroupKFold(n_splits=5)

class TrainConfig:
    window_size: int = 512
    stride: int = 256
    batch_size: int = 32
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 7
    n_splits: int = 5
    seed: int = 49

SEED = 8
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

NCPU = max(1, min(18, multiprocessing.cpu_count() - 4))
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

UCFG = UNetRegConfig()
TRCFG = TrainConfig()

# Artifact paths
CFG.artifacts_path.mkdir(parents=True, exist_ok=True)
FEAT_CACHE = str(CFG.artifacts_path / "sca_u2net_features_cache.pkl")
MODEL_DIR = str(CFG.artifacts_path / "models_sca_u2net")
NORM_FILE = str(CFG.artifacts_path / "sca_u2net_norm_stats.pkl")

# =============================================================================
# Parallel Feature Builder Utilities
# =============================================================================

def _build_one_well_safe(hw_path, tw_path, fi, di):
    try: 
        return utils.build_well_features(hw_path, tw_path, is_train=True, fi_imputer=fi, di_imputer=di)
    except Exception as e:
        print(f"\n[ERROR] Failed to compile features for well {hw_path}:")
        traceback.print_exc()
        return None

def build_all_features(train_dir, cache_path=None, n_jobs=None):
    if cache_path and Path(cache_path).exists():
        print(f"  Loading cached features from {cache_path}...")
        return joblib.load(cache_path)

    hw_paths = sorted(train_dir.glob('*__horizontal_well.csv'))
    all_wids = [p.stem.replace('__horizontal_well', '') for p in hw_paths]
    print(f"  Building spatial imputers ({len(all_wids)} wells)...")
    
    fi = utils.FormationPlaneKNN(all_wids, train_dir)
    di = utils.DenseANCCImputer(all_wids, train_dir)

    # Trigger JIT warm-up sequentially in parent process
    utils.warm_up_jit_compilers()

    args = []
    for hw_p in hw_paths:
        wid = hw_p.stem.replace('__horizontal_well', '')
        tw_p = train_dir / f'{wid}__typewell.csv'
        if tw_p.exists(): args.append((str(hw_p), str(tw_p)))

    if n_jobs is None: n_jobs = NCPU
    print(f"  Building features for {len(args)} wells (n_jobs={n_jobs})...")
    t0 = time.time()

    if n_jobs <= 1:
        results = []
        for hp, tp in tqdm(args, desc="Feature Engineering"):
            r = utils.build_well_features(hp, tp, is_train=True, fi_imputer=fi, di_imputer=di)
            if r is not None: results.append(r)
    else:
        from joblib import Parallel, delayed
        generator = Parallel(n_jobs=n_jobs, return_as="generator", prefer="processes")(
            delayed(_build_one_well_safe)(hp, tp, fi, di) for hp, tp in args
        )
        
        results = []
        for res in tqdm(generator, total=len(args), desc="Feature Engineering"):
            if res is not None:
                results.append(res)

    print(f"  Total: {len(results)} wells built in {time.time()-t0:.0f}s")
    if cache_path:
        joblib.dump(results, cache_path, compress=3)
        print(f"  Cached to {cache_path}")
    return results

# =============================================================================
# Pipeline Inference Modules
# =============================================================================

def predict_well_oof(model, well_data, norm_mean, norm_std, device, window_size=512, stride=256):
    model.eval()
    feat_raw = well_data['features']; L = len(feat_raw)
    feat = ((feat_raw - norm_mean) / norm_std).astype(np.float32); feat_t = feat.T

    if L <= window_size:
        pad_len = window_size - L
        feat_pad = np.pad(feat_t, ((0, 0), (0, pad_len)), mode='constant')
        with torch.no_grad():
            x = torch.from_numpy(feat_pad).unsqueeze(0).to(device)
            fuse, _ = model(x)
        return fuse[0, :L].cpu().numpy()

    starts = list(range(0, L - window_size + 1, stride))
    if starts[-1] + window_size < L: starts.append(L - window_size)

    pred_sum = np.zeros(L, dtype=np.float64); pred_cnt = np.zeros(L, dtype=np.float64)
    VAL_BATCH = 16
    with torch.no_grad():
        for bi in range(0, len(starts), VAL_BATCH):
            batch_starts = starts[bi:bi+VAL_BATCH]; bs = len(batch_starts)
            x_batch = np.stack([feat_t[:, s:s+window_size] for s in batch_starts])
            x = torch.from_numpy(x_batch).to(device)
            fuse, _ = model(x); out_np = fuse.cpu().numpy()
            for j, s in enumerate(batch_starts):
                pred_sum[s:s+window_size] += out_np[j]
                pred_cnt[s:s+window_size] += 1.0

    return (pred_sum / np.maximum(pred_cnt, 1.0)).astype(np.float32)

def postprocess_predictions(pred_delta, md_since, sg_win=61, sg_poly=3, apply_sg=True):
    if not apply_sg or len(pred_delta) < sg_win: return pred_delta
    try: return savgol_filter(pred_delta, sg_win, sg_poly).astype(np.float32)
    except Exception: return pred_delta

def predict_test_well(models, well_data, norm_mean, norm_std, device, window_size=512, stride=256, apply_sg=True):
    n_models = len(models); feat_raw = well_data['features']
    md_since = well_data['md_since']; last_tvt = well_data['last_tvt']
    L = len(feat_raw)

    all_fold_preds = np.zeros((n_models, L), dtype=np.float64)
    for fi, model in enumerate(models):
        model.to(device)
        pred_delta = predict_well_oof(model, well_data, norm_mean, norm_std, device, window_size, stride)
        all_fold_preds[fi] = pred_delta; model.cpu()

    ensemble_delta = all_fold_preds.mean(axis=0).astype(np.float32)
    ensemble_delta = postprocess_predictions(ensemble_delta, md_since, apply_sg=apply_sg)
    return ensemble_delta + last_tvt, ensemble_delta

# =============================================================================
# Cross-Validation & Model Training
# =============================================================================

def train_one_fold(fold, train_data, val_data, norm_mean, norm_std, device, cfg_t=None, cfg_u=None):
    cfg_t = cfg_t or TRCFG; cfg_u = cfg_u or UCFG
    fold_seed = cfg_t.seed + fold * 1000
    torch.manual_seed(fold_seed); np.random.seed(fold_seed)

    train_ds = WellSeqDataset(train_data, cfg_t.window_size, cfg_t.stride, norm_mean, norm_std)
    g = torch.Generator(); g.manual_seed(fold_seed)
    train_dl = DataLoader(train_ds, batch_size=cfg_t.batch_size, shuffle=True, pin_memory=True, drop_last=True, generator=g)

    model = SCA_U2Net_Reg(cfg_u).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg_t.lr, weight_decay=cfg_t.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg_t.epochs, eta_min=1e-6)

    best_val_rmse = float('inf'); best_state = None; patience_counter = 0

    for epoch in range(cfg_t.epochs):
        model.train(); train_loss_sum = 0.0; train_n = 0
        for batch in train_dl:
            feat_b, tgt_b, mask_b, pos_b = [t.to(device) for t in batch]
            optimizer.zero_grad()
            x_ch = feat_b.transpose(1, 2)
            fuse, sides = model(x_ch)
            loss = deep_supervision_rmse(fuse, sides, tgt_b, mask_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_sum += loss.item() * mask_b.sum().item()
            train_n += mask_b.sum().item()

        scheduler.step(); train_rmse = train_loss_sum / max(train_n, 1)

        model.eval(); all_true = []; all_pred = []
        for d in val_data:
            pred_delta = predict_well_oof(model, d, norm_mean, norm_std, device, cfg_t.window_size, cfg_t.stride)
            all_true.append(d['target']); all_pred.append(pred_delta)
        val_rmse = float(np.sqrt(np.mean((np.concatenate(all_true) - np.concatenate(all_pred)) ** 2)))

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0 or epoch == 0 or patience_counter == 0:
            print(f"    [UNet] Fold {fold} Epoch {epoch+1:02d}: train_rmse={train_rmse:.4f}, val_rmse={val_rmse:.4f}, best={best_val_rmse:.4f}")

        if patience_counter >= cfg_t.patience:
            print(f"    [UNet] Early stop at epoch {epoch+1}")
            break

    model.load_state_dict(best_state); model.eval()
    val_preds_best = {d['wid']: predict_well_oof(model, d, norm_mean, norm_std, device, cfg_t.window_size, cfg_t.stride) for d in val_data}
    return model, best_val_rmse, val_preds_best

def run_cv_training(all_well_data, device, cfg_t=None, cfg_u=None):
    cfg_t = cfg_t or TRCFG; cfg_u = cfg_u or UCFG; n_splits = cfg_t.n_splits
    wids = np.array([d['wid'] for d in all_well_data])
    gkf = GroupKFold(n_splits=n_splits); indices = np.arange(len(all_well_data))

    all_feats = np.concatenate([d['features'] for d in all_well_data], axis=0)
    global_mean = all_feats.mean(axis=0).astype(np.float32)
    global_std = np.maximum(all_feats.std(axis=0), 1e-6).astype(np.float32)
    del all_feats; gc.collect()

    model_dir = Path(MODEL_DIR); model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({'mean': global_mean, 'std': global_std}, NORM_FILE)

    models = []; oof_preds = {}; fold_rmses = []
    for fold, (train_idx, val_idx) in enumerate(gkf.split(indices, groups=wids)):
        print(f"\n  --- [UNet] Fold {fold} ({len(train_idx)} train, {len(val_idx)} val wells) ---")
        train_data = [all_well_data[i] for i in train_idx]
        val_data = [all_well_data[i] for i in val_idx]
        ckpt_path = model_dir / f"sca_u2net_fold{fold}.pt"

        if ckpt_path.exists():
            print(f"    [RESUME] Loading checkpoint: {ckpt_path}")
            model = SCA_U2Net_Reg(cfg_u).to(device)
            model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
            model.eval()
            val_preds = {d['wid']: predict_well_oof(model, d, global_mean, global_std, device, cfg_t.window_size, cfg_t.stride) for d in val_data}
            val_rmse = float(np.sqrt(np.mean((np.concatenate([d['target'] for d in val_data]) - np.concatenate(list(val_preds.values()))) ** 2)))
        else:
            model, val_rmse, val_preds = train_one_fold(fold, train_data, val_data, global_mean, global_std, device, cfg_t, cfg_u)
            torch.save(model.state_dict(), ckpt_path)

        models.append(model); fold_rmses.append(val_rmse); oof_preds.update(val_preds)

    cv_rmse_tvt = float(np.sqrt(np.mean((np.concatenate([d['target'] + d['last_tvt'] for d in all_well_data]) - np.concatenate([oof_preds[d['wid']] + d['last_tvt'] for d in all_well_data])) ** 2)))
    print(f"\n  SCA-U2Net CV (TVT): {cv_rmse_tvt:.4f}")
    return models, oof_preds, global_mean, global_std, cv_rmse_tvt

# =============================================================================
# Test Inference Block & Data Integrator
# =============================================================================

def build_test_features(test_dir, train_dir):
    test_paths = sorted(test_dir.glob('*__horizontal_well.csv'))
    train_wids = set(p.stem.replace('__horizontal_well', '') for p in train_dir.glob('*__horizontal_well.csv'))

    hw_paths = sorted(train_dir.glob('*__horizontal_well.csv'))
    all_wids = [p.stem.replace('__horizontal_well', '') for p in hw_paths]
    print(f"  Building spatial imputers ({len(all_wids)} wells)...")
    fi = utils.FormationPlaneKNN(all_wids, train_dir)
    di = utils.DenseANCCImputer(all_wids, train_dir)

    results = []
    for p in test_paths:
        wid = p.stem.replace('__horizontal_well', '')
        tw_p = test_dir / f'{wid}__typewell.csv'

        if not tw_p.exists() and wid in train_wids: tw_p = train_dir / f'{wid}__typewell.csv'
        if not tw_p.exists(): continue

        hw_test = pd.read_csv(p)
        if wid in train_wids:
            hw_train = pd.read_csv(train_dir / f'{wid}__horizontal_well.csv')
            hw_test['TVT_input'] = hw_train['TVT_input'].values
            tw_p = train_dir / f'{wid}__typewell.csv'

        tmp_path = str(Path('.') / f'{wid}__horizontal_well.csv')
        hw_test.to_csv(tmp_path, index=False)
        r = utils.build_well_features(tmp_path, str(tw_p), is_train=False, fi_imputer=fi, di_imputer=di)
        os.unlink(tmp_path)

        if r is not None:
            r['wid'] = wid; r['ev_indices'] = hw_test[hw_test['TVT_input'].isna()].index.tolist()
            results.append(r)
    return results

def run_test_inference(models, norm_mean, norm_std, device, train_dir, test_dir, window_size=512, stride=256, apply_sg=True):
    print("\n─── Test Inference ───")
    test_well_data = build_test_features(test_dir, train_dir)
    if not test_well_data: return None

    test_predictions = []
    for wd in test_well_data:
        pred_tvt, _ = predict_test_well(models, wd, norm_mean, norm_std, device, window_size, stride, apply_sg)
        test_predictions.append(pred_tvt)

    rows = []
    for wd, pred_tvt in zip(test_well_data, test_predictions):
        wid = wd['wid']; ev_indices = wd['ev_indices']
        for i, row_idx in enumerate(ev_indices): rows.append({'id': f"{wid}_{row_idx}", 'tvt': float(pred_tvt[i])})
    sub_df = pd.DataFrame(rows)

    sample_path = CFG.dataset_path / "sample_submission.csv"
    if sample_path.exists():
        sub_df = pd.read_csv(sample_path)[['id']].merge(sub_df, on='id', how='left')
        sub_df['tvt'] = sub_df['tvt'].fillna(sub_df['tvt'].median())

    sub_df[['id', 'tvt']].to_csv("submission.csv", index=False)
    print(f"  Submission saved: submission.csv ({len(sub_df)} rows)")
    return sub_df

# =============================================================================
# Main Pipeline Coordinator
# =============================================================================

if __name__ == "__main__":
    train_dir = CFG.dataset_path / "train"
    if not train_dir.exists():
        print(f"[ERROR] Training data not found at {train_dir}")
        raise SystemExit(1)

    print("─── Step 1: Feature Engineering ───")
    all_well_data = build_all_features(train_dir, cache_path=FEAT_CACHE)
    print(f"  Total wells with valid features: {len(all_well_data)}")

    print("\n─── Step 2: 5-Fold Cross-Validation ───")
    models, oof_preds, norm_mean, norm_std, cv_rmse = run_cv_training(all_well_data, DEVICE, TRCFG, UCFG)

    test_dir = CFG.dataset_path / "test"
    if test_dir.exists():
        run_test_inference(models, norm_mean, norm_std, DEVICE, train_dir, test_dir, TRCFG.window_size, TRCFG.stride, apply_sg=True)
    else:
        print("\n[INFO] Test data directory not detected. Skipping inference.") 
