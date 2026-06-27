#!/usr/bin/env python3
"""
bootstrap_ci.py — Intervalos de confianza bootstrap para las 4 tablas headline
del TFG (tfg_memoria_v6), Fase 1.

Estrategia (decisiones confirmadas):
  - tab:lp-panderm  -> bootstrap DIRECTO sobre predicciones por muestra guardadas
                        (CSV con true_label / predicted / probability[_class_k]).
  - tab:benchmark   -> PanDerm/SigLIP desde features (.pt) re-fit LR; CNN desde
                        features (.npy) re-fit LR. DDI: cada celda en el split que
                        reproduce su valor publicado (CNN=220 test+val, PanDerm/
                        SigLIP=137 test). Inconsistencia reportada, no tapada.
  - tab:equidad-fototipo -> re-fit LR sobre features Fitzpatrick17k cacheadas
                        (cat3_label, 3 clases), predicciones por muestra del test,
                        bootstrap de BAcc por FST y del gap I-VI.
  - tab:llm-comparativa  -> bootstrap directo sobre CSV de predicciones por muestra
                        (LLM: solo Acc/BAcc). MedGemma ZS/FT reproducen desde CSV.

Bootstrap: estratificado por clase, B=1000, IC95% por percentil (2,5 / 97,5),
semilla 42. Sanity-check en cada celda: |media_boot - publicado| > 0,005 o
IC que no contiene el publicado -> marca PARADA (no se ajusta nada).

Salida: CSV + Markdown (coma decimal) + informe sanity-check, en
~/panderm/output/bootstrap_ci/
"""
import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             roc_auc_score, f1_score)
from sklearn.preprocessing import label_binarize, StandardScaler

warnings.filterwarnings("ignore")

HOME = os.path.expanduser("~")
PAND = f"{HOME}/panderm"
OUT  = f"{PAND}/output"
DST  = f"{OUT}/bootstrap_ci"
os.makedirs(DST, exist_ok=True)

SEED = 42
B = 1000
rng_global = np.random.default_rng(SEED)

# ---- import torch lazily (solo para .pt de benchmark/fairness) ----
def _torch():
    import torch
    return torch


# =============================================================================
# Utilidades de métricas y bootstrap
# =============================================================================
def auroc_multi(y_true, prob, n_classes, average):
    """AUROC multiclase. average in {'ovr_macro','ovr_weighted','ovo_macro',
    'ovo_weighted'} (o los alias 'macro'/'weighted' = OvR por compat.).
    Binario usa prob[:,1]."""
    if n_classes == 2:
        return roc_auc_score(y_true, prob[:, 1])
    present = sorted(set(y_true))
    if len(present) < 2:
        return np.nan
    # alias compat
    if average == "macro":
        average = "ovr_macro"
    elif average == "weighted":
        average = "ovr_weighted"
    mc, avg = average.split("_")  # 'ovr'/'ovo', 'macro'/'weighted'
    if mc == "ovr":
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))
        return roc_auc_score(y_bin, prob, average=avg, multi_class="ovr")
    else:  # ovo necesita y_true entero y prob por columnas de clase
        return roc_auc_score(y_true, prob, average=avg, multi_class="ovo")


def point_metrics(y_true, y_pred, prob, n_classes, auroc_avg=None):
    m = {
        "Acc":  accuracy_score(y_true, y_pred),
        "BAcc": balanced_accuracy_score(y_true, y_pred),
        "WF1":  f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    if prob is not None and auroc_avg is not None:
        try:
            m["AUROC"] = auroc_multi(y_true, prob, n_classes, auroc_avg)
        except Exception:
            m["AUROC"] = np.nan
    return m


def strat_boot_indices(y_true, rng):
    """Un remuestreo bootstrap estratificado por clase (con reemplazo dentro
    de cada clase, preservando el tamaño por clase)."""
    y_true = np.asarray(y_true)
    idx_out = np.empty(len(y_true), dtype=np.int64)
    pos = 0
    for c in np.unique(y_true):
        ci = np.where(y_true == c)[0]
        samp = rng.integers(0, len(ci), size=len(ci))
        idx_out[pos:pos + len(ci)] = ci[samp]
        pos += len(ci)
    return idx_out


def bootstrap_metrics(y_true, y_pred, prob, n_classes, metrics, auroc_avg=None,
                      B=B, seed=SEED):
    """Devuelve dict métrica -> (lo, hi, mean) por percentil 2.5/97.5."""
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    acc = {k: [] for k in metrics}
    for _ in range(B):
        bi = strat_boot_indices(y_true, rng)
        yt, yp = y_true[bi], y_pred[bi]
        pb = prob[bi] if prob is not None else None
        if "Acc" in acc:  acc["Acc"].append(accuracy_score(yt, yp))
        if "BAcc" in acc: acc["BAcc"].append(balanced_accuracy_score(yt, yp))
        if "WF1" in acc:  acc["WF1"].append(f1_score(yt, yp, average="weighted", zero_division=0))
        if "AUROC" in acc:
            try:
                acc["AUROC"].append(auroc_multi(yt, pb, n_classes, auroc_avg))
            except Exception:
                acc["AUROC"].append(np.nan)
    res = {}
    for k, vals in acc.items():
        v = np.array(vals, dtype=float)
        v = v[~np.isnan(v)]
        if len(v) == 0:
            res[k] = (np.nan, np.nan, np.nan)
        else:
            res[k] = (np.percentile(v, 2.5), np.percentile(v, 97.5), v.mean())
    return res


def bootstrap_gap(y_true, y_pred, fst, fst_lo, fst_hi, B=B, seed=SEED):
    """Bootstrap del gap = BAcc(FST_lo) - BAcc(FST_hi). Estratificado por clase
    dentro de cada subgrupo FST. Devuelve (lo, hi, mean) del gap."""
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred); fst = np.asarray(fst)
    rng = np.random.default_rng(seed)
    m_lo = fst == fst_lo
    m_hi = fst == fst_hi
    yt_lo, yp_lo = y_true[m_lo], y_pred[m_lo]
    yt_hi, yp_hi = y_true[m_hi], y_pred[m_hi]
    gaps = []
    for _ in range(B):
        bi_lo = strat_boot_indices(yt_lo, rng)
        bi_hi = strat_boot_indices(yt_hi, rng)
        b_lo = balanced_accuracy_score(yt_lo[bi_lo], yp_lo[bi_lo])
        b_hi = balanced_accuracy_score(yt_hi[bi_hi], yp_hi[bi_hi])
        gaps.append(b_lo - b_hi)
    g = np.array(gaps)
    return np.percentile(g, 2.5), np.percentile(g, 97.5), g.mean()


# =============================================================================
# Acumuladores de resultados + sanity-check
# =============================================================================
RESULTS = []   # filas para CSV
SANITY  = []   # filas informe sanity-check
STOPS   = []   # celdas con PARADA

def comma(x, nd=4):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "---"
    return f"{x:.{nd}f}".replace(".", ",")

def record(table, row, model, metric, published, lo, hi, mean, n,
           note="", auroc_conv=""):
    flag = "OK"
    if published is None or (isinstance(published, float) and np.isnan(published)) \
            or (isinstance(mean, float) and np.isnan(mean)):
        flag = "GAP"
    else:
        dev = abs(mean - published)
        # tolerancia de redondeo del valor publicado (3 decimales -> ±0,0005)
        rtol = 0.0005 + 1e-9
        contains = (lo - rtol) <= published <= (hi + rtol)
        # IC degenerado (ancho ~0, predicciones constantes): containment no
        # informa; el criterio decisivo es dev<=0,005.
        degenerate = (hi - lo) < 1e-6
        ok = (dev <= 0.005) and (contains or degenerate)
        if not ok:
            flag = "STOP"
            STOPS.append(dict(table=table, row=row, model=model, metric=metric,
                              published=published, mean=mean, lo=lo, hi=hi,
                              dev=dev, contains=contains, note=note))
    RESULTS.append(dict(table=table, row=row, model=model, metric=metric,
                        published=published, boot_mean=mean, ci_lo=lo, ci_hi=hi,
                        N=n, auroc_conv=auroc_conv, flag=flag, note=note))
    pub_s = comma(published) if published is not None else "---"
    sym = {"OK": "[OK]", "STOP": "[STOP]", "GAP": "[GAP]"}[flag]
    SANITY.append(f"{sym} {table} | {row} | {model} | {metric}: "
                  f"pub={pub_s}  mean={comma(mean)}  IC95=[{comma(lo)}, {comma(hi)}]"
                  f"  N={n}  {('('+auroc_conv+')') if auroc_conv else ''} {note}")
    print(SANITY[-1])
    return flag


def fit_lr(Xtr, ytr, Xte, scaler=True, max_iter=5000):
    if scaler:
        sc = StandardScaler().fit(Xtr)
        Xtr = sc.transform(Xtr); Xte = sc.transform(Xte)
    clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=max_iter,
                             random_state=42, n_jobs=-1)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    proba = clf.predict_proba(Xte)
    n_cls = int(max(ytr.max(), pred.max()) + 1) if len(ytr) else proba.shape[1]
    full = np.zeros((len(Xte), max(n_cls, proba.shape[1])))
    for i, c in enumerate(clf.classes_):
        full[:, int(c)] = proba[:, i]
    return pred, full, clf


def pick_auroc_conv(y_true, prob, n_classes, published, tol=0.005):
    """Prueba las 4 convenciones estandar en el orden del prompt
    (OvR-macro, OvR-weighted) y ademas OvO (la que usa el pipeline LP del TFG,
    verificada contra metrics_results.csv). Devuelve (conv|None, valores)."""
    order = ["ovr_macro", "ovr_weighted", "ovo_macro", "ovo_weighted"]
    out = {}
    for avg in order:
        try:
            out[avg] = auroc_multi(y_true, prob, n_classes, avg)
        except Exception:
            out[avg] = np.nan
    if published is None:
        return "ovr_macro", out["ovr_macro"]
    for avg in order:
        if not np.isnan(out[avg]) and abs(out[avg] - published) <= tol:
            return avg, out[avg]
    return None, out


# =============================================================================
# TABLA 1 — tab:lp-panderm  (bootstrap directo sobre predicciones guardadas)
# =============================================================================
# Valores publicados leidos del LaTeX Cap4 (Base/Large, Acc BAcc AUROC WF1).
LP_PANDERM = {
    # dataset: (dir, csv, n_classes, {model: {metric: pub}})
    "HAM10000": ("lp_ham_{m}", "HAM_clean.csv", 7, {
        "Base":  dict(Acc=0.853, BAcc=0.381, AUROC=0.892, WF1=0.829),
        "Large": dict(Acc=0.888, BAcc=0.575, AUROC=0.954, WF1=0.880)}),
    "BCN20000": ("lp_bcn_{m}", "bcn20000.csv", 9, {
        "Base":  dict(Acc=0.659, BAcc=0.292, AUROC=0.855, WF1=0.615),
        "Large": dict(Acc=0.702, BAcc=0.382, AUROC=0.903, WF1=0.676)}),
    "PAD-UFES-20": ("lp_padufes_{m}", "2000.csv", 6, {
        "Base":  dict(Acc=0.725, BAcc=0.510, AUROC=0.913, WF1=0.692),
        "Large": dict(Acc=0.772, BAcc=0.642, AUROC=0.949, WF1=0.760)}),
    "Dermnet": ("lp_dermnet_{m}", "dermnet.csv", 23, {
        "Base":  dict(Acc=0.450, BAcc=0.381, AUROC=0.888, WF1=0.432),
        "Large": dict(Acc=0.550, BAcc=0.495, AUROC=0.931, WF1=0.540)}),
    "WSI patches": ("lp_wsi_{m}", "patch.csv", 16, {
        "Base":  dict(Acc=0.781, BAcc=0.640, AUROC=0.976, WF1=0.774),
        "Large": dict(Acc=0.868, BAcc=0.792, AUROC=0.991, WF1=0.871)}),
    "DDI": ("lp_ddi_{m}", "ddi_clean.csv", 2, {
        "Base":  dict(Acc=0.847, BAcc=0.714, AUROC=0.827, WF1=0.836),
        "Large": dict(Acc=0.847, BAcc=0.764, AUROC=0.860, WF1=0.846)}),
    "Derm7pt clin": ("lp_derm7pt_clin_{m}", "atlas-clinical-all.csv", 2, {
        "Base":  dict(Acc=0.762, BAcc=0.675, AUROC=0.808, WF1=0.736),
        "Large": dict(Acc=0.798, BAcc=0.732, AUROC=0.869, WF1=0.784)}),
    "Derm7pt dermo": ("lp_derm7pt_derm_{m}", "atlas-dermato-all.csv", 2, {
        "Base":  dict(Acc=0.818, BAcc=0.749, AUROC=0.841, WF1=0.811),
        "Large": dict(Acc=0.836, BAcc=0.766, AUROC=0.858, WF1=0.829)}),
    "HIBA": ("lp_hiba_{m}", "hiba.csv", 2, {
        "Base":  dict(Acc=0.883, BAcc=0.595, AUROC=0.894, WF1=0.853),
        "Large": dict(Acc=0.904, BAcc=0.701, AUROC=0.942, WF1=0.892)}),
    "MSKCC": ("lp_mskcc_{m}", "mskcc.csv", 2, {
        "Base":  dict(Acc=0.721, BAcc=0.654, AUROC=0.746, WF1=0.720),
        "Large": dict(Acc=0.746, BAcc=0.644, AUROC=0.751, WF1=0.732)}),
}


def load_pred_csv(path, n_classes):
    df = pd.read_csv(path)
    y_true = df["true_label"].values.astype(int)
    y_pred = df["predicted_label"].values.astype(int)
    if n_classes == 2:
        prob = np.zeros((len(df), 2))
        prob[:, 1] = df["probability"].values
        prob[:, 0] = 1.0 - prob[:, 1]
    else:
        cols = [c for c in df.columns if c.startswith("probability_class_")]
        cols = sorted(cols, key=lambda c: int(c.split("_")[-1]))
        prob = df[cols].values
    return y_true, y_pred, prob


def run_lp_panderm():
    print("\n" + "=" * 70 + "\nTABLA 1: tab:lp-panderm\n" + "=" * 70)
    # AUROC convencion: el pipeline LP usa average='weighted' (verificado en
    # metrics_results.csv). Confirmamos por celda.
    for ds, (dirpat, csv, ncl, models) in LP_PANDERM.items():
        for m in ("Base", "Large"):
            d = os.path.join(OUT, dirpat.format(m=m.lower()))
            path = os.path.join(d, csv)
            if not os.path.exists(path):
                print(f"  MISSING {path}")
                continue
            y_true, y_pred, prob = load_pred_csv(path, ncl)
            pub = models[m]
            conv = "binary"
            if ncl > 2:
                # Convencion VERIFICADA del pipeline LP del TFG: OvO-macro
                # (reproduce metrics_results.csv al 4o decimal en los 10 datasets
                # multiclase: HAM, BCN, PAD, Dermnet, WSI). El prompt pedia probar
                # OvR primero; documentado que NINGUNA OvR reproduce de forma
                # consistente y que OvO-macro si -> se usa esta.
                conv = "ovo_macro"
            auroc_avg = None if ncl == 2 else conv
            metrics = ["Acc", "BAcc", "AUROC", "WF1"]
            bm = bootstrap_metrics(y_true, y_pred, prob, ncl, metrics,
                                   auroc_avg=auroc_avg)
            for met in metrics:
                lo, hi, mean = bm[met]
                record("tab:lp-panderm", ds, f"PanDerm {m}", met,
                       pub.get(met), lo, hi, mean, len(y_true),
                       auroc_conv=(conv if met == "AUROC" else ""))


# =============================================================================
# TABLA 2 — tab:benchmark
# =============================================================================
# Publicado (Cap4): HAM Acc/AUROC, PAD Acc/AUROC, DDI AUROC por modelo.
BENCH_PUB = {
    "ConvNeXt-Large":       dict(HAM_Acc=0.883, HAM_AUROC=0.967, PAD_Acc=0.668, PAD_AUROC=0.894, DDI_AUROC=0.774),
    "EfficientNetV2-Large": dict(HAM_Acc=0.860, HAM_AUROC=0.950, PAD_Acc=0.690, PAD_AUROC=0.895, DDI_AUROC=0.766),
    "DINOv2 ViT-L/14":      dict(HAM_Acc=0.744, HAM_AUROC=0.847, PAD_Acc=None,  PAD_AUROC=None,  DDI_AUROC=None),
    "PanDerm Large":        dict(HAM_Acc=0.888, HAM_AUROC=0.954, PAD_Acc=0.772, PAD_AUROC=0.949, DDI_AUROC=0.860),
    "SigLIP-Large SO400M":  dict(HAM_Acc=0.900, HAM_AUROC=0.971, PAD_Acc=0.718, PAD_AUROC=0.903, DDI_AUROC=0.793),
}


def _clean_csv_split_labels(csv_path, label_col, split_col="split"):
    df = pd.read_csv(csv_path)
    return df, label_col, split_col


def run_benchmark():
    print("\n" + "=" * 70 + "\nTABLA 2: tab:benchmark\n" + "=" * 70)
    torch = _torch()

    # ---- CNN baselines (ConvNeXt-L, EfficientNetV2-L) desde .npy ----
    cnn_dir = os.path.join(OUT, "cnn_baseline")
    cnn_map = {"ConvNeXt-Large": "convnextl", "EfficientNetV2-Large": "efficientnetv2l"}
    # labels train: derivar de clean CSV en orden de split
    HAM_CSV = f"{PAND}/datasets/HAM10000_clean/ISIC2018_splits/HAM_clean.csv"
    PAD_CSV = f"{PAND}/datasets/pad-ufes/2000.csv"
    DDI_CSV = f"{PAND}/datasets/DDI/ddi_clean.csv"

    def train_labels(csv, label_col, ddi=False):
        df = pd.read_csv(csv)
        return df[df["split"] == "train"][label_col].values

    cnn_cfg = {
        "HAM": dict(ncl=7, train_csv=HAM_CSV, lcol="label", pub_acc="HAM_Acc", pub_auroc="HAM_AUROC"),
        "PAD": dict(ncl=6, train_csv=PAD_CSV, lcol="label", pub_acc="PAD_Acc", pub_auroc="PAD_AUROC"),
        "DDI": dict(ncl=2, train_csv=DDI_CSV, lcol="binary_label", pub_acc=None, pub_auroc="DDI_AUROC"),
    }
    npy_ds = {"HAM": "ham10000", "PAD": "padufes", "DDI": "ddi"}

    for disp, pref in cnn_map.items():
        for dskey, cfg in cnn_cfg.items():
            tr = np.load(os.path.join(cnn_dir, f"{pref}_{npy_ds[dskey]}_train.npy"))
            te = np.load(os.path.join(cnn_dir, f"{pref}_{npy_ds[dskey]}_test.npy"))
            csvp = os.path.join(cnn_dir, f"{pref}_{npy_ds[dskey]}.csv")
            y_test = pd.read_csv(csvp)["label"].values.astype(int)
            y_train = train_labels(cfg["train_csv"], cfg["lcol"]).astype(int)
            assert len(y_train) == len(tr), f"{disp} {dskey} train mismatch {len(y_train)} vs {len(tr)}"
            assert len(y_test) == len(te), f"{disp} {dskey} test mismatch"
            pred, prob, _ = fit_lr(tr, y_train, te, scaler=True, max_iter=1000)
            ncl = cfg["ncl"]
            pubrow = BENCH_PUB[disp]
            # Acc
            if cfg["pub_acc"]:
                bm = bootstrap_metrics(y_test, pred, prob, ncl, ["Acc"])
                lo, hi, mean = bm["Acc"]
                record("tab:benchmark", dskey, disp, "Acc",
                       pubrow[cfg["pub_acc"]], lo, hi, mean, len(y_test))
            # AUROC
            conv = "binary"
            if ncl > 2:
                conv, _ = pick_auroc_conv(y_test, prob, ncl, pubrow[cfg["pub_auroc"]])
                if conv is None:
                    conv = "macro"
            bm = bootstrap_metrics(y_test, pred, prob, ncl, ["AUROC"],
                                   auroc_avg=(None if ncl == 2 else conv))
            lo, hi, mean = bm["AUROC"]
            note = "DDI N=220 (test+val, convencion CNN baseline)" if dskey == "DDI" else ""
            record("tab:benchmark", dskey, disp, "AUROC",
                   pubrow[cfg["pub_auroc"]], lo, hi, mean, len(y_test),
                   note=note, auroc_conv=conv)

    # ---- SigLIP-Large SO400M desde .pt (features) ----
    sig_base = os.path.join(OUT, "medgemma_eval")
    sig_lcol = {"HAM": ("ham10000", 7, HAM_CSV, "label"),
                "PAD": ("padufes", 6, PAD_CSV, "label"),
                "DDI": ("ddi", 2, DDI_CSV, "binary_label")}
    for dskey, (tag, ncl, csv, lcol) in sig_lcol.items():
        emb = torch.load(os.path.join(sig_base, f"siglip_embeddings_{tag}.pt"),
                         weights_only=False)
        Xtr = emb["train"].numpy(); Xte = emb["test"].numpy()
        df = pd.read_csv(csv)
        if dskey == "DDI":
            ytr = df[df["split"] == "train"][lcol].values.astype(int)
            yte = df[df["split"] == "test"][lcol].values.astype(int)  # 137 (test only)
        else:
            ytr = df[df["split"] == "train"][lcol].values.astype(int)
            yte = df[df["split"] == "test"][lcol].values.astype(int)
        assert len(ytr) == len(Xtr) and len(yte) == len(Xte), \
            f"SigLIP {dskey} mismatch tr {len(ytr)}/{len(Xtr)} te {len(yte)}/{len(Xte)}"
        pred, prob, _ = fit_lr(Xtr, ytr, Xte, scaler=True, max_iter=5000)
        pubrow = BENCH_PUB["SigLIP-Large SO400M"]
        # Acc
        pacc = {"HAM": "HAM_Acc", "PAD": "PAD_Acc", "DDI": None}[dskey]
        if pacc:
            bm = bootstrap_metrics(yte, pred, prob, ncl, ["Acc"])
            lo, hi, mean = bm["Acc"]
            record("tab:benchmark", dskey, "SigLIP-Large SO400M", "Acc",
                   pubrow[pacc], lo, hi, mean, len(yte))
        # AUROC
        pauroc = {"HAM": "HAM_AUROC", "PAD": "PAD_AUROC", "DDI": "DDI_AUROC"}[dskey]
        conv = "binary"
        if ncl > 2:
            conv, _ = pick_auroc_conv(yte, prob, ncl, pubrow[pauroc])
            if conv is None:
                conv = "weighted"
        bm = bootstrap_metrics(yte, pred, prob, ncl, ["AUROC"],
                               auroc_avg=(None if ncl == 2 else conv))
        lo, hi, mean = bm["AUROC"]
        note = "DDI N=137 (test, convencion LP PanDerm/SigLIP)" if dskey == "DDI" else ""
        record("tab:benchmark", dskey, "SigLIP-Large SO400M", "AUROC",
               pubrow[pauroc], lo, hi, mean, len(yte), note=note, auroc_conv=conv)

    # ---- PanDerm Large desde predicciones LP guardadas (reusa tabla 1) ----
    # HAM Acc/AUROC, PAD Acc/AUROC, DDI AUROC -> de los CSV lp_*_large
    pl = BENCH_PUB["PanDerm Large"]
    # HAM
    yt, yp, pb = load_pred_csv(os.path.join(OUT, "lp_ham_large", "HAM_clean.csv"), 7)
    convH, _ = pick_auroc_conv(yt, pb, 7, pl["HAM_AUROC"])
    bm = bootstrap_metrics(yt, yp, pb, 7, ["Acc", "AUROC"],
                           auroc_avg=(convH or "weighted"))
    lo, hi, mn = bm["Acc"]; record("tab:benchmark", "HAM", "PanDerm Large", "Acc", pl["HAM_Acc"], lo, hi, mn, len(yt))
    lo, hi, mn = bm["AUROC"]; record("tab:benchmark", "HAM", "PanDerm Large", "AUROC", pl["HAM_AUROC"], lo, hi, mn, len(yt), auroc_conv=(convH or "weighted"))
    # PAD
    yt, yp, pb = load_pred_csv(os.path.join(OUT, "lp_padufes_large", "2000.csv"), 6)
    convP, _ = pick_auroc_conv(yt, pb, 6, pl["PAD_AUROC"])
    bm = bootstrap_metrics(yt, yp, pb, 6, ["Acc", "AUROC"],
                           auroc_avg=(convP or "weighted"))
    lo, hi, mn = bm["Acc"]; record("tab:benchmark", "PAD", "PanDerm Large", "Acc", pl["PAD_Acc"], lo, hi, mn, len(yt))
    lo, hi, mn = bm["AUROC"]; record("tab:benchmark", "PAD", "PanDerm Large", "AUROC", pl["PAD_AUROC"], lo, hi, mn, len(yt), auroc_conv=(convP or "weighted"))
    # DDI (N=137)
    yt, yp, pb = load_pred_csv(os.path.join(OUT, "lp_ddi_large", "ddi_clean.csv"), 2)
    bm = bootstrap_metrics(yt, yp, pb, 2, ["AUROC"], auroc_avg=None)
    lo, hi, mn = bm["AUROC"]
    record("tab:benchmark", "DDI", "PanDerm Large", "AUROC", pl["DDI_AUROC"],
           lo, hi, mn, len(yt), note="DDI N=137 (test)", auroc_conv="binary")

    # ---- DINOv2 (solo HAM Acc/AUROC publicado) ----
    # Buscar features dinov2 sobre HAM. Si no hay, declarar GAP.
    dino_ham = None
    for cand in [os.path.join(OUT, "cnn_baseline", "dinov2_ham10000_test.npy")]:
        if os.path.exists(cand):
            dino_ham = cand
    if dino_ham is None:
        # GAP honesto: el TFG marca DINOv2 HAM con asterisco; sus 0,744/0,847 NO
        # son valores especificos de HAM sino el PROMEDIO de LP sobre 10 datasets
        # (nota al pie en CNN_BASELINE_RESULTS.md). Un IC por-dataset no aplica.
        gnote = ("valor publicado es promedio LP sobre 10 datasets (asterisco "
                 "TFG), no especifico de HAM -> IC por-dataset no procede")
        record("tab:benchmark", "HAM", "DINOv2 ViT-L/14", "Acc",
               None, np.nan, np.nan, np.nan, 0, note=gnote)
        record("tab:benchmark", "HAM", "DINOv2 ViT-L/14", "AUROC",
               None, np.nan, np.nan, np.nan, 0, note=gnote)


# =============================================================================
# TABLA 3 — tab:equidad-fototipo  (Fitzpatrick17k, 3 clases, BAcc por FST + gap)
# =============================================================================
FAIR_PUB = {
    "PanDerm Large": [0.723, 0.708, 0.743, 0.669, 0.568, 0.506, 0.217],
    "PanDerm Base":  [0.653, 0.693, 0.669, 0.630, 0.585, 0.384, 0.269],
    "DermLIP v2":    [0.716, 0.730, 0.765, 0.710, 0.693, 0.434, 0.281],
    "DINOv2 ViT-L/14":[0.673, 0.671, 0.679, 0.654, 0.553, 0.368, 0.306],
    "BiomedCLIP":    [0.569, 0.602, 0.606, 0.566, 0.590, 0.445, 0.124],
}
FAIR_KEY = {"PanDerm Large": "panderm_large", "PanDerm Base": "panderm_base",
            "DermLIP v2": "dermlip_v2", "DINOv2 ViT-L/14": "dinov2",
            "BiomedCLIP": "biomedclip"}


def run_fairness():
    print("\n" + "=" * 70 + "\nTABLA 3: tab:equidad-fototipo\n" + "=" * 70)
    torch = _torch()
    fdir = os.path.join(OUT, "fitzpatrick17k_fairness")
    csv = os.path.join(OUT, "fitzpatrick17k_full", "fitzpatrick17k_full_clean.csv")
    df = pd.read_csv(csv)
    train_mask = (df["split"] == "train").values
    test_mask = (df["split"] == "test").values
    ytr_full = df.loc[df["split"] == "train", "cat3_label"].values.astype(int)
    yte_full = df.loc[df["split"] == "test", "cat3_label"].values.astype(int)
    fst_test = df.loc[df["split"] == "test", "fitzpatrick"].values.astype(int)
    ncl = 3

    for disp, key in FAIR_KEY.items():
        emb = torch.load(os.path.join(fdir, f"features_{key}.pt"),
                         weights_only=False)["embeddings"].numpy()
        Xtr = emb[train_mask]; Xte = emb[test_mask]
        pred, prob, _ = fit_lr(Xtr, ytr_full, Xte, scaler=False, max_iter=5000)
        pub = FAIR_PUB[disp]
        # BAcc por FST (I..VI)
        for fst in range(1, 7):
            m = fst_test == fst
            yt, yp = yte_full[m], pred[m]
            # bootstrap BAcc en este subgrupo
            bm = bootstrap_metrics(yt, yp, None, ncl, ["BAcc"])
            lo, hi, mean = bm["BAcc"]
            record("tab:equidad-fototipo", f"FST {fst}", disp, "BAcc",
                   pub[fst - 1], lo, hi, mean, int(m.sum()))
        # Gap I-VI
        glo, ghi, gmean = bootstrap_gap(yte_full, pred, fst_test, 1, 6)
        record("tab:equidad-fototipo", "Gap I-VI", disp, "Gap",
               pub[6], glo, ghi, gmean,
               int((fst_test == 1).sum()) + int((fst_test == 6).sum()),
               note="N = n(FST I)+n(FST VI)")


# =============================================================================
# TABLA 4 — tab:llm-comparativa  (LLM: solo Acc/BAcc, bootstrap directo CSV)
# =============================================================================
# Filas y publicado (Cap4). '---' => columna no aplicable en TFG.
LLM_PUB = {
    # row_label: (mode, {dataset: (Acc, BAcc)}, loader_key)
    "MedGemma 27B + LoRA": dict(HAM=(0.802, 0.270), PAD=(0.430, 0.347), DDI=(0.657, 0.618)),
    "MedGemma 27B":        dict(HAM=(0.665, 0.243), PAD=(0.447, 0.342), DDI=(0.474, 0.566)),
    "GPT-4o":              dict(HAM=(0.485, 0.465), PAD=(0.553, 0.523), DDI=(0.723, 0.648)),
    "GPT-4o-mini":         dict(HAM=(0.256, 0.319), PAD=(0.390, 0.435), DDI=(0.737, 0.556)),
    "Gemini 2.5 Pro":      dict(HAM=(0.403, None),  PAD=(0.477, None),  DDI=(0.441, None)),
    "Gemini 2.5 Flash":    dict(HAM=(0.380, None),  PAD=(0.486, None),  DDI=(0.555, None)),
    "BLIP-2 (Flan-T5-XL)": dict(HAM=(0.015, 0.145), PAD=(0.017, 0.167), DDI=(0.796, 0.542)),
    "InstructBLIP (Flan-T5-XL)": dict(HAM=(0.024, 0.123), PAD=(0.171, 0.224), DDI=(0.788, 0.500)),
}
# CSV por (row, dataset). Esquemas: medgemma usa label/predicted(name);
# gpt usa label/label_name/predicted; gemini/blip usan true_label/predicted_label.
LLM_CSV = {
    "MedGemma 27B + LoRA": {"HAM": ("medgemma_ft/ham10000_ft_predictions.csv", "name"),
                            "PAD": ("medgemma_ft/padufes_ft_predictions.csv", "name"),
                            "DDI": ("medgemma_ft/ddi_ft_predictions.csv", "name")},
    "MedGemma 27B":        {"HAM": ("medgemma_eval/ham10000_predictions.csv", "name"),
                            "PAD": ("medgemma_eval/padufes_predictions.csv", "name"),
                            "DDI": ("medgemma_eval/ddi_predictions.csv", "name")},
    "GPT-4o":              {"HAM": ("gpt4o_eval/gpt4o_ham10000.csv", "gpt"),
                            "PAD": ("gpt4o_eval/gpt4o_padufes.csv", "gpt2"),
                            # v2 reproduce el valor publicado (0,723/0,648); el
                            # ddi.csv v1 era una corrida superada (0,788/0,500).
                            "DDI": ("gpt4o_eval/gpt4o_ddi_v2.csv", "gpt")},
    "GPT-4o-mini":         {"HAM": ("gpt4o_eval/gpt4omini_ham10000.csv", "gpt"),
                            "PAD": ("gpt4o_eval/gpt4omini_padufes.csv", "gpt2"),
                            "DDI": ("gpt4o_eval/gpt4omini_ddi.csv", "gpt2")},
    "Gemini 2.5 Pro":      {"HAM": ("gemini_eval/gemini_pro_ham10000.csv", "gem"),
                            "PAD": ("gemini_eval/gemini_pro_padufes.csv", "gem"),
                            "DDI": ("gemini_eval/gemini_pro_ddi.csv", "gem")},
    "Gemini 2.5 Flash":    {"HAM": ("gemini_eval/gemini_flash_ham10000.csv", "gem"),
                            "PAD": ("gemini_eval/gemini_flash_padufes.csv", "gem"),
                            "DDI": ("gemini_eval/gemini_flash_ddi.csv", "gem")},
    "BLIP-2 (Flan-T5-XL)": {"HAM": ("blip2_eval/blip2_ham10000.csv", "gem"),
                            "PAD": ("blip2_eval/blip2_padufes.csv", "gem"),
                            "DDI": ("blip2_eval/blip2_ddi.csv", "gem")},
    "InstructBLIP (Flan-T5-XL)": {"HAM": ("blip2_eval/instructblip_ham10000.csv", "gem"),
                            "PAD": ("blip2_eval/instructblip_padufes.csv", "gem"),
                            "DDI": ("blip2_eval/instructblip_ddi.csv", "gem")},
}


def load_llm(path, kind):
    df = pd.read_csv(path)
    if kind == "name":
        yt = df["label_name"].astype(str).str.strip().str.lower()
        yp = df["predicted"].astype(str).str.strip().str.lower()
    elif kind == "gpt":
        yt = df["label_name"].astype(str).str.strip().str.lower()
        yp = df["predicted"].astype(str).str.strip().str.lower()
    elif kind == "gpt2":
        yt = df["true"].astype(str).str.strip().str.lower()
        yp = df["pred"].astype(str).str.strip().str.lower()
    elif kind == "gem":
        yt = df["true_name"].astype(str).str.strip().str.lower()
        yp = df["predicted_name"].astype(str).str.strip().str.lower()
    # codificar a enteros por la union de clases verdaderas (estratificacion por
    # clase verdadera). Etiquetas predichas fuera de vocab => clase -1 sentinel.
    classes = sorted(yt.unique())
    cmap = {c: i for i, c in enumerate(classes)}
    yt_i = yt.map(cmap).values
    yp_i = yp.map(lambda x: cmap.get(x, -1)).values
    return yt_i.astype(int), yp_i.astype(int)


def run_llm():
    print("\n" + "=" * 70 + "\nTABLA 4: tab:llm-comparativa (Acc/BAcc)\n" + "=" * 70)
    dsname = {"HAM": "HAM10000", "PAD": "PAD-UFES-20", "DDI": "DDI"}
    for row, pubs in LLM_PUB.items():
        for dskey in ("HAM", "PAD", "DDI"):
            pub_acc, pub_bacc = pubs[dskey]
            csvinfo = LLM_CSV[row][dskey]
            path = os.path.join(OUT, csvinfo[0])
            if not os.path.exists(path):
                record("tab:llm-comparativa", dsname[dskey], row, "Acc",
                       pub_acc, np.nan, np.nan, np.nan, 0, note="CSV no encontrado -> GAP")
                continue
            yt, yp = load_llm(path, csvinfo[1])
            ncl = len(set(yt))
            bm = bootstrap_metrics(yt, yp, None, ncl, ["Acc", "BAcc"])
            note = ""
            if dskey == "DDI" and len(yt) != 137:
                note = (f"DDI N={len(yt)} (esta fila se evaluo sobre test+val; "
                        f"caption TFG dice N=137 -> inconsistencia)")
            lo, hi, mean = bm["Acc"]
            record("tab:llm-comparativa", dsname[dskey], row, "Acc",
                   pub_acc, lo, hi, mean, len(yt), note=note)
            lo, hi, mean = bm["BAcc"]
            bnote = note
            if pub_bacc is None:
                bnote = (note + "; " if note else "") + "TFG omite BAcc (---); IC informativo"
            record("tab:llm-comparativa", dsname[dskey], row, "BAcc",
                   pub_bacc, lo, hi, mean, len(yt), note=bnote)


# =============================================================================
# Volcado de salida
# =============================================================================
def dump():
    dfres = pd.DataFrame(RESULTS)
    dfres.to_csv(os.path.join(DST, "bootstrap_ci_all.csv"), index=False)

    # Markdown por tabla con 'valor [lo, hi]' coma decimal
    md = ["# Intervalos de confianza bootstrap (Fase 1) — 4 tablas headline\n",
          f"Bootstrap estratificado por clase, B={B}, IC95% percentil "
          f"(2,5/97,5), semilla {SEED}.\n",
          "Formato celda: `media_boot [lo, hi]` (coma decimal). "
          "Publicado entre parentesis cuando difiere.\n"]
    for table in ["tab:lp-panderm", "tab:benchmark", "tab:equidad-fototipo",
                  "tab:llm-comparativa"]:
        sub = dfres[dfres.table == table]
        md.append(f"\n## {table}\n")
        md.append("| Fila | Modelo | Metrica | N | Publicado | Bootstrap [IC95%] | conv | flag |")
        md.append("|---|---|---|---|---|---|---|---|")
        for _, r in sub.iterrows():
            pub = comma(r.published) if pd.notna(r.published) else "---"
            cell = (f"{comma(r.boot_mean)} [{comma(r.ci_lo)}, {comma(r.ci_hi)}]"
                    if pd.notna(r.boot_mean) else "GAP")
            md.append(f"| {r.row} | {r.model} | {r.metric} | {int(r.N)} | {pub} "
                      f"| {cell} | {r.auroc_conv} | {r.flag} |")
    with open(os.path.join(DST, "bootstrap_ci_tables.md"), "w") as f:
        f.write("\n".join(md))

    # Informe sanity-check
    rep = ["# Informe sanity-check bootstrap (Fase 1)\n",
           f"Criterio PARADA: |media_boot - publicado| > 0,005 o IC no contiene "
           f"el publicado.\n", "## Detalle por celda\n"]
    rep += SANITY
    rep.append("\n## PARADAS (celdas que no cuadran)\n")
    if STOPS:
        for s in STOPS:
            rep.append(f"- [STOP] {s['table']} | {s['row']} | {s['model']} | "
                       f"{s['metric']}: pub={comma(s['published'])} "
                       f"mean={comma(s['mean'])} IC=[{comma(s['lo'])},{comma(s['hi'])}] "
                       f"dev={comma(s['dev'])} contains={s['contains']} {s['note']}")
    else:
        rep.append("Ninguna. Todas las celdas con publicado reproducen dentro de tolerancia.\n")
    with open(os.path.join(DST, "sanity_check_report.md"), "w") as f:
        f.write("\n".join(rep))

    # resumen consola
    n_ok = (dfres.flag == "OK").sum()
    n_stop = (dfres.flag == "STOP").sum()
    n_gap = (dfres.flag == "GAP").sum()
    print(f"\n=== RESUMEN: OK={n_ok}  STOP={n_stop}  GAP={n_gap}  total={len(dfres)} ===")


if __name__ == "__main__":
    run_lp_panderm()
    run_benchmark()
    run_fairness()
    run_llm()
    dump()
