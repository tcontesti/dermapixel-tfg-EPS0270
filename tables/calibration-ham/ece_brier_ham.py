#!/usr/bin/env python3
"""ECE (15 equal-width bins) + Brier on HAM10000 melanoma-vs-rest endpoint.
Reuses EXACTLY the DeLong/HAM pipeline: same cached embeddings, same SigLIP .pt,
same LogisticRegression(C=1.0,lbfgs,max_iter=5000,random_state=42), same CSV split.
Mandatory sanity: recomputed AUROC must reproduce published 0.9523/0.9495/0.9463.
NO retraining beyond the identical deterministic LP refit; CPU only.
"""
import os, json
import numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

HOME = os.path.expanduser('~')
OUT = f'{HOME}/panderm/output/ham10000_delong'
CSV = f'{HOME}/panderm/datasets/HAM10000_clean/ISIC2018_splits/HAM_clean.csv'
SIGLIP = f'{HOME}/panderm/output/medgemma_eval/siglip_embeddings_ham10000.pt'

PUBLISHED = {'PanDerm Large': 0.9523, 'SigLIP-Large': 0.9495, 'DermLIP v2': 0.9463}

def ece_equal_width(y, p, n_bins=15):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0; N = len(y); rows = []
    for b in range(n_bins):
        lo, hi = bins[b], bins[b+1]
        m = (p > lo) & (p <= hi) if b > 0 else (p >= lo) & (p <= hi)
        cnt = int(m.sum())
        if cnt == 0:
            rows.append((lo, hi, 0, np.nan, np.nan, 0.0)); continue
        conf = float(p[m].mean()); acc = float(y[m].mean())
        gap = abs(acc - conf); ece += (cnt / N) * gap
        rows.append((lo, hi, cnt, conf, acc, gap))
    return ece, rows

def main():
    df = pd.read_csv(CSV)
    tr_df = df[df.split == 'train']; te_df = df[df.split == 'test']
    ytr = (tr_df.binary_label.values == 1).astype(int)
    yte = (te_df.binary_label.values == 1).astype(int)
    assert len(tr_df) == 8207 and len(te_df) == 1232

    feats = {
        'PanDerm Large': ('npz', f'{OUT}/panderm_large_embeddings.npz'),
        'DermLIP v2':    ('npz', f'{OUT}/dermlip_v2_embeddings.npz'),
        'SigLIP-Large':  ('pt',  SIGLIP),
    }
    results = {}; per_bin = {}
    for name, (kind, path) in feats.items():
        if kind == 'npz':
            z = np.load(path); Xtr, Xte = z['train'], z['test']
        else:
            sg = torch.load(path, map_location='cpu')
            Xtr = sg['train'].float().numpy(); Xte = sg['test'].float().numpy()
        clf = LogisticRegression(C=1.0, solver='lbfgs', max_iter=5000,
                                 random_state=42, n_jobs=-1).fit(Xtr, ytr)
        p = clf.predict_proba(Xte)[:, 1]
        auroc = roc_auc_score(yte, p)
        ece, rows = ece_equal_width(yte, p, 15)
        brier = brier_score_loss(yte, p)
        results[name] = {'auroc': float(auroc), 'auroc_pub': PUBLISHED[name],
                         'ece': float(ece), 'brier': float(brier)}
        per_bin[name] = rows

    # sanity gate
    print('=== SANITY AUROC (published vs recomputed, tol 3 decimals) ===')
    ok = True
    for name, r in results.items():
        match = round(r['auroc'], 3) == round(r['auroc_pub'], 3)
        ok = ok and match
        print(f"  {name:14s} pub={r['auroc_pub']:.4f}  recomp={r['auroc']:.6f}  "
              f"round3 pub={round(r['auroc_pub'],3)} recomp={round(r['auroc'],3)}  -> {'OK' if match else 'MISMATCH'}")
    print(f"\nSANITY: {'PASS' if ok else 'FAIL'}")
    if not ok:
        print('GATE FAILED — NOT writing calibration outputs, STOP.')
        raise SystemExit(2)

    print('\n=== ECE (15 equal-width bins) + Brier ===')
    for name, r in results.items():
        print(f"  {name:14s} ECE={r['ece']:.4f}  Brier={r['brier']:.4f}")

    print('\n=== Per-bin reliability (PanDerm Large) ===')
    print('  bin            n     conf     acc      gap')
    for lo, hi, cnt, conf, acc, gap in per_bin['PanDerm Large']:
        if cnt == 0: continue
        print(f"  ({lo:.2f},{hi:.2f}]  {cnt:5d}  {conf:.4f}  {acc:.4f}  {gap:.4f}")

    out = {'task': 'HAM10000 melanoma vs rest — calibration (ECE 15 equal-width bins + Brier)',
           'n_test': int(len(yte)), 'n_pos': int(yte.sum()),
           'lp_config': 'LogisticRegression(C=1.0, lbfgs, max_iter=5000, random_state=42)',
           'sanity': 'PASS', 'models': results,
           'per_bin_panderm_large': [
               {'lo': lo, 'hi': hi, 'n': cnt,
                'conf': (None if np.isnan(conf) else conf),
                'acc': (None if np.isnan(acc) else acc), 'gap': gap}
               for (lo, hi, cnt, conf, acc, gap) in per_bin['PanDerm Large']],
           'per_bin_all': {name: [
               {'lo': lo, 'hi': hi, 'n': cnt,
                'conf': (None if np.isnan(conf) else conf),
                'acc': (None if np.isnan(acc) else acc), 'gap': gap}
               for (lo, hi, cnt, conf, acc, gap) in rows]
               for name, rows in per_bin.items()}}
    json.dump(out, open(f'{OUT}/calibration_ham_melanoma_results.json', 'w'), indent=2)
    print(f"\nsaved {OUT}/calibration_ham_melanoma_results.json")

if __name__ == '__main__':
    main()
