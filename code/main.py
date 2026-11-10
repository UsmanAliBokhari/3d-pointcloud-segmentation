"""
End-to-end pipeline using the functions written in notebooks/project.ipynb.
Runs even while functions are unimplemented (dummies are used and reported at the end;
results are only meaningful once no dummy is left).

Terminal (loads the SAVED notebook):
    python3 code/main.py --toy                 # synthetic scene, ~1 minute
    python3 code/main.py --max-points 2000000  # quick run on a real-data subset
    python3 code/main.py                       # full reproduction (what run.sh does)
Notebook:
    from main import run; run(toy=True)
"""
import os
import sys
import json
import argparse

import numpy as np
import pandas as pd

import utils
import scaffold
from scaffold import impl as I
from utils import CONFIG, RESULTS_DIR, FIG_DIR, DATA_DIR, Timer, set_seeds, make_toy_scene, feature_names


def load_data(cfg, toy):
    if toy:
        return make_toy_scene(30_000)
    paths = [os.path.join(DATA_DIR, f) for f in cfg['data_files']]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        sys.exit(f'Missing data files: {missing}\nRun:  bash data/download.sh')
    parts = [I.load_ply(p) for p in paths]
    return np.vstack([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def run(toy=False, max_points=None, skip_ablation=False, ablation_points=200_000, config=None):
    cfg = utils.load_config(config)
    team = dict(utils.TEAM)          # keep names set in the notebook
    CONFIG.update(cfg)
    utils.TEAM.update(team)
    seed = set_seeds(cfg['seed'])
    os.makedirs(FIG_DIR, exist_ok=True)
    sfx = '_toy' if toy else ''      # real-run outputs: metrics.csv, ablation.csv, ...
    scaffold._warned.clear()
    Timer.log.clear()

    # ---------------- Stage 1: data & neighbourhoods ----------------
    xyz, raw = load_data(cfg, toy)
    labels = I.remap_labels(raw, cfg['class_map'])
    max_pts = max_points or cfg.get('max_points')
    if max_pts and len(xyz) > max_pts:
        keep = np.sort(np.random.default_rng(seed).choice(len(xyz), max_pts, replace=False))
        xyz, labels = xyz[keep], labels[keep]
    print(f'{len(xyz):,} points, classes present: {np.unique(labels).tolist()}')

    with Timer('pyramid'):
        pyramid = I.build_pyramid(xyz)
    print('pyramid sizes:', [len(l.points) for l in pyramid])

    n_train = 200 if toy else cfg['n_train_per_class']
    train_idx = I.sample_training_set(labels, n_train, seed)
    test_mask = labels >= 0
    test_mask[train_idx] = False
    test_idx = np.flatnonzero(test_mask)
    print(f'train {len(train_idx):,}  test {len(test_idx):,}')

    # ---------------- Stage 2: features (training set only; test is on the fly) -------
    X_tr = I.extract_features(pyramid, xyz, train_idx)
    y_tr = labels[train_idx]

    # ---------------- Stage 3: classifier & evaluation ----------------
    model, cv = I.train_random_forest(X_tr, y_tr)
    cv.to_csv(os.path.join(RESULTS_DIR, f'cv_results{sfx}.csv'))
    print('best model:', getattr(model, 'n_estimators', '?'), 'trees, max depth', getattr(model, 'max_depth', '?'))

    y_pred = I.classify_cloud(model, pyramid, xyz, test_idx)
    y_true = labels[test_idx]
    per_class, summary = I.evaluate(y_true, y_pred)
    print(per_class.round(4)); print(json.dumps(summary, indent=1))

    rows = [(f'{c}/{m}', per_class.loc[c, m]) for c in per_class.index for m in per_class.columns]
    rows += list(summary.items())
    pd.DataFrame(rows, columns=['metric', 'value']).to_csv(os.path.join(RESULTS_DIR, f'metrics{sfx}.csv'), index=False)
    if not toy:
        I.compare_to_paper(per_class, summary).to_csv(os.path.join(RESULTS_DIR, 'paper_comparison.csv'))

    I.plot_confusion(y_true, y_pred, os.path.join(FIG_DIR, f'confusion{sfx}.png'))
    I.plot_cloud(xyz[test_idx], y_pred, os.path.join(FIG_DIR, f'prediction{sfx}.png'))
    I.plot_cloud(xyz[test_idx], y_pred, os.path.join(FIG_DIR, f'errors{sfx}.png'), errors=(y_pred != y_true))

    if not skip_ablation:
        sub = np.random.default_rng(seed).choice(test_idx, min(ablation_points, len(test_idx)), replace=False)
        X_te = I.extract_features(pyramid, xyz, sub)
        params = {k: getattr(model, k) for k in ('n_estimators', 'max_depth') if hasattr(model, k)}
        I.feature_ablation(X_tr, y_tr, X_te, labels[sub], feature_names(len(pyramid)), params, seed) \
            .to_csv(os.path.join(RESULTS_DIR, f'ablation{sfx}.csv'))

    I.save_timings(os.path.join(RESULTS_DIR, f'timings{sfx}.csv'))

    if scaffold._warned:
        print('\n!! These functions used DUMMY outputs in this run, so the numbers above are meaningless')
        print('   (status of all 21 functions: the status cell in the notebook, or python3 code/tests.py):')
        for name in [n for n in scaffold.SPEC if n in scaffold._warned]:
            o = scaffold.SPEC[name][0]
            print(f'   {name:24s} ({o}, {utils.who(o)})')
    else:
        print('\nAll functions implemented. Results written to results/.')
    return per_class, summary


def main():
    p = argparse.ArgumentParser(description='Hackel et al. 2016 reproduction')
    p.add_argument('--config', default=os.path.join(utils.ROOT, 'config.yaml'))
    p.add_argument('--toy', action='store_true', help='use the synthetic toy scene')
    p.add_argument('--max-points', type=int, default=None, help='random subset for quick runs')
    p.add_argument('--skip-ablation', action='store_true')
    p.add_argument('--ablation-points', type=int, default=200_000,
                   help='test points used for the ablation study (their features are stored)')
    a = p.parse_args()
    scaffold.load_notebook()
    run(a.toy, a.max_points, a.skip_ablation, a.ablation_points, a.config)


if __name__ == '__main__':
    main()
