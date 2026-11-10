"""
Shared helpers: paths, config, team names, seeds, data containers, timer, constants
and a synthetic toy scene. (Team functions are implemented in notebooks/project.ipynb.)

Nobody "owns" this file. Change it only after agreeing with the whole team.
"""
import os
import time
import random
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

# --------------------------------------------------------------------------------------
# Paths: ALWAYS relative to the project root (course rule: no absolute paths anywhere)
# --------------------------------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(ROOT, 'data')
RESULTS_DIR = os.path.join(ROOT, 'results')
FIG_DIR = os.path.join(RESULTS_DIR, 'figures')
CACHE_DIR = os.path.join(ROOT, 'cache')          # gitignored: cached feature matrices

# Paris-rue-Madame: the 6 classes of the paper's Table 2 (list position = our label id)
CLASS_NAMES = ['facade', 'ground', 'cars', 'motorcycles', 'traffic_signs', 'pedestrians']

# Default config. config.yaml (project root) overrides these values.
DEFAULT_CONFIG = {
    'seed': 42,
    'data_files': ['GT_Madame1_2.ply', 'GT_Madame1_3.ply'],
    # raw dataset class id -> our id (Serna et al. 2014, Table 1). Unlisted ids -> -1 (ignored)
    'class_map': {1: 0, 2: 1, 4: 2, 10: 3, 14: 4, 9: 5},
    'max_points': None,          # e.g. 2_000_000 for a quick laptop run; None = everything
    # --- paper Section 3.1 ---
    'base_voxel': 0.025,         # finest voxel edge (m)
    'n_levels': 9,               # 0.025 * 2^s for s = 0..8  ->  6.4 m
    'factor': 2.0,
    'k': 10,                     # neighbours per scale (point itself + 9)
    # --- height features: radius is NOT given in the paper (our documented choice) ---
    'cyl_radius_factor': 3.0,    # cylinder radius at level s = factor * voxel_size(s)
    # --- paper Sections 3.3 / 4 ---
    'n_train_per_class': 1000,
    'rf_grid': {'n_estimators': [25, 50, 100], 'max_depth': [10, 20, 30]},
    'cv_folds': 5,
    'chunk_size': 200_000,
    # who is P1 / P2 / P3 (the notebook's first cell can override this for a session)
    'team': {'P1': 'Usman', 'P2': 'Jana', 'P3': 'Divi'},
}


def load_config(path=None):
    cfg = dict(DEFAULT_CONFIG)
    path = path or os.path.join(ROOT, 'config.yaml')
    if os.path.exists(path):
        import yaml
        with open(path) as f:
            cfg.update(yaml.safe_load(f) or {})
    cfg['class_map'] = {int(k): int(v) for k, v in cfg['class_map'].items()}
    return cfg


CONFIG = load_config()
TEAM = dict(CONFIG['team'])      # {'P1': name, ...}; change in config.yaml or the notebook


def who(owner_id):
    """'P2' -> 'Jana' (whoever is currently assigned)."""
    return TEAM.get(owner_id, owner_id)


def detect_me():
    """Guess who is running this from `git config user.name` (e.g. 'Jana Peeters' -> 'Jana')."""
    import subprocess
    try:
        git_name = subprocess.run(['git', 'config', 'user.name'], capture_output=True, text=True,
                                  cwd=ROOT, timeout=5).stdout.strip().lower()
    except Exception:
        return None
    for name in TEAM.values():
        if str(name).lower() in git_name:
            return name
    return None


def resolve_owner(owner):
    """Accepts 'P2' or a name ('Jana', case-insensitive) and returns the P-id."""
    if owner in TEAM:
        return owner
    for pid, name in TEAM.items():
        if str(name).lower() == str(owner).lower():
            return pid
    raise ValueError(f'unknown owner {owner!r}; team is {TEAM}')


def set_seeds(seed=None):
    seed = CONFIG['seed'] if seed is None else seed
    random.seed(seed)
    np.random.seed(seed)
    return seed


# --------------------------------------------------------------------------------------
# Data containers
# --------------------------------------------------------------------------------------
@dataclass
class ScaleLevel:
    """One level of the multi-scale pyramid (paper Section 3.1, Fig. 2)."""
    voxel_size: float
    points: np.ndarray      # (M_s, 3) float64 voxel centroids at this level
    tree: cKDTree           # 3D kD-tree over `points`            -> kNN queries
    tree_xy: cKDTree        # 2D kD-tree over `points[:, :2]`     -> cylinder queries


class Timer:
    """with Timer('features'): ...   Times accumulate per name in Timer.log (Table 3)."""
    log = {}

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.t0
        Timer.log[self.name] = Timer.log.get(self.name, 0.0) + self.seconds


# --------------------------------------------------------------------------------------
# Constants shared by the notebook, tests and pipeline
# --------------------------------------------------------------------------------------
EPS = 1e-12

FEATURES_PER_SCALE = [
    'sum', 'omnivariance', 'eigenentropy', 'anisotropy', 'planarity', 'linearity',
    'surface_variation', 'sphericity', 'verticality',          # 9 covariance
    'mom1_e1', 'mom1_e2', 'mom2_e1', 'mom2_e2',                 # 4 moment
    'vertical_range', 'height_below', 'height_above',          # 3 height
]


def feature_names(n_levels=None):
    """Column names of X, scale-major: ['s0_sum', ..., 's0_height_above', 's1_sum', ...]."""
    n_levels = n_levels or CONFIG['n_levels']
    return [f's{s}_{f}' for s in range(n_levels) for f in FEATURES_PER_SCALE]


# Paper Table 2, Paris-rue-Madame, "Our Method": (recall, precision, F1)
PAPER_MADAME = {
    'facade': (0.9799, 0.9902, 0.9851), 'ground': (0.9692, 0.9934, 0.9811),
    'cars': (0.9786, 0.9086, 0.9423), 'motorcycles': (0.9796, 0.4792, 0.6435),
    'traffic_signs': (0.9939, 0.3403, 0.5070), 'pedestrians': (0.9987, 0.2414, 0.3888),
    'overall_accuracy': 0.9755, 'mean_recall': 0.9833, 'mean_f1': 0.7413,
}

# Colours used in the paper's figures (Fig. 4)
CLASS_COLORS = {'facade': '#9e9e9e', 'ground': '#ff8c00', 'cars': '#1f3fff',
                'motorcycles': '#ffa64d', 'traffic_signs': '#ff2fa0', 'pedestrians': '#a000c8'}


# --------------------------------------------------------------------------------------
# Synthetic toy street: for development and tests before/without the real dataset
# --------------------------------------------------------------------------------------
def make_toy_scene(n=30_000, seed=0):
    """Ground plane (dense near a 'scanner' at the origin), facade wall, car-shaped box,
    pole (traffic sign), motorcycle-sized box, a thin 'pedestrian' cylinder.
    Returns xyz (N,3) float64 and RAW class ids (as in the dataset: 1,2,4,14,10,9),
    so remap_labels() is exercised too."""
    rng = np.random.default_rng(seed)
    pts, cls = [], []

    def add(p, c):
        pts.append(p)
        cls.append(np.full(len(p), c))

    def box(lo, size, m):
        lo, size = np.asarray(lo, float), np.asarray(size, float)
        u = lo + rng.uniform(size=(m, 3)) * size
        face = rng.integers(0, 3, m)
        side = rng.random(m) < .5
        u[np.arange(m), face] = np.where(side, lo[face], (lo + size)[face])
        return u

    def cyl(cx, cy, rad, h, m):
        t, z = rng.uniform(0, 2 * np.pi, m), rng.uniform(0, h, m)
        return np.c_[cx + rad * np.cos(t), cy + rad * np.sin(t), z]

    m = n // 4
    r, a = np.abs(rng.normal(0, 6, m)), rng.uniform(0, 2 * np.pi, m)
    add(np.c_[r * np.cos(a), r * np.sin(a), rng.normal(0, .01, m)], 2)                     # ground
    add(np.c_[rng.uniform(-10, 10, m), 8 + rng.normal(0, .01, m), rng.uniform(0, 8, m)], 1)  # facade
    add(box([2, 2, 0], [4, 1.8, 1.4], m // 2), 4)                                          # car
    add(cyl(-3, 4, .05, 3, m // 6), 14)                                                     # sign pole
    add(box([-6, 2, 0], [1.8, .6, 1.1], m // 6), 10)                                       # motorcycle
    add(cyl(0, -3, .25, 1.75, m // 6), 9)                                                   # pedestrian
    return np.vstack(pts).astype(np.float64), np.concatenate(cls).astype(np.int64)
