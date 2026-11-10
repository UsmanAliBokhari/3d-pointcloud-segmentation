"""
Plumbing that connects the notebook to the tests and the pipeline. Nobody edits this.

* `impl` is a module object holding the team's CURRENT version of all 21 functions.
  It starts with placeholders. Every notebook cell decorated with `@task` replaces
  the placeholder with the notebook's version.
* While a function still raises NotImplementedError, `@task` falls back to a DUMMY
  (right shape, meaningless values), so the whole pipeline always runs.
* Tests put a function in STRICT mode, so a dummy never passes.
* `load_notebook()` executes the notebook's `#| export` cells. This is how
  `bash run.sh` and `python3 code/tests.py` use the code written in the notebook.
"""
import os
import json
import types
import functools

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestClassifier

import utils
from utils import CONFIG, CLASS_NAMES, ScaleLevel, make_toy_scene

NOTEBOOK = os.path.join(utils.ROOT, 'notebooks', 'project.ipynb')

impl = types.ModuleType('impl')     # impl.knn_neighbors(...) = the team's current version
REGISTRY = {}                       # name -> {'owner', 'stage', 'fn'}
STRICT = set()                      # names under test -> no dummy fallback
_warned = set()                     # names whose dummy was used in this run
SILENT_DUMMIES = False

# name -> (owner, stage), in pipeline order
SPEC = {
    'load_ply': ('P3', '1-data'), 'remap_labels': ('P3', '1-data'),
    'voxel_downsample': ('P1', '1-data'), 'build_pyramid': ('P1', '1-data'),
    'knn_neighbors': ('P2', '1-data'), 'cylinder_z_extent': ('P2', '1-data'),
    'sample_training_set': ('P2', '1-data'),
    'structure_tensor': ('P1', '2-features'), 'covariance_features': ('P1', '2-features'),
    'moment_features': ('P2', '2-features'), 'height_features': ('P2', '2-features'),
    'compute_point_features': ('P3', '2-features'), 'extract_features': ('P3', '2-features'),
    'evaluate': ('P1', '3-eval'), 'compare_to_paper': ('P1', '3-eval'), 'save_timings': ('P1', '3-eval'),
    'train_random_forest': ('P2', '3-eval'), 'classify_cloud': ('P2', '3-eval'),
    'plot_confusion': ('P3', '3-eval'), 'plot_cloud': ('P3', '3-eval'), 'feature_ablation': ('P3', '3-eval'),
}


# ====================================== dummies =====================================
def _pyramid(xyz, base=None, n_levels=None, factor=None):
    n_levels, base, factor = n_levels or CONFIG['n_levels'], base or CONFIG['base_voxel'], factor or CONFIG['factor']
    out = []
    for s in range(n_levels):
        pts = xyz[::2 ** s]
        out.append(ScaleLevel(base * factor ** s, pts, cKDTree(pts), cKDTree(pts[:, :2])))
    return out


def _knn(level, q, k=None):
    k = min(k or CONFIG['k'], len(level.points))
    return np.broadcast_to(level.points[:k], (len(q), k, 3)).copy()


def _sample(labels, n_per_class=None, seed=None):
    n = n_per_class or CONFIG['n_train_per_class']
    return np.concatenate([np.flatnonzero(labels == c)[:n] for c in np.unique(labels[labels >= 0])])


def _point_features(pyramid, q, k=None):
    W = np.random.default_rng(0).normal(size=(3, 16 * len(pyramid)))   # deterministic per point
    return np.sin(q @ W).astype(np.float32)


def _extract(pyramid, xyz, idx=None, chunk_size=None, cache_path=None):
    idx = np.arange(len(xyz)) if idx is None else idx
    return impl.compute_point_features(pyramid, xyz[idx])


def _evaluate(y_true, y_pred, class_names=None):
    names = class_names or CLASS_NAMES
    return (pd.DataFrame(0.0, index=names, columns=['recall', 'precision', 'f1']),
            {'overall_accuracy': 0.0, 'mean_recall': 0.0, 'mean_f1': 0.0})


def _train(X, y, grid=None, folds=None, seed=None):
    return RandomForestClassifier(n_estimators=5, max_depth=5, random_state=0).fit(X, y), pd.DataFrame()


def _classify(model, pyramid, xyz, idx=None, chunk_size=None):
    idx = np.arange(len(xyz)) if idx is None else idx
    return model.predict(impl.compute_point_features(pyramid, xyz[idx])).astype(np.int64)


DUMMIES = {
    'load_ply': lambda path: make_toy_scene(),
    'remap_labels': lambda raw_class, class_map: (np.asarray(raw_class) % 6).astype(np.int64),
    'voxel_downsample': lambda xyz, voxel_size: xyz[::2].copy(),
    'build_pyramid': _pyramid,
    'knn_neighbors': _knn,
    'cylinder_z_extent': lambda level, q, radius: (q[:, 2] - 1.0, q[:, 2] + 1.0),
    'sample_training_set': _sample,
    'structure_tensor': lambda neigh: (np.full((len(neigh), 3), 1 / 3),
                                       np.broadcast_to(np.eye(3), (len(neigh), 3, 3)).copy()),
    'covariance_features': lambda eigvals, eigvecs: np.zeros((len(eigvals), 9)),
    'moment_features': lambda neigh, q, eigvecs: np.zeros((len(neigh), 4)),
    'height_features': lambda z, zmin, zmax: np.zeros((len(z), 3)),
    'compute_point_features': _point_features,
    'extract_features': _extract,
    'evaluate': _evaluate,
    'compare_to_paper': lambda per_class, summary, paper=None: pd.DataFrame(columns=['ours', 'paper', 'diff']),
    'save_timings': lambda path: pd.DataFrame(columns=['seconds']),
    'train_random_forest': _train,
    'classify_cloud': _classify,
    'plot_confusion': lambda y_true, y_pred, path, class_names=None: np.zeros((len(class_names or CLASS_NAMES),) * 2),
    'plot_cloud': lambda xyz, labels, path, max_points=300_000, errors=None, class_names=None: 0,
    'feature_ablation': lambda *a, **k: pd.DataFrame(columns=['oa', 'mean_f1', 'drop_oa', 'drop_f1']),
}


# ====================================== @task =======================================
def task(fn):
    """Decorator for the 21 team functions in the notebook. Registers the function so
    that the tests and the pipeline use it, with a dummy fallback while unimplemented."""
    name = fn.__name__
    if name not in SPEC:
        raise NameError(f'@task: {name!r} is not one of the 21 project functions {list(SPEC)}')
    owner, stage = SPEC[name]
    dummy = DUMMIES[name]

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except NotImplementedError:
            if name in STRICT:
                raise
            if name not in _warned and not SILENT_DUMMIES:
                print(f'  [dummy] {name}() not implemented yet ({owner}, {utils.who(owner)}) -> using dummy output')
            _warned.add(name)
            return dummy(*args, **kwargs)

    REGISTRY[name] = {'owner': owner, 'stage': stage, 'fn': wrapper}
    setattr(impl, name, wrapper)
    return wrapper


def _placeholder(name):
    def f(*args, **kwargs):
        raise NotImplementedError
    f.__name__ = f.__qualname__ = name
    return task(f)


for _n in SPEC:
    _placeholder(_n)


# ================================= notebook loader ==================================
def export_cells(path=NOTEBOOK):
    """Source of every code cell whose first line starts with '#| export'."""
    with open(path) as f:
        nb = json.load(f)
    cells = []
    for c in nb['cells']:
        if c['cell_type'] != 'code':
            continue
        src = ''.join(c['source']) if isinstance(c['source'], list) else c['source']
        if src.lstrip().startswith('#| export'):
            cells.append(src)
    return cells


def load_notebook(path=NOTEBOOK):
    """Execute the notebook's '#| export' cells (the 21 functions) so `impl` holds the
    code written in the notebook. Used by run.sh / main.py / tests.py from a terminal.
    Remember to SAVE the notebook first."""
    import nb_env
    ns = dict(vars(nb_env))
    for src in export_cells(path):
        first = src.lstrip().splitlines()[0]
        try:
            exec(compile(src, f'<{os.path.basename(path)}: {first}>', 'exec'), ns)
        except Exception as e:
            raise RuntimeError(f'Error in notebook cell "{first}": {type(e).__name__}: {e}') from e
    return ns
