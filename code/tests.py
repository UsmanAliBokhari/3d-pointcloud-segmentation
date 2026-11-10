"""
Known-answer tests for every function, plus a status board. The functions themselves
are written in notebooks/project.ipynb. From a terminal, this file loads them from the
SAVED notebook; inside the notebook, check()/status() use the live versions.

    python3 code/tests.py                    # status board of all 21 functions
    python3 code/tests.py voxel_downsample   # one function, with the full error message
    python3 code/tests.py --owner P2         # only my functions (or --owner Jana)

Each test switches ITS function to strict mode (no dummy fallback), so a dummy never
passes. Tests build their own small inputs, so they don't depend on teammates' code:
you can make your function PASS while everything else is still a dummy.
"""
import os
import sys
import tempfile
import traceback

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import utils
import scaffold
from utils import ScaleLevel, Timer, CONFIG, make_toy_scene
from scaffold import REGISTRY, STRICT, impl as I

TMP = tempfile.mkdtemp(prefix='cvtests_')


def close(a, b, tol=1e-6, msg=''):
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape, f'{msg} shape {a.shape} != expected {b.shape}'
    assert np.allclose(a, b, atol=tol, rtol=1e-4), f'{msg}\n got      {np.round(a, 5)}\n expected {np.round(b, 5)}'


def manual_pyramid(xyz, n_levels=2, base=0.5):
    """Independent of build_pyramid so feature tests don't need P1's code."""
    lv = []
    for s in range(n_levels):
        pts = xyz[::2 ** s]
        lv.append(ScaleLevel(base * 2 ** s, pts, cKDTree(pts), cKDTree(pts[:, :2])))
    return lv


def level_from(points):
    points = np.asarray(points, float)
    return ScaleLevel(1.0, points, cKDTree(points), cKDTree(points[:, :2]))


# ===================================== STAGE 1 ======================================
def test_load_ply():
    from plyfile import PlyData, PlyElement
    rng = np.random.default_rng(0)
    n = 50
    arr = np.zeros(n, dtype=[('x', '>f4'), ('y', '>f4'), ('z', '>f4'), ('reflectance', '>f4'),
                             ('label', '>u4'), ('class', '>u4')])
    xyz = rng.uniform(-5, 5, (n, 3)).astype(np.float32)
    arr['x'], arr['y'], arr['z'] = xyz.T
    arr['class'] = rng.integers(0, 26, n)
    path = os.path.join(TMP, 'tiny.ply')
    PlyData([PlyElement.describe(arr, 'vertex')], text=False, byte_order='>').write(path)
    out_xyz, out_cls = I.load_ply(path)
    assert out_xyz.dtype == np.float64, f'xyz dtype {out_xyz.dtype}, expected float64'
    close(out_xyz, xyz, 1e-5, 'xyz')
    assert out_cls.dtype == np.int64, f'class dtype {out_cls.dtype}, expected int64'
    assert (out_cls == arr['class']).all(), 'class values differ'


def test_remap_labels():
    raw = np.array([1, 2, 4, 10, 14, 9, 0, 25, 1, 99])
    out = I.remap_labels(raw, {1: 0, 2: 1, 4: 2, 10: 3, 14: 4, 9: 5})
    assert out.dtype == np.int64, f'dtype {out.dtype}'
    close(out, [0, 1, 2, 3, 4, 5, -1, -1, 0, -1], msg='mapped labels')


def test_voxel_downsample():
    pts = np.array([[.1, .1, .1], [.3, .3, .3], [1.5, .2, .2], [-.2, 0, 0]])
    out = I.voxel_downsample(pts, 1.0)
    exp = np.array([[-.2, 0, 0], [.2, .2, .2], [1.5, .2, .2]])
    out = out[np.lexsort(out.T[::-1])]
    close(out, exp, msg='centroids (sorted by x)')
    big = np.random.default_rng(0).uniform(0, 10, (5000, 3))
    out = I.voxel_downsample(big, 2.5)
    n_vox = len(np.unique(np.floor(big / 2.5), axis=0))
    assert out.shape == (n_vox, 3), f'expected {n_vox} voxels, got {out.shape}'


def test_build_pyramid():
    xyz, _ = make_toy_scene(20_000)
    pyr = I.build_pyramid(xyz)
    assert len(pyr) == CONFIG['n_levels'], f'{len(pyr)} levels, expected {CONFIG["n_levels"]}'
    close([l.voxel_size for l in pyr], CONFIG['base_voxel'] * 2.0 ** np.arange(CONFIG['n_levels']), msg='voxel sizes')
    sizes = [len(l.points) for l in pyr]
    assert all(a >= b for a, b in zip(sizes, sizes[1:])), f'sizes must shrink: {sizes}'
    for l in pyr:
        assert isinstance(l, ScaleLevel)
        assert l.tree.n == len(l.points) and l.tree.m == 3, 'tree must index level.points (3D)'
        assert l.tree_xy.n == len(l.points) and l.tree_xy.m == 2, 'tree_xy must index points[:, :2]'
        keys = np.floor(l.points / l.voxel_size)
        assert len(np.unique(keys, axis=0)) == len(l.points), \
            f'level {l.voxel_size}: more than one point per voxel -> not voxel-downsampled'
    small = I.build_pyramid(xyz, base=0.1, n_levels=3, factor=2.0)
    close([l.voxel_size for l in small], [0.1, 0.2, 0.4], msg='custom arguments')


def test_knn_neighbors():
    lv = level_from([[x, 0, 0] for x in range(10)])
    nb = I.knn_neighbors(lv, np.array([[2.1, 0, 0]]), k=3)
    close(nb, [[[2, 0, 0], [3, 0, 0], [1, 0, 0]]], msg='3 nearest, nearest first')
    q = np.random.default_rng(0).uniform(0, 9, (20, 3))
    assert I.knn_neighbors(lv, q, k=10).shape == (20, 10, 3)
    assert I.knn_neighbors(level_from([[0, 0, 0], [1, 1, 1]]), q, k=10).shape == (20, 2, 3), \
        'k larger than the level -> use all its points'
    assert I.knn_neighbors(lv, q, k=1).shape == (20, 1, 3), 'k=1 must still be (Q,1,3)'


def test_cylinder_z_extent():
    lv = level_from([[0, 0, 0], [0, 0, 5], [0.2, -0.1, 3], [10, 10, -3]])
    q = np.array([[0.1, 0, 2], [10, 10, 0], [50, 50, 1]])
    zmin, zmax = I.cylinder_z_extent(lv, q, radius=1.0)
    close(zmin, [0, -3, 1], msg='zmin (empty cylinder -> own z)')
    close(zmax, [5, 0, 1], msg='zmax (own z included)')


def test_sample_training_set():
    labels = np.r_[np.zeros(5000), np.ones(50), -np.ones(1000), np.full(2000, 2)].astype(np.int64)
    idx = I.sample_training_set(labels, n_per_class=100, seed=1)
    counts = np.bincount(labels[idx] + 1, minlength=4)
    assert counts[0] == 0, 'sampled ignored (-1) points'
    close(counts[1:], [100, 50, 100], msg='samples per class 0,1,2')
    assert len(np.unique(idx)) == len(idx), 'duplicate indices'
    close(I.sample_training_set(labels, 100, seed=1), idx, msg='same seed must give same sample')
    assert not np.array_equal(np.sort(I.sample_training_set(labels, 100, seed=2)), np.sort(idx)), \
        'different seeds should give different samples'


# ===================================== STAGE 2 ======================================
def test_structure_tensor():
    rng = np.random.default_rng(0)
    neigh = rng.normal(size=(50, 10, 3)) * [3, 1, .1]
    w, v = I.structure_tensor(neigh)
    assert w.shape == (50, 3) and v.shape == (50, 3, 3), f'shapes {w.shape}, {v.shape}'
    assert (np.diff(w, axis=1) <= 1e-12).all(), 'eigenvalues must be DESCENDING'
    assert (w >= 0).all(), 'eigenvalues must be >= 0'
    d = neigh - np.median(neigh, axis=1, keepdims=True)
    C = np.einsum('qki,qkj->qij', d, d) / neigh.shape[1]
    close(w, np.linalg.eigvalsh(C)[:, ::-1], 1e-8, 'eigenvalues vs covariance around the MEDIAN')
    close(np.einsum('qji,qjk->qik', v, v), np.broadcast_to(np.eye(3), (50, 3, 3)), 1e-8, 'eigvecs orthonormal')
    close(v @ (w[:, :, None] * np.transpose(v, (0, 2, 1))), C, 1e-8, 'V diag(w) V^T == C (vector order matches values)')
    line = np.linspace(0, 1, 10)[:, None] * np.array([1, 1, 0]) / np.sqrt(2)
    _, v = I.structure_tensor(line[None])
    close(abs(v[0, :, 0]), [1 / np.sqrt(2), 1 / np.sqrt(2), 0], 1e-6, 'e1 of a line = its direction')


def test_covariance_features():
    eye = np.eye(3)
    vert_plane = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], float).T     # e3 = x -> vertical wall
    w = np.array([[2, 2, 0], [3, 0, 0], [1, 1, 1], [2, 2, 0], [0, 0, 0]], float)
    v = np.stack([eye, eye, eye, vert_plane, eye])
    f = I.covariance_features(w, v)
    assert f.shape == (5, 9), f'shape {f.shape}'
    assert np.isfinite(f).all(), 'non-finite values (all-zero eigenvalues must not crash)'
    n = {k: i for i, k in enumerate(utils.FEATURES_PER_SCALE[:9])}
    close(f[0, n['sum']], 4, msg='sum uses RAW eigenvalues')
    close(f[0, [n['planarity'], n['linearity'], n['sphericity'], n['verticality']]], [1, 0, 0, 0], 1e-6, 'horizontal plane')
    close(f[1, [n['linearity'], n['planarity'], n['anisotropy']]], [1, 0, 1], 1e-6, 'line')
    close(f[2, [n['sphericity'], n['omnivariance'], n['eigenentropy'], n['surface_variation']]],
          [1, 1 / 3, np.log(3), 1 / 3], 1e-6, 'sphere: sphericity, omnivariance, eigenentropy, surface variation')
    close(f[3, n['verticality']], 1, 1e-6, 'vertical wall -> verticality 1')


def test_moment_features():
    neigh = np.array([[[1, 0, 0], [2, 0, 0], [0, 3, 0]]], float)
    v = np.eye(3)[None]
    close(I.moment_features(neigh, np.zeros((1, 3)), v), [[3, 3, 5, 9]], msg='moments around origin')
    close(I.moment_features(neigh + 1, np.ones((1, 3)), v), [[3, 3, 5, 9]], msg='must use p_i - query point')


def test_height_features():
    close(I.height_features(np.array([2., 0.]), np.array([0., -3.]), np.array([5., 0.])),
          [[5, 2, 3], [3, 3, 0]], msg='[range, below, above]')


def _expected_features(pyr, q, k):
    blocks = []
    for lv in pyr:
        nb = I.knn_neighbors(lv, q, k)
        w, v = I.structure_tensor(nb)
        zmin, zmax = I.cylinder_z_extent(lv, q, CONFIG['cyl_radius_factor'] * lv.voxel_size)
        blocks += [I.covariance_features(w, v), I.moment_features(nb, q, v), I.height_features(q[:, 2], zmin, zmax)]
    return np.nan_to_num(np.hstack(blocks))


def test_compute_point_features():
    scaffold.SILENT_DUMMIES = True
    xyz, _ = make_toy_scene(3000)
    pyr = manual_pyramid(xyz, 2)
    q = xyz[:40]
    X = I.compute_point_features(pyr, q, k=5)
    assert X.shape == (40, 32), f'shape {X.shape}, expected (40, 16 * n_levels)'
    assert X.dtype == np.float32, f'dtype {X.dtype}, expected float32'
    assert np.isfinite(X).all(), 'non-finite features'
    close(X, _expected_features(pyr, q, 5), 1e-3, 'column order must be [cov 9 | moment 4 | height 3] per scale')


def test_extract_features():
    scaffold.SILENT_DUMMIES = True
    xyz, _ = make_toy_scene(1200)
    pyr = manual_pyramid(xyz, 2)
    idx = np.arange(0, 300, 3)
    X = I.extract_features(pyr, xyz, idx, chunk_size=7)
    assert X.dtype == np.float32 and X.shape == (100, 32), f'{X.dtype} {X.shape}'
    close(X, I.compute_point_features(pyr, xyz[idx]), 1e-4, 'chunked result != one-shot result')
    assert len(I.extract_features(pyr, xyz[:50], None, chunk_size=20)) == 50, 'idx=None -> all points'
    cache = os.path.join(TMP, 'sub', 'feat.npy')
    I.extract_features(pyr, xyz, idx, chunk_size=50, cache_path=cache)
    assert os.path.exists(cache), 'cache_path given but nothing saved'
    close(I.extract_features(pyr, xyz, idx, cache_path=cache), X, 1e-6, 'cached result differs')


# ===================================== STAGE 3 ======================================
def test_evaluate():
    names = ['a', 'b', 'c']
    df, s = I.evaluate(np.array([0, 0, 0, 1, 1, 2, -1]), np.array([0, 0, 1, 1, 1, 0, 2]), names)
    assert list(df.index) == names and set(['recall', 'precision', 'f1']) <= set(df.columns)
    close(df.loc[:, ['recall', 'precision', 'f1']].values,
          [[2 / 3, 2 / 3, 2 / 3], [1, 2 / 3, .8], [0, 0, 0]], 1e-6, 'per class [recall, precision, f1] (-1 ignored)')
    close([s['overall_accuracy'], s['mean_recall'], s['mean_f1']],
          [4 / 6, (2 / 3 + 1) / 3, (2 / 3 + .8) / 3], 1e-6, 'OA, mean recall, mean F1')


def test_compare_to_paper():
    P = utils.PAPER_MADAME
    names = utils.CLASS_NAMES
    per = pd.DataFrame([P[c] for c in names], index=names, columns=['recall', 'precision', 'f1'])
    summ = {'overall_accuracy': P['overall_accuracy'], 'mean_recall': P['mean_recall'], 'mean_f1': 0.8}
    df = I.compare_to_paper(per, summ)
    assert list(df.columns)[:3] == ['ours', 'paper', 'diff'], f'columns {list(df.columns)}'
    assert len(df) == 21, f'{len(df)} rows, expected 21'
    close(df.loc['cars/f1', ['ours', 'paper', 'diff']], [.9423, .9423, 0], 1e-9, 'cars/f1 row')
    close(df.loc['mean_f1', 'diff'], 0.8 - 0.7413, 1e-9, 'mean_f1 diff = ours - paper')


def test_save_timings():
    old = dict(Timer.log)
    Timer.log.clear(); Timer.log.update({'features': 1.0, 'training': 2.0})
    try:
        path = os.path.join(TMP, 'timings.csv')
        df = I.save_timings(path)
        assert os.path.exists(path), 'CSV not written'
        close(float(df.loc['total', 'seconds']), 3.0, msg='total seconds')
        assert 'features' in df.index and 'training' in df.index
    finally:
        Timer.log.clear(); Timer.log.update(old)


def _blobs(n=300, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n)
    return (rng.normal(size=(n, 6)) + 4 * np.eye(3, 6)[y]).astype(np.float32), y


def test_train_random_forest():
    X, y = _blobs()
    grid = {'n_estimators': [5, 10], 'max_depth': [2, None]}
    m, cv = I.train_random_forest(X, y, grid=grid, folds=3, seed=0)
    assert m.n_estimators in (5, 10) and m.max_depth in (2, None), 'model must use grid params'
    assert len(cv) == 4, f'cv_results should have one row per combination (4), got {len(cv)}'
    assert (m.predict(X) == y).mean() > .95, 'refit on all data -> should fit training data'
    m2, _ = I.train_random_forest(X, y, grid=grid, folds=3, seed=0)
    assert (m2.predict(X) == m.predict(X)).all(), 'same seed must give same model'


def test_classify_cloud():
    scaffold.SILENT_DUMMIES = True
    from sklearn.ensemble import RandomForestClassifier
    xyz, raw = make_toy_scene(1500)
    pyr = manual_pyramid(xyz, 2)
    m = RandomForestClassifier(10, random_state=0).fit(I.compute_point_features(pyr, xyz[::5]), raw[::5])
    idx = np.arange(0, 400, 2)
    out = I.classify_cloud(m, pyr, xyz, idx, chunk_size=13)
    assert out.shape == (200,) and out.dtype == np.int64, f'{out.shape} {out.dtype}'
    close(out, m.predict(I.compute_point_features(pyr, xyz[idx])), msg='chunked predictions differ')
    assert 'classification' in Timer.log, "wrap model.predict in Timer('classification')"


def test_plot_confusion():
    path = os.path.join(TMP, 'cm.png')
    cm = I.plot_confusion(np.array([0, 0, 1, 1, -1]), np.array([0, 1, 1, 1, 2]), path, ['a', 'b', 'c'])
    assert os.path.getsize(path) > 0, 'PNG not written'
    close(cm, [[.5, .5, 0], [0, 1, 0], [0, 0, 0]], 1e-9, 'row-normalised matrix (-1 ignored, absent class = 0)')


def test_plot_cloud():
    xyz, raw = make_toy_scene(5000)
    lab = raw % 6
    p = os.path.join(TMP, 'cloud.png')
    assert I.plot_cloud(xyz, lab, p, max_points=1000) == 1000, 'should draw max_points points'
    assert os.path.getsize(p) > 0, 'PNG not written'
    p2 = os.path.join(TMP, 'err.png')
    n = I.plot_cloud(xyz, lab, p2, max_points=1000, errors=np.arange(len(xyz)) % 7 == 0)
    assert 0 < n <= 1000 and os.path.getsize(p2) > 0, 'error figure'


def test_feature_ablation():
    rng = np.random.default_rng(0)
    names = utils.feature_names(2)
    X = rng.normal(size=(1200, 32)).astype(np.float32)
    hb = [names.index('s0_height_below'), names.index('s1_height_below')]
    y = (X[:, hb[0]] > 0).astype(int) + (X[:, hb[1]] > 0).astype(int)
    df = I.feature_ablation(X[:600], y[:600], X[600:], y[600:], names, {'n_estimators': 20}, 0)
    assert set(df.index) == set(utils.FEATURES_PER_SCALE), 'one row per feature type (16)'
    assert {'oa', 'mean_f1', 'drop_oa', 'drop_f1'} <= set(df.columns), f'columns {list(df.columns)}'
    assert df.index[0] == 'height_below', f'only height_below carries signal; top row is {df.index[0]}'
    assert df['drop_oa'].iloc[0] > .2 and (np.diff(df['drop_oa'].values) <= 1e-12).all(), 'sorted by drop_oa'


# ================================= runner / status board ============================
ORDER = ['load_ply', 'remap_labels', 'voxel_downsample', 'build_pyramid', 'knn_neighbors',
         'cylinder_z_extent', 'sample_training_set', 'structure_tensor', 'covariance_features',
         'moment_features', 'height_features', 'compute_point_features', 'extract_features',
         'evaluate', 'compare_to_paper', 'save_timings', 'train_random_forest', 'classify_cloud',
         'plot_confusion', 'plot_cloud', 'feature_ablation']


def _run(name, verbose=False):
    """Run one test with `name` in strict mode. Returns ('PASS'|'TODO'|'FAIL', message)."""
    STRICT.add(name)
    silent = scaffold.SILENT_DUMMIES
    try:
        globals()[f'test_{name}']()
        return 'PASS', ''
    except NotImplementedError:
        return 'TODO', 'not implemented (still raises NotImplementedError)'
    except AssertionError as e:
        return 'FAIL', str(e) if not verbose else traceback.format_exc()
    except Exception as e:
        return 'FAIL', f'{type(e).__name__}: {e}' if not verbose else traceback.format_exc()
    finally:
        STRICT.discard(name)
        scaffold.SILENT_DUMMIES = silent


def check(name):
    """Notebook-friendly: test one function and print the result (full error on FAIL)."""
    st, msg = _run(name, verbose=False)
    o = REGISTRY[name]['owner']
    icon = {'PASS': '✅', 'TODO': '⏳', 'FAIL': '❌'}[st]
    print(f'{icon} {st}  {name}  ({o}, {utils.who(o)})')
    if st == 'FAIL':
        print(msg)
    elif st == 'TODO':
        print('   still raises NotImplementedError: implement it in the cell above, run that cell, then this one')
    return None


def status(owner=None, names=None, verbose=False):
    """Status board. owner: 'P2' or a name like 'Jana' (None = everyone)."""
    owner = utils.resolve_owner(owner) if owner else None
    rows = []
    for name in names or ORDER:
        info = REGISTRY[name]
        if owner and info['owner'] != owner:
            continue
        st, msg = _run(name, verbose)
        rows.append({'stage': info['stage'], 'function': name, 'owner': info['owner'],
                     'name': utils.who(info['owner']), 'status': st,
                     'message': msg if verbose else msg.split('\n')[0][:90]})
    df = pd.DataFrame(rows)
    with pd.option_context('display.max_colwidth', 200, 'display.width', 250):
        print(df.to_string(index=False))
    done = (df.status == 'PASS').sum()
    print(f'\n{done}/{len(df)} passing  |  ' + '  '.join(
        f'{o} {utils.who(o)}: {(df[df.owner == o].status == "PASS").sum()}/{(df.owner == o).sum()}'
        for o in sorted(df.owner.unique())))
    return df


if __name__ == '__main__':
    scaffold.load_notebook()
    args = sys.argv[1:]
    scaffold.SILENT_DUMMIES = True
    if args and args[0] == '--owner':
        status(owner=args[1])
    elif args:
        for a in args:
            st, msg = _run(a, verbose=True)
            print(f'{st:4s}  {a}\n{msg}')
    else:
        status()
