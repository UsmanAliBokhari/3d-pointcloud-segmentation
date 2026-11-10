# Project Guide: reproducing *Fast Semantic Segmentation of 3D Point Clouds with Strongly Varying Density*

> Hackel, T., Wegner, J. D., Schindler, K. (2016). ISPRS Annals III-3, 177–184. doi:10.5194/isprsannals-III-3-177-2016
> Computer Vision course project, VUB. **Our internal deadline: Friday 20 November 2026** (official deadline 21-12-2026).

**Contents**

1. [Quick start](#1-quick-start)
2. [Computer vision concepts from zero](#2-computer-vision-concepts-from-zero)
3. [The paper, step by step](#3-the-paper-step-by-step)
4. [What we reproduce and the target numbers](#4-what-we-reproduce-and-the-target-numbers)
5. [How the code is organised](#5-how-the-code-is-organised)
6. [Division of work](#6-division-of-work)
7. [How to work: dummies, tests, workflow](#7-how-to-work-dummies-tests-workflow)
8. [Timeline (10 days)](#8-timeline-10-days)
9. [Environment setup](#9-environment-setup)
10. [Git and GitHub](#10-git-and-github)
11. [Course requirements and submission checklist](#11-course-requirements-and-submission-checklist)
12. [Decisions and differences from the paper](#12-decisions-and-differences-from-the-paper)
13. [Oral defense: questions everyone must be able to answer](#13-oral-defense-questions-everyone-must-be-able-to-answer)

---

## 1. Quick start

```bash
git clone https://github.com/UsmanAliBokhari/3d-pointcloud-segmentation.git
cd 3d-pointcloud-segmentation
bash install_MAC.sh          # or: bash install_Linux.sh
source .venv/bin/activate
jupyter lab notebooks/project.ipynb      # kernel "Python (cv-repro)" -> Run All Cells
```

**All work happens in `notebooks/project.ipynb`.** It contains all 21 functions in pipeline order. Each function has an explanation, a code cell and a test cell. Your job: find your 7 functions (the status board at the top lists them with your name), replace `raise NotImplementedError` with your code, and run the test cell until it says `✅ PASS`. Then save, commit and push.

---

## 2. Computer vision concepts from zero

### 2.1 Point clouds and LiDAR
An image is a regular grid of pixels. A **point cloud** is just a list of 3D points `(x, y, z)`. There is no grid, no fixed order, and no notion of "the pixel to the left". We get one from a **LiDAR** scanner, which fires laser pulses and measures how long each takes to bounce back, giving a distance. Combined with the pulse direction (and, on a car, GPS/IMU positioning), every return becomes a 3D point. A single street scan easily has tens of millions of points.

There are two acquisition setups in the paper:

| | **MLS: Mobile Laser Scanning** | **TLS: Terrestrial Laser Scanning** |
|---|---|---|
| Scanner | On a car driving down the street | On a tripod, static |
| Density | Fairly uniform along the road | **Strongly varying**: very dense near the tripod, sparse far away |
| In the paper | Paris-rue-Madame, Paris-rue-Cassette (Table 2) | 18 scans across Europe (Table 4) |

**Why does density vary so much for TLS?** The scanner sends rays at fixed angular steps. Rays spread out with distance, so the point spacing grows linearly with distance and the density (points per m²) falls **quadratically**. Far surfaces also return weaker signals (more missing points), and with few viewpoints, objects hide each other (**occlusion**).

### 2.2 Semantic segmentation
"Segmentation" means splitting data into meaningful parts. "Semantic" means each part gets a *class name*. For images, this is colouring every pixel as road, car, sky and so on. Here we give **every 3D point** a label such as *façade, ground, car, motorcycle, traffic sign, pedestrian*. It is a **per-point classification** problem.

### 2.3 Supervised classification pipeline
The classic recipe (used before deep learning took over 3D) is:

1. **Features:** turn each point into a fixed-length vector of numbers that describe it (here 144 numbers).
2. **Classifier:** learn from labelled examples a function `features → class`, then apply it to all points.

The paper's contribution is entirely in step 1: *which* features to use and how to compute them *fast*.

### 2.4 Neighbourhoods: kNN vs. radius search
One point alone says nothing about shape. Its **neighbourhood** does: are the nearby points on a flat sheet, along a thin line, or scattered in a blob? There are two ways to define "nearby":

* **Radius search:** all points within *r* metres. A fixed physical scale, which is conceptually right. But with varying density, the same *r* gives 3 points far from the scanner and 30,000 near it.
* **k-nearest neighbours (kNN):** the *k* closest points, whatever their distance. Density-adaptive: it automatically looks further in sparse regions. The paper uses kNN with **k = 10**.

### 2.5 kD-trees: making neighbour search fast
Brute force means comparing a query against all *N* points, O(N) per query. For 20 M queries that is 20 M × 20 M comparisons, which is impossible. A **kD-tree** recursively splits space in two along x, then y, then z, and so on, like a 3D binary search. Finding neighbours then costs about **O(log N)** per query. The paper used FLANN; we use `scipy.spatial.cKDTree` (`tree.query(points, k=10, workers=-1)` uses all CPU cores).

### 2.6 Voxel grid downsampling
A **voxel** is a 3D pixel: a small cube. The **voxel-grid filter** cuts space into cubes of edge *s* and replaces all points inside each cube by their **centroid** (average). The result has at most one point per cube. It is fast and memory-light, since only non-empty voxels exist. It also **evens out density**: dense areas lose many points, sparse areas almost none.

### 2.7 The scale pyramid (the paper's key idea)
Ten neighbours in the original cloud span only a few centimetres. That is enough to see surface shape, but not enough to see that a point belongs to a 3 m tall pole or a building. Large-scale context would normally need k = thousands, which is too slow.

The trick: **downsample repeatedly**, doubling the voxel size each time: 2.5 cm → 5 cm → 10 cm → … → 6.4 m (**9 levels**). Then take the 10 nearest neighbours *in every level*:

```
level 0 (2.5 cm voxels):  10 neighbours span ~ 10 cm   -> fine surface detail
level 4 (40 cm voxels):   10 neighbours span ~ 1.5 m   -> object parts
level 8 (6.4 m voxels):   10 neighbours span ~ 25 m    -> scene context
```

Each coarse point stands for a whole cube of original points, so 10 coarse points represent a large region well enough. Two things get cheaper: k stays tiny, and coarse levels have far fewer points to search. That is how the paper gets features over 9 scales in minutes instead of hours. Note that the **query points always come from the original cloud**: every original point gets classified, but its neighbours are looked up in each level.

### 2.8 Covariance, eigenvalues and shape
Take the 10 neighbours, subtract their centre, and compute the 3×3 **covariance matrix** (the paper calls it the *structure tensor*):

$$C = \frac{1}{k}\sum_{i}(p_i - \bar p)(p_i - \bar p)^\top$$

Its **eigenvectors** e1, e2, e3 are the main directions in which the points spread. Its **eigenvalues** λ1 ≥ λ2 ≥ λ3 ≥ 0 say *how much* they spread along each direction. Think of fitting an ellipsoid to the points:

| Shape | Eigenvalues | Example | Feature that lights up |
|---|---|---|---|
| Line | λ1 ≫ λ2 ≈ λ3 ≈ 0 | pole, cable, tree trunk | **linearity** = (λ1−λ2)/λ1 ≈ 1 |
| Plane | λ1 ≈ λ2 ≫ λ3 ≈ 0 | wall, road | **planarity** = (λ2−λ3)/λ1 ≈ 1 |
| Blob | λ1 ≈ λ2 ≈ λ3 | bush, noise | **sphericity** = λ3/λ1 ≈ 1 |

**e3** (the direction of least spread) is the **surface normal**. **Verticality** = 1 − |e3 · (0,0,1)| is ≈ 0 for the ground (normal points up) and ≈ 1 for walls (normal is horizontal). Eigenvalues are normalised to sum to 1, so the features don't depend on absolute density.

### 2.9 Random Forest
A **decision tree** asks a sequence of yes/no questions ("planarity at scale 3 > 0.7?", "height_below at scale 6 > 2 m?") and ends in a class. A single tree overfits. A **Random Forest** trains many trees (here ~50), each on a random sample of the data and considering random subsets of features at each split, and lets them **vote**. The paper chose it because it is:

* fast and easy to parallelise
* naturally multi-class
* able to output probabilities
* known to work well on point clouds

**Gini impurity** is the criterion each split tries to reduce: how mixed the classes are in a node.

### 2.10 Hyperparameters, grid search, cross-validation
Number of trees and maximum depth are **hyperparameters**: they are not learned, we choose them. **Grid search** tries every combination. **5-fold cross-validation** splits the training set into 5 parts, trains on 4 and validates on 1, rotates 5 times and averages. Then the best setting is **refit on all training data**. This avoids picking settings that only look good on one lucky split.

### 2.11 Class imbalance and balanced sampling
A street scan is ~70% façade and ground, and pedestrians are a tiny fraction. Points close to the scanner are also over-represented. Training on raw frequencies would teach the classifier "always say façade". The paper instead trains on **1000 random points per class**. That is a balanced, tiny training set, and it is enough.

### 2.12 Evaluation metrics (Table 2)

| Metric | Question it answers | Formula |
|---|---|---|
| **Recall** (per class) | Of all real cars, how many did we find? | TP / (TP + FN) |
| **Precision** (per class) | Of everything we called "car", how much really is a car? | TP / (TP + FP) |
| **F1** (per class) | Balance of both | 2PR / (P + R) |
| **Overall accuracy (OA)** | Fraction of all points correct | dominated by big classes |
| **Mean class recall / mean F1** | Average over classes | every class counts equally |

**Why is pedestrian precision only 0.24 while recall is 0.999?** The test set has millions of façade points and only a few thousand pedestrian points. If just 0.1% of façade points get mislabelled "pedestrian", those wrong points outnumber the real pedestrians. Recall stays high, precision collapses. The row-normalised **confusion matrix** (rows = true class, columns = predicted) makes this visible.

---

## 3. The paper, step by step

**Problem.** Label every point of huge, unstructured point clouds whose density varies strongly, and do it fast (prior work: hours per scan).

**Insight.** The bottleneck is neighbour search at many scales. Approximate the large neighbourhoods with a downsampled pyramid. That makes multi-scale features cheap, so you can use *all* scales instead of picking a single "best" one (Weinmann et al. 2013 picked one per point).

**Pipeline**

```
             ┌──────────────────────────── Stage 1: data & neighbourhoods ──────────────┐
 .ply file → │ load xyz + labels → voxel pyramid (9 levels) → kD-tree per level         │
             │   for each original point:  kNN (k=10) in every level                    │
             │                             vertical cylinder (radius) in every level    │
             └──────────────────────────────────────────────────────────────────────────┘
             ┌──────────────────────────── Stage 2: features (Table 1) ─────────────────┐
             │ per level: covariance -> eigenvalues/vectors -> 9 covariance features    │
             │            + 4 moment features + 3 height features  = 16                 │
             │ 16 × 9 levels = 144-D feature vector per point                           │
             └──────────────────────────────────────────────────────────────────────────┘
             ┌──────────────────────────── Stage 3: classify & evaluate ────────────────┐
             │ 1000 pts/class → Random Forest (grid search + 5-fold CV)                 │
             │ predict every point (features computed on the fly, not stored)           │
             │ precision / recall / F1 / OA / mean recall  vs. Table 2                  │
             └──────────────────────────────────────────────────────────────────────────┘
```

**Section 3.1: neighbourhood approximation.** kNN rather than radius, because of varying density. Voxel grid with smallest spacing **2.5 cm**, **9 levels**, factor **2** (max 6.4 m), **k = 10**. Each level has far fewer points than the one before, so the whole pyramid needs at most ~2× the memory of the finest level. After downsampling, density is even enough that **radius** search works, which is used for the **vertical-cylinder** height features (kNN is unreliable for cylinders).

**Section 3.2: features (Table 1), per scale.**

| Group | Feature | Formula (λ normalised, except Sum) | Intuition |
|---|---|---|---|
| covariance | Sum | λ1+λ2+λ3 | overall spread (raw λ) |
| | Omnivariance | (λ1λ2λ3)^(1/3) | volume-like spread |
| | Eigenentropy | −Σ λi ln λi | disorder of the spread |
| | Anisotropy | (λ1−λ3)/λ1 | how directional |
| | Planarity | (λ2−λ3)/λ1 | flat sheet |
| | Linearity | (λ1−λ2)/λ1 | thin line |
| | Surface variation | λ3/(λ1+λ2+λ3) | curvature / roughness |
| | Sphericity | λ3/λ1 | blob |
| | Verticality | 1 − \|⟨[0,0,1], e3⟩\| | wall vs ground |
| moment | 1st order, axis 1 / 2 | Σ⟨pᵢ−p, e1⟩, Σ⟨pᵢ−p, e2⟩ | is the point at the edge of its neighbourhood? (occlusion boundaries, crease edges) |
| | 2nd order, axis 1 / 2 | Σ⟨pᵢ−p, e1⟩², Σ⟨pᵢ−p, e2⟩² | spread along the axes |
| height | Vertical range | z_max − z_min | how tall is the column here |
| | Height below | z − z_min | ≈ height above ground (**most important feature**) |
| | Height above | z_max − z | how much is above me |

Total: 16 × 9 = **144 dimensions**. Colour and intensity are deliberately **not** used: not always available, and unreliable under different lighting.

**Section 3.2.1: optional histogram descriptors** (A-SHOT, A-SC3D). These are approximate versions of SHOT and 3D Shape Context, computed only for points likely to lie on contours. They help little (+1.7 F1 points on TLS) and were **not used for the Paris datasets**, so we skip them (mention this in the report).

**Section 3.3: classification.** Random Forest, Gini, parameters by 5-fold CV (they found **50 trees, depth ≈ 30**). Training set downsampled to roughly uniform resolution / balanced. At test time, features are computed **on the fly** and never stored (memory).

**Section 4: experiments.**
* MLS: Paris-rue-Madame and Paris-rue-Cassette, **1000 training samples per class**, test on all other points (~2·10⁷ and ~1.2·10⁷).
* TLS: train on 8 scans, test on 10.
* Implementation: C++, PCL, FLANN, OpenMP, on a 6-core Xeon with 64 GB RAM (classification < 16 GB).

**Section 4.2: results on MLS.** OA 97.6% (Madame), 95.4% (Cassette). Mean F1 is more than 20 points higher than the baseline. The authors attribute the gain to using the full pyramid of scales. **Feature ablation:** removing *z − z_min* costs ~5 points; any other single feature < 1 point. **Timing (Table 3):** 267 s total vs 23,092 s for the baseline.

**Conclusion and future work.** Fast and accurate. Future directions: CRF smoothing, line features, deep networks on point clouds (which is exactly what happened next: PointNet in 2017; the same authors then released the Semantic3D benchmark).

---

## 4. What we reproduce and the target numbers

**Primary target: Table 2, Paris-rue-Madame, base features (no A-SHOT/A-SC3D).**

| Class | Recall | Precision | F1 |
|---|---|---|---|
| Façade | 0.9799 | 0.9902 | 0.9851 |
| Ground | 0.9692 | 0.9934 | 0.9811 |
| Cars | 0.9786 | 0.9086 | 0.9423 |
| Motorcycles | 0.9796 | 0.4792 | 0.6435 |
| Traffic signs | 0.9939 | 0.3403 | 0.5070 |
| Pedestrians | 0.9987 | 0.2414 | 0.3888 |
| **Overall accuracy** | | | **0.9755** |
| **Mean class recall** | | | **0.9833** |
| **Mean F1** | | | **0.7413** |

**Also reproduced:** the feature-relevance test (§4.2, `results/ablation.csv`) and a timing table like Table 3 (`results/timings.csv`).

**Not reproduced (explain in report):**
* Paris-rue-Cassette: we could not find a current public download.
* The TLS scans of Table 4: not public in that form. Semantic3D is their later public successor, optional only if we have time.
* A-SHOT/A-SC3D: not used by the paper on MLS data.

**Expectation:** Python is slower than the authors' C++. Our test run measured **~11 s per 100k points** for all 144 features (2-core machine). The full 20 M point cloud is therefore roughly **20–40 minutes** of feature extraction on a laptop. For development, use `--max-points 2000000`.

---

## 5. How the code is organised

```
3d-pointcloud-segmentation/
├── notebooks/project.ipynb   ★ ALL OUR WORK: 21 functions + explanations + test cells + visual checks
├── README.md                 course-required: citation, names, how to run, hardware, expected numbers
├── run.sh                    single command that reproduces everything (course requirement)
├── install_MAC.sh            one-command environment setup (macOS)
├── install_Linux.sh          one-command environment setup (Linux)
├── requirements.txt          pinned before submission
├── config.yaml               every experiment setting + team names
├── algorithm.txt             pseudocode (course requirement)
├── paper.pdf                 the paper
├── code/                     plumbing: nobody needs to edit these
│   ├── utils.py              paths, config, team names, ScaleLevel, Timer, constants, toy scene
│   ├── scaffold.py           @task decorator, dummy outputs, loads the notebook's functions
│   ├── nb_env.py             imports available inside the notebook's function cells
│   ├── tests.py              known-answer tests + status board
│   ├── main.py               the pipeline (what run.sh calls; reads the SAVED notebook)
│   ├── export_code.py        writes implementation.py from the notebook (run.sh does this)
│   └── implementation.py     GENERATED plain-Python copy of the notebook's functions (for graders)
├── data/                     download.sh, README-data.txt, data_link.txt (.ply files are gitignored)
├── results/                  metrics.csv, paper_comparison.csv, ablation.csv, timings.csv, figures/
├── report/REPORT_OUTLINE.md  report skeleton with section owners
└── docs/PROJECT_GUIDE.md     this document
```

**How the notebook connects to everything else.** Every function cell starts with `#| export` and has the `@task` decorator.
* **In the notebook:** running a cell registers your version, so the test cells, the status board and `run(toy=True)` immediately use it. There is no need to save first.
* **In a terminal:** `bash run.sh`, `python3 code/main.py` and `python3 code/tests.py` read the `#| export` cells from the **saved** notebook. So save (Ctrl/Cmd + S) before using the terminal.
* `run.sh` also regenerates `code/implementation.py`, a readable `.py` copy for the graders. It satisfies the course's `code/` requirement. Never edit it by hand.

**Why one shared notebook doesn't turn into merge hell** (we tested this with three simulated teammates editing and pushing at the same time):
1. Outputs and execution counts are stripped on commit (`nbstripout`). Git then only sees the code you changed.
2. Each person edits only their own function cells. Those are different places in the file, so git merges them automatically.
3. If a real conflict happens anyway (e.g. two people edited the scratch cell), `nbdime` keeps the notebook valid. The conflict shows up inside that one cell. See §10.4.

### The data contract (shapes passed between functions)

| Object | Type / shape | Made by | Used by |
|---|---|---|---|
| `xyz` | float64 (N, 3), metres | `load_ply` | everything |
| `raw_class` → `labels` | int64 (N,), 0..5, **−1 = ignore** | `load_ply` → `remap_labels` | sampling, evaluation |
| `pyramid` | list of 9 `ScaleLevel(voxel_size, points, tree, tree_xy)` | `build_pyramid` | neighbour queries |
| `neigh` | float64 (Q, k, 3) neighbour coordinates | `knn_neighbors` | features |
| `zmin, zmax` | float64 (Q,) each | `cylinder_z_extent` | `height_features` |
| `eigvals, eigvecs` | (Q, 3) descending, (Q, 3, 3) columns | `structure_tensor` | covariance + moment features |
| `X` | float32 (Q, 144), order = `feature_names()` | `compute_point_features` | classifier |
| `train_idx` | int64 indices | `sample_training_set` | training |
| `model` | fitted `RandomForestClassifier` | `train_random_forest` | `classify_cloud` |

**Don't change a contract alone.** If you need to, agree in the group chat first and update the docstring and test together.

---

## 6. Division of work

**Principle: everybody touches all three stages.** Each person implements functions in **data/neighbourhoods**, **features** *and* **classification/evaluation**, so everyone learns the whole CV pipeline. Bilal explicitly asked not to have one coder and one report writer, and the peer evaluation asks for individual contributions. 21 functions, **7 each**:

| Stage | **P1** | **P2** | **P3** |
|---|---|---|---|
| **1. Data & neighbourhoods** | `voxel_downsample`, `build_pyramid` | `knn_neighbors`, `cylinder_z_extent`, `sample_training_set` | `load_ply`, `remap_labels` (+ verify class ids, `data/download.sh`) |
| **2. Features** | `structure_tensor`, `covariance_features` | `moment_features`, `height_features` | `compute_point_features`, `extract_features` |
| **3. Classify & evaluate** | `evaluate`, `compare_to_paper`, `save_timings` | `train_random_forest`, `classify_cloud` | `plot_confusion`, `plot_cloud`, `feature_ablation` |
| **You will be able to explain** | voxel grids and the pyramid; covariance, eigenvalues and all 9 shape features; every metric | kD-trees, kNN vs radius; moments and height features; Random Forest, Gini, grid search and CV | the dataset and labels; how 144-D vectors are assembled and why chunking; confusion matrix, failure cases, ablation |

**Current assignment: P1 = Usman, P2 = Jana, P3 = Divi.** To swap, change `team:` in `config.yaml` (permanent, for everyone) or the `P1/P2/P3` strings in the notebook's first cell (just your session). The status board and dummy warnings then show the new names.

**Report and deliverables**

| Item | Owner | Reviewer |
|---|---|---|
| 3a Data acquisition (+ `data/README-data.txt`) | P3 | P1 |
| 3b Preprocessing (voxel pyramid, sampling) | P1 | P2 |
| 3c Algorithm: neighbourhoods + features | P1 & P2 (their functions) | P3 |
| 3c Algorithm: classifier, on-the-fly classification | P2 | P1 |
| 3d Differences from the paper (§12 of this guide) | P3 | all |
| 4 Quantitative results (Table 2/3 style tables) | P1 | P3 |
| 4 Qualitative results (figures, failure cases) | P3 | P2 |
| 5 Discussion (why our numbers differ) | all, P2 edits | all |
| 1–2 Title, summary (≤ 200 words), 6 reproducibility checklist | P2 | all |
| `algorithm.txt` | each writes the part for their functions | P1 merges |
| `README.md`, `requirements.txt` pinning, clean-clone test | P3 (Linux) + P1/P2 (Mac) | all |

**Pairing rule for learning:** before you start a function, read the docstring of the function *before* and *after* yours in the pipeline. When you push something, ping a teammate to review it (5 minutes). The oral defense can ask anyone about anything.

---

## 7. How to work: dummies, tests, workflow

### The notebook, top to bottom
1. **Team cell:** `P1 = "Usman"`, `P2 = "Jana"`, `P3 = "Divi"`. `ME` stays empty: it is detected from your `git config user.name` (so nobody commits a change to that line).
2. **Setup cell:** imports (don't edit).
3. **Status board:** all 21 functions with owner, name and PASS / TODO / FAIL, then *your* 7 functions.
4. **Stage 1 → 2 → 3:** for each function, a short explanation, the code cell (`#| export`, `@task`, docstring with inputs/outputs/hints, `raise NotImplementedError`) and a test cell `check('name')`. After each stage there is a **visual check** (plots of the pyramid, neighbourhoods, features, predictions).
5. **Full pipeline:** `run(toy=True)` and the final status board.
6. **Scratch area:** your experiments. Empty it before committing.

### Dummies: nobody waits for anybody
While a function still raises `NotImplementedError`, `@task` returns a **dummy output with the correct shape**. For example, the dummy `knn_neighbors` returns the first k points for everyone, and the dummy `covariance_features` returns zeros. So **Run All Cells always works from day one**. As functions get implemented, every later cell automatically uses the real ones. At the end, `run()` lists which functions still used dummies (results are meaningless until none are left).

### Tests: does my function do what it should?
Each function has a **known-answer test** (in `code/tests.py`). Tests build small hand-made inputs, so **your test does not depend on your teammates' code**. A test runs *its* function in strict mode, so a dummy can never pass.

```
✅ PASS  knn_neighbors  (P2, Jana)
⏳ TODO  moment_features  (P2, Jana)
❌ FAIL  covariance_features  (P1, Usman)
horizontal plane
 got      [0. 1. 0. 1.]
 expected [1. 0. 0. 0.]
```

The FAIL message names the case and shows what you returned vs what was expected. Run `status(verbose=True)` for full tracebacks.

> `compute_point_features`, `extract_features` and `classify_cloud` check *assembly*. They compare your output to what the team's current component functions produce (real or dummy), so they can pass before the components are done.

From a terminal (reads the saved notebook): `python3 code/tests.py`, `python3 code/tests.py knn_neighbors`, `python3 code/tests.py --owner Jana`.

### Workflow for one function
1. `git pull --rebase` (in a terminal), then open the notebook, **Kernel → Restart & Run All**.
2. Read the explanation and docstring of your function. Read the paper section it names, and the function before and after yours.
3. Replace `raise NotImplementedError` with your code. Run the cell, then the test cell below it. Repeat until `✅ PASS`.
4. Re-run the status board and `run(toy=True)` at the bottom: everything still works?
5. Save, empty the scratch cell, commit and push (§10.3).

## 8. Timeline (10 days)

| Day | Date | Everyone | Milestone |
|---|---|---|---|
| Tue | 10 Nov | Usman creates repo, pushes skeleton, adds collaborators | repo live |
| Wed | 11 Nov | Clone + install, read this guide and the paper, run `main.py --toy`. P3: `bash data/download.sh`, check class ids with `np.unique` | everyone's setup works |
| Thu | 12 Nov | Stage-1 functions (each person their own) | stage-1 tests PASS |
| Fri | 13 Nov | Stage-2 functions | stage-2 tests PASS |
| Sat–Sun | 14–15 Nov | Stage-3 functions; 15-min call: each explains their functions to the others | **21/21 PASS** |
| Mon | 16 Nov | Integration on real data: `bash run.sh --max-points 2000000`; fix speed issues (cylinder search!) | real-data numbers |
| Tue | 17 Nov | Full run (`bash run.sh`), ablation, figures, timings. Write report sections in parallel | results final |
| Wed | 18 Nov | Report draft complete, `algorithm.txt`, README with expected numbers | draft |
| Thu | 19 Nov | **Clean-clone test** on one Mac and the Linux machine; pin `requirements.txt`; review report; oral practice (§13) | release candidate |
| Fri | 20 Nov | Make zip, final read-through, send to Bilal.Moussa.Fares@vub.be | **submitted** |

If something slips, the official deadline (21-12-2026) is a safety net, not a plan.

---

## 9. Environment setup

**macOS (Intel or Apple Silicon):** `bash install_MAC.sh`
**Linux:** `bash install_Linux.sh`

Both scripts:
1. find Python ≥ 3.9 (the Mac script installs Python 3.12 with Homebrew if needed)
2. create `.venv/` and install `requirements.txt`
3. register a Jupyter kernel **"Python (cv-repro)"**
4. enable `nbstripout` (strips notebook outputs from commits)
5. run the status board and a toy pipeline as a smoke test

Afterwards, in every new terminal: `source .venv/bin/activate`.

**Hardware.** No GPU is needed: this is a CPU method (kD-trees + Random Forest). 16 GB RAM is comfortable for the full dataset because test features are computed chunk by chunk. 8 GB: use `--max-points` or a smaller `chunk_size` in `config.yaml`. **Google Colab is not needed.** If someone's laptop is too slow, Colab works: open the notebook there and run the optional "Colab" cell at the top (clones the repo and installs requirements).

**Troubleshooting**

* `xcrun: error: invalid active developer path` → run `xcode-select --install`.
* `ensurepip is not available` → run `sudo apt install python3-venv`.
* Kernel not shown in Jupyter → restart JupyterLab and pick "Python (cv-repro)".

---

## 10. Git and GitHub

### 10.1 One-time: put the skeleton in the repository (Usman)
The repository already exists: **https://github.com/UsmanAliBokhari/3d-pointcloud-segmentation** (public, `main` branch, contains a LICENSE commit). Clone it and copy the project files in, so the LICENSE history is kept:

```bash
# 1. clone the (almost empty) repo somewhere, e.g. ~/projects
cd ~/projects
git clone https://github.com/UsmanAliBokhari/3d-pointcloud-segmentation.git
# 2. unzip the skeleton somewhere else, e.g. ~/Downloads
cd ~/Downloads && unzip 3d-pointcloud-segmentation.zip
# 3. copy everything (including hidden files like .gitignore) into the clone
cp -R ~/Downloads/3d-pointcloud-segmentation/. ~/projects/3d-pointcloud-segmentation/
# 4. set up the environment INSIDE the clone (also configures the notebook git tools)
cd ~/projects/3d-pointcloud-segmentation
bash install_MAC.sh            # or install_Linux.sh
# 5. commit and push
git add .
git status                     # check: no .venv/, no .ply files
git commit -m "Project skeleton: stubs, tests, notebook, docs, install scripts"
git push
```

Then **Settings → Collaborators → Add people** → add both teammates. Even on a public repo, only collaborators can push.

**Authentication:** GitHub doesn't accept your password over HTTPS. Use `gh auth login` (easiest), or create a *Personal access token* (Settings → Developer settings → Tokens) and paste it as the password, or set up an SSH key.

> **Public repo note:** other groups can see the code. That's fine for us (each group has a different paper), but don't post the repo link in course channels before the deadline. **Never commit the dataset**: it's ~hundreds of MB (GitHub's limit is 100 MB per file) and its licence forbids redistribution. `.gitignore` already excludes it.

### 10.2 One-time: teammates

```bash
git clone https://github.com/UsmanAliBokhari/3d-pointcloud-segmentation.git
cd 3d-pointcloud-segmentation
git config user.name  "Jana Peeters"     # must CONTAIN your team name (Usman / Jana / Divi): the notebook uses it for ME
git config user.email "you@example.com"
bash install_MAC.sh                      # or install_Linux.sh: run it INSIDE the clone
```

Running the install script inside the clone matters. It configures `nbstripout` and `nbdime` for this repository. Without them, notebook outputs get committed and conflicts become likely.

### 10.3 Every work session

```bash
cd 3d-pointcloud-segmentation
git pull --rebase                       # 1. get teammates' work FIRST (before opening the notebook)
source .venv/bin/activate
jupyter lab notebooks/project.ipynb     # 2. Kernel -> Restart & Run All, implement, test until PASS
                                        # 3. empty the scratch cell, SAVE (Cmd/Ctrl + S)
python3 code/tests.py --owner Jana      # 4. optional: same board from the terminal
git add notebooks/project.ipynb         # 5. (plus any other file you deliberately changed)
git commit -m "P2: knn_neighbors (test passes)"
git pull --rebase                       # 6. merge anything pushed meanwhile
git push                                # 7. share
```

**After step 6**, if a teammate's work came in, reload the notebook in Jupyter (**File → Reload Notebook from Disk**). Otherwise your open tab still shows the old version, and the next save would overwrite their work. Simplest habit: close the notebook tab before `git pull`, and reopen it after.

**Commit messages:** `P<n>: <what>`, e.g. `P1: covariance_features, all 9 features, test passes`. These double as evidence for the peer evaluation.

### 10.4 If `git pull --rebase` reports a conflict
This only happens if two people changed the same cell (e.g. both left something in the scratch area). Thanks to nbdime, the notebook stays a valid notebook, and the conflict appears **inside that cell**:

```
<<<<<<< local
# jana was here
=======
# divi experiment
>>>>>>> remote
```

1. Open the notebook in Jupyter, find the cell with `<<<<<<<` (Ctrl/Cmd + F), keep the right version, delete the markers, save.
2. In the terminal:
   ```bash
   git add notebooks/project.ipynb
   git rebase --continue
   git push
   ```

Panic button: `git rebase --abort` puts you back where you were before the pull.

### 10.5 Rules
1. **Pull before you open the notebook; reload after pulling.** Push when a test passes. Small commits, often.
2. **Only edit inside your own function cells.** Don't add, delete or move cells in the stage sections. If you need a new helper function, define it inside your function cell.
3. **Empty the scratch cell before committing**, or use a personal copy `notebooks/scratch_<name>.ipynb` (gitignored).
4. Don't change the team cell's `P1/P2/P3` without agreeing in the chat. Leave `ME` empty.
5. Files in `code/`, `config.yaml`: only after agreeing in the chat.
6. Never push a notebook where **Restart & Run All** crashes.
7. Optional, if you prefer reviews: work on a branch (`git switch -c p2-knn`, then `git push -u origin p2-knn`) and open a Pull Request on GitHub for a teammate to merge.

---

## 11. Course requirements and submission checklist

From *ProjectDetails.pdf* and Bilal's announcement:

* **Reproduce** the paper's results: same or similar data, same kind of tests.
* **Code** must run with a **single command** (`bash run.sh`), **relative paths only**, no editing of files needed.
* **README** with paper citation, contact names, how to run, hardware used, **expected metric numbers**.
* **requirements.txt**, **algorithm.txt** (polished pseudocode, also sent separately to the instructor), **report.pdf**, **paper.pdf**, **results/** (tables + figures), data or `data_link.txt` + download instructions in `data/README-data.txt`.
* **Seeds** set and reported; **hardware and runtimes** reported.
* **Report structure:**
  1. Title, authors, article
  2. Summary (≤ 200 words)
  3. Implementation: data acquisition, preprocessing, algorithm, differences from paper
  4. Results: quantitative tables like the paper; qualitative figures incl. **failure cases**
  5. Discussion: comparison to the paper, reasons for differences
  6. Reproducibility checklist
* **Submission:** one zip (`3d-pointcloud-segmentation.zip`) without large raw data, emailed to **Bilal.Moussa.Fares@vub.be**.
* **AI use is allowed, but you must be able to explain everything at the oral defense.** There will be a **peer evaluation** of individual contributions.

**Final checklist (Thursday 19 Nov)**

- [ ] Fresh `git clone` into a new folder → `bash install_*.sh` → `bash run.sh` works on Mac **and** Linux
- [ ] `grep -rnE "/Users/|/home/|C:\\\\" code/ run.sh` finds nothing
- [ ] Notebook: Restart & Run All → final status board 21/21 PASS (and `python3 code/tests.py` in a terminal)
- [ ] `requirements.txt` pinned to exact versions
- [ ] README: names, citation, hardware, expected numbers (copy from `results/metrics.csv`)
- [ ] `algorithm.txt` final
- [ ] `report.pdf` (all 6 sections, summary ≤ 200 words, checklist included)
- [ ] `results/`: metrics.csv, paper_comparison.csv, ablation.csv, timings.csv, figures (prediction + errors + confusion)
- [ ] Run `bash run.sh` once in the folder you zip, so `code/implementation.py` (generated, gitignored) is included
- [ ] Zip excludes `.venv/`, `data/*.ply`, `cache/`, `.git/`:
  `zip -r 3d-pointcloud-segmentation.zip . -x ".venv/*" ".git/*" "data/*.ply" "cache/*" "*__pycache__*"`

---

## 12. Decisions and differences from the paper

Write these in report section 3d. Each is marked in the code.

| # | Paper | Ours | Why |
|---|---|---|---|
| 1 | C++, PCL, FLANN, ETH RF library, OpenMP | Python, NumPy, SciPy `cKDTree`, scikit-learn | course setting; same algorithms |
| 2 | Centre of neighbourhood = medoid | coordinate-wise median | cheap approximation, same robustness idea |
| 3 | Eigenvalues normalised to sum 1 | all features except *Sum* use normalised λ; *Sum* uses raw λ | otherwise *Sum* is constantly 1 |
| 4 | Cylinder radius not given | radius = 3 × voxel size of the level (`config.yaml`) | must choose something; could tune |
| 5 | Exact radius search for cylinders | (if implemented) 2D grid-cell approximation | speed in Python |
| 6 | Same train/test split as Weinmann et al. 2015 | 1000 random points per class (seed 42), test = all other labelled points | exact split not published |
| 7 | Paris-rue-Cassette and TLS scans also evaluated | Madame only | Cassette not downloadable now, TLS scans not public |
| 8 | A-SHOT / A-SC3D | not implemented | paper did not use them on MLS data |
| 9 | Grid search ranges not given | 25/50/100 trees × depth 10/20/30 | includes the paper's 50 / 30 |

---

## 13. Oral defense: questions everyone must be able to answer

1. What is a point cloud, how does LiDAR produce one, and why does TLS density vary so much?
2. What is semantic segmentation of a point cloud? What are the classes here?
3. kNN vs radius search: which does the paper use for what, and why?
4. What is a kD-tree, and what does it make faster?
5. Explain the voxel-grid filter. Why does it even out density?
6. Explain the scale pyramid. Why is "10 neighbours at 9 levels" so much cheaper than large neighbourhoods? Which points are classified: original or downsampled?
7. What is the covariance matrix of a neighbourhood? What do its eigenvalues and eigenvectors mean? Draw the line/plane/blob cases.
8. Define planarity, linearity, verticality. What value does each take on a road, a wall, a pole?
9. What do the moment and height features add? Why is z − z_min the most important feature?
10. Why 144 features? (16 per scale × 9 scales.)
11. How does a Random Forest work? What is Gini impurity? Why a RF here?
12. What are grid search and 5-fold cross-validation for?
13. Why train on only 1000 points per class? What goes wrong with raw class frequencies?
14. Define precision, recall, F1, overall accuracy, mean class recall. Why is pedestrian precision so low but recall so high?
15. Why are features computed on the fly at test time? (Hint: 20 M × 144 × 4 bytes.)
16. Our numbers vs the paper's: what differs, and what are plausible reasons? (§12)
17. Why is our implementation slower than the paper's, and where is the bottleneck?
18. What would you do next? (CRF smoothing, deep learning on points, e.g. PointNet; GPU.)

---

*Sources: the paper; ProjectDetails.pdf; Serna et al., "Paris-rue-Madame database", ICPRAM 2014; dataset page https://people.cmm.minesparis.psl.eu/users/serna/rueMadameDataset.html*
