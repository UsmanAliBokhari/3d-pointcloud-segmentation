# Fast Semantic Segmentation of 3D Point Clouds with Strongly Varying Density: a reproduction

Reproduction of

> T. Hackel, J. D. Wegner, K. Schindler. **Fast semantic segmentation of 3D point clouds with strongly varying density.** *ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences*, III-3, 177–184, 2016. doi:[10.5194/isprsannals-III-3-177-2016](https://doi.org/10.5194/isprsannals-III-3-177-2016) (`paper.pdf`)

Computer Vision course project, Vrije Universiteit Brussel, 2026–27. Repository: https://github.com/UsmanAliBokhari/3d-pointcloud-segmentation

| Role | Name | Contact |
|---|---|---|
| P1 | Usman Ali Bokhari | _email_ |
| P2 | Jana _(surname)_ | _email_ |
| P3 | Divi _(surname)_ | _email_ |

## Team: first-time setup (do this once)

```bash
git clone https://github.com/UsmanAliBokhari/3d-pointcloud-segmentation.git
cd 3d-pointcloud-segmentation
git config user.name "Jana Peeters"   # ⚠️ IMPORTANT: must CONTAIN your team name (Usman / Jana / Divi)
bash install_MAC.sh                   # Linux: bash install_Linux.sh  (run it INSIDE the clone)
source .venv/bin/activate
jupyter lab notebooks/project.ipynb   # or open the folder in VS Code; kernel: .venv / "Python (cv-repro)"
```

> **Why the git name matters:** the notebook detects who you are from `git config user.name`, so the status board can show *your* 7 functions. Check yours with `git config user.name`. If it doesn't contain your team name, set it as above (it applies to this repo only).

**New to the project? Read [`docs/PROJECT_GUIDE.md`](docs/PROJECT_GUIDE.md) first.** It explains the paper and every computer vision concept from scratch, the work split, the timeline, and the git workflow.

## What it does
Every point of a 3D LiDAR point cloud gets a semantic label (façade, ground, car, motorcycle, traffic sign, pedestrian):

1. A 9-level voxel-grid pyramid (2.5 cm → 6.4 m) approximates multi-scale neighbourhoods.
2. 144 geometric features are computed per point: eigenvalue-based shape, moments and height, at 9 scales.
3. A Random Forest, trained on 1000 points per class, classifies the cloud.

Dataset: Paris-rue-Madame (MINES ParisTech).

## How to run

```bash
git clone https://github.com/UsmanAliBokhari/3d-pointcloud-segmentation.git && cd 3d-pointcloud-segmentation
```

```bash
bash install_MAC.sh        # macOS   (or: bash install_Linux.sh on Linux) - one time
bash run.sh                # downloads the data if needed, runs everything, writes results/
```

**Development happens in `notebooks/project.ipynb`** (all 21 functions, explanations and tests). Open it with `jupyter lab notebooks/project.ipynb`.

Quick variants (activate the environment first: `source .venv/bin/activate`):

```bash
python3 code/main.py --toy                    # synthetic scene, about 1 minute, no download
bash run.sh --max-points 2000000              # 2 M point random subset of the real data
python3 code/tests.py                         # implementation status (reads the saved notebook)
```

Settings are in `config.yaml`. All paths are relative to the project root. The random seed is `42` (`config.yaml: seed`), applied to Python, NumPy, training-point sampling, cross-validation folds and the Random Forest.

## Outputs (`results/`)

| File | Content |
|---|---|
| `metrics.csv` | per-class recall / precision / F1, overall accuracy, mean recall, mean F1 |
| `paper_comparison.csv` | our numbers vs. the paper's Table 2 |
| `ablation.csv` | feature-relevance test (paper §4.2) |
| `timings.csv` | runtime per stage + hardware (paper Table 3) |
| `cv_results.csv` | grid search / 5-fold cross-validation scores |
| `figures/` | predictions, misclassified points (failure cases), confusion matrix |

## Expected numbers
_To be filled in after the final run (from `results/metrics.csv`)._

| | Ours | Paper (Table 2, Madame) |
|---|---|---|
| Overall accuracy | _ | 0.9755 |
| Mean class recall | _ | 0.9833 |
| Mean F1 | _ | 0.7413 |

## Hardware used
_e.g. MacBook Air M2, 8-core CPU, 16 GB RAM, macOS 15; runtime of `bash run.sh`: … min. No GPU needed._

## Repository layout
`notebooks/project.ipynb` (**all implementation**) · `code/` (`main.py` pipeline and `tests.py` read the notebook; `implementation.py` is a generated plain-Python copy) · `data/` (download script and data docs; the .ply files are not included) · `results/` · `docs/` (project guide) · `report/` · `algorithm.txt` (pseudocode) · `paper.pdf`.

## Contributing (team)
We work on `main`: `git pull --rebase` → implement your cells in the notebook until their tests pass → save → `git commit -m "P<n>: …"` → `git pull --rebase` → `git push`. Details are in the guide, §10.

## Data licence
Paris-rue-Madame database: MINES ParisTech 3D mobile laser scanner dataset from Madame street in Paris. © MINES ParisTech, CC BY-NC-ND 3.0. MINES ParisTech does not endorse our uses or conclusions. A. Serna, B. Marcotegui, F. Goulette, J.-E. Deschaud, *Paris-rue-Madame database: a 3D mobile laser scanner dataset for benchmarking urban detection, segmentation and classification methods*, ICPRAM 2014.
