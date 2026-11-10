# Report outline (becomes report.pdf)

Structure required by the course template. Owner writes, reviewer checks. Fill in by **Wed 18 Nov**.

## 1. Title, authors, article (P2)
Our title, three names, the full citation of Hackel et al. 2016.

## 2. Summary, max 200 words (P2, everyone reviews)
Problem → method in one sentence each (pyramid, 144 features, RF) → dataset → our main numbers vs the paper → one-sentence conclusion.

## 3. Implementation (in our own words, not too deep)
### 3a Data acquisition (P3, reviewer P1)
Paris-rue-Madame: what it is (MLS, 2 files, ~20 M points, fields), where it comes from (link, `data/download.sh`), licence, class mapping table (raw id → our class), which classes are ignored.

### 3b Preprocessing (P1, reviewer P2)
No reflectance; optional subsampling; voxel-grid filter; pyramid table (level, voxel size, #points on our data); training sampling (1000/class, seed).

### 3c Algorithm (P1: covariance/eigen; P2: neighbours, moments, height, RF; P3: assembly and chunking; reviewer P3)
Pipeline figure (from the guide), neighbourhoods (kNN per level, cylinder), feature table, Random Forest + grid search + CV, on-the-fly classification. Refer to `algorithm.txt`.

### 3d Differences from the original paper (P3, everyone)
Copy and complete the table in `docs/PROJECT_GUIDE.md` §12.

## 4. Results
### Quantitative (P1, reviewer P3)
- Table like the paper's Table 2: ours vs paper (`results/paper_comparison.csv`)
- Timing table like Table 3 + hardware (`results/timings.csv`)
- Feature ablation (`results/ablation.csv`): is height_below also our most important feature?
- Chosen hyperparameters + CV score (`results/cv_results.csv`)

### Qualitative (P3, reviewer P2)
- Prediction figure (like Fig. 4), ground truth next to it
- Failure cases: misclassified points in red; describe *where* errors happen (object borders? motorcycles vs cars? pedestrians next to facades?)
- Confusion matrix

## 5. Discussion (all, P2 edits)
Where we match and where we don't, and plausible reasons: different train/test split, cylinder radius choice, median vs medoid, random sampling variance (try 2–3 seeds if time), class mapping (fast pedestrians?), Python approximations. Why low precision on small classes. Runtime comparison and why.

## 6. Reproducibility checklist (P2)
| Item | Status |
|---|---|
| Code: provided and documented (README) | ✓ `code/`, `README.md` |
| Data: download script provided | ✓ `data/download.sh`, `data/README-data.txt` |
| Environment | ✓ `requirements.txt` (pinned), `install_MAC.sh`, `install_Linux.sh` |
| Run instructions: single command | ✓ `bash run.sh` |
| Random seeds | 42 (Python, NumPy, sampling, CV folds, Random Forest), set in `config.yaml` |
| Expected outputs | OA …, mean recall …, mean F1 … (`README.md`) |
