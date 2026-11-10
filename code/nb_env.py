"""Everything the notebook's function cells can use without importing it themselves.
The notebook does `from nb_env import *`; the terminal loader runs the cells in this
same environment, so a cell behaves the same in both places."""
import os
import json
import time
import platform
import subprocess

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (precision_recall_fscore_support, confusion_matrix, accuracy_score,
                             f1_score)

import utils
from utils import (CONFIG, CLASS_NAMES, CLASS_COLORS, PAPER_MADAME, FEATURES_PER_SCALE, EPS,
                   ScaleLevel, Timer, feature_names, make_toy_scene, set_seeds,
                   ROOT, DATA_DIR, RESULTS_DIR, FIG_DIR, CACHE_DIR)
import scaffold
from scaffold import task, impl

# make every team function callable by name (placeholders/dummies until a cell defines it)
for _n in scaffold.SPEC:
    globals()[_n] = getattr(impl, _n)
