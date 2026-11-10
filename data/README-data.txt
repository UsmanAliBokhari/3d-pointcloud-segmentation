DATA: Paris-rue-Madame database (MINES ParisTech)
==================================================

Paper reproduced: Hackel, Wegner & Schindler (2016), Table 2 uses this dataset.

What it is
  Mobile laser scan (LARA2-3D MLS system, car-mounted) of ~160 m of rue Madame, Paris 6e,
  acquired 8 February 2013. Two annotated files, ~10 million points each:
      GT_Madame1_2.ply   GT_Madame1_3.ply
  Binary big-endian PLY. Per point: x, y, z (Lambert 93 / IGN1969, with X0 = 650976 m and
  Y0 = 6861466 m already subtracted), reflectance, label (object id), class (category id).

How to get it (automatic)
  bash data/download.sh
  -> downloads http://cmm.mines-paristech.fr/Projects/TerraMobilita/rueMadame_database.zip
     and extracts both .ply files into data/

How to get it (manual, if the link fails)
  1. Open https://people.cmm.minesparis.psl.eu/users/serna/rueMadameDataset.html
  2. Download the zip, place it in data/ and run  bash data/download.sh  again.

Class ids we use (Serna et al. 2014, Table 1) -> our label
  1 Facade -> 0     2 Ground -> 1        4 Cars -> 2
  10 Motorcycles -> 3   14 Traffic signs -> 4   9 Pedestrians -> 5
  every other id (background, noise, poles, plants, ...) -> -1 = ignored
  (set in config.yaml: class_map. TO VERIFY: whether "Fast pedestrian" (22) and
   "Pedestrian + something" (24) should also map to pedestrians.)

Preprocessing (done in code, nothing is stored)
  - reflectance is not used (purely geometric features, paper Section 3.2)
  - optional random subset (--max-points) for quick runs
  - voxel-grid pyramid: 9 levels, 0.025 m ... 6.4 m (notebook stage 1: build_pyramid)
  - training set: 1000 random points per class (seeded); test = all other labelled points

Licence and required attribution
  Creative Commons BY-NC-ND 3.0 (non-commercial, no modification of the data).
  Results must mention: "Paris-rue-Madame database: MINES ParisTech 3D mobile laser scanner
  dataset from Madame street in Paris", (c) MINES ParisTech, which does not endorse our
  uses or conclusions. Cite:
  A. Serna, B. Marcotegui, F. Goulette, J.-E. Deschaud. Paris-rue-Madame database: a 3D
  mobile laser scanner dataset for benchmarking urban detection, segmentation and
  classification methods. ICPRAM 2014.
  -> Because of size and licence, the .ply files are NOT committed to git / the zip.
