# LiDAR Compensation Experiments

Experiments presented in Chapter 5 of the paper.
# LiDAR Point Cloud Compensation under Observation-to-Control Latency in UAV Autonomous Flight

Python experiment scripts and recorded results for evaluating planar UAV LiDAR motion compensation in AirSim. The experiments compare uncompensated LiDAR observations (`raw`) with motion-compensated observations (`corrected` or `ours`) using estimated motion histories and polynomial prediction.

The repository contains translation and yaw-rotation experiments, component ablations, and circular and figure-eight flight scripts. The accompanying manuscript is **LiDAR Point Cloud Compensation under Observation-to-Control Latency in UAV Autonomous Flight**, by **Grori Park and Pilsu Jung** (supplied draft dated November 10, 2026). Section and table references below refer to that draft. A verified publication link and final citation can be added after publication; the template DOI is not treated as a publication identifier.

## Repository structure

```text
.
├── Q1/
│   ├── translation/
│   └── rotation/
├── Q2/
│   ├── translation/
│   └── rotation/
├── Q3/
│   ├── ablation1/
│   └── ablation2/
├── Q4/
│   ├── translation/
│   └── rotation/
├── Discussion/
│   └── combined_motion/
│       ├── circle/
│       └── figure_eight/
├── FILE_INDEX.md
└── DIRECTORY_RENAMES.md
```

`Q1`–`Q4` correspond to the research questions in manuscript Section 5.1. `Discussion` supports the parameter and motion analyses in Section 7. The table separates their intended scientific purposes from what is actually included in this snapshot.

## Mapping to the manuscript

| Folder | Scientific purpose | Manuscript reference | Included files and coverage |
|---|---|---|---|
| `Q1/` | Establish whether compensation reduces relative position and bearing errors compared with raw LiDAR. | Sections 5.2 and 5.3.1; Table 1 | Translation and rotation scripts plus saved speed/rate sweeps. The manuscript's basic comparison uses 10 m/s translation, 30 degrees/s rotation, and 0.2 s latency; the current scripts instead default to 25 m/s, 75 degrees/s, and 0.3 s latency. The bundled sweeps are not identified as the dedicated Table 1 runs. |
| `Q2/` | Evaluate sensitivity to increasing observation-to-control latency. | Sections 5.2 and 5.3.2; Figures 4–5 | Translation and rotation scripts with a configurable `LATENCY` constant. The paper varies latency from 0.15 to 0.30 s; separately identified latency-sweep datasets are absent here. |
| `Q3/ablation1/` | Measure the contribution of the observation, command-generation, and command-application compensation intervals. | Section 5.3.3; Table 2 | Entry point for scan-only, scan-plus-generation, and full compensation. Shared engine resides in `Q3/ablation2/`; recorded ablation results are absent. |
| `Q3/ablation2/` | Measure the contribution of point-specific timing, polynomial motion prediction, and rotation-aware coordinate transformation. | Section 5.3.3; Figure 6 | Ablation entry point, shared engine, and S-path tracking code. See the implementation-label note below; recorded results are absent. |
| `Q4/` | Evaluate compensation as UAV linear and angular speeds increase. | Section 5.2; Tables 3–4 | Saved translation sweeps at 5–25 m/s and rotation sweeps at 15–75 degrees/s, with raw and derived analyses. These match the paper's sweep ranges; individual table values have not been independently reproduced. |
| `Discussion/combined_motion/figure_eight/` | Evaluate polynomial-degree sensitivity under changing translation and rotation. | Section 7.1; Table 6 | Figure-eight runner with configurable `POLY_DEGREE`; the paper tests degrees 1–5 at N=10. Degree-sweep results are absent. |
| `Discussion/combined_motion/circle/` and `figure_eight/` | Evaluate different flight trajectories, including continuous turns and alternating turns. | Section 7.2; Table 7 | Circular and figure-eight runners. Translation at 5 m/s and rotation at 15 degrees/s are represented by the corresponding Q4 folders. Recorded combined-motion runs are absent. |
| `Discussion/combined_motion/figure_eight/analyze_n_comparison.py` | Evaluate how the number of historical motion measurements affects compensation and processing cost. | Section 7.3; Table 8 | Comparison utility for N=10, 20, 30, 40, and 50 with degree 2. The historical input runs it references are absent. |

**Snapshot limitation:** all 119 corresponding files in Q1, Q2, and Q4 are byte-for-byte identical, including the six equivalent path updates made during renaming. Their folder labels describe different research questions, but the bundled copies do not establish three independently acquired datasets. In particular, the Q2 speed/rate folders do not encode the manuscript's latency sweep. Preserve acquisition settings and identify the intended runs before claiming reproduction of the paper tables.

See [FILE_INDEX.md](FILE_INDEX.md) for a linked description of every supplied file, and [DIRECTORY_RENAMES.md](DIRECTORY_RENAMES.md) for the original-to-English directory map.

## Experiment scripts

The following translation and rotation scripts occur in each of Q1, Q2, and Q4.

| Relative path within each Q folder | Role |
|---|---|
| `translation/make__.py` | Runs straight translational flight and compares raw and corrected XY coordinates against the static `Cylinder_2` target. |
| `translation/make__static.py` | Runs translational UAV flight with a moving `Cube_1` target following a straight shuttle trajectory. Despite its historical filename, the target is dynamic. |
| `rotation/make__rotate.py` | Runs yaw-rotation trials and compares raw and compensated target bearings against control-time ground truth for `Cylinder_2`. |
| `translation/reset_test_drone.py` | Connects to AirSim, repositions and resets `Follower0`, enables API control, arms it, and commands a short takeoff. |
| `rotation/reset_test_drone.py` | The same simulator reset and takeoff helper, provided beside the rotation script. |
| `translation/experiment_results/analyze_filtered.py` | Recomputes position and bearing statistics from saved translation scans; exports retained and excluded scans for a configurable position-error threshold. |
| `rotation/experiment_results/analyze_filtered.py` | Recomputes rotation statistics with a 20 m position-error threshold, or without exclusion when `--no-filter` is passed. |

Current primary-script constants specify 100 trials, a 0.1 s scan time, 0.3 s latency, 10 motion-history samples, and a second-degree polynomial. Translation trials last 10 s and rotation trials last 20 s. The saved sweeps cover UAV speeds of 5, 10, 15, 20, and 25 m/s and yaw rates of 15, 30, 45, 60, and 75 degrees/s. Use each run's `settings.csv` as the record of its acquisition configuration; current script constants are not a substitute for those settings.

### Ablation experiments (Q3)

| File | Role |
|---|---|
| `Q3/ablation1/make_ablation_1.py` | Entry point for compensation-interval comparisons: scan only, scan plus generation delay, and full compensation through application time. |
| `Q3/ablation2/make_ablation_2.py` | Entry point for sequential component removal: Full, without T1, without T1/T3, without T1/T2/T3, and Raw. |
| `Q3/ablation2/make_ablation_common.py` | Shared configuration, motion-history regression, scan reconstruction, compensation, paired evaluation, and result export. T1 denotes point-specific acquisition timing, T2 polynomial prediction, and T3 the final coordinate rotation. |
| `Q3/ablation2/fixed_s_path.py` | Fixed S-shaped path geometry and velocity tracking with curvature feedforward and lateral feedback. |

The paper presents the technical-component comparison by incrementally adding components to Raw. The supplied implementation instead labels variants by sequential removal: `Full`, `w/o T1`, `w/o T1,T3`, `w/o T1,T2,T3`, and `Raw`. Reversing this removal list does not yield the same intermediate component combinations as adding T1, then T2, then T3. Verify the intended Figure 6 variants before using this engine to reproduce that figure; the README retains the actual code labels.

The ablation engine reconstructs uniform scans in a static scene using independent ground-truth poses. These are reconstructed observations, rather than native rolling LiDAR measurements. Variants share the same scan and application-time ground truth.

Only source code is included for Q3. The first entry point imports `make_ablation_common`, which is located in the sibling `ablation2` directory; add that directory to `PYTHONPATH` when running ablation 1. Its docstring also refers to `make_ablation_README.md`, which is absent from this snapshot.

### Combined motion (Discussion)

| File under `Discussion/combined_motion/` | Role |
|---|---|
| `circle/make_circle_static.py` | Runs one circular flight per trial with tangent yaw, observing the static `Cylinder_2` target. The configured circle diameter is 100 m. |
| `figure_eight/make__figure8_static.py` | Runs figure-eight flight with changing translation and yaw, observing static `Cylinder_2`. The configured path spans 90 m by 45 m. |
| `figure_eight/analyze_n_comparison.py` | Compares motion-history sizes N=10, 20, 30, 40, and 50 using saved runs; exports accuracy, processing-time statistics, plots, and a Korean analysis report. |
| `circle/reset_test_drone.py`, `figure_eight/reset_test_drone.py` | Simulator reset and takeoff helpers. |

Only source code is supplied for the Discussion experiments. The figure-eight comparison references specific historical run directories that are not included, so its inputs must be supplied before execution. Both flight scripts currently use 0.2 s latency and a 0.1 s scan time. The simulator scene must contain the expected target; configure returned LiDAR points to represent that target. Object-center ground truth and the observed surface centroid can differ for a finite-size target.

## Recorded data and file meanings

Within each translation or rotation experiment, `experiment_results/` separates original recorded runs from derived analyses.

| Directory or filename | Meaning |
|---|---|
| `translation/experiment_results/make__/uav_speed_<v>_mps/` | Original straight-flight dataset for UAV speed `<v>` m/s. |
| `rotation/experiment_results/make__rotate/yaw_rate_<r>_deg_s/` | Original rotation dataset for yaw rate `<r>` degrees/s. |
| `settings.csv` | Name/value pairs recording the experiment configuration. This is an experiment log, not an AirSim `settings.json` file. |
| `scans.csv` | Scan-level observations, motion values, raw/corrected coordinates, ground-truth values, and errors. Column sets differ between translation and rotation. |
| `trials.csv` | Trial completion status, error messages, scan counts, elapsed time, and trial-level error statistics. |
| `summary.csv` inside an original run | Overall run completion and error summary. |
| `filtered_3m/` | Translation analysis retaining scans with planar position error at most 3 m. |
| `filtered_20m/` | Translation or rotation analysis retaining scans with planar position error at most 20 m. |
| `unfiltered/` | Rotation analysis using all scans without position-error exclusion. |
| `summary.csv` inside an analysis directory | Statistics by speed or angular rate and method, including scan counts, pooled scan means, and equally weighted trial means. |
| `trial_means.csv` | Trial-level means and retained-scan counts for each condition and method. |
| `speed<v>_<method>_normal_scans.csv` | Retained translation scans at speed `<v>` for `raw` or `corrected`. |
| `speed<v>_<method>_excluded_scans.csv` | Translation scans excluded by the directory's error threshold. |
| `rate<r>_<method>_normal_scans.csv` | Retained rotation scans at yaw rate `<r>`. |
| `rate<r>_<method>_excluded_scans.csv` | Rotation scans excluded by the 20 m position-error threshold. |
| `report.md` | Human-readable analysis tables and filtering notes; supplied reports are in Korean. |

`raw` denotes the uncompensated baseline from a scan, and `corrected`/`ours` denotes the compensated estimate. Position units are meters; bearing units are degrees. Planar position error is the Euclidean XY distance to ground truth. Bearing error uses the wrapped angular difference. Averaging X and Y MAEs is a different quantity from Euclidean position error.

The analysis scripts filter raw and corrected estimates **independently**. Their retained scan sets can therefore differ; filtered comparisons are not necessarily paired on identical retained observations. Distinguish pooled scan means from equal-trial means when reporting paper results, and disclose the threshold and exclusion counts.

## Running the code

Use Python with `airsim` and `numpy` for simulation experiments. Plotting and comparison utilities also use `matplotlib` and `pandas`; the ablation engine optionally exports Excel using `openpyxl`. Exact dependency versions are not recorded in this snapshot.

AirSim must already be running with a compatible multirotor scene, the expected vehicle (`Follower0`), LiDAR sensor (`Follower0_LidarSensor`), and the target object named in the chosen script. The scene assets and AirSim configuration are not included. Primary experiment timing comments assume AirSim `ClockSpeed=1`; altitude constants use NED coordinates.

Example commands from the repository root:

```powershell
# Run saved-data analyses (no simulator connection required).
python Q1/translation/experiment_results/analyze_filtered.py --threshold 3
python Q1/translation/experiment_results/analyze_filtered.py --threshold 20
python Q1/rotation/experiment_results/analyze_filtered.py
python Q1/rotation/experiment_results/analyze_filtered.py --no-filter

# Run a simulation experiment after preparing AirSim and reviewing constants.
python Q1/translation/make__.py
python Q1/rotation/make__rotate.py
python Discussion/combined_motion/circle/make_circle_static.py
python Discussion/combined_motion/figure_eight/make__figure8_static.py

# Inspect ablation options; make the shared engine importable for ablation 1.
$env:PYTHONPATH = (Resolve-Path Q3/ablation2).Path
python Q3/ablation1/make_ablation_1.py --help
python Q3/ablation2/make_ablation_2.py --help
```

Analysis commands overwrite the corresponding derived analysis outputs. Simulation scripts create timestamped output directories; they do not automatically rebuild the named speed/rate sweep directories bundled here. To collect a sweep, adjust the selected script's constants for each condition and retain the generated settings with its results.

## Snapshot and naming notes

All 39 Korean-named subdirectories were renamed to English. Six analysis scripts were updated to read the new speed/rate directory names. Existing English filenames, dataset contents, and the experiment equations were retained. Original CSV and report bytes were preserved during renaming, and all 30 Python sources passed syntax parsing. Simulator execution and scientific result reproduction were not performed as part of the naming and documentation update.

Before publication, add a verified publication link, final bibliographic citation, and an author-selected code/data license. Supply or identify the distinct Q1 baseline and Q2 latency runs, and the missing Q3 and Discussion results. The draft's publisher-template copyright text is not used to infer a repository license. Section 6 (the interception application case study) is outside the supplied Q1–Discussion folder coverage.
