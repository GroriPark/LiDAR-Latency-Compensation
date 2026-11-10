# LiDAR Point Cloud Compensation under Observation-to-Control Latency in UAV Autonomous Flight

Experiment code accompanying the manuscript by **Grori Park and Pilsu Jung**. The method predicts UAV motion from historical linear and angular velocities and compensates LiDAR observations into the UAV coordinate frame at control time. `raw` denotes the uncompensated baseline; `corrected` or `ours` denotes the compensated estimate.

## Repository structure

```text
chapter5/
|-- Discussion/
|   `-- combined_motion/
|       |-- circle/
|       |   |-- make_circle_static.py
|       |   `-- reset_test_drone.py
|       `-- figure_eight/
|           |-- analyze_n_comparison.py
|           |-- make__figure8_static.py
|           `-- reset_test_drone.py
|-- Q1/
|   |-- rotation/
|   |   |-- make__rotate.py
|   |   `-- reset_test_drone.py
|   `-- translation/
|       |-- make__.py
|       |-- make__static.py
|       `-- reset_test_drone.py
|-- Q2/
|   |-- rotation/
|   |   |-- make__rotate.py
|   |   `-- reset_test_drone.py
|   `-- translation/
|       |-- make__.py
|       |-- make__static.py
|       `-- reset_test_drone.py
|-- Q3/
|   |-- ablation1/
|   |   `-- make_ablation_1.py
|   `-- ablation2/
|       |-- fixed_s_path.py
|       |-- make_ablation_2.py
|       `-- make_ablation_common.py
|-- Q4/
|   |-- rotation/
|   |   |-- make__rotate.py
|   |   `-- reset_test_drone.py
|   `-- translation/
|       |-- make__.py
|       |-- make__static.py
|       `-- reset_test_drone.py
|-- DIRECTORY_RENAMES.md
|-- FILE_INDEX.md
`-- README.md
```

This repository provides **24 Python source files** for AirSim experiments. It includes simulation runners, ablation utilities, vehicle reset helpers, and a history-size analysis script. Experimental datasets and recorded result reports are not distributed here. Simulation runners generate results when executed.

[FILE_INDEX.md](FILE_INDEX.md) describes all 24 current Python source files individually. [DIRECTORY_RENAMES.md](DIRECTORY_RENAMES.md) gives the current directory tree and the original-to-English names for directories that still exist.

## Experiments and manuscript references

| Directory | Experiment purpose | Manuscript reference | How to configure or use it |
|---|---|---|---|
| `Q1/` | Basic relative position and bearing accuracy versus raw LiDAR. | Sections 5.2 and 5.3.1; Table 1 | The paper uses 10 m/s translation, 30 degrees/s rotation, and 0.2 s latency. Set the runner constants accordingly; current defaults differ. No recorded Table 1 runs are included. |
| `Q2/` | Sensitivity to observation-to-control latency. | Sections 5.2 and 5.3.2; Figures 4-5 | Vary `LATENCY` to evaluate the paper's 0.15-0.30 s range. No recorded latency sweep is included. |
| `Q3/ablation1/` | Contribution of compensation over observation, command generation, and command application. | Section 5.3.3; Table 2 | Entry point for scan-only, scan-plus-generation, and full compensation. Imports the shared engine in `ablation2`. |
| `Q3/ablation2/` | Contribution of point-specific timing, motion prediction, and rotation-aware coordinate transformation. | Section 5.3.3; Figure 6 | Includes the shared ablation engine and S-shaped path geometry. Actual code variants are described below. |
| `Q4/` | Sensitivity to UAV linear and angular speeds. | Section 5.2; Tables 3-4 | Vary `SPEED` over 5, 10, 15, 20, and 25 m/s, or `YAW_RATE_DEG_S` over 15, 30, 45, 60, and 75 degrees/s. No recorded sweep is included. |
| `Discussion/combined_motion/figure_eight/` | Polynomial-degree sensitivity. | Section 7.1; Table 6 | Vary `POLY_DEGREE` from 1 to 5 with `HISTORY_SAMPLES=10`. |
| `Discussion/combined_motion/` | Translation, rotation, circular, and figure-eight trajectories. | Section 7.2; Table 7 | Circular and figure-eight runners are here. Use the translation/rotation runners for the simple motion cases. |
| `Discussion/combined_motion/figure_eight/analyze_n_comparison.py` | Number of historical motion measurements and processing cost. | Section 7.3; Table 8 | Compares N=10, 20, 30, 40, and 50 with degree 2. Requires historical input runs that are not included. |

Q1, Q2, and Q4 retain separate folders to express their different research questions. The five corresponding simulation/reset scripts have identical contents across these three folders. Configure the constants and preserve each generated run's settings to distinguish the experiments; the folder name alone does not select a research-question-specific configuration.

## Selecting an experiment

Review the constants near the top of each runner before execution. Q1, Q2, and Q4 use copies of the same runners; their folder names do not automatically apply different settings.

| Experiment | Runner | Parameters to set |
|---|---|---|
| Q1: translation baseline comparison | `Q1/translation/make__.py` | `SPEED=10.0`, `LATENCY=0.2`; retain N=10 and degree 2. |
| Q1: rotation baseline comparison | `Q1/rotation/make__rotate.py` | `YAW_RATE_DEG_S=30.0`, `LATENCY=0.2`; retain N=10 and degree 2. |
| Q2: latency sensitivity | The translation/rotation runners under `Q2/` | Vary `LATENCY` across the manuscript's 0.15-0.30 s range and record the motion settings for each run. |
| Q4: linear-speed sensitivity | `Q4/translation/make__.py` | Repeat at `SPEED` values 5, 10, 15, 20, and 25 m/s. |
| Q4: angular-speed sensitivity | `Q4/rotation/make__rotate.py` | Repeat at `YAW_RATE_DEG_S` values 15, 30, 45, 60, and 75. |
| Discussion: polynomial degree | `Discussion/combined_motion/figure_eight/make__figure8_static.py` | Set `HISTORY_SAMPLES=10`; repeat with `POLY_DEGREE` from 1 to 5. |
| Discussion: history size | The same figure-eight runner | Set `POLY_DEGREE=2`; repeat with `HISTORY_SAMPLES` values 10, 20, 30, 40, and 50. |

These settings identify the manuscript experiments; they do not constitute a verified reproduction of its numerical results. Preserve all generated settings and review the implementation notes below.

## Script roles

The first four entries below apply to Q1, Q2, and Q4.

| Script | Role |
|---|---|
| `translation/make__.py` | Straight translational flight; compares raw and corrected XY estimates against static `Cylinder_2`. |
| `translation/make__static.py` | Translational flight with a dynamic `Cube_1` shuttle target. The historical filename contains `static`, but the target moves; this is an additional scenario rather than the paper's static-target setup. |
| `rotation/make__rotate.py` | Yaw-rotation trials; compares raw and compensated target bearings against control-time ground truth for `Cylinder_2`. |
| `translation/reset_test_drone.py`, `rotation/reset_test_drone.py` | Connect to AirSim, reset/reposition `Follower0`, enable API control, arm, and command a short takeoff. |
| `Q3/ablation1/make_ablation_1.py` | Starts the compensation-interval ablation. |
| `Q3/ablation2/make_ablation_2.py` | Starts the technical-component ablation. |
| `Q3/ablation2/make_ablation_common.py` | Shared configuration, motion regression, reconstructed scans, compensation, paired metrics, and exports. |
| `Q3/ablation2/fixed_s_path.py` | Fixed S-path geometry and feedback velocity tracking. |
| `Discussion/combined_motion/circle/make_circle_static.py` | One circle per trial with tangent yaw and static `Cylinder_2`; configured diameter is 100 m. |
| `Discussion/combined_motion/figure_eight/make__figure8_static.py` | Figure-eight flight observing static `Cylinder_2`; configured path spans 90 by 45 m. |
| `Discussion/combined_motion/figure_eight/analyze_n_comparison.py` | Reads saved runs for different history sizes and exports accuracy/processing statistics, figures, and a Korean report. |
| `Discussion/combined_motion/*/reset_test_drone.py` | Simulator reset and takeoff helpers beside the combined-motion runners. |

The ablation code labels variants as `Full`, `w/o T1`, `w/o T1,T3`, `w/o T1,T2,T3`, and `Raw`. T1 is point-specific acquisition timing, T2 polynomial motion prediction, and T3 the final coordinate rotation. The manuscript describes incrementally adding timing, prediction, then rotation; reversing the supplied removal list produces different intermediate combinations. Verify the intended Figure 6 variants before claiming numerical reproduction.

The ablation engine reconstructs uniform scans of a static scene from independent ground-truth poses; these are not native rolling LiDAR measurements. Variants share the same scan and application-time ground truth. The ablation1 docstring references `make_ablation_README.md`, which is not included.

## Requirements and execution

Use Python with `airsim` and `numpy`. Plotting/comparison code also uses `matplotlib` and `pandas`; ablation Excel export uses `openpyxl`. Exact dependency versions are not recorded.

Before running simulation scripts, start AirSim with the vehicle `Follower0`, LiDAR sensor `Follower0_LidarSensor`, and the target object required by the selected script. Scene assets and AirSim configuration are not supplied. Configure the scene so returned LiDAR points represent the intended target. Altitude uses NED coordinates; the primary timing comments assume AirSim `ClockSpeed=1`.

The Q1/Q2/Q4 primary runners currently default to 100 trials, a 0.1 s scan time, 0.3 s latency, N=10, and polynomial degree 2. The straight-flight runner uses 25 m/s and 10 s trials; the rotation runner uses 75 degrees/s and 20 s trials. Combined-motion runners currently use 0.2 s latency. Review and adjust constants for the desired manuscript experiment before execution.

Run these examples from the repository root:

```powershell
# Simulation runners: require AirSim and the configured scene.
python Q1/translation/make__.py
python Q1/rotation/make__rotate.py
python Discussion/combined_motion/circle/make_circle_static.py
python Discussion/combined_motion/figure_eight/make__figure8_static.py

# Shared ablation engine must be importable by the ablation1 entry point.
$env:PYTHONPATH = (Resolve-Path Q3/ablation2).Path
python Q3/ablation1/make_ablation_1.py --help
python Q3/ablation2/make_ablation_2.py --help
```

Simulation runners create their output directories as needed and write timestamped runs. To collect speed, latency, polynomial-degree, or history-size sweeps, adjust the selected runner's constants for each condition and retain its generated configuration alongside the results. The N-comparison utility requires its historical datasets; update its source-folder mapping to match newly collected runs before executing it.

## Generated experiment outputs

Result directories are created at runtime and are not included in the current repository. Typical output files are:

| Output file | Meaning |
|---|---|
| `settings.csv` | Recorded experiment configuration; this is not an AirSim `settings.json` file. |
| `scans.csv` | Scan-level observations, motion values, raw/compensated estimates, ground truth, and errors. |
| `trials.csv` | Trial completion status, scan counts, elapsed time, and trial-level error statistics. |
| `summary.csv` | Aggregate completion and error statistics for the run. |

Actual columns and additional plots/exports depend on the runner. Position units are meters and bearing units are degrees. Euclidean XY position error differs from the average of X and Y MAEs. Distinguish scan-weighted and equal-trial averages when reporting results.

## Publication information

Add a verified publication link, final citation, and an author-selected repository license when available. The manuscript template DOI and copyright text are not used as a verified publication identifier or code/data license. Section 6's interception case study is outside this folder's coverage.

Section, figure, and table numbers refer to the supplied manuscript draft dated November 10, 2026. Simulator execution and numerical reproduction of the paper results have not been verified for this repository snapshot.

