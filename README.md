# MRI-to-DTI Synthesis

Code for the paper **"Deep Learning-Based MRI-to-DTI Synthesis for Inferring White Matter Connectivity from Structural Scans"** (Purav N Khincha, Pranavi Sistla, Mahitha G — PES University).

We predict the full diffusion tensor at every voxel from a T1-weighted MRI scan using a 3D U-Net with a log-Euclidean output, which guarantees that every predicted tensor is symmetric positive-definite. FA and MD are computed from the predicted tensor. We compare four variants:

| Variant | Description | Notebook |
|---|---|---|
| V1 | Single-head baseline | `notebooks/03_training/train_V1_single_head.ipynb` |
| V2 | Dual head (eigenvalues + 6D rotation) with directional loss | `notebooks/03_training/train_V2_dual_head.ipynb` |
| V3 | V2 + conditional PatchGAN discriminator | `notebooks/03_training/train_V3_adversarial.ipynb` |
| V4 | V2 + background-suppression loss | `notebooks/03_training/train_V4_background_suppression.ipynb` |

## Results (43 held-out test subjects)

| Variant | FA MSE | FA SSIM | MD SSIM | Cos. (WM) |
|---|---|---|---|---|
| V1 | 0.0092 | 0.073 | 0.630 | 0.621 |
| **V2** | **0.0065** | **0.152** | **0.842** | **0.802** |
| V4 | 0.0083 | 0.110 | 0.558 | 0.716 |

V3 was evaluated only under the earlier (voxel-size) pipeline; see the paper, Section 6.2. Full per-subject results are in `results/`.

## Repository layout

```
notebooks/
  01_preprocessing/        HCP and ADNI preprocessing (DICOM -> NIfTI, denoising, skull-stripping, tensor fitting)
  02_harmonization_qc/     frozen train/val/test split + T1/DTI alignment QC
  03_training/             one notebook per variant (V1-V4), including evaluation
  04_evaluation/           cross-variant statistics and paper figures
  archive/                 earlier experiments, kept for reference only
results/                   test metrics, training histories and QC output per variant
splits/split.csv           frozen subject-level split used for all reported results
figures/                   script that draws Fig. 1
paper/                     LaTeX source and figures of the camera-ready paper
```

## Running

All notebooks were developed and run on Kaggle (GPU P100 / T4 for training, CPU for preprocessing and QC).

1. Run the preprocessing notebooks to produce `processed_hcp/` and `processed_adni/{AD,MCI,CN}/` folders, and save them as Kaggle datasets.
2. Run the harmonization/QC notebook with those datasets attached. It writes `split.csv` and the alignment report.
3. Run each training notebook with the processed data and `split.csv` attached. Each one trains, evaluates on the test set and writes the files in `results/`.

Install dependencies with `pip install -r requirements.txt`. Tensor fitting uses **dipy 1.9.0**, which requires `numpy<2`.

## Data

This repository contains **no imaging data**. The data come from:

- **HCP** (Human Connectome Project, WU-Minn HCP 1200 release): https://db.humanconnectome.org
- **ADNI** (Alzheimer's Disease Neuroimaging Initiative): https://adni.loni.usc.edu

Both require a data use agreement. Apply for access and download the data yourself.

## Citation

```
P. N. Khincha, P. Sistla, and M. G, "Deep Learning-Based MRI-to-DTI Synthesis for Inferring
White Matter Connectivity from Structural Scans," 24th OITS International Conference on Information Technology, 2026.
```
