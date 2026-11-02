# Trust the Ranking, Not the Threshold

Code and data for **"Trust the Ranking, Not the Threshold: Selective Prediction in
Vision-Language Models Under Distribution Shift"** (Udit Deo and Balasubramanian Raman,
IIT Roorkee).

The study measures how the single-pass confidence of five open vision-language models
(Qwen2.5-VL-3B/7B, Qwen2-VL-2B, InternVL3-2B, SmolVLM-2.2B) behaves on MMBench-perception
under a 21-condition ImageNet-C grid. It replicates the analysis on A-OKVQA and checks it
on the naturally degraded DisasterVQA. The repository has two layers:

1. **Analysis (CPU, minutes).** The released per-condition signal tables in `outputs/`
   reproduce every table and figure in the paper with `bash reproduce.sh`.
2. **Extraction (GPU).** `scripts/extract.py` regenerates those tables from the public
   benchmarks and model checkpoints.

## Quick start: reproduce the tables and figures

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-analysis.txt
QUICK=1 bash reproduce.sh     # all tables and figures except the permutation p-values
bash reproduce.sh             # full run; the permutation tests take ~45 min per model
```

Results go to `results/` (text and CSV) and `figures/` (PDF and PNG).

| Paper item | Script | Notes |
| --- | --- | --- |
| Table 1 (ECE, worst AUROC, label) | `analysis/boundary_table.py`, `analysis/ece_bootstrap.py` | Delta-ECE CIs and P use a question-clustered bootstrap that re-selects the worst condition in every replicate |
| Table 2 (advantage over random) | `analysis/boundary_table.py` | "@50%" at severity 5 |
| Table 3 (threshold transfer) | `analysis/risk_transfer_table.py --cluster` | 5% and 10% targets |
| Table 4 (coverage at fixed risk) | `analysis/coverage_gain.py --cluster` | |
| Table 5 (gain beyond confidence) | `analysis/gain_beyond_confidence.py` | `--nperm 2000` gives the selection-aware p-value; `--nbins 5/20` for the bin sweep |
| Table 6 (A-OKVQA) | `boundary_table.py`, `ece_bootstrap.py`, `risk_transfer_table.py`, `gain_beyond_confidence.py` | on the `*_aokvqa` tables |
| Table 7 (DisasterVQA) | `boundary_table.py`, `gain_beyond_confidence.py` | single condition; 50/50 question split for the CI |
| Table C1 (option order) | `analysis/analyze_circular.py` | per-rotation ranges and rotation-averaged risk |
| App. B debiased ECE | `analysis/ece_bootstrap.py` | `debiased_delta` column |
| App. B feature scaling | `analysis/gain_beyond_confidence.py --scale` | |
| Figs. 2-6 | `analysis/make_figures.py` | Fig. 1 is a TikZ diagram in the manuscript |
| Sec. 4.4 hidden-state drift | `scripts/drift_analysis.py` | needs hidden states; see below |

All analyses share one convention: 15 equal-width ECE bins, a calibration cutoff of 0.03,
and a fixed 50/50 split of question indices (clean-condition indices in file order,
permuted with `numpy.random.RandomState(0)`). A question therefore sits in the same half
under every condition.

## Released data

`outputs/features_<model>[_<benchmark>]/<corruption>_s<severity>.parquet` holds one row
per question per condition with these columns:

| Column | Meaning |
| --- | --- |
| `index` | question id (see `data/question_ids/`) |
| `corruption`, `severity` | `clean`/0, or fog, snow, defocus_blur, brightness at 1-5 |
| `pred_letter`, `gt_letter`, `correct` | prediction, answer, correctness |
| `conf`, `margin`, `entropy` | output-family signals (Eq. A1-A2) |
| `lens_agree_frac` | logit-lens layer agreement, last 8 layers (Eq. A3-A4) |
| `attn_img_entropy`, `attn_img_mass` | attention geometry over image tokens (Eq. A5) |

`outputs/features_{qwen25vl_3b,internvl3_2b}_circular/` hold the CircularEval runs (App. C).
Each row is one rotation of one question. `o0`-`o3` are the option probabilities mapped back
to the original option order.

The tables contain no images or question text, only ids and model outputs. Hidden states
(several GB) are not included; `scripts/extract.py` writes them as `<tag>_hidden.npz`
when you re-run an extraction.

`data/question_ids/` lists the sampled questions for every benchmark.

## Re-running the extraction

```bash
pip install -r requirements.txt     # install the torch build for your CUDA first
```

**MMBench.** Place `MMBench_DEV_EN.tsv` (the official MMBench dev split from
[open-compass/MMBench](https://github.com/open-compass/MMBench); md5
`b6caf1133a01c6bb705cf753bb527ed8`) in `data/`, then confirm you get the same questions:

```bash
python scripts/check_subsets.py --tsv data/MMBench_DEV_EN.tsv
```

**A-OKVQA** (`HuggingFaceM4/A-OKVQA`, validation) and **DisasterVQA** (`QCRI/DisasterVQA`)
load from the Hugging Face Hub. `python scripts/download_aokvqa.py` saves an offline copy of
A-OKVQA.

One config per model and benchmark lives in `configs/`:

```bash
python scripts/extract.py --config configs/qwen25vl_3b.yaml                 # 21 conditions
python scripts/extract.py --config configs/smolvlm_aokvqa.yaml
python scripts/extract.py --config configs/qwen2vl_2b_disastervqa.yaml      # clean only
python scripts/extract_circular.py --config configs/qwen25vl_3b.yaml --subset 50 --types --check
python scripts/extract_circular.py --config configs/qwen25vl_3b.yaml        # App. C
python scripts/drift_analysis.py --config configs/qwen25vl_3b.yaml          # needs *_hidden.npz
```

Extraction is restartable: conditions whose table already exists are skipped. Precision
is fixed per model. We used float32 for Qwen2-VL-2B and InternVL3-2B because float16
produced NaN activations on our Pascal (P5000) GPU. Qwen2.5-VL-7B ran in bfloat16 on an
RTX A5000. Exact numbers can shift slightly with the GPU, the transformers version and
the checkpoint revision. The released tables are the ones the paper reports.

## Layout

```
saif/          evaluation harness: data loaders, corruptions, one-pass signal extractor, metrics
scripts/       extraction (extract.py, extract_circular.py), data helpers, drift analysis
analysis/      table and figure scripts (CPU only)
configs/       one YAML per model x benchmark
outputs/       released per-condition signal tables
data/          question_ids/ (and your MMBench TSV)
reproduce.sh   runs every analysis
```

## Citation

See `CITATION.cff`. Licence: MIT (code); the signal tables are released under CC BY 4.0.
The benchmarks and model checkpoints keep their own licences.
