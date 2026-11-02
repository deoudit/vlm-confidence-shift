#!/usr/bin/env bash
# Recompute every table and figure in the paper from the released signal tables.
# CPU only; needs requirements-analysis.txt.
#   bash reproduce.sh          full run (the permutation tests take ~45 min per model)
#   QUICK=1 bash reproduce.sh  skip the permutation tests (everything else in a few minutes)
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results figures

MMB="outputs/features_qwen25vl_7b outputs/features_qwen25vl_3b outputs/features_qwen2vl_2b outputs/features_internvl3_2b outputs/features_smolvlm_instruct"
AOK="outputs/features_qwen2vl_2b_aokvqa outputs/features_smolvlm_instruct_aokvqa"
DIS="outputs/features_qwen2vl_2b_disastervqa outputs/features_smolvlm_instruct_disastervqa"
names() { for d in "$@"; do basename "$d"; done; }
NPERM=2000; [ "${QUICK:-0}" = "1" ] && NPERM=0

echo "== Tables 1, 2, 6, 7 (ECE, worst-case AUROC, ECE-AUROC r, advantage over random)"
python analysis/boundary_table.py --dirs $MMB $AOK $DIS          | tee results/boundary_table.txt
echo "== Tables 1, 6 (Delta-ECE bootstrap CIs, P, debiased estimate)"
python analysis/ece_bootstrap.py --root outputs --dirs $(names $MMB $AOK) \
       --out results/ece_bootstrap.csv                           | tee results/ece_bootstrap.txt
echo "== Table 3 (clean -> severity-5 threshold transfer)"
python analysis/risk_transfer_table.py --dirs $MMB $AOK --cluster | tee results/risk_transfer.txt
echo "== Table 4 (attainable coverage, confidence vs combiner)"
python analysis/coverage_gain.py --dirs $MMB --cluster            | tee results/coverage_gain.txt
echo "== Tables 5, 6, 7 (best gain beyond confidence, selection-aware p-value)"
python analysis/gain_beyond_confidence.py --dirs $MMB $AOK $DIS --nperm $NPERM \
                                                                 | tee results/gain_beyond_confidence.txt
echo "== App. B (standardized-feature combiner)"
python analysis/gain_beyond_confidence.py --dirs $MMB --nperm $NPERM --scale \
                                                                 | tee results/gain_beyond_confidence_scaled.txt
echo "== Table C1 (option-order robustness)"
python analysis/analyze_circular.py --dirs outputs/features_qwen25vl_3b_circular \
       outputs/features_internvl3_2b_circular --out results/circular.csv | tee results/circular.txt
echo "== Figures 2-6"
python analysis/make_figures.py --root outputs --out figures
echo "done: results/ and figures/"
