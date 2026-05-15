#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PROJECT_ROOT}/venv_SOUL/bin/python"
MAIN_SCRIPT="${SCRIPT_DIR}/main_flexible.py"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

EPOCHS="${EPOCHS:-200}"
KFOLD="${KFOLD:-5}"
DEVICE="${DEVICE:-auto}"
COMMON_ARGS=(
  --epoch "${EPOCHS}"
  --k_fold "${KFOLD}"
  --device "${DEVICE}"
)

RUN_SUFFIX="${RUN_SUFFIX:-$(date +%Y%m%d_%H%M%S)}"

run_experiment() {
  local run_name="$1"
  shift

  echo
  echo "============================================================"
  echo "Running: ${run_name}"
  echo "Omics: $*"
  echo "============================================================"

  "${PYTHON_BIN}" "${MAIN_SCRIPT}" \
    "${COMMON_ARGS[@]}" \
    --run_name "${run_name}_${RUN_SUFFIX}" \
    --omics "$@"
}

# Config 1: 7Omics (GE(chromatin + methylation)TMP + pathway)
# Interpreted as all available 7 omics stems in final_dataset.
run_experiment "flexible_7omics_ge_chrom_meth_tmp_pathway" \
  genomics_mutation epigenomics_chromatin epigenomics_methylation transcriptomics_expression metabolomics_profile proteomics_reverse_phase pathway

# Config 2: 3Omics (G(mutation)ET) + pathway
run_experiment "flexible_3omics_gmutation_et_pathway" \
  genomics_mutation epigenomics_methylation transcriptomics_expression pathway

# Config 3: pathway only
run_experiment "flexible_pathway_only" \
  pathway


echo

echo "All configured runs completed."
