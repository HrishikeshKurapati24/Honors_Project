import os
import shutil
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from benchmarking_common import ensure_dir, write_json
from benchmarking_common.drug_features import (
    build_fingerprint_table,
    build_hkl_graphs_from_smiles,
    copy_graph_subset,
    load_graph_keys,
    load_smiles_txt,
    repair_graph_hkl_directory,
)
from benchmarking_common.splits import canonicalize_response_pairs, normalize_identifier


def _read_indexed_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.map(normalize_identifier)
    return df


def _read_feature_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    first_col = df.columns[0]
    df = df.rename(columns={first_col: "id"})
    df["id"] = df["id"].map(normalize_identifier)
    df = df.set_index("id")
    return df


def _save_feature_table(df: pd.DataFrame, output_path: str, index_name: str) -> None:
    out = df.copy()
    out.index = out.index.map(normalize_identifier)
    out.index.name = index_name
    out.to_csv(output_path)


def _load_physicochemical_table(path: str, id_column: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if id_column is None:
        id_column = df.columns[0]
    df = df.rename(columns={id_column: "drug_id"})
    df["drug_id"] = df["drug_id"].map(normalize_identifier)
    df = df.set_index("drug_id")
    return df


def _shared_physicochemical_table(root_dir: str) -> pd.DataFrame | None:
    shared_path = os.path.join(root_dir, "data", "GDSC", "Processed data", "pubchem_physiochemical_properties_1.csv")
    if not os.path.isfile(shared_path):
        return None
    return _load_physicochemical_table(shared_path, id_column="PUBCHEM_CID")


def _shared_smiles_map(root_dir: str) -> Dict[str, str]:
    return load_smiles_txt(os.path.join(root_dir, "data", "GDSC", "Processed data", "pubchem_smiles_1.txt"))


def _filter_tables(
    response_pairs: pd.DataFrame,
    cell_tables: Dict[str, pd.DataFrame],
    drug_tables: Dict[str, pd.DataFrame],
    available_graph_drugs: Iterable[str],
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], List[str], List[str]]:
    common_cells = set(response_pairs["cell_id"])
    for table in cell_tables.values():
        common_cells &= set(table.index)

    common_drugs = set(response_pairs["drug_id"])
    for table in drug_tables.values():
        common_drugs &= set(table.index)
    common_drugs &= set(map(str, available_graph_drugs))

    common_cells = sorted(common_cells)
    common_drugs = sorted(common_drugs)

    filtered_pairs = response_pairs[
        response_pairs["cell_id"].isin(common_cells) & response_pairs["drug_id"].isin(common_drugs)
    ].reset_index(drop=True)
    filtered_pairs = canonicalize_response_pairs(filtered_pairs)

    filtered_cells = {name: table.loc[common_cells] for name, table in cell_tables.items()}
    filtered_drugs = {name: table.loc[common_drugs] for name, table in drug_tables.items()}
    return filtered_pairs, filtered_cells, filtered_drugs, common_cells, common_drugs


def _write_prepared_dataset(
    output_dir: str,
    response_pairs: pd.DataFrame,
    cell_tables: Dict[str, pd.DataFrame],
    drug_tables: Dict[str, pd.DataFrame],
    graph_source_dir: str | None,
    metadata: Dict,
    extra_tables: Dict[str, pd.DataFrame] | None = None,
    extra_files: Dict[str, str] | None = None,
) -> Dict:
    ensure_dir(output_dir)
    response_pairs.to_csv(os.path.join(output_dir, "response_pairs.csv"), index=False)

    for name, table in cell_tables.items():
        _save_feature_table(table, os.path.join(output_dir, f"{name}.csv"), "cell_id")

    for name, table in drug_tables.items():
        _save_feature_table(table, os.path.join(output_dir, f"{name}.csv"), "drug_id")

    graph_output_dir = os.path.join(output_dir, "drug_graph_feat")
    if graph_source_dir and os.path.isdir(graph_source_dir):
        copy_graph_subset(graph_source_dir, graph_output_dir, response_pairs["drug_id"].unique())

    aux_dir = ensure_dir(os.path.join(output_dir, "aux"))
    if extra_tables:
        for name, table in extra_tables.items():
            _save_feature_table(table, os.path.join(aux_dir, f"{name}.csv"), "id")
    if extra_files:
        for relative_name, source_path in extra_files.items():
            if os.path.isfile(source_path):
                shutil.copy2(source_path, os.path.join(aux_dir, relative_name))

    metadata = dict(metadata)
    metadata.update(
        {
            "cell_count": int(response_pairs["cell_id"].nunique()),
            "drug_count": int(response_pairs["drug_id"].nunique()),
            "pair_count": int(len(response_pairs)),
        }
    )
    write_json(os.path.join(output_dir, "metadata.json"), metadata)
    return metadata


def prepare_3omics_dataset1(root_dir: str, benchmark_dir: str, output_dir: str) -> Dict:
    dataset_dir = os.path.join(benchmark_dir, "dataset-1")
    mutation = _read_indexed_csv(os.path.join(dataset_dir, "Celline", "genomic_mutation_34673_demap_features.csv"))
    expression = _read_indexed_csv(os.path.join(dataset_dir, "Celline", "genomic_expression_561celllines_697genes_demap_features.csv"))
    methylation = _read_indexed_csv(os.path.join(dataset_dir, "Celline", "genomic_methylation_561celllines_808genes_demap_features.csv"))
    similarity = _read_indexed_csv(os.path.join(dataset_dir, "Celline", "cellline_pathway_activity_PROGENy_561.csv"))
    ic50 = pd.read_csv(os.path.join(dataset_dir, "Celline", "GDSC_IC50.csv"), index_col=0)
    thresholds = pd.read_csv(os.path.join(dataset_dir, "Drug", "drug_threshold.csv"))
    thresholds["DrugID"] = thresholds["DrugID"].astype(str)
    threshold_map = {
        str(row.DrugID): (normalize_identifier(row.pubchem), float(row.IC50))
        for row in thresholds.itertuples(index=False)
    }

    graph_dir = os.path.join(dataset_dir, "Drug", "drug_graph_feat")
    available_graphs = set(load_graph_keys(graph_dir))

    # GraphCDR's native 3-omics setup ships a 338-drug physicochemical table
    # that covers the full 222-drug benchmark universe. The shared GDSC table
    # only covers 182 of those drugs, so prefer the dataset-local file here.
    local_phys_path = os.path.join(dataset_dir, "Drug", "pubchem_physiochemical_properties_338.csv")
    if os.path.isfile(local_phys_path):
        physico = _load_physicochemical_table(local_phys_path, id_column="PUBCHEM_CID")
    else:
        shared_phys = _shared_physicochemical_table(root_dir)
        if shared_phys is None:
            raise FileNotFoundError(
                "3OmicsBenchmarking/dataset-1 is missing both the local 338-drug "
                "physicochemical table and the shared GDSC physicochemical table."
            )
        physico = shared_phys

    rows: List[Dict] = []
    for drug_label, values in ic50.iterrows():
        drug_numeric_id = str(drug_label).split(":")[-1]
        if drug_numeric_id not in threshold_map:
            continue
        pubchem_id, threshold = threshold_map[drug_numeric_id]
        if pubchem_id not in available_graphs:
            continue
        for cell_id, ln_ic50 in values.items():
            if pd.isna(ln_ic50):
                continue
            rows.append(
                {
                    "cell_id": normalize_identifier(cell_id),
                    "drug_id": pubchem_id,
                    "label": int(float(ln_ic50) < threshold),
                }
            )
    response_pairs = canonicalize_response_pairs(pd.DataFrame(rows))
    cell_tables = {
        "genomics_mutation": mutation,
        "transcriptomics_expression": expression,
        "epigenomics_methylation": methylation,
        "similarity": similarity,
    }
    drug_tables = {"physicochemical": physico}
    response_pairs, filtered_cells, filtered_drugs, _, _ = _filter_tables(
        response_pairs=response_pairs,
        cell_tables=cell_tables,
        drug_tables=drug_tables,
        available_graph_drugs=available_graphs,
    )
    metadata = {
        "benchmark": "3OmicsBenchmarking",
        "dataset": "dataset-1",
        "models": ["SOULCDR", "GraphCDR", "RedCDR"],
        "omics_for_soulcdr": [
            "genomics_mutation",
            "transcriptomics_expression",
            "epigenomics_methylation",
        ],
    }
    return _write_prepared_dataset(
        output_dir=output_dir,
        response_pairs=response_pairs,
        cell_tables=filtered_cells,
        drug_tables=filtered_drugs,
        graph_source_dir=graph_dir,
        metadata=metadata,
        extra_files={"drug_threshold.csv": os.path.join(dataset_dir, "Drug", "drug_threshold.csv")},
    )


def prepare_3omics_dataset2(root_dir: str, benchmark_dir: str, output_dir: str) -> Dict:
    dataset_dir = os.path.join(benchmark_dir, "dataset-2")
    mutation = _read_feature_csv(os.path.join(dataset_dir, "genomics_mutation.csv"))
    expression = _read_feature_csv(os.path.join(dataset_dir, "transcriptomics_expression.csv"))
    methylation = _read_feature_csv(os.path.join(dataset_dir, "epigenomics_methylation.csv"))
    similarity = _read_feature_csv(os.path.join(dataset_dir, "similarity.csv"))
    physico = _load_physicochemical_table(os.path.join(dataset_dir, "physicochemical.csv"))
    response_pairs = canonicalize_response_pairs(pd.read_csv(os.path.join(dataset_dir, "response_pairs.csv")))
    graph_dir = os.path.join(root_dir, "final_dataset", "drug_graph_feat")
    if not os.path.isdir(graph_dir):
        graph_dir = os.path.join(dataset_dir, "drug_graph_feat")

    response_pairs, filtered_cells, filtered_drugs, _, _ = _filter_tables(
        response_pairs=response_pairs,
        cell_tables={
            "genomics_mutation": mutation,
            "transcriptomics_expression": expression,
            "epigenomics_methylation": methylation,
            "similarity": similarity,
        },
        drug_tables={"physicochemical": physico},
        available_graph_drugs=load_graph_keys(graph_dir),
    )
    metadata = {
        "benchmark": "3OmicsBenchmarking",
        "dataset": "dataset-2",
        "models": ["SOULCDR", "DeepCDR", "GraphCDR", "RedCDR"],
        "omics_for_soulcdr": [
            "genomics_mutation",
            "transcriptomics_expression",
            "epigenomics_methylation",
        ],
    }
    return _write_prepared_dataset(
        output_dir=output_dir,
        response_pairs=response_pairs,
        cell_tables=filtered_cells,
        drug_tables=filtered_drugs,
        graph_source_dir=graph_dir,
        metadata=metadata,
    )

PREPARE_FUNCTIONS = {
    ("3OmicsBenchmarking", "dataset-1"): prepare_3omics_dataset1,
    ("3OmicsBenchmarking", "dataset-2"): prepare_3omics_dataset2
}


def prepare_benchmark_dataset(root_dir: str, benchmark_name: str, dataset_name: str) -> Dict:
    benchmark_dir = os.path.join(root_dir, benchmark_name)
    output_dir = os.path.join(benchmark_dir, "prepared", dataset_name)
    ensure_dir(output_dir)
    return PREPARE_FUNCTIONS[(benchmark_name, dataset_name)](root_dir, benchmark_dir, output_dir)
