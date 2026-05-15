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
from benchmarking_common.smiles_bpe import DeepTTCBPEEncoder, encode_smiles_table, load_smiles_frame
from benchmarking_common.splits import canonicalize_response_pairs, normalize_identifier
from benchmarking_common.strict_contract import (
    STRICT_DISABLED_MODELS,
    STRICT_ENABLED_MODELS,
    STRICT_GRAPH_INPUTS,
    STRICT_PREDICTIVE_INPUTS,
)


def _deduplicate_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.index.is_unique:
        return df
    return df.groupby(level=0, sort=True).mean()


def _read_indexed_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.map(normalize_identifier)
    return _deduplicate_index(df)


def _read_feature_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    first_col = df.columns[0]
    df = df.rename(columns={first_col: "id"})
    df["id"] = df["id"].map(normalize_identifier)
    df = df.set_index("id")
    return _deduplicate_index(df)


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
    return _deduplicate_index(df)


def _shared_physicochemical_table(root_dir: str) -> pd.DataFrame | None:
    shared_path = os.path.join(root_dir, "data", "GDSC", "Processed data", "pubchem_physiochemical_properties_1.csv")
    if not os.path.isfile(shared_path):
        return None
    return _load_physicochemical_table(shared_path, id_column="PUBCHEM_CID")


def _shared_smiles_map(root_dir: str) -> Dict[str, str]:
    return load_smiles_txt(os.path.join(root_dir, "data", "GDSC", "Processed data", "pubchem_smiles_1.txt"))


def _build_strict_smiles_table(root_dir: str, local_smiles_path: str | None, drug_ids: Iterable[str]) -> pd.DataFrame:
    smiles_map: Dict[str, str] = {}
    if local_smiles_path and os.path.isfile(local_smiles_path):
        smiles_map.update(load_smiles_txt(local_smiles_path))
    smiles_map.update({key: value for key, value in _shared_smiles_map(root_dir).items() if key not in smiles_map})

    rows = []
    for drug_id in sorted(set(map(str, drug_ids))):
        smiles = smiles_map.get(str(drug_id))
        if smiles:
            rows.append({"drug_id": str(drug_id), "smiles": str(smiles)})
    if not rows:
        return pd.DataFrame(columns=["smiles"]).set_index(pd.Index([], name="drug_id"))
    return pd.DataFrame(rows).set_index("drug_id")


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


def _prepare_3omics_dataset1_from_source(
    root_dir: str,
    dataset_dir: str,
    output_dir: str,
    benchmark_name: str,
    models: List[str],
    extra_metadata: Dict | None = None,
) -> Dict:
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
        "benchmark": benchmark_name,
        "dataset": "dataset-1",
        "models": models,
        "omics_for_fusecdr": [
            "genomics_mutation",
            "transcriptomics_expression",
            "epigenomics_methylation",
        ],
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    return _write_prepared_dataset(
        output_dir=output_dir,
        response_pairs=response_pairs,
        cell_tables=filtered_cells,
        drug_tables=filtered_drugs,
        graph_source_dir=graph_dir,
        metadata=metadata,
        extra_files={"drug_threshold.csv": os.path.join(dataset_dir, "Drug", "drug_threshold.csv")},
    )


def prepare_3omics_dataset1(root_dir: str, benchmark_dir: str, output_dir: str) -> Dict:
    dataset_dir = os.path.join(benchmark_dir, "dataset-1")
    return _prepare_3omics_dataset1_from_source(
        root_dir=root_dir,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        benchmark_name="3OmicsBenchmarking",
        models=["FUSECDR", "GraphCDR", "RedCDR"],
    )


def _prepare_3omics_strict_dataset1_from_ccle(
    root_dir: str,
    dataset_dir: str,
    output_dir: str,
) -> Dict:
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)

    mutation = _read_indexed_csv(os.path.join(dataset_dir, "Celline", "genomic_mutation_34673_demap_features.csv"))
    expression = _read_indexed_csv(
        os.path.join(dataset_dir, "Celline", "genomic_expression_561celllines_697genes_demap_features.csv")
    )
    methylation = _read_indexed_csv(
        os.path.join(dataset_dir, "Celline", "genomic_methylation_561celllines_808genes_demap_features.csv")
    )
    similarity = _read_indexed_csv(os.path.join(dataset_dir, "Celline", "cellline_pathway_activity_PROGENy_561.csv"))
    ccle_response = pd.read_csv(os.path.join(dataset_dir, "CCLE", "CCLE_response.csv"))
    thresholds = pd.read_csv(os.path.join(dataset_dir, "Drug", "drug_threshold.csv"))

    ccle_response["cell_id"] = ccle_response["DepMap_ID"].map(normalize_identifier)
    ccle_response["drug_id"] = ccle_response["pubchem"].astype(str).map(normalize_identifier)
    ccle_response["Z_SCORE"] = ccle_response["Z_SCORE"].astype(float)

    thresholds["drug_id"] = thresholds["pubchem"].astype(str).map(normalize_identifier)
    threshold_values = thresholds[["drug_id", "IC50"]].copy()
    threshold_values["threshold"] = threshold_values["IC50"].astype(float)
    threshold_values["threshold_source"] = "drug_threshold_ic50"
    threshold_values = threshold_values[["drug_id", "threshold", "threshold_source"]]

    missing_threshold_drugs = sorted(set(ccle_response["drug_id"]) - set(threshold_values["drug_id"]))
    if missing_threshold_drugs:
        fallback = (
            ccle_response[ccle_response["drug_id"].isin(missing_threshold_drugs)]
            .groupby("drug_id", as_index=False)["Z_SCORE"]
            .median()
            .rename(columns={"Z_SCORE": "threshold"})
        )
        fallback["threshold_source"] = "ccle_zscore_median"
        threshold_values = pd.concat([threshold_values, fallback], ignore_index=True)

    threshold_values = threshold_values.drop_duplicates(subset=["drug_id"], keep="first")
    merged = ccle_response.merge(threshold_values, on="drug_id", how="inner")
    merged["label"] = (merged["Z_SCORE"] > merged["threshold"]).astype(int)
    response_pairs = canonicalize_response_pairs(merged[["cell_id", "drug_id", "label"]])

    graph_dir = os.path.join(dataset_dir, "CCLE", "drug_graph_feat")
    available_graphs = set(load_graph_keys(graph_dir))

    local_phys_path = os.path.join(dataset_dir, "CCLE", "physicochemical.csv")
    if os.path.isfile(local_phys_path):
        physico = _load_physicochemical_table(local_phys_path, id_column="pubchem_id")
    else:
        local_phys_path = os.path.join(dataset_dir, "CCLE", "pubchem_physiochemical_properties_338.csv")
        if os.path.isfile(local_phys_path):
            physico = _load_physicochemical_table(local_phys_path, id_column="PUBCHEM_CID")
        else:
            local_phys_path = os.path.join(dataset_dir, "Drug", "pubchem_physiochemical_properties_338.csv")
            if os.path.isfile(local_phys_path):
                physico = _load_physicochemical_table(local_phys_path, id_column="PUBCHEM_CID")
            else:
                shared_phys = _shared_physicochemical_table(root_dir)
                if shared_phys is None:
                    raise FileNotFoundError(
                        "3OmicsStrictBenchmarking/dataset-1 is missing CCLE/physicochemical.csv, "
                        "the local 338-drug physicochemical table, and the shared GDSC physicochemical table."
                    )
                physico = shared_phys

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

    threshold_summary = threshold_values.copy()
    threshold_summary["used_explicit_threshold"] = (
        threshold_summary["threshold_source"] == "drug_threshold_ic50"
    ).astype(int)
    threshold_summary = threshold_summary.set_index("drug_id").loc[sorted(filtered_drugs["physicochemical"].index)]

    metadata = {
        "benchmark": "3OmicsStrictBenchmarking",
        "dataset": "dataset-1",
        "models": list(STRICT_ENABLED_MODELS),
        "omics_for_fusecdr": [
            "genomics_mutation",
            "transcriptomics_expression",
            "epigenomics_methylation",
        ],
        "cell_graph_source": "similarity.csv",
        "drug_similarity_graph_source": "physicochemical.csv",
        "drug_structure_source": "drug_graph_feat",
        "response_graph_source": "train_pairs",
        "graph_builder": "topk_directed_cosine",
        "strict_predictive_inputs": list(STRICT_PREDICTIVE_INPUTS),
        "strict_graph_inputs": list(STRICT_GRAPH_INPUTS),
        "disabled_models_pending_strict_alignment": list(STRICT_DISABLED_MODELS),
        "response_metric": "CCLE_Z_SCORE",
        "label_rule": "label = int(Z_SCORE > per_drug_threshold)",
        "threshold_priority": [
            "drug_threshold.csv IC50",
            "per-drug median Z_SCORE fallback",
        ],
        "explicit_threshold_drug_count": int(
            (threshold_summary["threshold_source"] == "drug_threshold_ic50").sum()
        ),
        "median_threshold_drug_count": int(
            (threshold_summary["threshold_source"] == "ccle_zscore_median").sum()
        ),
    }

    metadata = _write_prepared_dataset(
        output_dir=output_dir,
        response_pairs=response_pairs,
        cell_tables=filtered_cells,
        drug_tables=filtered_drugs,
        graph_source_dir=graph_dir,
        metadata=metadata,
        extra_tables={"ccle_thresholds": threshold_summary},
        extra_files={"drug_threshold.csv": os.path.join(dataset_dir, "Drug", "drug_threshold.csv")},
    )
    metadata = dict(metadata)
    metadata["notes"] = metadata.get("notes", []) + [
        "Dataset-1 strict prep now uses CCLE_response.csv as the response source.",
        "For drugs missing explicit entries in drug_threshold.csv, the per-drug median CCLE Z_SCORE is used as the fallback threshold.",
        "DeepTTC auxiliary inputs are not emitted; strict runners use the common strict predictive inputs directly.",
    ]
    stale_files = [
        os.path.join(output_dir, "pathway.csv"),
        os.path.join(output_dir, "aux", "smiles_token_ids.csv"),
        os.path.join(output_dir, "aux", "smiles_attention_mask.csv"),
    ]
    for stale_path in stale_files:
        if os.path.isfile(stale_path):
            os.remove(stale_path)
    write_json(os.path.join(output_dir, "metadata.json"), metadata)
    return metadata


def prepare_3omics_strict_dataset1(root_dir: str, benchmark_dir: str, output_dir: str) -> Dict:
    dataset_dir = os.path.join(benchmark_dir, "dataset-1")
    return _prepare_3omics_strict_dataset1_from_ccle(
        root_dir=root_dir,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
    )


def prepare_3omics_strict_dataset2(root_dir: str, benchmark_dir: str, output_dir: str) -> Dict:
    dataset_dir = os.path.join(benchmark_dir, "dataset-2")
    mutation = _read_feature_csv(os.path.join(dataset_dir, "genomics_mutation.csv"))
    expression = _read_feature_csv(os.path.join(dataset_dir, "transcriptomics_expression.csv"))
    methylation = _read_feature_csv(os.path.join(dataset_dir, "epigenomics_methylation.csv"))
    similarity = _read_feature_csv(os.path.join(dataset_dir, "similarity.csv"))
    physico = _load_physicochemical_table(os.path.join(dataset_dir, "physicochemical.csv"))
    response_pairs = canonicalize_response_pairs(pd.read_csv(os.path.join(dataset_dir, "response_pairs.csv")))
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
        "benchmark": "3OmicsStrictBenchmarking",
        "dataset": "dataset-2",
        "models": list(STRICT_ENABLED_MODELS),
        "omics_for_fusecdr": [
            "genomics_mutation",
            "transcriptomics_expression",
            "epigenomics_methylation",
        ],
        "cell_graph_source": "similarity.csv",
        "drug_similarity_graph_source": "physicochemical.csv",
        "drug_structure_source": "drug_graph_feat",
        "response_graph_source": "train_pairs",
        "graph_builder": "topk_directed_cosine",
        "strict_predictive_inputs": list(STRICT_PREDICTIVE_INPUTS),
        "strict_graph_inputs": list(STRICT_GRAPH_INPUTS),
        "disabled_models_pending_strict_alignment": list(STRICT_DISABLED_MODELS),
    }
    metadata = _write_prepared_dataset(
        output_dir=output_dir,
        response_pairs=response_pairs,
        cell_tables=filtered_cells,
        drug_tables=filtered_drugs,
        graph_source_dir=graph_dir,
        metadata=metadata,
    )
    metadata = dict(metadata)
    metadata["notes"] = metadata.get("notes", []) + [
        "Strict prep does not emit DeepTTC auxiliary inputs; DeepTTC now uses the strict predictive inputs directly in its runner.",
    ]
    stale_files = [
        os.path.join(output_dir, "pathway.csv"),
        os.path.join(output_dir, "aux", "smiles_token_ids.csv"),
        os.path.join(output_dir, "aux", "smiles_attention_mask.csv"),
    ]
    for stale_path in stale_files:
        if os.path.isfile(stale_path):
            os.remove(stale_path)
    write_json(os.path.join(output_dir, "metadata.json"), metadata)
    return metadata


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
        "models": ["FUSECDR", "DeepCDR", "GraphCDR", "RedCDR"],
        "omics_for_fusecdr": [
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


def prepare_gadrp_native_dataset(root_dir: str, benchmark_dir: str, output_dir: str) -> Dict:
    del benchmark_dir
    dataset_dir = os.path.join(root_dir, "benchmark models", "GADRP-main", "data")
    cell_dir = os.path.join(dataset_dir, "cell_line")
    drug_dir = os.path.join(dataset_dir, "drug")
    pair_dir = os.path.join(dataset_dir, "pair")

    mapper = pd.read_csv(os.path.join(cell_dir, "common_cell_lines_mapper.csv"))
    mapper["cell_line"] = mapper["cell_line"].map(normalize_identifier)
    mapper["depMapID"] = mapper["depMapID"].map(normalize_identifier)
    cell_to_depmap = {
        str(row.cell_line): str(row.depMapID)
        for row in mapper.itertuples(index=False)
        if str(row.cell_line) and str(row.depMapID)
    }

    expression = _read_feature_csv(os.path.join(cell_dir, "transcriptomics_expression.csv"))
    cnv = _read_feature_csv(os.path.join(cell_dir, "genomics_cnv.csv"))
    mirna = _read_feature_csv(os.path.join(cell_dir, "transcriptomics_miRNA.csv"))
    methylation = _read_feature_csv(os.path.join(cell_dir, "epigenomics_methylation_data.csv"))
    fingerprint = _load_physicochemical_table(
        os.path.join(drug_dir, "881_dim_fingerprint.csv"),
        id_column="Unnamed: 0",
    )
    physicochemical = _load_physicochemical_table(
        os.path.join(drug_dir, "269_dim_physicochemical.csv"),
        id_column="pubchem_cid",
    )

    response_raw = pd.read_csv(os.path.join(pair_dir, "response_pairs.csv"))
    response_raw = response_raw.rename(columns={"ccle_name": "cell_id", "pubchem_cid": "drug_id"})
    response_raw["cell_id"] = response_raw["cell_id"].map(normalize_identifier).map(cell_to_depmap)
    response_raw["drug_id"] = response_raw["drug_id"].map(normalize_identifier)
    response_pairs = canonicalize_response_pairs(response_raw[["cell_id", "drug_id", "label"]])

    similarity = pd.concat(
        [
            mirna.add_prefix("mirna__"),
            methylation.add_prefix("methylation__"),
        ],
        axis=1,
        join="inner",
    )

    cell_tables = {
        "transcriptomics_expression": expression,
        "genomics_cnv": cnv,
        "transcriptomics_miRNA": mirna,
        "epigenomics_methylation": methylation,
        "similarity": similarity,
    }
    drug_tables = {
        "drug_fingerprint": fingerprint,
        "physicochemical": physicochemical,
    }

    response_pairs, filtered_cells, filtered_drugs, _, _ = _filter_tables(
        response_pairs=response_pairs,
        cell_tables=cell_tables,
        drug_tables=drug_tables,
        available_graph_drugs=fingerprint.index,
    )
    metadata = {
        "benchmark": "GADRPBenchmarking",
        "dataset": "dataset-1",
        "models": ["FUSECDR", "GADRP"],
        "omics_for_fusecdr": ["transcriptomics_expression", "genomics_cnv"],
        "native_prediction_modalities": ["transcriptomics_expression", "genomics_cnv"],
        "native_similarity_modalities": ["transcriptomics_miRNA", "epigenomics_methylation"],
        "drug_input": "fingerprint",
        "drug_similarity_source": "269_dim_physicochemical",
    }
    return _write_prepared_dataset(
        output_dir=output_dir,
        response_pairs=response_pairs,
        cell_tables=filtered_cells,
        drug_tables=filtered_drugs,
        graph_source_dir=None,
        metadata=metadata,
    )


def prepare_gadrp_feature_fair_dataset(root_dir: str, benchmark_dir: str, output_dir: str) -> Dict:
    del benchmark_dir
    dataset_dir = os.path.join(root_dir, "benchmark models", "GADRP-main", "data")
    cell_dir = os.path.join(dataset_dir, "cell_line")
    drug_dir = os.path.join(dataset_dir, "drug")
    pair_dir = os.path.join(dataset_dir, "pair")

    mapper = pd.read_csv(os.path.join(cell_dir, "common_cell_lines_mapper.csv"))
    mapper["cell_line"] = mapper["cell_line"].map(normalize_identifier)
    mapper["depMapID"] = mapper["depMapID"].map(normalize_identifier)
    cell_to_depmap = {
        str(row.cell_line): str(row.depMapID)
        for row in mapper.itertuples(index=False)
        if str(row.cell_line) and str(row.depMapID)
    }

    expression = _read_feature_csv(os.path.join(cell_dir, "transcriptomics_expression.csv"))
    cnv = _read_feature_csv(os.path.join(cell_dir, "genomics_cnv.csv"))
    mirna = _read_feature_csv(os.path.join(cell_dir, "transcriptomics_miRNA.csv"))
    methylation = _read_feature_csv(os.path.join(cell_dir, "epigenomics_methylation_data.csv"))
    gsva_similarity = _read_feature_csv(os.path.join(cell_dir, "GSVA_pathway_scores_similarity.csv"))
    fingerprint = _load_physicochemical_table(
        os.path.join(drug_dir, "881_dim_fingerprint.csv"),
        id_column="Unnamed: 0",
    )
    physicochemical = _load_physicochemical_table(
        os.path.join(drug_dir, "269_dim_physicochemical.csv"),
        id_column="pubchem_cid",
    )

    response_raw = pd.read_csv(os.path.join(pair_dir, "response_pairs.csv"))
    response_raw = response_raw.rename(columns={"ccle_name": "cell_id", "pubchem_cid": "drug_id"})
    response_raw["cell_id"] = response_raw["cell_id"].map(normalize_identifier).map(cell_to_depmap)
    response_raw["drug_id"] = response_raw["drug_id"].map(normalize_identifier)
    response_pairs = canonicalize_response_pairs(response_raw[["cell_id", "drug_id", "label"]])

    cell_tables = {
        "transcriptomics_expression": expression,
        "genomics_cnv": cnv,
        "transcriptomics_miRNA": mirna,
        "epigenomics_methylation": methylation,
        "similarity": gsva_similarity,
    }
    drug_tables = {
        "drug_fingerprint": fingerprint,
        "physicochemical": physicochemical,
    }

    response_pairs, filtered_cells, filtered_drugs, _, _ = _filter_tables(
        response_pairs=response_pairs,
        cell_tables=cell_tables,
        drug_tables=drug_tables,
        available_graph_drugs=fingerprint.index,
    )
    metadata = {
        "benchmark": "GADRPFeatureFairBenchmarking",
        "dataset": "dataset-1",
        "models": ["FUSECDR", "GADRP"],
        "omics_for_fusecdr": ["transcriptomics_expression", "genomics_cnv"],
        "native_prediction_modalities": ["transcriptomics_expression", "genomics_cnv"],
        "drug_input": "fingerprint",
        "cell_similarity_source": "GSVA_pathway_scores_similarity",
        "drug_similarity_source": "269_dim_physicochemical",
        "gadrp_cell_similarity_source": ["transcriptomics_miRNA", "epigenomics_methylation"],
        "gadrp_drug_similarity_source": "269_dim_physicochemical",
    }
    return _write_prepared_dataset(
        output_dir=output_dir,
        response_pairs=response_pairs,
        cell_tables=filtered_cells,
        drug_tables=filtered_drugs,
        graph_source_dir=None,
        metadata=metadata,
    )


def prepare_deepttc_benchmark_dataset(root_dir: str, benchmark_dir: str, output_dir: str) -> Dict:
    del benchmark_dir
    dataset_dir = os.path.join(root_dir, "benchmark models", "DeepTTC-main", "benchmark formatted dataset")

    response_pairs = canonicalize_response_pairs(
        pd.read_csv(os.path.join(dataset_dir, "response_pairs.csv"))
    )
    expression = _read_feature_csv(os.path.join(dataset_dir, "transcriptomics_expression.csv"))
    pathway = _read_feature_csv(os.path.join(dataset_dir, "pathway.csv"))
    physicochemical = _load_physicochemical_table(
        os.path.join(dataset_dir, "physicochemical.csv"),
        id_column="pubchem_id",
    )
    smiles = load_smiles_frame(os.path.join(dataset_dir, "smiles_data.csv"))

    common_cells = sorted(set(response_pairs["cell_id"]) & set(expression.index) & set(pathway.index))
    common_drugs = sorted(set(response_pairs["drug_id"]) & set(physicochemical.index) & set(smiles.index))

    response_pairs = canonicalize_response_pairs(
        response_pairs[
            response_pairs["cell_id"].isin(common_cells) & response_pairs["drug_id"].isin(common_drugs)
        ]
    )
    expression = expression.loc[common_cells]
    pathway = pathway.loc[common_cells]
    similarity = pathway.copy()
    physicochemical = physicochemical.loc[common_drugs]
    smiles = smiles.loc[common_drugs]

    encoder = DeepTTCBPEEncoder(os.path.join(root_dir, "benchmark models", "DeepTTC-main"))
    encoded = encode_smiles_table(smiles, encoder)

    metadata = {
        "benchmark": "DeepTTCBenchmarking",
        "dataset": "dataset-1",
        "models": ["FUSECDR", "DeepTTC"],
        "omics_for_fusecdr": ["transcriptomics_expression"],
        "prediction_modalities": ["transcriptomics_expression", "smiles"],
        "cell_similarity_source": "pathway.csv",
        "drug_similarity_source": "physicochemical.csv",
        "drug_input": "smiles_bpe",
    }
    return _write_prepared_dataset(
        output_dir=output_dir,
        response_pairs=response_pairs,
        cell_tables={
            "transcriptomics_expression": expression,
            "pathway": pathway,
            "similarity": similarity,
        },
        drug_tables={"physicochemical": physicochemical},
        graph_source_dir=None,
        metadata=metadata,
        extra_tables={
            "smiles_data": smiles,
            "smiles_token_ids": encoded.token_ids,
            "smiles_attention_mask": encoded.attention_mask,
        },
    )

PREPARE_FUNCTIONS = {
    ("3OmicsBenchmarking", "dataset-1"): prepare_3omics_dataset1,
    ("3OmicsBenchmarking", "dataset-2"): prepare_3omics_dataset2,
    ("3OmicsStrictBenchmarking", "dataset-1"): prepare_3omics_strict_dataset1,
    ("3OmicsStrictBenchmarking", "dataset-2"): prepare_3omics_strict_dataset2,
    ("GADRPBenchmarking", "dataset-1"): prepare_gadrp_native_dataset,
    ("GADRPFeatureFairBenchmarking", "dataset-1"): prepare_gadrp_feature_fair_dataset,
    ("DeepTTCBenchmarking", "dataset-1"): prepare_deepttc_benchmark_dataset,
}


def prepare_benchmark_dataset(root_dir: str, benchmark_name: str, dataset_name: str) -> Dict:
    benchmark_dir = os.path.join(root_dir, benchmark_name)
    output_dir = os.path.join(benchmark_dir, "prepared", dataset_name)
    ensure_dir(output_dir)
    return PREPARE_FUNCTIONS[(benchmark_name, dataset_name)](root_dir, benchmark_dir, output_dir)
