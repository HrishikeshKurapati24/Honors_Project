import copy
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Batch

from benchmarking_common import resolve_device, set_seed
from benchmarking_common.metrics import compute_binary_metrics
from benchmarking_common.results import (
    load_completed_folds,
    load_saved_predictions,
    save_fold_result,
    save_model_outputs,
)
from benchmarking_common.strict_contract import (
    validate_strict_model_contract,
    validate_strict_prepared_metadata,
    validate_strict_train_edge_array,
)
from benchmark_wrappers.common import (
    build_flat_mask,
    build_label_vector,
    build_prediction_rows_from_mask,
    build_pyg_graphs,
    build_shared_similarity_graphs,
    build_strict_train_edge_array,
    load_external_module,
    load_fold_bundle_tables,
    load_hkl_features,
    load_prepared_dataset,
    resolve_fold_ids,
    scope_entities_for_split,
    sorted_cell_drug_ids,
)


def _load_graphcdr_model(root_dir: str):
    prog_dir = os.path.join(root_dir, "benchmark models", "GraphCDR", "prog")
    return load_external_module(
        module_name="benchmark_graphcdr_shared_model",
        file_path=os.path.join(prog_dir, "model_shared_graph.py"),
        extra_paths=[prog_dir],
    )


def _empty_pairs() -> pd.DataFrame:
    return pd.DataFrame(columns=["cell_id", "drug_id", "label"])


def _concat_pairs(*tables: pd.DataFrame) -> pd.DataFrame:
    non_empty = [table for table in tables if table is not None and not table.empty]
    if not non_empty:
        return _empty_pairs()
    return pd.concat(non_empty, ignore_index=True).drop_duplicates(["cell_id", "drug_id"], keep="first")


def _build_scope(
    dataset,
    graph_dir: str,
    scope_pairs: pd.DataFrame,
    train_pairs: pd.DataFrame,
    val_pairs: pd.DataFrame,
    test_pairs: pd.DataFrame,
    device: torch.device,
    top_k: int,
) -> Dict:
    cell_ids, drug_ids = sorted_cell_drug_ids(scope_pairs)
    mutation_df = dataset["tables"]["genomics_mutation"].loc[cell_ids]
    expression_df = dataset["tables"]["transcriptomics_expression"].loc[cell_ids]
    methylation_df = dataset["tables"]["epigenomics_methylation"].loc[cell_ids]
    similarity_df = dataset["tables"]["similarity"].loc[cell_ids]
    physicochemical_df = dataset["tables"]["physicochemical"].loc[drug_ids]

    mutation = torch.from_numpy(mutation_df.to_numpy(dtype="float32")).unsqueeze(1).unsqueeze(1).to(device)
    gexpr = torch.from_numpy(expression_df.to_numpy(dtype="float32")).to(device)
    methylation = torch.from_numpy(methylation_df.to_numpy(dtype="float32")).to(device)

    graph_features = load_hkl_features(graph_dir, drug_ids)
    graphs = build_pyg_graphs(drug_ids, graph_features)
    drug_batch = Batch.from_data_list(graphs).to(device)
    atom_shape = graphs[0].x.shape[-1]

    graph_edge_index_dict = build_shared_similarity_graphs(
        cell_similarity_features=similarity_df.to_numpy(dtype=np.float32),
        drug_similarity_features=physicochemical_df.to_numpy(dtype=np.float32),
        top_k=top_k,
        device=device,
    )

    label_pos = build_label_vector(cell_ids, drug_ids, scope_pairs).to(device)
    train_mask = build_flat_mask(cell_ids, drug_ids, train_pairs).to(device)
    val_mask = build_flat_mask(cell_ids, drug_ids, val_pairs).to(device)
    test_mask = build_flat_mask(cell_ids, drug_ids, test_pairs).to(device)

    train_edge = build_strict_train_edge_array(
        cell_ids,
        drug_ids,
        train_pairs,
        mirror=True,
        sort_by_label=False,
    )
    validate_strict_train_edge_array(
        cell_ids=cell_ids,
        drug_ids=drug_ids,
        train_pairs=train_pairs,
        train_edge=train_edge,
        mirrored=True,
    )
    train_edge = torch.tensor(train_edge, dtype=torch.long, device=device)

    return {
        "cell_ids": cell_ids,
        "drug_ids": drug_ids,
        "mutation_df": mutation_df,
        "expression_df": expression_df,
        "methylation_df": methylation_df,
        "mutation": mutation,
        "gexpr": gexpr,
        "methylation": methylation,
        "drug_batch": drug_batch,
        "atom_shape": atom_shape,
        "label_pos": label_pos,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
        "train_edge": train_edge,
        "cell_similarity_edge": graph_edge_index_dict[("cell", "similar_to", "cell")],
        "drug_similarity_edge": graph_edge_index_dict[("drug", "similar_to", "drug")],
    }


def run(
    root_dir: str,
    prepared_dir: str,
    split_dir: str,
    results_dir: str,
    device: str = "auto",
    seed: int = 0,
    epochs: int = 350,
    lr: float = 1e-3,
    alpha: float | None = None,
    alph: float = 0.30,
    beta: float = 0.30,
    hidden_channels: int = 256,
    output_channels: int = 100,
    top_k: int = 10,
    fold_ids: List[int] | None = None,
) -> Dict:
    del hidden_channels
    set_seed(seed)
    runtime_device = resolve_device(device)
    module = _load_graphcdr_model(root_dir)
    alpha = alph if alpha is None else alpha

    dataset = load_prepared_dataset(prepared_dir)
    validate_strict_prepared_metadata(prepared_dir, dataset["metadata"])
    validate_strict_model_contract(
        "GraphCDR",
        predictive_inputs=(
            "genomics_mutation.csv",
            "transcriptomics_expression.csv",
            "epigenomics_methylation.csv",
            "drug_graph_feat/",
        ),
        graph_inputs=("similarity.csv", "physicochemical.csv", "train_pairs"),
    )
    graph_dir = os.path.join(prepared_dir, "drug_graph_feat")
    fold_metrics = []
    prediction_rows_by_fold = {}
    completed_folds = {int(row["fold"]): row for row in load_completed_folds(results_dir)}

    for fold in resolve_fold_ids(split_dir, fold_ids):
        if fold in completed_folds:
            print(f"> Fold {fold} already complete. Reusing saved outputs.", flush=True)
            fold_metrics.append(completed_folds[fold])
            prediction_rows_by_fold[fold] = load_saved_predictions(results_dir, fold)
            continue
        set_seed(seed + fold * 1000)
        bundle = load_fold_bundle_tables(split_dir, fold)
        protocol = bundle["manifest"].get("protocol", "random")
        train_pairs = bundle["train"]
        val_pairs = bundle["val"]
        test_pairs = bundle["test"]

        if protocol == "random":
            full_pairs = dataset["response_pairs"]
            train_scope = _build_scope(dataset, graph_dir, full_pairs, train_pairs, val_pairs, test_pairs, runtime_device, top_k)
            val_scope = train_scope
            test_scope = train_scope
        else:
            train_cells, train_drugs = scope_entities_for_split(bundle, "train")
            val_cells, val_drugs = scope_entities_for_split(bundle, "val")
            test_cells, test_drugs = scope_entities_for_split(bundle, "test")

            train_scope = _build_scope(
                dataset,
                graph_dir,
                train_pairs,
                train_pairs,
                _empty_pairs(),
                _empty_pairs(),
                runtime_device,
                top_k,
            )
            val_scope = _build_scope(
                dataset,
                graph_dir,
                _concat_pairs(train_pairs, val_pairs),
                train_pairs,
                val_pairs,
                _empty_pairs(),
                runtime_device,
                top_k,
            )
            test_scope = _build_scope(
                dataset,
                graph_dir,
                _concat_pairs(train_pairs, test_pairs),
                train_pairs,
                _empty_pairs(),
                test_pairs,
                runtime_device,
                top_k,
            )

        model = module.GraphCDRSharedGraph(
            hidden_channels=256,
            encoder=module.Encoder(output_channels, 256),
            summary=module.Summary(output_channels, 256),
            feat=module.NodeRepresentationSharedGraph(
                train_scope["atom_shape"],
                train_scope["expression_df"].shape[-1],
                train_scope["methylation_df"].shape[-1],
                output_channels,
            ),
            index=len(train_scope["cell_ids"]),
        ).to(runtime_device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
        criterion = nn.BCELoss()
        best_val_auc = -1.0
        best_state = None
        for epoch in range(epochs):
            print(f"> Fold {fold} - Epoch {epoch + 1}/{epochs}", flush=True)
            model.index = len(train_scope["cell_ids"])
            model.train()
            optimizer.zero_grad()
            pos_z, neg_z, summary_pos, summary_neg, pos_adj = model(
                train_scope["drug_batch"].x,
                train_scope["drug_batch"].edge_index,
                train_scope["drug_batch"].batch,
                train_scope["mutation"],
                train_scope["gexpr"],
                train_scope["methylation"],
                train_scope["train_edge"].detach().cpu().numpy(),
                train_scope["cell_similarity_edge"],
                train_scope["drug_similarity_edge"],
            )
            dgi_pos = model.loss(pos_z, neg_z, summary_pos)
            dgi_neg = model.loss(neg_z, pos_z, summary_neg)
            pos_loss = criterion(pos_adj[train_scope["train_mask"]], train_scope["label_pos"][train_scope["train_mask"]])
            loss = (1 - alpha - beta) * pos_loss + alpha * dgi_pos + beta * dgi_neg
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                model.index = len(val_scope["cell_ids"])
                model.eval()
                _, _, _, _, predictions = model(
                    val_scope["drug_batch"].x,
                    val_scope["drug_batch"].edge_index,
                    val_scope["drug_batch"].batch,
                    val_scope["mutation"],
                    val_scope["gexpr"],
                    val_scope["methylation"],
                    val_scope["train_edge"].detach().cpu().numpy(),
                    val_scope["cell_similarity_edge"],
                    val_scope["drug_similarity_edge"],
                )
                val_metrics = compute_binary_metrics(val_scope["label_pos"][val_scope["val_mask"]], predictions[val_scope["val_mask"]])
                if val_metrics["auc"] > best_val_auc:
                    best_val_auc = val_metrics["auc"]
                    best_state = copy.deepcopy(model.state_dict())

        if best_state is None:
            raise RuntimeError("Strict GraphCDR runner failed to capture a best checkpoint")

        model.load_state_dict(best_state)
        model.index = len(test_scope["cell_ids"])
        model.eval()
        with torch.no_grad():
            _, _, _, _, best_predictions_tensor = model(
                test_scope["drug_batch"].x,
                test_scope["drug_batch"].edge_index,
                test_scope["drug_batch"].batch,
                test_scope["mutation"],
                test_scope["gexpr"],
                test_scope["methylation"],
                test_scope["train_edge"].detach().cpu().numpy(),
                test_scope["cell_similarity_edge"],
                test_scope["drug_similarity_edge"],
            )
            best_predictions = best_predictions_tensor.detach().cpu()
        test_metrics = compute_binary_metrics(test_scope["label_pos"][test_scope["test_mask"]], best_predictions[test_scope["test_mask"].cpu()])
        fold_metric = {
            "fold": fold,
            "best_val_auc": float(best_val_auc),
            "auc": test_metrics["auc"],
            "aupr": test_metrics["aupr"],
            "f1": test_metrics["f1"],
            "acc": test_metrics["acc"],
        }
        prediction_rows = build_prediction_rows_from_mask(
            cell_ids=test_scope["cell_ids"],
            drug_ids=test_scope["drug_ids"],
            flat_mask=test_scope["test_mask"].cpu(),
            flat_labels=test_scope["label_pos"].cpu(),
            flat_predictions=best_predictions,
        )
        fold_metrics.append(fold_metric)
        prediction_rows_by_fold[fold] = prediction_rows
        save_fold_result(results_dir, fold, fold_metric, prediction_rows)

    return save_model_outputs(
        model_results_dir=results_dir,
        fold_metrics=fold_metrics,
        prediction_rows_by_fold=prediction_rows_by_fold,
        metadata={
            "model": "GraphCDR",
            "prepared_dir": prepared_dir,
            "split_dir": split_dir,
            "config": {"lr": lr, "alpha": alpha, "beta": beta, "top_k": top_k},
        },
    )
