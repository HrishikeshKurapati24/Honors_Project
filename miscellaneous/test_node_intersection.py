import sys

sys.path.append(".")

from data_load import dataload

ROOT = "/Volumes/Work/Semester - 6/Honors/CDRP models testing/new_data"
DRUG_DIR = ROOT + "/GDSC/Drug/drug_graph_feat"
# Keep this aligned with main.py default in the node module folder.
RESPONSE_FILE = ROOT + "/GDSC/Processed data/response_labels.csv"


def run_loader(*, genomics_csv, use_proteomics, use_metabolomics, use_pathway):
    return dataload(
        drug_feature_dir=DRUG_DIR,
        response_file=RESPONSE_FILE,
        genomics_csv=genomics_csv,
        epigenomics_csv=ROOT + "/CCLE/Processed data/DNA_methylation_CpG_clusters_Vista_enhancers_with_mutation_data.csv",
        transcriptomics_csv=ROOT + "/CCLE/Processed data/gene_expression_data.csv",
        proteomics_csv=ROOT + "/CCLE/Processed data/reverse_phase_protein_array_data.csv",
        metabolomics_csv=ROOT + "/CCLE/Processed data/metabolomics_data.csv",
        pathway_csv=ROOT + "/CCLE/Processed data/cell_pathway_scores_from_gene_expression.csv",
        use_genomics=True,
        use_epigenomics=True,
        use_transcriptomics=True,
        use_proteomics=use_proteomics,
        use_metabolomics=use_metabolomics,
        use_pathway=use_pathway,
    )


def assert_three_omics_loader_behavior():
    print("\n=== TEST: 3 OMICS MODE (proteomics/metabolomics/pathway disabled) ===")
    res3 = run_loader(
        genomics_csv=ROOT + "/CCLE/Processed data/mutation_data.csv",
        use_proteomics=False,
        use_metabolomics=False,
        use_pathway=False,
    )
    assert not res3[1].empty, "Genomics should be present in 3-omics mode."
    assert not res3[2].empty, "Epigenomics should be present in 3-omics mode."
    assert not res3[3].empty, "Transcriptomics should be present in 3-omics mode."
    assert res3[4].empty, "Proteomics should be empty when disabled."
    assert res3[5].empty, "Metabolomics should be empty when disabled."
    assert res3[6].empty, "Pathway should be empty when disabled."
    assert len(res3[7]) > 0, "Expected non-empty response pairs in 3-omics mode."
    assert res3[8] > 0 and res3[9] > 0, "Expected positive counts for cells and drugs."
    print("3-omics checks passed.")


def assert_six_omics_loader_behavior():
    print("\n=== TEST: 6 OMICS MODE (all enabled) ===")
    res6 = run_loader(
        genomics_csv=ROOT + "/CCLE/Processed data/chromatin_profiling.csv",
        use_proteomics=True,
        use_metabolomics=True,
        use_pathway=True,
    )
    assert not res6[1].empty, "Genomics should be present in 6-omics mode."
    assert not res6[2].empty, "Epigenomics should be present in 6-omics mode."
    assert not res6[3].empty, "Transcriptomics should be present in 6-omics mode."
    assert not res6[4].empty, "Proteomics should be present in 6-omics mode."
    assert not res6[5].empty, "Metabolomics should be present in 6-omics mode."
    assert not res6[6].empty, "Pathway should be present in 6-omics mode."
    assert len(res6[7]) > 0, "Expected non-empty response pairs in 6-omics mode."
    assert res6[8] > 0 and res6[9] > 0, "Expected positive counts for cells and drugs."
    print("6-omics checks passed.")


if __name__ == "__main__":
    assert_three_omics_loader_behavior()
    assert_six_omics_loader_behavior()
    print("\nAll node intersection loader assertions passed.")
