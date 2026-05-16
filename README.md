# FUSE-CDR (Flexible Omics Unified-representation Learning for Cancer Drug Response)

**Author:** Hrishikesh Kurapati  
**Institution:** IIIT Sri City  
**Guide/Supervisor:** Dr. Amilpur Santhosh  

---

## 🚀 Overview

**FUSE-CDR** is a flexible heterogeneous graph learning framework designed to predict cancer drug responses. It achieves this by combining **hierarchical multi-omics fusion** with **dual-branch graph representation learning** (GraphSAGE and HGT), specifically modeling the intricate relationships between different omics layers, drugs, and cancer cell lines.

**Key Contributions:**
*   **Heterogeneous Graph Construction:** Explicitly models drug–drug, cell–cell, and drug–cell edges, bridging seen and unseen entities.
*   **Hierarchical Multi-Omics Fusion:** Employs a cross-modal sequential attention mechanism aligning with biological regulatory structures to fuse up to 6 omics modalities.
*   **Dual-Branch Encoder:** Merges local structural pattern recognition (GraphSAGE) with global long-range dependency modeling (Heterogeneous Graph Transformer).
*   **Supervised Contrastive Learning (SCL):** Increases representation robustness, pulling matching response pairs together and pushing mismatched pairs apart.

FUSE-CDR achieves a state-of-the-art **AUC of 0.9359** on the benchmark dataset random split and demonstrates robust inductive generalization with an **AUC of 0.5847** in the strictly disjoint "Unseen Both" setting. In comprehensive evaluations across 5 foundational baselines, this represents a **12.31% relative improvement** over GraphCDR and **1.35%** over RedCDR in inductive settings.

---

## 📖 Problem Context

### What is Cancer Drug Response Prediction (CDRP)?
CDRP is a core challenge in precision oncology. The goal is simple: given a specific **cancer cell line** and a **drug molecule**, predict the therapeutic sensitivity (e.g., sensitive or resistant, measured traditionally via metrics like IC50). 

### Why It Matters
Identifying effective drugs for individual patients accelerates drug discovery and supports personalized treatment strategies. However, existing methods struggle due to noisy high-dimensional multi-omics data, the complex heterogeneous nature of biological graphs (i.e. treating cells and drugs identically), and importantly, severely degraded generalization when predicting for completely novel drugs or cell lines (the inductive setting).

---

## 🗺️ Project Roadmap

*   [x] **Phase 1:** Representation learning
*   [x] **Phase 2 (Latest Completed):** Graph modeling + contrastive learning
*   [ ] **Phase 3:** Explainability
*   [ ] **Phase 4:** Clinical validation

---

## 🏗️ Architecture

![Architecture of FUSE-CDR](architecture_diagram.jpg)

1.  **Drug Representation Learning:** Uses Graph Isomorphism Networks (GIN) to process SMILES-derived molecular graphs into compact 256-dimensional embeddings.
2.  **Cell Representation Learning:** Utilizes intra-category attention to compress single omics features, followed by a hierarchical attention mechanism bridging **Genomics** $\rightarrow$ **Epigenomics (Methylation)** $\rightarrow$ **Transcriptomics** $\rightarrow$ **Epigenomics (Chromatin)** $\rightarrow$ **Metabolomics** $\rightarrow$ **Proteomics** $\rightarrow$ **Pathway activity**.
3.  **Heterogeneous Graph Construction:** Integrates independent embeddings into a unified multi-relational graph via top-10 cosine similarities (cell-cell and drug-drug), establishing communication bridges during prediction.
4.  **Dual-Branch Graph Encoder:** GraphSAGE effectively samples node neighborhoods to understand local clusters while the Heterogeneous Graph Transformer (HGT) attends to distinct relation types and long-scale distances. 

---

## 💻 Getting Started

This repository provides all code required to reproduce experimental setups, evaluate baselines, and develop further upon FUSE-CDR.

### 1. Requirements

Ensure you are running Python 3.11+ and PyTorch 2.4+.

```bash
# Clone the repository
git clone https://github.com/yourusername/FUSE-CDR.git
cd FUSE-CDR

# Install dependencies
pip install -r requirements.txt
```
*(Key dependencies include PyTorch Geometric, DeepChem, RDKit, hickle, h5py, and standard scientific libraries)*

### 2. Execution Commands

Run the model on the flexible setup:
```bash
python "flexible model/main_flexible.py" 
# or via shell script for specific configs
cd "flexible model"
bash run_flexible_selected_configs.sh
```

Run benchmarks on the unified 3-omics datasets:
```bash
cd 3OmicsStrictBenchmarking

# Run specific inductive evaluations
python run_unseen_both.py --datasets dataset-2
python run_unseen_cells.py --datasets dataset-2
python run_unseen_drugs.py --datasets dataset-2
```

---

## 📁 Project Structure

*   `3OmicsStrictBenchmarking/` - Full benchmarking pipeline, strict evaluation protocols, and automated tuning for FUSE-CDR and baselines.
*   `flexible model/` - Core implementation code of the 7-modality FUSE-CDR framework.
*   `benchmark_wrappers/` - Unified runners for baseline models (GraphCDR, RedCDR, GADRP, DeepTTC, GraphDRP).
*   `benchmarking_common/` - Shared hyperparameter tuning, model logic, and dataset preparation routines.
*   `data/` & `final_dataset/` - Raw inputs (CCLE, GDSC, SMILES, etc.) and fully processed intermediate graphs and omics tables.
*   `Research paper/` - Complete LaTeX source code, resulting PDFs, and plotting scripts for figures. 

---

## 📊 Data Pipeline

Data flows from broad external repositories down directly to model inputs:
1.  **Sources:** Integrated primarily from **CCLE (DepMap)** and **GDSC1**. 
2.  **Omics Preprocessing:** Missing values are imputed using KNN ($k=5$). We apply biological feature selection based on the MSigDB Hallmark gene sets. 
3.  **Drug Preprocessing:** SMILES are converted into molecular graphs using DeepChem, while global physicochemical attributes are extracted to parameterize drug--drug relational edges.

---

## 🔬 Experimental Setup

We evaluate FUSE-CDR across multiple setups, designed around rigorous clinical evaluation norms:

*   **Custom 7-Modalities Dataset:** Used largely for model development and architecture validation (425 cells, 298 drugs).
*   **Custom 3-Omics Dataset & GraphCDR Benchmark:** Strict unified benchmark datasets to conduct fair comparisons representing Mutation, Expression, and Methylation data. 
*   **Evaluation Protocol:** 5-fold cross-validation. We break splits down by:
    *   **Random Split:** Classic performance measurement.
    *   **Unseen Cells / Unseen Drugs:** One group is completely novel.
    *   **Unseen Both:** Strict inductive split where the evaluation pairings have no entities overlapping the training set.
*   **Metrics:** Main metrics relied upon are AUC (Area Under the Receiver Operating Characteristic Curve) and AUPR (Area Under the Precision-Recall Curve).

---

## 📈 Results

| Model | Setting | AUC | AUPR |
| :--- | :--- | :--- | :--- |
| **GADRP** | Random (Dataset-2) | 0.8407 $\pm$ 0.0138 | 0.7852 $\pm$ 0.0195 |
| **DeepTTC** | Random (Dataset-2) | 0.8808 $\pm$ 0.0165 | 0.8269 $\pm$ 0.0261 |
| **GraphDRP** | Random (Dataset-2) | 0.9218 $\pm$ 0.0039 | 0.8941 $\pm$ 0.0057 |
| **GraphCDR** | Random (Dataset-2) | 0.9315 $\pm$ 0.0007 | 0.9079 $\pm$ 0.0019 |
| **RedCDR** | Random (Dataset-2) | 0.9319 $\pm$ 0.0006 | 0.9087 $\pm$ 0.0012 |
| **FUSE-CDR** | Random (Dataset-2) | **0.9359 $\pm$ 0.0010** | **0.9156 $\pm$ 0.0021** |

*\*Benchmark Random setup indicates competitive predictive matching across identical data splits*

**Key Takeaways:**
*   **Unmatched Inductive Performance:** FUSE-CDR demonstrates superior generalization in the strictly disjoint "Unseen Both" setting, outperforming previous SOTA benchmarks.
*   **Modal Synergy:** The flexible design reveals that selective modality combinations (e.g., Pathway + Expression + Proteomics) can reach optimal performance (0.9226 AUC) while reducing computational redundancy.

---

## 📚 Baselines / References

FUSE-CDR was systematically modeled and benchmarked against the following foundational architectures for CDRP:
*   [GraphCDR](https://github.com/liuxuan666/GraphCDR) (Multi-omics Graph Neural Network)
*   [RedCDR](https://github.com/mhxu1998/RedCDR) (Relational Decomposition Graph Neural Network)
*   [GraphDRP](https://github.com/hosseinshn/GraphDRP) (Graph Convolutional Networks for Drug Response)
*   [DeepTTC](https://github.com/qiaoyun-li/DeepTTC) (Transformer-based Therapeutic Candidate Prediction)
*   [GADRP](https://github.com/flora619/GADRP) (Graph Convolutional Networks and Autoencoders)

---

## 💡 Key Insights

*   **Pathway Summarization Wins:** Evaluating multi-omics independently revealed that deriving high-level Pathway activity scores serves as the most potent signal relative to throwing unstructured omics into dense layers.
*   **Contrastive regularizations stabilize embeddings:** During severe class-imbalance, the Supervised Contrastive Learning objective prevents representational collapse, proving decisive upon unseen nodes. 
*   **Global interactions limit locality:** Only testing against immediate neighbored graphs limits model potential. Combining GraphSAGE (local) with an HGT (global) solves graph saturation without overfitting.

---

## 🔮 Future Work

*   **Explainability:** Tracing predictions backwards through the cell-encoder to identify which specific pathways or biological omics triggered therapeutic sensitivities.
*   **Clinical Validation:** Projecting the algorithm toward heavily in-vivo or clinical trial datasets bypassing strictly cell-line distributions.
*   **Additional Biological Networks:** Constructing richer heterogeneous edges beyond raw structural and biological pathways. 

---

## 📄 License & Citation

Licensed under **MIT License**.
*(If you found this software useful in your research, consider citing the upcoming FUSE-CDR publication once archived.)*