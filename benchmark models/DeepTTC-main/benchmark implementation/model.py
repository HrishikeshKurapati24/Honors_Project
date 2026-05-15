import importlib.util
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


def _load_original_helper():
    helper_path = Path(__file__).resolve().parents[1] / "model_helper.py"
    spec = importlib.util.spec_from_file_location("_deepttc_original_helper", helper_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_ORIGINAL_HELPER = _load_original_helper()
Embeddings = _ORIGINAL_HELPER.Embeddings
Encoder_MultipleLayers = _ORIGINAL_HELPER.Encoder_MultipleLayers

# --- Main Model Classes ---

class DrugTransformer(nn.Module):
    def __init__(self):
        super(DrugTransformer, self).__init__()
        input_dim_drug = 2586
        transformer_emb_size_drug = 128
        transformer_dropout_rate = 0.1
        transformer_n_layer_drug = 8
        transformer_intermediate_size_drug = 512
        transformer_num_attention_heads_drug = 8
        
        self.emb = Embeddings(input_dim_drug, transformer_emb_size_drug, 50, transformer_dropout_rate)
        self.encoder = Encoder_MultipleLayers(
            transformer_n_layer_drug,
            transformer_emb_size_drug,
            transformer_intermediate_size_drug,
            transformer_num_attention_heads_drug,
            0.1, 0.1
        )

    def forward(self, v_d, v_mask):
        v_d = v_d.long()
        v_mask = v_mask.long()
        ex_e_mask = v_mask.unsqueeze(1).unsqueeze(2)
        ex_e_mask = (1.0 - ex_e_mask) * -10000.0
        emb = self.emb(v_d)
        encoded_layers = self.encoder(emb.float(), ex_e_mask.float())
        return encoded_layers[:, 0]

class GeneMLP(nn.Module):
    def __init__(self, input_dim_gene=17737):
        super(GeneMLP, self).__init__()
        mlp_hidden_dims_gene = [1024, 256, 64]
        hidden_dim_gene = 256
        
        dims = [input_dim_gene] + mlp_hidden_dims_gene + [hidden_dim_gene]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
        self.predictor = nn.Sequential(*layers)

    def forward(self, v):
        return self.predictor(v)

class DeepTTC_Model(nn.Module):
    def __init__(self, gene_input_dim=17737):
        super(DeepTTC_Model, self).__init__()
        self.drug_model = DrugTransformer()
        self.gene_model = GeneMLP(gene_input_dim)
        
        self.dropout = nn.Dropout(0.1)
        hidden_dims = [1024, 1024, 512]
        # Drug embedding (128) + Gene embedding (256) = 384
        input_dim = 128 + 256
        
        dims = [input_dim] + hidden_dims + [1]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                layers.append(self.dropout)
        self.classifier = nn.Sequential(*layers)

    def forward(self, v_drug, v_mask, v_gene):
        v_d = self.drug_model(v_drug, v_mask)
        v_g = self.gene_model(v_gene)
        
        v_f = torch.cat((v_d, v_g), 1)
        return self.classifier(v_f)
