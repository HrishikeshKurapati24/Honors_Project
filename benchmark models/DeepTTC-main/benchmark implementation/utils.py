import os
import pandas as pd
import numpy as np
import codecs
import torch
from torch.utils import data

try:
    from subword_nmt.apply_bpe import BPE  # type: ignore
except ImportError:
    class BPE:
        def __init__(self, codes_stream, merges=-1, separator=''):
            del separator
            self.cache = {}
            self.bpe_codes = {}
            for line in codes_stream:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) != 2:
                    continue
                rank = len(self.bpe_codes)
                if merges >= 0 and rank >= merges:
                    break
                self.bpe_codes[(parts[0], parts[1])] = rank

        @staticmethod
        def _pairs(tokens):
            return {(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)}

        def _encode_word(self, word):
            cached = self.cache.get(word)
            if cached is not None:
                return cached
            tokens = tuple(word)
            while len(tokens) > 1:
                ranked_pairs = [(self.bpe_codes[pair], pair) for pair in self._pairs(tokens) if pair in self.bpe_codes]
                if not ranked_pairs:
                    break
                _, best_pair = min(ranked_pairs, key=lambda item: item[0])
                merged = []
                idx = 0
                while idx < len(tokens):
                    if idx < len(tokens) - 1 and tokens[idx] == best_pair[0] and tokens[idx + 1] == best_pair[1]:
                        merged.append(best_pair[0] + best_pair[1])
                        idx += 2
                    else:
                        merged.append(tokens[idx])
                        idx += 1
                tokens = tuple(merged)
            encoded = ' '.join(tokens)
            self.cache[word] = encoded
            return encoded

        def process_line(self, line):
            words = line.strip().split()
            if not words:
                return ''
            return ' '.join(self._encode_word(word) for word in words)

class BPE_Encoder:
    def __init__(self, vocab_dir):
        # Paths to vocab files in the ESPF folder
        self.vocab_path = os.path.join(vocab_dir, "ESPF/drug_codes_chembl_freq_1500.txt")
        self.sub_csv_path = os.path.join(vocab_dir, "ESPF/subword_units_map_chembl_freq_1500.csv")
        
        if not os.path.exists(self.vocab_path):
            raise FileNotFoundError(f"BPE codes not found at {self.vocab_path}")
            
        bpe_codes_drug = codecs.open(self.vocab_path)
        self.dbpe = BPE(bpe_codes_drug, merges=-1, separator='')
        
        sub_csv = pd.read_csv(self.sub_csv_path)
        idx2word_d = sub_csv['index'].values
        self.words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))
        self.max_d = 50

    def encode(self, smile):
        t1 = self.dbpe.process_line(smile).split()
        try:
            i1 = np.asarray([self.words2idx_d[i] for i in t1])
        except:
            i1 = np.array([0])

        l = len(i1)
        if l < self.max_d:
            i = np.pad(i1, (0, self.max_d - l), 'constant', constant_values=0)
            input_mask = ([1] * l) + ([0] * (self.max_d - l))
        else:
            i = i1[:self.max_d]
            input_mask = [1] * self.max_d
        
        return i, np.asarray(input_mask)

class DeepTTC_Dataset(data.Dataset):
    def __init__(self, response_df, expression_df, drug_smiles_df, encoder, target_gene_dim=17737):
        self.response = response_df
        self.expression = expression_df # Index should be Cell Line Name
        self.drug_smiles = drug_smiles_df.set_index('Name') # Mapping from Drug Name to SMILES
        self.encoder = encoder
        self.target_gene_dim = target_gene_dim

        # Filter to ensure intersection
        valid_cells = set(self.expression.index)
        valid_drugs = set(self.drug_smiles.index)
        
        self.indices = []
        for i, row in self.response.iterrows():
            if row['CELL_LINE_NAME'] in valid_cells and row['DRUG_NAME'] in valid_drugs:
                self.indices.append(i)
        
        print(f"Dataset initialized with {len(self.indices)} / {len(self.response)} samples.")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        row = self.response.iloc[self.indices[index]]
        cell_name = row['CELL_LINE_NAME']
        drug_name = row['DRUG_NAME']
        label = row['label']
        
        # 1. Map -1/1 labels to 0/1 for BCE loss
        label = 1.0 if label == 1 else 0.0
        
        # 2. Get Drug Encoding
        smile = self.drug_smiles.loc[drug_name, 'smiles']
        v_d, v_mask = self.encoder.encode(smile)
        
        # 3. Get Expression Vector
        v_p = self.expression.loc[cell_name].values.astype(np.float32)
        if v_p.shape[0] < self.target_gene_dim:
            v_p = np.pad(v_p, (0, self.target_gene_dim - v_p.shape[0]), mode='constant')
        elif v_p.shape[0] > self.target_gene_dim:
            v_p = v_p[:self.target_gene_dim]
        
        return torch.LongTensor(v_d), torch.FloatTensor(v_mask), torch.FloatTensor(v_p), torch.FloatTensor([label])

def load_formatted_data(data_dir):
    response_df = pd.read_csv(os.path.join(data_dir, "response_pairs.csv"))
    expression_df = pd.read_csv(os.path.join(data_dir, "transcriptomics_expression.csv"), index_col=0)
    drug_smiles_df = pd.read_csv(os.path.join(data_dir, "smiles_data.csv"))
    
    # Clean expression index if necessary (handled by script usually)
    return response_df, expression_df, drug_smiles_df
