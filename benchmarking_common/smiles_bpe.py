import codecs
import os
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from benchmarking_common.splits import normalize_identifier


@dataclass
class EncodedSmilesTable:
    token_ids: pd.DataFrame
    attention_mask: pd.DataFrame


def _get_pairs(tokens: Tuple[str, ...]) -> set[Tuple[str, str]]:
    return {(tokens[idx], tokens[idx + 1]) for idx in range(len(tokens) - 1)}


class SimpleBPE:
    def __init__(self, codes_stream, merges: int = -1, separator: str = ""):
        del separator
        self.cache: Dict[str, str] = {}
        self.bpe_codes: Dict[Tuple[str, str], int] = {}
        for line in codes_stream:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            rank = len(self.bpe_codes)
            if merges >= 0 and rank >= merges:
                break
            self.bpe_codes[(parts[0], parts[1])] = rank

    def _encode_word(self, word: str) -> str:
        cached = self.cache.get(word)
        if cached is not None:
            return cached

        tokens = tuple(word)
        if not tokens:
            self.cache[word] = ""
            return ""

        while len(tokens) > 1:
            pairs = _get_pairs(tokens)
            ranked_pairs = [(self.bpe_codes[pair], pair) for pair in pairs if pair in self.bpe_codes]
            if not ranked_pairs:
                break
            _, best_pair = min(ranked_pairs, key=lambda item: item[0])
            first, second = best_pair
            merged_tokens = []
            idx = 0
            while idx < len(tokens):
                if idx < len(tokens) - 1 and tokens[idx] == first and tokens[idx + 1] == second:
                    merged_tokens.append(first + second)
                    idx += 2
                else:
                    merged_tokens.append(tokens[idx])
                    idx += 1
            tokens = tuple(merged_tokens)

        encoded = " ".join(tokens)
        self.cache[word] = encoded
        return encoded

    def process_line(self, line: str) -> str:
        words = line.strip().split()
        if not words:
            return ""
        return " ".join(self._encode_word(word) for word in words)


class DeepTTCBPEEncoder:
    def __init__(self, deepttc_root: str):
        vocab_dir = os.path.join(deepttc_root, "ESPF")
        vocab_path = os.path.join(vocab_dir, "drug_codes_chembl_freq_1500.txt")
        sub_csv_path = os.path.join(vocab_dir, "subword_units_map_chembl_freq_1500.csv")

        bpe_codes_drug = codecs.open(vocab_path)
        self.dbpe = SimpleBPE(bpe_codes_drug, merges=-1, separator="")

        sub_csv = pd.read_csv(sub_csv_path)
        idx2word_d = sub_csv["index"].values
        self.words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))
        self.max_len = 50

    def encode(self, smile: str) -> Tuple[np.ndarray, np.ndarray]:
        tokens = self.dbpe.process_line(smile).split()
        try:
            token_ids = np.asarray([self.words2idx_d[token] for token in tokens], dtype=np.int64)
        except Exception:
            token_ids = np.array([0], dtype=np.int64)

        length = len(token_ids)
        if length < self.max_len:
            padded = np.pad(token_ids, (0, self.max_len - length), "constant", constant_values=0)
            mask = np.asarray(([1] * length) + ([0] * (self.max_len - length)), dtype=np.int64)
        else:
            padded = token_ids[: self.max_len]
            mask = np.ones(self.max_len, dtype=np.int64)
        return padded, mask


def load_smiles_frame(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    id_col = "pubchem_id" if "pubchem_id" in df.columns else df.columns[0]
    smiles_col = "smiles" if "smiles" in df.columns else df.columns[1]
    out = df[[id_col, smiles_col]].rename(columns={id_col: "drug_id", smiles_col: "smiles"})
    out["drug_id"] = out["drug_id"].map(normalize_identifier)
    out["smiles"] = out["smiles"].astype(str)
    out = out.drop_duplicates("drug_id", keep="first").set_index("drug_id")
    return out


def encode_smiles_table(
    smiles_df: pd.DataFrame,
    encoder: DeepTTCBPEEncoder,
    drug_ids: Iterable[str] | None = None,
) -> EncodedSmilesTable:
    if drug_ids is not None:
        drug_ids = [normalize_identifier(drug_id) for drug_id in drug_ids]
        smiles_df = smiles_df.loc[drug_ids]

    token_rows: Dict[str, np.ndarray] = {}
    mask_rows: Dict[str, np.ndarray] = {}
    for drug_id, row in smiles_df.iterrows():
        token_ids, attention_mask = encoder.encode(row["smiles"])
        token_rows[str(drug_id)] = token_ids
        mask_rows[str(drug_id)] = attention_mask

    token_df = pd.DataFrame.from_dict(token_rows, orient="index")
    mask_df = pd.DataFrame.from_dict(mask_rows, orient="index")
    token_df.index.name = "drug_id"
    mask_df.index.name = "drug_id"
    return EncodedSmilesTable(token_ids=token_df, attention_mask=mask_df)
