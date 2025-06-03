#!/usr/bin/env python3
"""Compute lexical‑ and context‑level similarity between human and LLM identifiers.

Usage:
    python evaluate_naming.py llama_repos_variables_snippets_proccesed.json
"""
import json, argparse, pathlib, statistics, itertools, warnings
from typing import List, Tuple
from tqdm import tqdm

import torch, torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel

# ────────────────────────────────────────────────────────────────────────────────
# Embedding helpers
# ────────────────────────────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=0
@torch.inference_mode()
def embed(texts: List[str], model_id: str, device: str = CUDA_VISIBLE_DEVICES) -> torch.Tensor:
    """Return ℓ2‑normalised CLS embeddings (n × d)."""
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModel.from_pretrained(model_id).to(device).eval()
    batch = tok(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    out = mdl(**batch).last_hidden_state[:, 0]          # CLS token
    return F.normalize(out, p=2, dim=1).cpu()

NAME_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # 384‑d, NL focus
CTX_MODEL  = "microsoft/unixcoder-base"                 # 768‑d, code focus

# ────────────────────────────────────────────────────────────────────────────────
# Core evaluation
# ────────────────────────────────────────────────────────────────────────────────

def pairwise_scores(h_names: List[str], l_names: List[str], h_ctx_snips: List[str], l_ctx_snips: List[str]) -> List[float]:
    """Return cosine‑similarity per aligned identifier (average of two models)."""
    # 1) Name‑only embedding
    emb_h_n  = embed(h_names, NAME_MODEL)
    emb_l_n  = embed(l_names, NAME_MODEL)
    sim_name = (emb_h_n * emb_l_n).sum(1)

    # 2) Name‑in‑context embedding – we concatenate name + snippet for richer signal
    ctx_h = [f"{n} | {c}" for n, c in zip(h_names, h_ctx_snips)]
    ctx_l = [f"{n} | {c}" for n, c in zip(l_names, l_ctx_snips)]

    emb_h_c  = embed(ctx_h, CTX_MODEL)
    emb_l_c  = embed(ctx_l, CTX_MODEL)
    sim_ctx  = (emb_h_c * emb_l_c).sum(1)

    return ((sim_name + sim_ctx) / 2).tolist()

# ────────────────────────────────────────────────────────────────────────────────
# Utility to extract aligned name pairs from JSON node
# ────────────────────────────────────────────────────────────────────────────────

def extract_pairs(node: dict, baseline: bool) -> Tuple[List[str], List[str], List[str]]:
    """Return (human_names, llm_names, context_snippets). Lengths are equal."""
    h_names = node["variables"]
    l_names = node["llm_variables"]

    if not h_names or not l_names:
        raise ValueError(f"Empty variable lists in {node['file_path']}")


    # Align by position. If lengths differ, truncate to shorter length and warn.
    k = min(len(h_names), len(l_names))
    if len(h_names) != len(l_names):
        warnings.warn(f"Variable count mismatch in {node['file_path']} – truncating to {k}.")
    h_names = h_names[:k]
    l_names = l_names[:k]

    # Use 3‑line context around each var occurrence if available; fallback to full fn.
    if baseline:
        # Anonymized code is used for baseline evaluation
         l_context = node.get("anonymized_code", "")
         # k-length list of var_i for i in 1..k to replace llm_names
         l_names = [f"var_{i+1}" for i in range(k)]
    else:
        l_context = node.get("llm_code", "")  # LLM-generated code snippet
    h_context = node.get("code", "")  # simplest: full snippet
    l_ctx_list = [l_context] * k
    h_ctx_list = [h_context] * k
    return h_names, l_names, h_ctx_list, l_ctx_list

# ────────────────────────────────────────────────────────────────────────────────
# Main CLI
# ────────────────────────────────────────────────────────────────────────────────

def main(json_path: pathlib.Path, baseline: bool = False):
    data = json.loads(json_path.read_text())

    all_scores = []
    for repo, entries in data.items():
        for node in tqdm(entries, desc=repo):
            try:
                human_names, llm_names, h_ctx, l_ctx = extract_pairs(node, baseline)
                scores = pairwise_scores(human_names, llm_names, h_ctx, l_ctx)
                all_scores.extend(scores)
            except Exception as e:
                warnings.warn(f"Error processing {node['file_path']}: {e}")
                continue

    mean_sim = statistics.mean(all_scores)
    median_sim = statistics.median(all_scores)
    print(f"• N pairs      : {len(all_scores):,}")
    print(f"• Mean cosine  : {mean_sim:.3f}")
    print(f"• Median cosine: {median_sim:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_file", type=pathlib.Path)
    parser.add_argument("--baseline", type=bool, default=False)
    args = parser.parse_args()
    main(args.json_file, args.baseline)