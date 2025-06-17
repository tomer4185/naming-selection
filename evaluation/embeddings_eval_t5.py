import argparse
import json
import pathlib
import re
import warnings
import torch, torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, T5EncoderModel
import statistics

MODEL_ID = "Salesforce/codet5p-220m"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_DIR = "../hf_cache"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    cache_dir=CACHE_DIR,
    trust_remote_code=True,
)
model = T5EncoderModel.from_pretrained(
    MODEL_ID,
    cache_dir=CACHE_DIR,
    torch_dtype=torch.float32,
    device_map="auto",
    trust_remote_code=True
).to(DEVICE).eval()

@torch.inference_mode()
def embed(texts):
    tok = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    tok = {k: v.to(model.device) for k, v in tok.items()}
    out = model(**tok).last_hidden_state.mean(dim=1)
    return F.normalize(out, p=2, dim=1).cpu()

def extract_pairs(entry, baseline):
    h_names = entry["variables"]
    if baseline == "basic":
        b_names = [f"var_{i+1}" for i in range(len(h_names))]
    elif baseline == "gibberish":
        b_names = entry["gibberish_variables"]
    elif baseline == "random":
        b_names = entry["random_variables"]
    else:
        raise ValueError(f"Unknown baseline type: {baseline}")
        
    llm_names = entry["llm_variables"]
    return h_names, b_names, llm_names

def score_pairs(names1, names2):
    return (embed(names1) * embed(names2)).sum(1).tolist()

def evaluate(json_path, baseline="basic"):
    data = json.loads(pathlib.Path(json_path).read_text())
    baseline_scores = []
    llm_scores = []

    for repo, entries in data.items():
        for entry in tqdm(entries, desc=repo):  # batch size 1
            h, b, llm = extract_pairs(entry, baseline)
            if llm is None:
                continue
            llm_scores.extend(score_pairs(h, llm))
            baseline_scores.extend(score_pairs(h, b))
        
    # LLM
    print(f"----- LLM -----")
    print(f"• N pairs: {len(llm_scores)}")
    print(f"• Mean   : {statistics.mean(llm_scores):.4f}")
    print(f"• Median : {statistics.median(llm_scores):.4f}")
    
    # Baseline
    print(f"----- Baseline -----")
    print(f"• N pairs: {len(baseline_scores)}")
    print(f"• Mean   : {statistics.mean(baseline_scores):.4f}")
    print(f"• Median : {statistics.median(baseline_scores):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--json_file", type=pathlib.Path)
    parser.add_argument("--baseline_type", type=str, default="gibberish", choices=["basic", "gibberish", "random"])
    parser.add_argument('--cache', required=False, default="../hf_cache", type=str)
    args = parser.parse_args()
    evaluate(args.json_file, args.baseline_type)
