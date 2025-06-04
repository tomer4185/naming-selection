import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import re


_models: dict = {
    "qwen": "Qwen/Qwen3-8B",
    "llama": "meta-llama/Llama-3.1-8B-Instruct"
}

def _parse_json(raw: str):
    raw = "{'full_code': " + raw

    depth = 0
    end = None
    for i, ch in enumerate(raw):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                end = i
                break
    if end is None:
        return None

    snippet = raw[:end+1]
    snippet = snippet.replace("'", '\"')
    
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        return None
    


def _regex_extract_variables(raw: str):
    """
    Look for the first occurrence of
      'variable': ['a', ...]  or  'variables': ["a", ...]
    and return the list of items inside the brackets.
    """
    # Match either 'variable' or 'variables', then colon, then [ ... ]
    pattern = r"""['"]variable(?:s)?['"]\s*:\s*\[\s*([^\]]*?)\s*\]"""
    m = re.search(pattern, raw)
    if not m:
        return []
    inside = m.group(1)
    # Now pull out each quoted item
    return re.findall(r"""['"]([^'"]+)['"]""", inside)

class SnippetDataset(Dataset):
    def __init__(self, data: dict):
        self.data = data
        self.index = []  # list of (repo, idx)
        for repo, snippets in data.items():
            for i in range(len(snippets)):
                self.index.append((repo, i))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        repo, i = self.index[idx]
        return repo, i, self.data[repo][i]

class VariableRenamer:
    PROMPT_TEMPLATE = """
You are a Python code assistant helping with completing variable naming for given code snippet. Given a function with anonymized variables (var_1, var_2, etc.), replace each var_i with a meaningful variable name, matching the full context.
You should return **ONLY** a JSON text with two keys: "full_code" and "variables". Beside the json, your response shouldn't contain any explanations, markdown, or extra text.

For example:

Input
```python
def add(var_1, var_2):
    var_3 = var_1 + var_2
    return var_3
```
Output:
{"full_code": "def add(a, b):\n    result = a + b\n    return result", "variables": ["a", "b", "result"]}

Now process:
```python
"""

    def __init__(self, model: str, cache_dir: str):
        model_name = _models.get(model, model)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map='auto',
            torch_dtype=torch.float16,
            cache_dir=cache_dir,
            low_cpu_mem_usage=True
        )

        self.pipe = pipeline(
            'text-generation',
            model=self.model,
            tokenizer=self.tokenizer,
            trust_remote_code=True,
            device_map='auto',
            max_new_tokens=256,
            return_full_text=False  # to avoid repeating the input text
        )


    def rename(self, code: str, expected_count: int, max_retries: int=2):
        """
        Rename variables in the given code using the LLM.
        @param code: The anonymized code to process.
        @param expected_count: The expected number of variables to be renamed.
        @param max_retries: Number of retries if the output does not match expectations.
        @return: Tuple of (new_code, new_variables), None if failed.
        """
        prompt = self.PROMPT_TEMPLATE  + code + "\n```\n\Output:\n{'full_code': "  # Engage the LLM to generate a JSON output
        print(f"Code: {code}")
        for i in range(max_retries + 1):
            print(f"Attempt {i + 1}")
            result = self.pipe(prompt, num_return_sequences=1)[0]['generated_text']
            parsed = _parse_json(result)
            fallback_vars = _regex_extract_variables(result)
            if parsed and 'full_code' in parsed and 'variables' in parsed:  # JSON output extrection
                if isinstance(parsed['variables'], list) and len(parsed['variables']) == expected_count:
                    return parsed['full_code'], parsed['variables']
                else: 
                    print(f"Expected {expected_count} variables, but got variablses = {parsed['variables']}. Retrying...")
            elif len(fallback_vars) == expected_count:  # Fallback to regex extraction
                print(f"Fallback extraction succeeded: {fallback_vars}")
                # substitute var_1, var_2, … in the anonymized code
                full_code = code
                for idx, name in enumerate(fallback_vars, start=1):
                    full_code = re.sub(rf"\bvar_{idx}\b", name, full_code)
                return full_code, fallback_vars
            else:
                suffix = "" if i < max_retries else " Retrying..."
                print(f"LLM output did not match expected format.{suffix}")
        return None, None


if __name__ == '__main__':
    import os
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True, help='Input JSON file', type=str)
    parser.add_argument('--output', '-o', required=False, help='Output JSON file', type=str)
    parser.add_argument('--model', '-m',  default="qwen", type=str, required=False, choices=["qwen", "llama"])
    parser.add_argument('--cache', required=False, default="../hf_cache", type=str)
    parser.add_argument('--batch-size', required=False, type=int, default=1)
    args = parser.parse_args()
    
    output = args.output if args.output else 'proccesed.json'
    output = "../data/" + args.model + '_' + output
    # Load data, using output as checkpoint if exists, in case of resuming failed attempt
    if os.path.exists(output):
        with open(output, 'r') as f:
            data = json.load(f)
    else:
        with open(args.input, 'r') as f:
            data = json.load(f)

    dataset = SnippetDataset(data)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda x: x[0]
    )

    os.makedirs(args.cache, exist_ok=True)
    renamer = VariableRenamer(args.model, args.cache)
    for repo, idx, snippet in loader:
        # Skip already-processed snippets
        if 'llm_code' in snippet and 'llm_variables' in snippet:
            continue

        code = snippet['anonymized_code']
        expected = len(snippet['variables'])
        new_code, new_vars = renamer.rename(code, expected_count=expected)
        snippet['llm_code'] = new_code
        snippet['llm_variables'] = new_vars
        # Checkpoint after each snippet
        with open(output, 'w') as f:
            json.dump(data, f, indent=2)
    print(f"Saved output to {output}")
