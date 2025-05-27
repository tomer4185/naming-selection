import os
import json
import re
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def count_words(var: str) -> int:
    if '_' in var:
        return len(var.split('_'))
    parts = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?![a-z])', var)
    return len(parts) if parts else 1

def detect_style(var: str) -> str:
    # 1) All‐caps (constants), e.g. A_NUMBER or MAX_VALUE
    if var.isupper():
        return 'UPPERCASE'
    # 2) snake_case (must have at least one underscore, but not all-caps)
    if '_' in var:
        return 'snake_case'
    # 3) camelCase (lower initial, then uppercase interior)
    if re.match(r'^[a-z]+([A-Z][a-z]+)+$', var):
        return 'camelCase'
    # 4) PascalCase (upper initial, then uppercase interior)
    if re.match(r'^[A-Z][a-z]+([A-Z][a-z]+)*$', var):
        return 'PascalCase'
    # 5) single_word (all lowercase, no underscores)
    if var.islower():
        return 'single_word'
    # 6) anything else
    return 'other'


def analyze(input_json: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    with open(input_json, 'r') as f:
        data = json.load(f)
    
    if '_' in input_json:
        # Assuming input_json is in the format "model_....json"
        model = input_json.split('_')[0] + '_'
    else:
        model = ''

    records = []
    for repo, snippets in data.items():
        for snippet in snippets:
            orig_vars = snippet.get('variables') or []
            llm_vars  = snippet.get('llm_variables') or []
            if not llm_vars:
                continue
            for tag, var_list in (('original', orig_vars), ('llm', llm_vars)):
                for v in var_list:
                    records.append({
                        'repo': repo,
                        'type': tag,
                        'variable': v,
                        'length': len(v),
                        'words': count_words(v),
                        'style': detect_style(v)
                    })

    df = pd.DataFrame(records)
    # ——— Summaries ———
    summary = df.groupby('type').agg(
        mean_length=('length','mean'),
        mean_words =('words','mean'),
        count      =('variable','count')
    ).reset_index()

    style_dist = df.groupby(['type','style']).size().unstack(fill_value=0)
    # map each type to a distinct color
    color_map = {
        'original': 'tab:blue',
        'llm'     : 'tab:orange',
    }
    colors = [color_map[t] for t in summary['type']]

    # ——— Plot: Mean Length ———
    fig, ax = plt.subplots()
    ax.bar(summary['type'], summary['mean_length'], color=colors)
    ax.set_title('Mean Variable Name Length')
    ax.set_xlabel('Type')
    ax.set_ylabel('Length')
    fig.savefig(os.path.join(output_dir,f'{model}mean_length.png'))
    plt.close(fig)

    # ——— Plot: Mean Words ———
    fig, ax = plt.subplots()
    ax.bar(summary['type'], summary['mean_words'], color=colors)
    ax.set_title('Mean # of Words per Name')
    ax.set_xlabel('Type')
    ax.set_ylabel('Words')
    fig.savefig(os.path.join(output_dir,f'{model}mean_words.png'))
    plt.close(fig)

    # ——— Plot: Style Distribution ———
    fig, ax = plt.subplots()
    style_dist.plot(kind='bar', stacked=True, ax=ax)
    ax.set_title('Naming Style Distribution')
    ax.set_xlabel('Type')
    ax.set_ylabel('Count')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir,f'{model}style_distribution.png'))
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and plot variable naming statistics")
    parser.add_argument("-i", "--input", required=True, help="Path to input JSON file")
    parser.add_argument("-o", "--output_dir", default="analysis_results", help="Directory to save plots")
    args = parser.parse_args()
    analyze(args.input, args.output_dir)
