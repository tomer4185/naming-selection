import os
import json
import re
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import statistics


def padded_hamming_distance(s1, s2, pad_char=" "):
    max_len = max(len(s1), len(s2))
    s1 = s1.ljust(max_len, pad_char)
    s2 = s2.ljust(max_len, pad_char)
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))

def count_words(var: str) -> int:
    if not var:
        return 0
    var = var[1:] if var.startswith('_') else var  # ignore leading underscore
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
        return 'single word'
    # 6) anything else
    return 'other'

def visualize_word_length_distribution(output_dir, model, df):
    from collections import Counter

    def extract_word_lengths(var):
        # Strip leading underscores
        var = var[1:] if var.startswith('_') else var

        if '_' in var:
            words = var.split('_')
        else:
            words = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?![a-z])', var)
            if not words:  # fallback
                words = [var]
        return [len(w) for w in words]

    length_counter = {
        "Original": Counter(),
        "LLM": Counter()
    }

    for _, row in df.iterrows():
        var = row["variable"]
        tag = row["type"]
        word_lengths = extract_word_lengths(var)
        length_counter[tag].update(word_lengths)

    all_lengths = sorted(set(length_counter["Original"].keys()) | set(length_counter["LLM"].keys()))
    orig_counts = [length_counter["Original"].get(l, 0) for l in all_lengths]
    llm_counts  = [length_counter["LLM"].get(l, 0) for l in all_lengths]

    bar_width = 0.4
    x = list(range(len(all_lengths)))
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar([i - bar_width/2 for i in x], orig_counts, width=bar_width, label="Original", color="tab:blue")
    ax.bar([i + bar_width/2 for i in x], llm_counts, width=bar_width, label="LLM", color="tab:orange", hatch="//")

    ax.set_xticks(x)
    ax.set_xticklabels(all_lengths)
    ax.set_xlabel("Word Length")
    ax.set_ylabel("Number of Words")
    ax.set_title("Distribution of Word Lengths in Variable Names")
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{model}word_length_distribution.png"))
    plt.close(fig)


def visualize_number_of_words_cdf(output_dir, model, df):
    fig, ax = plt.subplots(figsize=(8, 5))

    for name, group in df.groupby("type"):
        word_counts = group["words"].sort_values().values
        cdf = [i / len(word_counts) for i in range(1, len(word_counts) + 1)]
        ax.step(word_counts, cdf, where='post', label=name)

    ax.set_xlabel("Number of Words in Variable Name")
    ax.set_ylabel("Cumulative Fraction (CDF)")
    ax.set_title("CDF of Number of Words in Variable Names")
    ax.legend(title="Variable Source")
    ax.grid(True, linestyle='--', alpha=0.5)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{model}words_cdf.png"))
    plt.close(fig)

def visualize_variables_naming_distribution(output_dir, model, df):
    grouped = df.groupby(["type", "length", "words"]).size().reset_index(name="count")

    lengths = sorted(df["length"].unique())
    word_counts = sorted(df["words"].unique())
    count_lookup = {
        (row["type"], row["length"], row["words"]): row["count"]
        for _, row in grouped.iterrows()
    }

    types = ["Original", "LLM"]
    bar_width = 0.35
    n_lengths = len(lengths)

    # Use a consistent color mapping for “number of words”
    cmap = plt.get_cmap("tab10")
    color_map_words = { wc: cmap(i % 10) for i, wc in enumerate(word_counts) }

    x_positions = list(range(n_lengths))

    fig, ax = plt.subplots(figsize=(10, 6))

    bottoms = {
        "Original": [0] * n_lengths,
        "LLM":      [0] * n_lengths
    }

    for wc in word_counts:
        heights_orig = []
        heights_llm = []
        for length in lengths:
            heights_orig.append(count_lookup.get(("Original", length, wc), 0))
            heights_llm.append(count_lookup.get(("LLM",      length, wc), 0))

        orig_x = [x - bar_width/2 for x in x_positions]
        llm_x  = [x + bar_width/2 for x in x_positions]

        # Plot “Original” stacked slices for this word‐count
        ax.bar(
            orig_x,
            heights_orig,
            bar_width,
            bottom=bottoms["Original"],
            color=color_map_words[wc],
            edgecolor="white"
        )
        # Update the bottom positions for “Original” 
        bottoms["Original"] = [
            bottoms["Original"][i] + heights_orig[i]
            for i in range(n_lengths)
        ]

        # Plot LLM stacked slices for this word‐count
        ax.bar(
            llm_x,
            heights_llm,
            bar_width,
            bottom=bottoms["LLM"],
            color=color_map_words[wc],
            hatch="//",
            edgecolor="white"
        )
        # Update the bottom positions for LLM
        bottoms["LLM"] = [
            bottoms["LLM"][i] + heights_llm[i]
            for i in range(n_lengths)
        ]

    # 4f) Final formatting for the combined histogram
    ax.set_xlabel("Variables Length")
    ax.set_ylabel("Namber of Variables")
    ax.set_title("Variables Distribution by Length and Number of Words")

    # Put x‐ticks at 0..n_lengths−1, labeled by the actual lengths
    ax.set_xticks(x_positions)
    ax.set_xticklabels(lengths, rotation=45)

    # Legend for “number of words” (colors)
    handles = [
        Patch(facecolor=color_map_words[wc], label=f'{wc} word{"s" if wc != 1 else ""}')
        for wc in word_counts
    ]
    ax.legend(
        handles=handles,
        title="Number of Words",
        bbox_to_anchor=(1.05, 1),
        loc="upper left"
    )

    ax.text(
        0.01, 0.99,
        "Left bar = Original     Right bar (striped) = LLM",
        transform=ax.transAxes,
        fontsize=9,
        va="top"
    )

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{model}length_vs_words_histogram.png"))
    plt.close(fig)
    
def visualize_hamming_distribution(output_dir, model, distances):
    if not distances:
        print("No variable pairs found to compute Hamming distance.")
        return

    mean_dist = statistics.mean(distances)
    median_dist = statistics.median(distances)
    max_dist = max(distances)
    print(f"Average Hamming distance between original and LLM variable names: {mean_dist:.2f}")
    print(f"Median  Hamming distance between original and LLM variable names: {median_dist}")
    print(f"Maximum Hamming distance between original and LLM variable names: {max_dist}")

    bins = list(range(0, max_dist + 2))  # +2 so that the final integer shows fully

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(distances, bins=bins, edgecolor="black", color="skyblue", alpha=0.7)

    ax.axvline(mean_dist, color="red", linestyle="--", linewidth=1.5, label=f"Mean = {mean_dist:.2f}")
    ax.axvline(median_dist, color="green", linestyle=":", linewidth=1.5, label=f"Median = {median_dist}")

    ax.set_xlabel("Hamming Distance")
    ax.set_ylabel("Number of Variable Pairs")
    ax.set_title("Distribution of Hamming Distances\n(Original vs. LLM variable names)")

    ax.legend(loc="upper right")

    fig.tight_layout()
    out_path = os.path.join(output_dir, f"{model}hamming_distribution.png")
    fig.savefig(out_path)
    plt.close(fig)
    
    
def visualize_variables_styles_distribution(output_dir, model, df):
    style_color_map = {
        'snake_case': 'tab:green',
        'single word': 'tab:orange',
        'UPPERCASE': 'tab:blue',
        'camelCase': 'tab:red',
        'PascalCase': 'tab:purple',
        'other': 'tab:gray'
    }
    
    style_dist = df.groupby(["type", "style"]).size().unstack(fill_value=0)

    types = style_dist.index.tolist()
    n_types = len(types)

    fig, axes = plt.subplots(1, n_types, figsize=(6 * n_types, 6))
    if n_types == 1:
        axes = [axes]
    
    fig.suptitle("Style Distribution", fontsize=17, y=0.97)

    for ax, t in zip(axes, types):
        counts = style_dist.loc[t]
        # Exclude zero‐count styles
        non_zero = counts[counts > 0]
        labels = non_zero.index.tolist()
        sizes = non_zero.values.tolist()
        # Use the consistent color mapping
        pie_colors = [style_color_map[label] for label in labels]
        
        def autopct_filter(pct):
            return f'{pct:.1f}%' if pct > 1 else ''
        
        wedges, _, _ = ax.pie(
            sizes,
            labels=None,
            autopct=autopct_filter,
            startangle=90,
            colors=pie_colors
        )

        # Add legend for style labels
        ax.legend(
            wedges,
            labels,
            title="Style",
            loc="center left",
            bbox_to_anchor=(1, 0.5)
        )
        ax.set_title(t, fontsize=13)

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{model}style_distribution.png")
    fig.savefig(out_path)
    plt.close(fig)

def analyze(input_json: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    with open(input_json, 'r') as f:
        data = json.load(f)
    
    if '_' in input_json:
        # Assuming input_json is in the format "path/model_....json"
        model = os.path.basename(input_json).split('_')[0] + '_'
    else:
        model = ''

    records = []
    distances = []  # to collect Hamming distances

    for repo, snippets in data.items():
        for snippet in snippets:
            orig_vars = snippet.get('variables') or []
            llm_vars  = snippet.get('llm_variables') or []
            if not llm_vars:
                continue

            # compute Hamming distances for each pair of original vs. llm variable
            for orig_v, llm_v in zip(orig_vars, llm_vars):
                distances.append(padded_hamming_distance(orig_v, llm_v))

            for tag, var_list in (('Original', orig_vars), ('LLM', llm_vars)):
                for v in var_list:
                    records.append({
                        'repo': repo,
                        'type': tag,
                        'variable': v,
                        'length': len(v.replace("_", "")), # ignore underscores in length
                        'words': count_words(v),
                        'style': detect_style(v)
                    })

    df = pd.DataFrame(records)
        
    # Ensure we have data before plotting
    if df.empty:
        print("No variable data to plot.")
        return

    visualize_variables_naming_distribution(output_dir, model, df)
    visualize_number_of_words_cdf(output_dir, model, df)
    visualize_word_length_distribution(output_dir, model, df)
    visualize_variables_styles_distribution(output_dir, model, df)
    visualize_hamming_distribution(output_dir, model, distances)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and plot variable naming statistics")
    parser.add_argument("-i", "--input", required=True, help="Path to input JSON file")
    parser.add_argument("-o", "--output_dir", default="../analysis_results/", help="Directory to save plots")
    args = parser.parse_args()
    analyze(args.input, args.output_dir)
