import os
import ast
import json
import argparse
import logging
import re
import random
from typing import Dict, List, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('repo_variable_extraction.log')
    ]
)
logger = logging.getLogger(__name__)


def extract_variables_from_code(code: str) -> List[str]:
    """
    Extract variable names from Python code string using AST
    """
    try:
        tree = ast.parse(code)
        visitor = VariableVisitor()
        visitor.visit(tree)
        return list(visitor.variables)
    except Exception as e:
        logger.error(f"Error processing code: {e}")
        return []


class VariableVisitor(ast.NodeVisitor):
    def __init__(self):
        # a list to hold names in the order we see them…
        self.variables = []
        # …and a set to dedupe
        self._seen = set()

    def visit_Name(self, node):
        # only record on assignment (Store context):
        if isinstance(node.ctx, ast.Store) and node.id not in self._seen:
            self._seen.add(node.id)
            self.variables.append(node.id)
        self.generic_visit(node)

    def visit_arg(self, node):
        # record function arguments, too
        if node.arg not in self._seen:
            self._seen.add(node.arg)
            self.variables.append(node.arg)
        self.generic_visit(node)



def replace_variables_in_code(code: str, variables: List[str]) -> str:
    modified = code
    var_map = {v: f"var_{i+1}" for i, v in enumerate(variables)}
    for name, repl in var_map.items():
        pattern = r'\b' + re.escape(name) + r'\b'
        modified = re.sub(pattern, repl, modified)
    return modified


def get_random_lines_snippet(lines: List[str], n: int) -> str:
    # Filter out import statements
    non_import_idxs = [i for i, ln in enumerate(lines) if not re.match(r'^\s*(import|from)\b', ln)]
    if len(non_import_idxs) <= n:
        start = 0
    else:
        start = random.choice(non_import_idxs)
        if start + n > len(lines):
            start = len(lines) - n
    return ''.join(lines[start:start+n])


def get_random_function_snippet(lines: List[str], max_len: int = 10) -> str:
    """
    Extract a random function snippet, but limit to max_len lines
    """
    code = ''.join(lines)
    try:
        tree = ast.parse(code)
        funcs = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        if not funcs:
            return ''
        fn = random.choice(funcs)
        start = fn.lineno - 1
        end = getattr(fn, 'end_lineno', fn.lineno)
        # Limit snippet to max_len lines
        snippet_end = min(end, start + max_len)
        return ''.join(lines[start:snippet_end])
    except Exception as e:
        logger.error(f"Error extracting function snippet: {e}")
        return ''


def process_repo(
    repo_dir: str,
    snippet_method: str,
    snippet_length: int
) -> List[Dict[str, Any]]:
    results = []
    for root, _, files in os.walk(repo_dir):
        for file in files:
            if not file.endswith('.py'):
                continue
            path = os.path.join(root, file)
            rel = os.path.relpath(path, repo_dir)
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    all_lines = f.readlines()
                total = len(all_lines)

                # Select code snippet
                if snippet_method == 'first':
                    snippet = ''.join(all_lines[:snippet_length])
                elif snippet_method == 'random_lines':
                    snippet = get_random_lines_snippet(all_lines, snippet_length)
                elif snippet_method == 'random_function':
                    snippet = get_random_function_snippet(all_lines, snippet_length)
                else:
                    snippet = ''.join(all_lines[:snippet_length])

                if not snippet.strip():
                    logger.info(f"Skipped {rel} - empty snippet")
                    continue

                vars = extract_variables_from_code(snippet)
                if not vars:
                    logger.info(f"Skipped {rel} - no variables found in snippet")
                    continue

                anon = replace_variables_in_code(snippet, vars)
                info = {
                    'file_path': rel,
                    'code': snippet,
                    'variables': vars,
                    'anonymized_code': anon,
                    'lines_processed': len(snippet.splitlines()),
                    'total_lines': total
                }
                results.append(info)
                logger.info(f"Processed {rel} - {len(vars)} vars in snippet")
            except Exception as e:
                logger.error(f"Error processing {path}: {e}")
    return results


def main():
    p = argparse.ArgumentParser(description='Extract variable snippets from Python repos')
    p.add_argument('-d', '--dir', help='Base directory of repos')
    p.add_argument('-r', '--repos', nargs='+', help='Specific repo dirs')
    p.add_argument('--snippet-method', choices=['first', 'random_lines', 'random_function'], default='random_function',
                   help='Method to select code snippet')
    p.add_argument('--snippet-length', type=int, default=5,
                   help='Number of lines for snippet (ignored for random_function)')
    p.add_argument('-o', '--output', default='repos_variables_snippets.json', help='Output JSON file')
    args = p.parse_args()

    repos = []
    if args.dir:
        repos = [os.path.join(args.dir, d) for d in os.listdir(args.dir) if os.path.isdir(os.path.join(args.dir, d))]
    else:
        repos = args.repos or []

    all_data = {}
    for repo_path in repos:
        name = os.path.basename(repo_path)
        data = process_repo(repo_path, args.snippet_method, args.snippet_length)
        if data:
            all_data[name] = data

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=2)
    logger.info(f"Results saved to {args.output}")

if __name__ == '__main__':
    main()
