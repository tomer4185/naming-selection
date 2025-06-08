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


def is_valid_python_code(code: str) -> bool:
    """Check if code string is valid Python syntax"""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def extract_variables_from_code(code: str) -> List[str]:
    """Extract variable names from code, with better error handling"""
    if not code.strip():
        return []

    try:
        tree = ast.parse(code)
        visitor = VariableVisitor()
        visitor.visit(tree)
        return list(visitor.variables)
    except SyntaxError as e:
        logger.debug(f"Syntax error in code snippet: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error processing code: {e}")
        return []


class VariableVisitor(ast.NodeVisitor):
    def __init__(self):
        self.variables = []
        self._seen = set()

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Store) and node.id not in self._seen:
            self._seen.add(node.id)
            self.variables.append(node.id)
        self.generic_visit(node)

    def visit_arg(self, node):
        if node.arg not in self._seen:
            self._seen.add(node.arg)
            self.variables.append(node.arg)
        self.generic_visit(node)


def replace_variables_in_code(code: str, variables: List[str]) -> str:
    """Replace variable names with generic names"""
    modified = code
    var_map = {v: f"var_{i + 1}" for i, v in enumerate(variables)}
    for name, repl in var_map.items():
        pattern = r'\b' + re.escape(name) + r'\b'
        modified = re.sub(pattern, repl, modified)
    return modified


def normalize_indentation(lines: List[str]) -> List[str]:
    """Remove common leading whitespace from all lines"""
    if not lines:
        return lines

    # Find minimum indentation (ignoring empty lines)
    min_indent = float('inf')
    for line in lines:
        if line.strip():  # Skip empty lines
            indent = len(line) - len(line.lstrip())
            min_indent = min(min_indent, indent)

    if min_indent == float('inf'):
        return lines

    # Remove common indentation
    normalized = []
    for line in lines:
        if line.strip():
            normalized.append(line[min_indent:])
        else:
            normalized.append(line)

    return normalized


def get_random_lines_snippet(lines: List[str], n: int) -> str:
    """Get a random snippet of n lines, avoiding imports and ensuring valid syntax"""
    non_import_idxs = [i for i, ln in enumerate(lines)
                       if not re.match(r'^\s*(import|from)\b', ln.strip()) and ln.strip()]

    if len(non_import_idxs) < n:
        # If not enough non-import lines, try from the beginning
        snippet_lines = lines[:n]
    else:
        # Try multiple random starting points to find valid syntax
        max_attempts = 10
        for _ in range(max_attempts):
            start_idx = random.choice(non_import_idxs)
            end_idx = min(start_idx + n, len(lines))
            snippet_lines = lines[start_idx:end_idx]

            # Normalize indentation and check if valid
            normalized_lines = normalize_indentation(snippet_lines)
            snippet = ''.join(normalized_lines)

            if is_valid_python_code(snippet):
                return snippet

        # Fallback: try from beginning of file
        snippet_lines = lines[:n]

    normalized_lines = normalize_indentation(snippet_lines)
    return ''.join(normalized_lines)


def extract_complete_function(lines: List[str], func_node: ast.FunctionDef) -> str:
    """Extract a complete function definition"""
    start_line = func_node.lineno - 1  # Convert to 0-based indexing

    # Find the actual end of the function by looking for the next top-level definition
    # or end of file
    end_line = len(lines)
    if hasattr(func_node, 'end_lineno') and func_node.end_lineno:
        end_line = func_node.end_lineno
    else:
        # Fallback: find next function/class at same indentation level
        func_indent = len(lines[start_line]) - len(lines[start_line].lstrip())
        for i in range(start_line + 1, len(lines)):
            line = lines[i]
            if line.strip() and not line.startswith(' ' * (func_indent + 1)):
                # Found line at same or less indentation
                if re.match(r'^\s*(def|class|if __name__|import|from)\b', line):
                    end_line = i
                    break

    func_lines = lines[start_line:end_line]
    return ''.join(func_lines)


def get_up_to_n_function_snippets(lines: List[str], n_funcs: int = 10, max_len: int = 10) -> List[str]:
    """Extract function snippets with better handling of complete functions"""
    code = ''.join(lines)
    try:
        tree = ast.parse(code)
        funcs = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        if not funcs:
            return []

        selected_funcs = random.sample(funcs, min(n_funcs, len(funcs)))
        snippets = []

        for fn in selected_funcs:
            try:
                complete_func = extract_complete_function(lines, fn)

                # If function is too long, truncate but try to keep it valid
                if len(complete_func.splitlines()) > max_len:
                    func_lines = complete_func.splitlines(keepends=True)
                    truncated = ''.join(func_lines[:max_len])

                    # Add a pass statement if we truncated in the middle of a function
                    if not is_valid_python_code(truncated):
                        # Try to find a good breaking point
                        for i in range(max_len - 1, 0, -1):
                            candidate = ''.join(func_lines[:i])
                            if is_valid_python_code(candidate):
                                truncated = candidate
                                break
                        else:
                            # Last resort: add pass to make it valid
                            truncated += "    pass\n"

                    snippets.append(truncated)
                else:
                    snippets.append(complete_func)

            except Exception as e:
                logger.debug(f"Error extracting function {fn.name}: {e}")
                continue

        return snippets

    except Exception as e:
        logger.error(f"Error extracting function snippets: {e}")
        return []


def process_repo(repo_dir: str, snippet_method: str, snippet_length: int) -> List[Dict[str, Any]]:
    """Process repository with improved error handling"""
    results = []
    processed_files = 0
    skipped_files = 0

    for root, _, files in os.walk(repo_dir):
        for file in files:
            if not file.endswith('.py'):
                continue

            path = os.path.join(root, file)
            rel = os.path.relpath(path, repo_dir)

            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    all_lines = f.readlines()

                if not all_lines:
                    continue

                total = len(all_lines)

                # Generate snippets based on method
                snippets = []
                if snippet_method == 'random_function':
                    snippets = get_up_to_n_function_snippets(all_lines, n_funcs=10, max_len=snippet_length)
                elif snippet_method == 'random_lines':
                    snippet = get_random_lines_snippet(all_lines, snippet_length)
                    if snippet.strip():
                        snippets = [snippet]
                else:  # 'first' or fallback
                    first_lines = all_lines[:snippet_length]
                    snippet = ''.join(first_lines)
                    if snippet.strip():
                        snippets = [snippet]

                if not snippets:
                    logger.debug(f"No valid snippets found in {rel}")
                    skipped_files += 1
                    continue

                # Process each snippet
                file_had_valid_snippet = False
                for snippet in snippets:
                    if not snippet.strip():
                        continue

                    vars = extract_variables_from_code(snippet)
                    if not vars:
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
                    file_had_valid_snippet = True

                if file_had_valid_snippet:
                    processed_files += 1
                    logger.debug(f"Processed {rel}")
                else:
                    skipped_files += 1

            except Exception as e:
                logger.error(f"Error processing {path}: {e}")
                skipped_files += 1

    logger.info(f"Processed {processed_files} files successfully, skipped {skipped_files} files")
    return results


def main():
    p = argparse.ArgumentParser(description='Extract variable snippets from Python repos')
    p.add_argument('-d', '--dir', help='Base directory of repos')
    p.add_argument('-r', '--repos', nargs='+', help='Specific repo dirs')
    p.add_argument('--snippet-method', choices=['first', 'random_lines', 'random_function'], default='random_function',
                   help='Method to select code snippet')
    p.add_argument('--snippet-length', type=int, default=10,
                   help='Number of lines for snippet or max lines per function')
    p.add_argument('-o', '--output', default='../data/anonymized_variables.json', help='Output JSON file')
    p.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = p.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    repos = []
    if args.dir:
        if not os.path.exists(args.dir):
            logger.error(f"Directory {args.dir} does not exist")
            return
        repos = [os.path.join(args.dir, d) for d in os.listdir(args.dir)
                 if os.path.isdir(os.path.join(args.dir, d))]
    else:
        repos = args.repos or []

    if not repos:
        logger.error("No repositories specified or found")
        return

    logger.info(f"Processing {len(repos)} repositories")

    all_data = {}
    total_snippets = 0

    for repo_path in repos:
        if not os.path.exists(repo_path):
            logger.warning(f"Repository path {repo_path} does not exist, skipping")
            continue

        name = os.path.basename(repo_path)
        logger.info(f"Processing repository: {name}")

        data = process_repo(repo_path, args.snippet_method, args.snippet_length)
        if data:
            all_data[name] = data
            total_snippets += len(data)
            logger.info(f"Extracted {len(data)} snippets from {name}")
        else:
            logger.warning(f"No valid snippets found in {name}")

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=2)

    logger.info(f"Results saved to {args.output}")
    logger.info(f"Total snippets extracted: {total_snippets}")


if __name__ == '__main__':
    main()

# import os
# import ast
# import json
# import argparse
# import logging
# import re
# import random
# from typing import Dict, List, Any
#
# # Set up logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.StreamHandler(),
#         logging.FileHandler('repo_variable_extraction.log')
#     ]
# )
# logger = logging.getLogger(__name__)
#
#
# def extract_variables_from_code(code: str) -> List[str]:
#     try:
#         tree = ast.parse(code)
#         visitor = VariableVisitor()
#         visitor.visit(tree)
#         return list(visitor.variables)
#     except Exception as e:
#         logger.error(f"Error processing code: {e}")
#         return []
#
#
# class VariableVisitor(ast.NodeVisitor):
#     def __init__(self):
#         self.variables = []
#         self._seen = set()
#
#     def visit_Name(self, node):
#         if isinstance(node.ctx, ast.Store) and node.id not in self._seen:
#             self._seen.add(node.id)
#             self.variables.append(node.id)
#         self.generic_visit(node)
#
#     def visit_arg(self, node):
#         if node.arg not in self._seen:
#             self._seen.add(node.arg)
#             self.variables.append(node.arg)
#         self.generic_visit(node)
#
#
# def replace_variables_in_code(code: str, variables: List[str]) -> str:
#     modified = code
#     var_map = {v: f"var_{i+1}" for i, v in enumerate(variables)}
#     for name, repl in var_map.items():
#         pattern = r'\b' + re.escape(name) + r'\b'
#         modified = re.sub(pattern, repl, modified)
#     return modified
#
#
# def get_random_lines_snippet(lines: List[str], n: int) -> str:
#     non_import_idxs = [i for i, ln in enumerate(lines) if not re.match(r'^\s*(import|from)\b', ln)]
#     if len(non_import_idxs) <= n:
#         start = 0
#     else:
#         start = random.choice(non_import_idxs)
#         if start + n > len(lines):
#             start = len(lines) - n
#     return ''.join(lines[start:start+n])
#
#
# def get_up_to_n_function_snippets(lines: List[str], n_funcs: int = 10, max_len: int = 10) -> List[str]:
#     code = ''.join(lines)
#     try:
#         tree = ast.parse(code)
#         funcs = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
#         if not funcs:
#             return []
#
#         selected_funcs = random.sample(funcs, min(n_funcs, len(funcs)))
#         snippets = []
#         for fn in selected_funcs:
#             start = fn.lineno - 1
#             end = getattr(fn, 'end_lineno', fn.lineno)
#             snippet_end = min(end, start + max_len)
#             snippets.append(''.join(lines[start:snippet_end]))
#         return snippets
#     except Exception as e:
#         logger.error(f"Error extracting function snippets: {e}")
#         return []
#
#
# def process_repo(repo_dir: str, snippet_method: str, snippet_length: int) -> List[Dict[str, Any]]:
#     results = []
#     for root, _, files in os.walk(repo_dir):
#         for file in files:
#             if not file.endswith('.py'):
#                 continue
#             path = os.path.join(root, file)
#             rel = os.path.relpath(path, repo_dir)
#             try:
#                 with open(path, 'r', encoding='utf-8') as f:
#                     all_lines = f.readlines()
#                 total = len(all_lines)
#
#                 if snippet_method == 'random_function':
#                     snippets = get_up_to_n_function_snippets(all_lines, n_funcs=10, max_len=snippet_length)
#                     if not snippets:
#                         logger.info(f"Skipped {rel} - no function snippets found")
#                         continue
#                 elif snippet_method == 'random_lines':
#                     snippet = get_random_lines_snippet(all_lines, snippet_length)
#                     snippets = [snippet]
#                 else:  # 'first' or fallback
#                     snippets = [''.join(all_lines[:snippet_length])]
#
#                 for snippet in snippets:
#                     if not snippet.strip():
#                         continue
#                     vars = extract_variables_from_code(snippet)
#                     if not vars:
#                         continue
#
#                     anon = replace_variables_in_code(snippet, vars)
#                     info = {
#                         'file_path': rel,
#                         'code': snippet,
#                         'variables': vars,
#                         'anonymized_code': anon,
#                         'lines_processed': len(snippet.splitlines()),
#                         'total_lines': total
#                     }
#                     results.append(info)
#                     logger.info(f"Processed {rel} - {len(vars)} vars in snippet")
#
#             except Exception as e:
#                 logger.error(f"Error processing {path}: {e}")
#     return results
#
#
# def main():
#     p = argparse.ArgumentParser(description='Extract variable snippets from Python repos')
#     p.add_argument('-d', '--dir', help='Base directory of repos')
#     p.add_argument('-r', '--repos', nargs='+', help='Specific repo dirs')
#     p.add_argument('--snippet-method', choices=['first', 'random_lines', 'random_function'], default='random_function',
#                    help='Method to select code snippet')
#     p.add_argument('--snippet-length', type=int, default=10,
#                    help='Number of lines for snippet or max lines per function')
#     p.add_argument('-o', '--output', default='../data/anonymized_variables.json', help='Output JSON file')
#     args = p.parse_args()
#
#     repos = []
#     if args.dir:
#         repos = [os.path.join(args.dir, d) for d in os.listdir(args.dir) if os.path.isdir(os.path.join(args.dir, d))]
#     else:
#         repos = args.repos or []
#
#     all_data = {}
#     for repo_path in repos:
#         name = os.path.basename(repo_path)
#         data = process_repo(repo_path, args.snippet_method, args.snippet_length)
#         if data:
#             all_data[name] = data
#
#     with open(args.output, 'w', encoding='utf-8') as f:
#         json.dump(all_data, f, indent=2)
#     logger.info(f"Results saved to {args.output}")
#
#
# if __name__ == '__main__':
#     main()
#
#
#
