import os
import ast
import json
import argparse
import logging
import re
from typing import Dict, List, Set, Any, Tuple, Optional
from collections import defaultdict

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


class VariableVisitor(ast.NodeVisitor):
    """AST Node visitor that extracts variable names from Python code"""

    def __init__(self):
        self.variables = set()

    def visit_Name(self, node):
        # Check if this name is being assigned to (a variable definition)
        if isinstance(node.ctx, ast.Store):
            self.variables.add(node.id)
        self.generic_visit(node)

    def visit_arg(self, node):
        # Extract function arguments
        self.variables.add(node.arg)
        self.generic_visit(node)


def extract_variables_from_code(code: str) -> List[str]:
    """
    Extract variable names from Python code string using AST

    Args:
        code: Python code as a string

    Returns:
        List of variable names found in the code
    """
    try:
        # Parse the code into an AST
        tree = ast.parse(code)

        # Visit the AST to extract variables
        visitor = VariableVisitor()
        visitor.visit(tree)

        # Convert set to sorted list
        return sorted(list(visitor.variables))
    except SyntaxError as e:
        logger.error(f"Syntax error in code: {str(e)}")
        return []
    except Exception as e:
        logger.error(f"Error processing code: {str(e)}")
        return []


def replace_variables_in_code(code: str, variables: List[str]) -> str:
    """
    Replace variable names in code with 'var_x' pattern

    Args:
        code: Original Python code
        variables: List of variable names to replace

    Returns:
        Code with variable names replaced
    """
    try:
        # Sort variables by length (descending) to avoid partial replacements
        sorted_vars = sorted(variables, key=len, reverse=True)

        # Create a modified copy of the code
        modified_code = code

        # Create a mapping of variable names to their replacement
        var_map = {var: f"var_{i + 1}" for i, var in enumerate(sorted_vars)}

        # Replace variable names in the code
        for var_name, replacement in var_map.items():
            # Use regex to replace whole word variable names only
            pattern = r'\b' + re.escape(var_name) + r'\b'
            modified_code = re.sub(pattern, replacement, modified_code)

        return modified_code
    except Exception as e:
        logger.error(f"Error replacing variables: {str(e)}")
        return code


def process_repo(repo_dir: str) -> List[Dict[str, Any]]:
    """
    Process a repository and extract variables from first 15 lines of all Python files

    Args:
        repo_dir: Path to the repository directory

    Returns:
        List of dictionaries containing file info
    """
    result = []

    try:
        # Walk through all files in the repository
        for root, _, files in os.walk(repo_dir):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, repo_dir)

                    try:
                        # Read only the first 15 lines of the file
                        with open(file_path, 'r', encoding='utf-8') as f:
                            lines = []
                            for _ in range(15):  # Read only 15 lines
                                line = f.readline()
                                if not line:  # If we reach EOF before 15 lines
                                    break
                                lines.append(line)

                            # Join the first 15 lines (or fewer if file is shorter)
                            code = ''.join(lines)

                            # Store total line count for reference
                            with open(file_path, 'r', encoding='utf-8') as f:
                                total_lines = sum(1 for _ in f)

                        # Extract variables from the code
                        variables = extract_variables_from_code(code)

                        # Replace variables in the code
                        anonymized_code = replace_variables_in_code(code, variables)

                        # Only add the file information to the result if variables were found
                        if variables:  # Skip empty variable lists
                            file_info = {
                                "file_path": relative_path,
                                "code": code,
                                "variables": variables,
                                "anonymized_code": anonymized_code,
                                "lines_processed": min(15, total_lines),
                                "total_lines": total_lines
                            }

                            result.append(file_info)
                            logger.info(
                                f"Processed {relative_path} - Found {len(variables)} variables in first 15 lines (total lines: {total_lines})")
                        else:
                            logger.info(f"Skipped {relative_path} - No variables found in first 15 lines")
                    except Exception as e:
                        logger.error(f"Error processing file {file_path}: {str(e)}")
                        # Skip files with errors as they have empty variable lists
                        # (Uncomment the below code if you want to include error files in the output)
                        # file_info = {
                        #     "file_path": relative_path,
                        #     "code": "# Error reading file",
                        #     "variables": [],
                        #     "anonymized_code": "# Error processing file",
                        #     "error": str(e)
                        # }
                        # result.append(file_info)
    except Exception as e:
        logger.error(f"Error processing repository {repo_dir}: {str(e)}")

    return result


def process_repos(base_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Process multiple repositories and extract variables

    Args:
        base_dir: Path to the directory containing repositories

    Returns:
        Dictionary with repositories as keys and lists of file info dictionaries as values
    """
    result = {}

    # List all subdirectories in the base directory
    try:
        subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

        logger.info(f"Found {len(subdirs)} potential repositories in {base_dir}")

        # Process each subdirectory as a repository
        for repo_name in subdirs:
            repo_path = os.path.join(base_dir, repo_name)
            logger.info(f"Processing repository: {repo_name}")

            # Process the repository
            repo_data = process_repo(repo_path)

            # Only add repositories with non-empty results
            if repo_data:
                result[repo_name] = repo_data
                logger.info(f"Completed repository {repo_name} - Found {len(repo_data)} Python files with variables")
            else:
                logger.info(f"Skipped repository {repo_name} - No Python files with variables found")
    except Exception as e:
        logger.error(f"Error listing repositories in {base_dir}: {str(e)}")

    return result


def process_repos_list(repos_dir_list: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Process a list of repository directories

    Args:
        repos_dir_list: List of paths to repository directories

    Returns:
        Dictionary with repositories as keys and lists of file info dictionaries as values
    """
    result = {}

    for repo_path in repos_dir_list:
        if os.path.isdir(repo_path):
            repo_name = os.path.basename(repo_path)
            logger.info(f"Processing repository: {repo_name}")

            # Process the repository
            repo_data = process_repo(repo_path)

            # Only add repositories with non-empty results
            if repo_data:
                result[repo_name] = repo_data
                logger.info(f"Completed repository {repo_name} - Found {len(repo_data)} Python files with variables")
            else:
                logger.info(f"Skipped repository {repo_name} - No Python files with variables found")
        else:
            logger.warning(f"Skipping {repo_path} - Not a directory")

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Extract variables from the first 15 lines of Python files in repositories')

    # Input methods
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('-d', '--dir', help='Directory containing repositories')
    input_group.add_argument('-r', '--repos', nargs='+', help='List of repository directories')

    # Output
    parser.add_argument('-o', '--output', default='repo_variables_anonymized.json', help='Output JSON file')
    parser.add_argument('--pretty', action='store_true', help='Pretty print JSON output')

    args = parser.parse_args()

    # Process repositories
    if args.dir:
        logger.info(f"Processing repositories in directory: {args.dir}")
        result = process_repos(args.dir)
    else:
        logger.info(f"Processing specified repositories: {args.repos}")
        result = process_repos_list(args.repos)

    # Write result to JSON file
    indent = 2 if args.pretty else None
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=indent)

    logger.info(f"Results written to {args.output}")
    print(f"Processed {len(result)} repositories. Results saved to {args.output}")

    # Print sample output structure
    print("\nOutput structure example:")
    print("{\n  \"repo_name\": [")
    print("    {\n      \"file_path\": \"path/to/file.py\",")
    print("      \"code\": \"# First 15 lines of code...\",")
    print("      \"variables\": [\"var1\", \"var2\", ...],")
    print("      \"anonymized_code\": \"# Code with variables replaced...\",")
    print("      \"lines_processed\": 15,")
    print("      \"total_lines\": 250")
    print("    },\n    ...\n  ],\n  ...\n}")


if __name__ == "__main__":
    main()