import os
import ast
import argparse
import json
from collections import Counter
import re
from typing import Dict, List, Set, Any, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('variable_extraction.log')
    ]
)
logger = logging.getLogger(__name__)


class VariableVisitor(ast.NodeVisitor):
    """AST Node visitor that extracts variable names from Python code"""

    def __init__(self):
        self.variables = set()
        self.function_params = set()
        self.function_names = set()
        self.class_names = set()
        self.imports = set()
        self.current_scope = []

    def visit_Name(self, node):
        # Check if this name is being assigned to (a variable definition)
        if isinstance(node.ctx, ast.Store):
            # Skip private variables if needed
            # if not node.id.startswith('_'):
            self.variables.add(node.id)
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        # Add function name
        self.function_names.add(node.name)

        # Push a new scope
        self.current_scope.append(node.name)

        # Extract parameters
        for arg in node.args.args:
            self.function_params.add(arg.arg)

        # Visit children
        self.generic_visit(node)

        # Pop the scope
        self.current_scope.pop()

    def visit_ClassDef(self, node):
        # Add class name
        self.class_names.add(node.name)

        # Push a new scope
        self.current_scope.append(node.name)

        # Visit children
        self.generic_visit(node)

        # Pop the scope
        self.current_scope.pop()

    def visit_Import(self, node):
        for name in node.names:
            self.imports.add(name.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module:
            self.imports.add(node.module)
        for name in node.names:
            self.imports.add(name.name)
        self.generic_visit(node)


def extract_variables_from_file(file_path: str) -> Dict[str, Set[str]]:
    """
    Extract variable names from a Python file using AST

    Args:
        file_path: Path to the Python file

    Returns:
        Dictionary with sets of variables, function parameters, function names,
        class names, and imports
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Parse the file into an AST
        tree = ast.parse(content)

        # Visit the AST to extract variables
        visitor = VariableVisitor()
        visitor.visit(tree)

        return {
            'variables': visitor.variables,
            'function_params': visitor.function_params,
            'function_names': visitor.function_names,
            'class_names': visitor.class_names,
            'imports': visitor.imports
        }
    except SyntaxError as e:
        logger.error(f"Syntax error in {file_path}: {str(e)}")
        return {
            'variables': set(),
            'function_params': set(),
            'function_names': set(),
            'class_names': set(),
            'imports': set()
        }
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        return {
            'variables': set(),
            'function_params': set(),
            'function_names': set(),
            'class_names': set(),
            'imports': set()
        }


def extract_variables_from_folder(folder_path: str) -> Dict[str, Dict[str, Set[str]]]:
    """
    Extract variables from all Python files in a folder (and subfolders)

    Args:
        folder_path: Path to the folder containing Python files

    Returns:
        Dictionary mapping file paths to their extracted variables
    """
    results = {}

    # Walk through the directory
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    logger.info(f"Processing {file_path}")
                    variables = extract_variables_from_file(file_path)
                    results[file_path] = variables
                except Exception as e:
                    logger.error(f"Failed to process {file_path}: {str(e)}")

    return results


def analyze_variable_patterns(variables: List[str]) -> Dict[str, Any]:
    """
    Analyze patterns in variable names

    Args:
        variables: List of variable names

    Returns:
        Dictionary with analysis results
    """
    results = {}

    # Count variable lengths
    lengths = [len(var) for var in variables]
    results['avg_length'] = sum(lengths) / len(lengths) if lengths else 0
    results['min_length'] = min(lengths) if lengths else 0
    results['max_length'] = max(lengths) if lengths else 0

    # Check for naming conventions
    snake_case = sum(1 for var in variables if re.match(r'^[a-z][a-z0-9_]*$', var))
    camel_case = sum(
        1 for var in variables if re.match(r'^[a-z][a-zA-Z0-9]*$', var) and not re.match(r'^[a-z][a-z0-9_]*$', var))
    pascal_case = sum(1 for var in variables if re.match(r'^[A-Z][a-zA-Z0-9]*$', var))

    total = len(variables)
    results['naming_conventions'] = {
        'snake_case': snake_case / total if total else 0,
        'camel_case': camel_case / total if total else 0,
        'pascal_case': pascal_case / total if total else 0,
        'other': (total - snake_case - camel_case - pascal_case) / total if total else 0
    }

    # Most common prefixes and suffixes
    prefixes = Counter()
    suffixes = Counter()
    for var in variables:
        if len(var) >= 3:
            prefixes[var[:2]] += 1
            suffixes[var[-2:]] += 1

    results['common_prefixes'] = dict(prefixes.most_common(10))
    results['common_suffixes'] = dict(suffixes.most_common(10))

    return results


def process_variables(extracted_data: Dict[str, Dict[str, Set[str]]]) -> Dict[str, Any]:
    """
    Process and analyze extracted variables from multiple files

    Args:
        extracted_data: Dictionary of extracted variables by file

    Returns:
        Dictionary with analysis results
    """
    # Combine variables from all files
    all_variables = set()
    all_function_params = set()
    all_function_names = set()
    all_class_names = set()
    all_imports = set()

    file_count = len(extracted_data)

    for file_path, data in extracted_data.items():
        all_variables.update(data['variables'])
        all_function_params.update(data['function_params'])
        all_function_names.update(data['function_names'])
        all_class_names.update(data['class_names'])
        all_imports.update(data['imports'])

    # Convert sets to lists for JSON serialization
    results = {
        'summary': {
            'file_count': file_count,
            'unique_variable_count': len(all_variables),
            'unique_function_param_count': len(all_function_params),
            'unique_function_count': len(all_function_names),
            'unique_class_count': len(all_class_names),
            'unique_import_count': len(all_imports)
        },
        'variable_analysis': analyze_variable_patterns(list(all_variables)),
        'function_param_analysis': analyze_variable_patterns(list(all_function_params)),
        'top_variables': list(all_variables)[:100],  # Limiting to 100 to avoid too much output
        'top_function_params': list(all_function_params)[:100],
        'top_function_names': list(all_function_names)[:100],
        'top_class_names': list(all_class_names)[:100],
        'top_imports': list(all_imports)[:100]
    }

    return results


def export_detailed_results(extracted_data: Dict[str, Dict[str, Set[str]]], output_file: str):
    """
    Export detailed results to a JSON file

    Args:
        extracted_data: Dictionary of extracted variables by file
        output_file: Path to the output JSON file
    """
    # Convert sets to lists for JSON serialization
    serializable_data = {}
    for file_path, data in extracted_data.items():
        serializable_data[file_path] = {
            key: list(value) for key, value in data.items()
        }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Extract variable names from Python files')
    parser.add_argument('folder', help='Folder containing Python files')
    parser.add_argument('-o', '--output', default='variable_analysis.json',
                        help='Output JSON file for the analysis results')
    parser.add_argument('-d', '--detailed', default='variables_detailed.json',
                        help='Output JSON file for detailed variables by file')
    parser.add_argument('--exclude-dirs', nargs='+', default=[],
                        help='Directories to exclude (relative to the input folder)')
    args = parser.parse_args()

    logger.info(f"Scanning Python files in {args.folder}")
    extracted_data = extract_variables_from_folder(args.folder)
    logger.info(f"Found {len(extracted_data)} Python files")

    # Process and analyze the data
    results = process_variables(extracted_data)

    # Write summary results to JSON
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Analysis results written to {args.output}")

    # Export detailed results
    export_detailed_results(extracted_data, args.detailed)
    logger.info(f"Detailed results written to {args.detailed}")

    # Print summary to console
    print("\n===== Variable Extraction Summary =====")
    print(f"Processed {results['summary']['file_count']} Python files")
    print(f"Found {results['summary']['unique_variable_count']} unique variables")
    print(f"Found {results['summary']['unique_function_count']} unique functions")
    print(f"Found {results['summary']['unique_class_count']} unique classes")
    print("\nVariable Naming Conventions:")
    conventions = results['variable_analysis']['naming_conventions']
    print(f"  Snake Case: {conventions['snake_case']:.1%}")
    print(f"  Camel Case: {conventions['camel_case']:.1%}")
    print(f"  Pascal Case: {conventions['pascal_case']:.1%}")
    print(f"  Other: {conventions['other']:.1%}")

    print("\nMost Common Variable Name Prefixes:")
    for prefix, count in list(results['variable_analysis']['common_prefixes'].items())[:5]:
        print(f"  {prefix}: {count}")

    print("\nExample Variable Names:")
    for var in list(results['top_variables'])[:10]:
        print(f"  {var}")


if __name__ == "__main__":
    main()