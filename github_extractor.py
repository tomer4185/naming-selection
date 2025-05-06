import os
import sys
import requests
import argparse
from urllib.parse import urljoin, urlparse
import logging
import time
import concurrent.futures

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('github_extractor.log')
    ]
)
logger = logging.getLogger(__name__)


def convert_to_raw_url(github_url):
    """
    Convert a GitHub URL to its corresponding raw content URL.

    This function processes a GitHub repository URL and attempts to convert it
    to a raw content URL format. It supports both standard GitHub URLs with
    a "blob" path as well as other variations. If the URL is already in the
    raw content format or not relevant to the expected patterns, the original
    URL is returned unchanged.

    :param github_url: The URL of the GitHub resource to be converted.
    :type github_url: str
    :return: A modified URL pointing to the raw content, or the original
        URL if conversion is not applicable.
    :rtype: str
    """
    # Parse the URL
    parsed_url = urlparse(github_url)
    path_parts = parsed_url.path.strip('/').split('/')

    # Check if it's a regular GitHub URL
    if parsed_url.netloc == 'github.com':
        # Check if it's already a raw URL
        if parsed_url.netloc == 'raw.githubusercontent.com':
            return github_url

        # Regular GitHub repository file
        if len(path_parts) >= 5 and path_parts[2] == 'blob':
            # Extract user, repo, branch and filepath
            user = path_parts[0]
            repo = path_parts[1]
            branch = path_parts[3]
            filepath = '/'.join(path_parts[4:])
            return f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/{filepath}"

    # If it's not a standard GitHub file URL, try the generic approach
    # Replace 'blob/' with 'raw/' in the URL path
    if '/blob/' in github_url:
        return github_url.replace('/blob/', '/raw/')

    return github_url


def download_python_file(url, output_dir='.', repo_name=None):
    """
    Download a Python file from a GitHub URL
    """
    try:
        # Convert to raw URL if needed
        raw_url = convert_to_raw_url(url)

        # Get the filename from the URL
        filename = os.path.basename(urlparse(raw_url).path)
        if not filename.endswith('.py'):
            filename += '.py'

        # If repo_name is provided, create a subfolder for the repository
        if repo_name:
            output_dir = os.path.join(output_dir, repo_name)

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Download the file
        response = requests.get(raw_url, timeout=10)

        if response.status_code == 200:
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(response.text)
            logger.info(f"Downloaded {filename} to {output_dir}")
            return filepath
        else:
            logger.warning(f"Failed to download {raw_url}, status code: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Error downloading {url}: {str(e)}")
        return None


def download_repository_py_files(repo_url, branch='main', output_dir='.', organize_by_repo=True):
    """
    Download all Python files from a GitHub repository
    """
    try:
        # Parse the URL to get user and repo
        parsed_url = urlparse(repo_url)
        path_parts = parsed_url.path.strip('/').split('/')

        if len(path_parts) < 2:
            logger.error(f"Invalid repository URL: {repo_url}")
            return []

        user = path_parts[0]
        repo = path_parts[1]
        repo_name = f"{user}_{repo}" if organize_by_repo else None

        # Use GitHub API to get repository contents
        api_url = f"https://api.github.com/repos/{user}/{repo}/git/trees/{branch}?recursive=1"

        headers = {}
        # Add GitHub token if available as environment variable
        if 'GITHUB_TOKEN' in os.environ:
            headers['Authorization'] = f"token {os.environ['GITHUB_TOKEN']}"

        logger.info(f"Fetching repository contents for {user}/{repo}")
        response = requests.get(api_url, headers=headers, timeout=30)

        if response.status_code == 403 and 'rate limit exceeded' in response.text.lower():
            logger.warning("GitHub API rate limit exceeded. Consider using a GitHub token.")
            logger.warning("Set the GITHUB_TOKEN environment variable with your personal access token.")
            logger.warning("Waiting for 60 seconds before continuing...")
            time.sleep(60)  # Wait a bit before potentially trying again
            return []

        if response.status_code != 200:
            logger.error(f"Failed to fetch repository contents for {repo_url}, status code: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return []

        # Extract Python files
        py_files = []
        repo_data = response.json()

        if 'tree' not in repo_data:
            logger.error(f"No 'tree' found in repository data for {repo_url}")
            return []

        for item in repo_data['tree']:
            if item['type'] == 'blob' and item['path'].endswith('.py'):
                file_url = f"https://github.com/{user}/{repo}/blob/{branch}/{item['path']}"
                download_path = download_python_file(file_url, output_dir, repo_name)
                if download_path:
                    py_files.append(download_path)

        logger.info(f"Downloaded {len(py_files)} Python files from {repo_url}")
        return py_files
    except Exception as e:
        logger.error(f"Error processing repository {repo_url}: {str(e)}")
        return []


def process_repo_url(repo_url, branch, output_dir, organize_by_repo):
    """Process a single repository URL - used for parallel processing"""
    try:
        logger.info(f"Processing repository: {repo_url}")
        return download_repository_py_files(repo_url, branch, output_dir, organize_by_repo)
    except Exception as e:
        logger.error(f"Error processing {repo_url}: {str(e)}")
        return []


def process_multiple_repositories(repo_urls, branch='main', output_dir='.', max_workers=5, organize_by_repo=True):
    """
    Process multiple repository URLs in parallel
    """
    total_files = 0
    processed_repos = 0

    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all repository processing tasks
        future_to_url = {
            executor.submit(process_repo_url, url, branch, output_dir, organize_by_repo): url
            for url in repo_urls
        }

        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                files = future.result()
                total_files += len(files)
                processed_repos += 1
                logger.info(
                    f"Completed {processed_repos}/{len(repo_urls)} repositories. Progress: {processed_repos / len(repo_urls) * 100:.1f}%")
            except Exception as e:
                logger.error(f"Repository {url} generated an exception: {str(e)}")

    return total_files


def read_repo_urls_from_file(file_path):
    """Read repository URLs from a file, one URL per line"""
    try:
        with open(file_path, 'r') as f:
            urls = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
        return urls
    except Exception as e:
        logger.error(f"Error reading URLs from file {file_path}: {str(e)}")
        return []


def main():
    parser = argparse.ArgumentParser(description='Download Python files from multiple GitHub repositories')

    # Input methods
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('-u', '--urls', nargs='+', help='List of GitHub repository URLs')
    input_group.add_argument('-f', '--file', help='File containing GitHub repository URLs (one per line)')

    # Additional options
    parser.add_argument('-o', '--output', default='github_extracted', help='Output directory for downloaded files')
    parser.add_argument('-b', '--branch', default='main', help='Branch to download from (default: main)')
    parser.add_argument('-w', '--workers', type=int, default=5, help='Number of parallel workers (default: 5)')
    parser.add_argument('-s', '--single', action='store_true',
                        help='Single file mode, treats URLs as Python file URLs instead of repositories')
    parser.add_argument('--no-organize', action='store_true', help='Do not organize files by repository')

    args = parser.parse_args()

    # Get repository URLs
    repo_urls = []
    if args.file:
        repo_urls = read_repo_urls_from_file(args.file)
        if not repo_urls:
            logger.error(f"No valid URLs found in {args.file}")
            return
    else:
        repo_urls = args.urls

    logger.info(f"Processing {len(repo_urls)} {'files' if args.single else 'repositories'}")

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Process based on mode
    if args.single:
        # Single file mode
        total_files = 0
        for url in repo_urls:
            if download_python_file(url, args.output):
                total_files += 1
        logger.info(f"Downloaded {total_files} Python files")
    else:
        # Repository mode
        total_files = process_multiple_repositories(
            repo_urls,
            args.branch,
            args.output,
            args.workers,
            not args.no_organize
        )
        logger.info(f"Completed processing {len(repo_urls)} repositories")
        logger.info(f"Total Python files downloaded: {total_files}")


if __name__ == "__main__":
    main()