import os
import sys
import requests
import argparse
from urllib.parse import urljoin, urlparse
import logging
import time
import concurrent.futures
from datetime import datetime, timedelta
import json

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


class GitHubRateLimiter:
    """Handle GitHub API rate limiting"""

    def __init__(self):
        self.reset_time = None
        self.remaining_requests = None
        self.last_check = None

    def check_rate_limit(self, headers):
        """Update rate limit info from response headers"""
        if 'x-ratelimit-remaining' in headers:
            self.remaining_requests = int(headers['x-ratelimit-remaining'])
        if 'x-ratelimit-reset' in headers:
            self.reset_time = datetime.fromtimestamp(int(headers['x-ratelimit-reset']))
        self.last_check = datetime.now()

    def should_wait(self):
        """Check if we should wait due to rate limiting"""
        if self.remaining_requests is not None and self.remaining_requests < 5:
            return True
        return False

    def wait_for_reset(self):
        """Wait until rate limit resets"""
        if self.reset_time:
            wait_time = (self.reset_time - datetime.now()).total_seconds()
            if wait_time > 0:
                logger.warning(f"Rate limit reached. Waiting {wait_time:.0f} seconds until reset...")
                time.sleep(wait_time + 5)  # Add 5 seconds buffer
            else:
                logger.info("Rate limit should have reset, continuing...")
        else:
            logger.warning("Rate limit reached but no reset time available. Waiting 60 seconds...")
            time.sleep(60)


# Global rate limiter instance
rate_limiter = GitHubRateLimiter()


def get_github_headers():
    """Get headers for GitHub API requests"""
    headers = {
        'Accept': 'application/vnd.github.v3+json',
        'User-Agent': 'GitHub-Python-Extractor'
    }

    # Add GitHub token if available
    if 'GITHUB_TOKEN' in os.environ:
        headers['Authorization'] = f"token {os.environ['GITHUB_TOKEN']}"
        logger.info("Using GitHub token for authentication")
    else:
        logger.warning("No GitHub token found. Rate limits will be more restrictive.")

    return headers


def make_github_request(url, timeout=30):
    """Make a GitHub API request with rate limit handling"""
    headers = get_github_headers()

    # Check if we should wait before making the request
    if rate_limiter.should_wait():
        rate_limiter.wait_for_reset()

    try:
        response = requests.get(url, headers=headers, timeout=timeout)

        # Update rate limit info
        rate_limiter.check_rate_limit(response.headers)

        # Handle rate limit responses
        if response.status_code == 403:
            if 'rate limit exceeded' in response.text.lower():
                logger.warning("Rate limit exceeded in response")
                rate_limiter.wait_for_reset()
                # Retry once after waiting
                response = requests.get(url, headers=headers, timeout=timeout)
                rate_limiter.check_rate_limit(response.headers)

        return response

    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed for {url}: {str(e)}")
        return None


def convert_to_raw_url(github_url):
    """Convert a GitHub URL to its corresponding raw content URL."""
    parsed_url = urlparse(github_url)
    path_parts = parsed_url.path.strip('/').split('/')

    if parsed_url.netloc == 'github.com':
        if parsed_url.netloc == 'raw.githubusercontent.com':
            return github_url

        if len(path_parts) >= 5 and path_parts[2] == 'blob':
            user = path_parts[0]
            repo = path_parts[1]
            branch = path_parts[3]
            filepath = '/'.join(path_parts[4:])
            return f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/{filepath}"

    if '/blob/' in github_url:
        return github_url.replace('/blob/', '/raw/')

    return github_url


def download_python_file(url, output_dir='.', repo_name=None):
    """Download a Python file from a GitHub URL"""
    try:
        raw_url = convert_to_raw_url(url)
        filename = os.path.basename(urlparse(raw_url).path)
        if not filename.endswith('.py'):
            filename += '.py'

        if repo_name:
            output_dir = os.path.join(output_dir, repo_name)

        os.makedirs(output_dir, exist_ok=True)

        # Add delay between file downloads to be respectful
        time.sleep(0.1)

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


def parse_repo_url(repo_url):
    """Parse repository URL to extract user and repo name"""
    parsed_url = urlparse(repo_url)
    path_parts = parsed_url.path.strip('/').split('/')

    if len(path_parts) < 2:
        raise ValueError(f"Invalid repository URL: {repo_url}")

    user = path_parts[0]
    repo = path_parts[1]

    # Handle URLs with additional path components (like /tree/main/src)
    if len(path_parts) > 2:
        logger.info(f"URL contains path beyond repo: {'/'.join(path_parts[2:])}")

    return user, repo


def download_repository_py_files(repo_url, branch='main', output_dir='.', organize_by_repo=True, max_files=15):
    """Download Python files from a GitHub repository (limited to max_files)"""
    try:
        user, repo = parse_repo_url(repo_url)
        repo_name = f"{user}_{repo}" if organize_by_repo else None

        # Use GitHub API to get repository contents
        api_url = f"https://api.github.com/repos/{user}/{repo}/git/trees/{branch}?recursive=1"

        logger.info(f"Fetching repository contents for {user}/{repo}")
        response = make_github_request(api_url)

        if response is None:
            logger.error(f"Failed to make request for {repo_url}")
            return []

        if response.status_code != 200:
            logger.error(f"Failed to fetch repository contents for {repo_url}, status code: {response.status_code}")
            logger.error(f"Response: {response.text[:500]}")  # Truncate long responses
            return []

        # Extract Python files (limited to max_files)
        py_files = []
        repo_data = response.json()

        if 'tree' not in repo_data:
            logger.error(f"No 'tree' found in repository data for {repo_url}")
            return []

        # Filter and limit Python files
        python_files = [item for item in repo_data['tree']
                        if item['type'] == 'blob' and item['path'].endswith('.py')]

        # Limit to max_files
        python_files = python_files[:max_files]

        logger.info(f"Found {len(python_files)} Python files to download from {user}/{repo} (limited to {max_files})")

        # Add delay between API calls to be respectful
        time.sleep(0.5)

        for item in python_files:
            file_url = f"https://github.com/{user}/{repo}/blob/{branch}/{item['path']}"
            download_path = download_python_file(file_url, output_dir, repo_name)
            if download_path:
                py_files.append(download_path)

            # Small delay between downloads
            time.sleep(0.2)

        logger.info(f"Downloaded {len(py_files)} Python files from {repo_url}")
        return py_files
    except Exception as e:
        logger.error(f"Error processing repository {repo_url}: {str(e)}")
        return []


def process_repo_url(repo_url, branch, output_dir, organize_by_repo, max_files):
    """Process a single repository URL - used for parallel processing"""
    try:
        logger.info(f"Processing repository: {repo_url}")
        return download_repository_py_files(repo_url, branch, output_dir, organize_by_repo, max_files)
    except Exception as e:
        logger.error(f"Error processing {repo_url}: {str(e)}")
        return []


def process_multiple_repositories(repo_urls, branch='main', output_dir='.', max_workers=3, organize_by_repo=True,
                                  max_files=15):
    """Process multiple repository URLs with reduced parallelism to respect rate limits"""
    total_files = 0
    processed_repos = 0

    # Reduced max_workers to be more respectful of rate limits
    actual_workers = min(max_workers, 3) if 'GITHUB_TOKEN' not in os.environ else max_workers

    logger.info(f"Using {actual_workers} workers for processing")

    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=actual_workers) as executor:
        # Submit all repository processing tasks
        future_to_url = {
            executor.submit(process_repo_url, url, branch, output_dir, organize_by_repo, max_files): url
            for url in repo_urls
        }

        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                files = future.result()
                total_files += len(files)
                processed_repos += 1
                progress = processed_repos / len(repo_urls) * 100
                logger.info(f"Completed {processed_repos}/{len(repo_urls)} repositories. Progress: {progress:.1f}%")

                # Add a small delay between repository processing
                time.sleep(1)

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
    parser.add_argument('-w', '--workers', type=int, default=3, help='Number of parallel workers (default: 3)')
    parser.add_argument('-m', '--max-files', type=int, default=20,
                        help='Maximum number of Python files to download per repository (default: 15)')
    parser.add_argument('-s', '--single', action='store_true',
                        help='Single file mode, treats URLs as Python file URLs instead of repositories')
    parser.add_argument('--no-organize', action='store_true', help='Do not organize files by repository')
    parser.add_argument('-t', '--token', help='GitHub personal access token for API authentication')

    args = parser.parse_args()

    # Set GitHub token if provided via command line
    if args.token:
        os.environ['GITHUB_TOKEN'] = args.token
        logger.info("GitHub token provided via command line")

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
    if not args.single:
        logger.info(f"Maximum files per repository: {args.max_files}")

    # Check for GitHub token
    if 'GITHUB_TOKEN' not in os.environ:
        logger.warning("No GitHub token detected. You may hit rate limits quickly.")
        logger.warning("Consider setting GITHUB_TOKEN environment variable or using -t option for better performance.")

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Process based on mode
    if args.single:
        # Single file mode
        total_files = 0
        for url in repo_urls:
            if download_python_file(url, args.output):
                total_files += 1
            time.sleep(0.1)  # Small delay between downloads
        logger.info(f"Downloaded {total_files} Python files")
    else:
        # Repository mode
        total_files = process_multiple_repositories(
            repo_urls,
            args.branch,
            args.output,
            args.workers,
            not args.no_organize,
            args.max_files
        )
        logger.info(f"Completed processing {len(repo_urls)} repositories")
        logger.info(f"Total Python files downloaded: {total_files}")


if __name__ == "__main__":
    main()
