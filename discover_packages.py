"""
discover_packages.py

Discovers Python packages and extracts their official metadata (documentation,
repository, homepage, and summary) using:
1. PyPI Simple Index API (PEP 691 JSON API) for discovering package names.
2. PyPI JSON API (https://pypi.org/pypi/<package>/json) for detailed package metadata.
"""

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import urllib.error
import urllib.request

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("discover_packages")

# PyPI asks API consumers to set a descriptive User-Agent
USER_AGENT = "learn-functions-crawler/0.1.0 (https://github.com/learn-functions)"
PYPI_SIMPLE_URL = "https://pypi.org/simple/"
PYPI_PACKAGE_JSON_URL = "https://pypi.org/pypi/{package}/json"

# Common keywords used in project_urls keys for doc links
DOC_URL_PATTERNS = [
    re.compile(r"^doc(s|umentation)?$", re.IGNORECASE),
    re.compile(r".*documentation.*", re.IGNORECASE),
    re.compile(r".*docs.*", re.IGNORECASE),
    re.compile(r".*reference.*", re.IGNORECASE),
    re.compile(r".*api.*", re.IGNORECASE),
]

SOURCE_URL_PATTERNS = [
    re.compile(r"^(source|source code|repository|code|repo)$", re.IGNORECASE),
    re.compile(r".*(source|repository|github|gitlab).*", re.IGNORECASE),
]

HOMEPAGE_URL_PATTERNS = [
    re.compile(r"^(home|homepage|home page|website)$", re.IGNORECASE),
]


@dataclass
class PackageMetadata:
    """Normalized metadata for a discovered package."""
    name: str
    version: str
    summary: Optional[str] = None
    documentation_url: Optional[str] = None
    homepage_url: Optional[str] = None
    source_url: Optional[str] = None
    all_project_urls: Dict[str, str] = field(default_factory=dict)
    discovered_at: float = field(default_factory=time.time)


def _make_request(url: str, headers: Optional[Dict[str, str]] = None) -> bytes:
    """Helper to perform HTTP GET requests with custom headers and error handling."""
    req_headers = {"User-Agent": USER_AGENT}
    if headers:
        req_headers.update(headers)

    req = urllib.request.Request(url, headers=req_headers)
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.read()


def fetch_pypi_index_names(limit: Optional[int] = None) -> List[str]:
    """
    Fetch package names from PyPI Simple Index API (PEP 691 JSON).
    Note: PyPI hosts 500k+ packages, so response is large (~30-50MB JSON).
    """
    logger.info("Fetching registered packages from PyPI Simple Index (PEP 691)...")
    headers = {"Accept": "application/vnd.pypi.simple.v1+json"}
    
    try:
        raw_data = _make_request(PYPI_SIMPLE_URL, headers=headers)
        data = json.loads(raw_data.decode("utf-8"))
        projects = data.get("projects", [])
        names = [p["name"] for p in projects if "name" in p]
        logger.info("Discovered %d total packages on PyPI.", len(names))
        if limit and limit > 0:
            names = names[:limit]
            logger.info("Limited to top %d packages.", len(names))
        return names
    except Exception as exc:
        logger.error("Failed to query PyPI Simple API: %s", exc)
        return []


def _extract_url_by_patterns(urls: Dict[str, str], patterns: List[re.Pattern]) -> Optional[str]:
    """Find the first URL whose label matches any of the given regex patterns."""
    for pattern in patterns:
        for label, url in urls.items():
            if url and pattern.match(label.strip()):
                return url.strip()
    return None


def fetch_package_metadata(package_name: str) -> Optional[PackageMetadata]:
    """
    Query the PyPI JSON API for a given package and extract normalized links.
    """
    url = PYPI_PACKAGE_JSON_URL.format(package=package_name)
    try:
        raw_json = _make_request(url)
        payload = json.loads(raw_json.decode("utf-8"))
    except urllib.error.HTTPError as http_err:
        if http_err.code == 404:
            logger.warning("Package '%s' not found on PyPI (404).", package_name)
        else:
            logger.warning("HTTP %d error fetching metadata for '%s'.", http_err.code, package_name)
        return None
    except Exception as exc:
        logger.warning("Failed fetching metadata for '%s': %s", package_name, exc)
        return None

    info: Dict[str, Any] = payload.get("info", {})
    name = info.get("name") or package_name
    version = info.get("version", "unknown")
    summary = info.get("summary")

    # Extract declared URLs from project_urls and fallbacks
    project_urls: Dict[str, str] = info.get("project_urls") or {}

    doc_url = _extract_url_by_patterns(project_urls, DOC_URL_PATTERNS)
    # PyPI historically also had an info.docs_url field
    if not doc_url and info.get("docs_url"):
        doc_url = info.get("docs_url")

    source_url = _extract_url_by_patterns(project_urls, SOURCE_URL_PATTERNS)
    homepage_url = _extract_url_by_patterns(project_urls, HOMEPAGE_URL_PATTERNS)
    if not homepage_url and info.get("home_page"):
        homepage_url = info.get("home_page")

    return PackageMetadata(
        name=name,
        version=version,
        summary=summary,
        documentation_url=doc_url,
        homepage_url=homepage_url,
        source_url=source_url,
        all_project_urls=project_urls,
    )


def discover_and_collect(
    package_names: List[str],
    delay_seconds: float = 0.1,
) -> List[PackageMetadata]:
    """
    Iterate over package names, fetching metadata for each with courteous rate limiting.
    """
    results: List[PackageMetadata] = []
    total = len(package_names)

    for idx, name in enumerate(package_names, start=1):
        logger.info("[%d/%d] Fetching metadata for '%s'...", idx, total, name)
        meta = fetch_package_metadata(name)
        if meta:
            logger.info(
                "  -> Found '%s' v%s | Docs: %s",
                meta.name,
                meta.version,
                meta.documentation_url or "(None)",
            )
            results.append(meta)
        if delay_seconds > 0 and idx < total:
            time.sleep(delay_seconds)

    return results


def save_discovered_packages(metadata_list: List[PackageMetadata], output_path: Path) -> None:
    """Save the metadata records into a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = [asdict(item) for item in metadata_list]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serialized, f, indent=2)
    logger.info("Saved %d package records to %s", len(metadata_list), output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discover Python packages and extract documentation URLs from PyPI."
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "-p", "--packages",
        nargs="+",
        help="Explicit list of package names to inspect (e.g. numpy pandas torch requests).",
    )
    group.add_argument(
        "--from-simple-index",
        action="store_true",
        help="Discover package names by querying PyPI Simple Index API.",
    )
    parser.add_argument(
        "-l", "--limit",
        type=int,
        default=10,
        help="Max number of packages to process when using --from-simple-index (default: 10).",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("data/discovered_packages.json"),
        help="Output path for the discovered package metadata JSON (default: data/discovered_packages.json).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.2,
        help="Polite delay in seconds between HTTP requests (default: 0.2s).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Determine which package names to discover/query
    if args.packages:
        package_names = args.packages
    elif args.from_simple_index:
        package_names = fetch_pypi_index_names(limit=args.limit)
    else:
        # Default starter seed of popular packages to learn with immediately
        package_names = [
            "numpy",
            "pandas",
            "scipy",
            "requests",
            "torch",
            "scikit-learn",
            "matplotlib",
            "flask",
        ]
        logger.info(
            "No packages specified. Using starter seed list (%d packages): %s",
            len(package_names),
            ", ".join(package_names),
        )

    if not package_names:
        logger.error("No packages found to process.")
        sys.exit(1)

    metadata_list = discover_and_collect(package_names, delay_seconds=args.delay)
    save_discovered_packages(metadata_list, args.output)


if __name__ == "__main__":
    main()
