"""
collect_documentation.py

Step 2 in the learn-functions pipeline:
Retrieves and indexes Python API symbols (functions, methods, classes, modules)
from discovered packages by:
1. Reading and validating package records from data/discovered_packages.json.
2. Validating URLs and rejecting placeholders ("UNKNOWN", "404 NOT FOUND", etc.).
3. Normalizing project links (classifying repository vs doc vs homepage URLs).
4. Running fallback discovery (inspecting GitHub README / Pages when doc URL is missing).
5. Probing and decoding Sphinx 'objects.inv' inventories using standard zlib.
6. Extracting canonical Python symbols with full URLs.
7. Tracking explicit collection statuses (documentation_found, no_documentation_url,
   invalid_url, collection_failed).
8. Deduplicating records without cross-package collision.
"""

import argparse
import enum
import json
import logging
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("collect_documentation")

USER_AGENT = "learn-functions-collector/0.1.0 (https://github.com/learn-functions)"

# Placeholders frequently found in malformed PyPI metadata
INVALID_PLACEHOLDERS = {
    "unknown",
    "none",
    "null",
    "n/a",
    "na",
    "todo",
    "404",
    "404 not found",
    "not found",
    "tbd",
    "example.com",
    "http://",
    "https://",
    "",
}

# Regex for repository hosts
REPO_HOST_REGEX = re.compile(
    r"^https?://(www\.)?(github\.com|gitlab\.com|bitbucket\.org|codeberg\.org)/([a-zA-Z0-9_\-\.]+)/([a-zA-Z0-9_\-\.]+)",
    re.IGNORECASE,
)

# Regex for known doc hosts or doc path keywords
DOC_HOST_REGEX = re.compile(
    r"(readthedocs\.io|github\.io|gitbook\.io|docs\.python\.org|docs\.|documentation\.)",
    re.IGNORECASE,
)

# Regex to find doc links inside Markdown/RST README content
README_DOC_LINK_REGEX = re.compile(
    r"https?://[^\s\)\>\"\'\`]+(?:\.readthedocs\.io|[a-zA-Z0-9_\-]+\.github\.io/[a-zA-Z0-9_\-]+|/docs?/[a-zA-Z0-9_\-/]+)",
    re.IGNORECASE,
)


class CollectionStatus(str, enum.Enum):
    DOCUMENTATION_FOUND = "documentation_found"
    NO_DOCUMENTATION_URL = "no_documentation_url"
    INVALID_URL = "invalid_url"
    COLLECTION_FAILED = "collection_failed"


@dataclass
class DocSymbol:
    """A documented Python API element (function, class, method, module)."""
    canonical_name: str
    symbol_type: str  # function, method, class, module, etc.
    package: str
    url: str
    source_inventory: str


@dataclass
class PackageCollectionReport:
    """Per-package collection result and status tracker."""
    package_name: str
    package_version: str
    status: CollectionStatus
    status_reason: str
    resolved_documentation_url: Optional[str] = None
    repository_url: Optional[str] = None
    symbols_collected: int = 0
    collection_time: float = field(default_factory=time.time)


def is_valid_url(url: Optional[str]) -> bool:
    """Validate that a URL has a valid scheme and hostname, and isn't a placeholder."""
    if not url or not isinstance(url, str):
        return False

    cleaned = url.strip()
    if cleaned.lower() in INVALID_PLACEHOLDERS:
        return False

    try:
        parsed = urllib.parse.urlparse(cleaned)
        if parsed.scheme.lower() not in ("http", "https"):
            return False
        if not parsed.netloc or "." not in parsed.netloc:
            return False
        return True
    except Exception:
        return False


def normalize_project_urls(
    documentation_url: Optional[str],
    homepage_url: Optional[str],
    source_url: Optional[str],
    all_project_urls: Dict[str, str],
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Separate and classify URLs into: (doc_url, repo_url, homepage_url).
    Fixes cases where repo URL was entered as homepage, or doc URL was in all_project_urls.
    """
    candidates = {}

    # Gather all declared URLs
    if documentation_url:
        candidates["doc"] = documentation_url.strip()
    if source_url:
        candidates["source"] = source_url.strip()
    if homepage_url:
        candidates["home"] = homepage_url.strip()
    for k, v in all_project_urls.items():
        if isinstance(v, str) and is_valid_url(v):
            candidates[k.lower()] = v.strip()

    final_repo: Optional[str] = None
    final_doc: Optional[str] = None
    final_home: Optional[str] = None

    # Classify each candidate
    for label, url in candidates.items():
        if not is_valid_url(url):
            continue

        is_repo = bool(REPO_HOST_REGEX.search(url))
        is_doc = bool(DOC_HOST_REGEX.search(url)) or any(
            w in label for w in ("doc", "api", "reference", "manual", "guide")
        )

        if is_repo and not final_repo:
            final_repo = url
        elif is_doc and not final_doc and not is_repo:
            final_doc = url
        elif not final_home and not is_repo:
            final_home = url

    # If documentation_url was already valid and explicit, prioritize it
    if is_valid_url(documentation_url) and not REPO_HOST_REGEX.search(documentation_url):
        final_doc = documentation_url.strip()

    # If source_url was valid, prioritize it
    if is_valid_url(source_url) and REPO_HOST_REGEX.search(source_url):
        final_repo = source_url.strip()

    return final_doc, final_repo, final_home


def fetch_github_fallback_doc_url(repo_url: str) -> Optional[str]:
    """
    Inspect a GitHub repository's README to discover documentation links.
    """
    match = REPO_HOST_REGEX.match(repo_url)
    if not match or "github.com" not in match.group(2).lower():
        return None

    owner, repo = match.group(3), match.group(4)
    # Strip any trailing .git or subpaths
    repo = repo.removesuffix(".git").split("/")[0]

    logger.info("  [Fallback] Inspecting GitHub README for %s/%s...", owner, repo)

    # Candidate raw README locations on GitHub
    readme_candidates = [
        f"https://raw.githubusercontent.com/{owner}/{repo}/HEAD/README.md",
        f"https://raw.githubusercontent.com/{owner}/{repo}/HEAD/README.rst",
        f"https://raw.githubusercontent.com/{owner}/{repo}/main/README.md",
        f"https://raw.githubusercontent.com/{owner}/{repo}/master/README.md",
    ]

    for raw_url in readme_candidates:
        try:
            req = urllib.request.Request(raw_url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=8) as resp:
                content = resp.read().decode("utf-8", errors="replace")

            # Search for doc links in content
            matches = README_DOC_LINK_REGEX.findall(content)
            for m in matches:
                clean_m = m.rstrip(".)]>")
                if is_valid_url(clean_m) and not clean_m.endswith(".svg") and not clean_m.endswith(".png"):
                    logger.info("  [Fallback] Discovered doc link in README: %s", clean_m)
                    return clean_m

            break  # Found README, no doc link found inside
        except urllib.error.HTTPError as err:
            if err.code in (404, 400):
                continue
            break
        except Exception:
            break

    # Also check if GitHub Pages exists: https://<owner>.github.io/<repo>/
    pages_url = f"https://{owner.lower()}.github.io/{repo}/"
    try:
        req = urllib.request.Request(pages_url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                logger.info("  [Fallback] Discovered active GitHub Pages at %s", pages_url)
                return pages_url
    except Exception:
        pass

    return None


def fetch_url_bytes(url: str, timeout: int = 10) -> bytes:
    """Fetch binary content from a URL with custom user-agent."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def probe_sphinx_inventory(base_doc_url: str) -> Tuple[Optional[bytes], Optional[str]]:
    """
    Probe candidate locations for Sphinx 'objects.inv' starting from base doc URL.
    Returns (raw_inventory_bytes, successful_url).
    """
    clean_base = base_doc_url.rstrip("/")

    candidates = [
        f"{clean_base}/objects.inv",
        f"{clean_base}/en/latest/objects.inv",
        f"{clean_base}/en/stable/objects.inv",
        f"{clean_base}/stable/objects.inv",
        f"{clean_base}/latest/objects.inv",
    ]

    for cand in candidates:
        try:
            data = fetch_url_bytes(cand, timeout=10)
            if b"# Sphinx inventory version" in data[:100]:
                return data, cand
        except urllib.error.HTTPError as err:
            if err.code == 404:
                continue
            logger.debug("HTTP %d fetching %s", err.code, cand)
        except Exception as exc:
            logger.debug("Error probing %s: %s", cand, exc)

    return None, None


def parse_sphinx_inventory(
    inv_bytes: bytes,
    inv_url: str,
    package_name: str,
    target_types: Optional[Set[str]] = None,
) -> List[DocSymbol]:
    """
    Decompresses and parses a Sphinx v2 objects.inv file.
    Format per line after zlib:
    name domain:role priority uri dispname
    """
    symbols: List[DocSymbol] = []
    base_url = inv_url.rsplit("/", 1)[0] + "/"

    # Find where zlib compressed stream starts
    marker = b"The remainder of this file is compressed using zlib.\n"
    parts = inv_bytes.split(marker, 1)
    if len(parts) < 2:
        # Fallback to carriage return newline
        marker_crlf = b"The remainder of this file is compressed using zlib.\r\n"
        parts = inv_bytes.split(marker_crlf, 1)
        if len(parts) < 2:
            raise ValueError("Invalid Sphinx inventory format: missing zlib header marker.")

    compressed_body = parts[1]
    decompressed = zlib.decompress(compressed_body).decode("utf-8", errors="replace")

    # Mapping of domain:role to standard symbol type
    role_mapping = {
        "py:function": "function",
        "py:method": "method",
        "py:class": "class",
        "py:module": "module",
        "py:exception": "exception",
        "py:data": "data",
        "py:attribute": "attribute",
    }

    if target_types is None:
        target_types = {"function", "method", "class", "module"}

    for line in decompressed.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # Format: <name> <domain:role> <priority> <uri> <dispname>
        tokens = line.split(maxsplit=4)
        if len(tokens) < 4:
            continue

        sym_name, role, _prio, uri = tokens[0], tokens[1], tokens[2], tokens[3]

        symbol_type = role_mapping.get(role)
        if not symbol_type or symbol_type not in target_types:
            continue

        # Sphinx uri shorthand: '$' at the end means substitute the symbol name
        if uri.endswith("$"):
            uri = uri[:-1] + sym_name

        full_url = urllib.parse.urljoin(base_url, uri)

        symbols.append(
            DocSymbol(
                canonical_name=sym_name,
                symbol_type=symbol_type,
                package=package_name,
                url=full_url,
                source_inventory=inv_url,
            )
        )

    return symbols


def collect_package_documentation(
    pkg: Dict[str, Any],
    symbol_types: Set[str],
) -> Tuple[PackageCollectionReport, List[DocSymbol]]:
    """
    Process a single package record:
    1. Validate and normalize URLs.
    2. Fallback to repository README if doc URL missing.
    3. Locate & parse Sphinx objects.inv.
    4. Return report and discovered DocSymbols.
    """
    pkg_name = pkg.get("name", "unknown")
    pkg_version = pkg.get("version", "unknown")
    raw_doc_url = pkg.get("documentation_url")
    raw_home_url = pkg.get("homepage_url")
    raw_source_url = pkg.get("source_url")
    all_urls = pkg.get("all_project_urls") or {}

    # Step 1: Normalize & validate declared URLs
    doc_url, repo_url, _home_url = normalize_project_urls(
        raw_doc_url, raw_home_url, raw_source_url, all_urls
    )

    # Check for invalid URL placeholder
    if raw_doc_url and not is_valid_url(raw_doc_url):
        logger.warning("[%s] Rejected invalid documentation URL placeholder: %r", pkg_name, raw_doc_url)
        return (
            PackageCollectionReport(
                package_name=pkg_name,
                package_version=pkg_version,
                status=CollectionStatus.INVALID_URL,
                status_reason=f"Declared URL '{raw_doc_url}' is a placeholder or malformed",
                resolved_documentation_url=None,
                repository_url=repo_url,
            ),
            [],
        )

    # Step 2: Fallback discovery if no documentation URL declared
    if not doc_url and repo_url:
        doc_url = fetch_github_fallback_doc_url(repo_url)

    # If still no documentation URL discovered
    if not doc_url:
        logger.info("[%s] No documentation URL declared or discovered.", pkg_name)
        return (
            PackageCollectionReport(
                package_name=pkg_name,
                package_version=pkg_version,
                status=CollectionStatus.NO_DOCUMENTATION_URL,
                status_reason="No documentation URL found in PyPI metadata or repository README",
                resolved_documentation_url=None,
                repository_url=repo_url,
            ),
            [],
        )

    # Step 3: Probe Sphinx objects.inv
    logger.info("[%s] Probing Sphinx inventory at %s ...", pkg_name, doc_url)
    inv_bytes, inv_url = probe_sphinx_inventory(doc_url)

    if not inv_bytes or not inv_url:
        # Crucial requirement: Never treat a failed collection as proof that a package has no documentation!
        logger.warning(
            "[%s] Collection failed: Could not locate Sphinx objects.inv at %s (may use non-Sphinx docs or require JS)",
            pkg_name,
            doc_url,
        )
        return (
            PackageCollectionReport(
                package_name=pkg_name,
                package_version=pkg_version,
                status=CollectionStatus.COLLECTION_FAILED,
                status_reason=f"Documentation URL exists ({doc_url}) but Sphinx objects.inv was not found or unreachable",
                resolved_documentation_url=doc_url,
                repository_url=repo_url,
            ),
            [],
        )

    # Step 4: Parse inventory symbols
    try:
        symbols = parse_sphinx_inventory(inv_bytes, inv_url, pkg_name, target_types=symbol_types)
        logger.info("[%s] Extracted %d symbols from %s", pkg_name, len(symbols), inv_url)
        return (
            PackageCollectionReport(
                package_name=pkg_name,
                package_version=pkg_version,
                status=CollectionStatus.DOCUMENTATION_FOUND,
                status_reason="Successfully parsed Sphinx inventory",
                resolved_documentation_url=doc_url,
                repository_url=repo_url,
                symbols_collected=len(symbols),
            ),
            symbols,
        )
    except Exception as exc:
        logger.warning("[%s] Failed parsing inventory at %s: %s", pkg_name, inv_url, exc)
        return (
            PackageCollectionReport(
                package_name=pkg_name,
                package_version=pkg_version,
                status=CollectionStatus.COLLECTION_FAILED,
                status_reason=f"Failed parsing inventory: {exc}",
                resolved_documentation_url=doc_url,
                repository_url=repo_url,
            ),
            [],
        )


def run_documentation_collection(
    input_file: Path,
    output_file: Path,
    symbol_types: Set[str],
    delay_seconds: float = 0.2,
) -> None:
    """Main orchestration pipeline for reading packages and collecting documentation."""
    if not input_file.exists():
        logger.error("Input file not found: %s. Run discover_packages.py first!", input_file)
        sys.exit(1)

    with open(input_file, "r", encoding="utf-8") as f:
        packages = json.load(f)

    logger.info("Loaded %d packages from %s", len(packages), input_file)

    reports: List[PackageCollectionReport] = []
    # Deduplication map keyed by (package_name, canonical_name)
    deduped_symbols: Dict[Tuple[str, str], DocSymbol] = {}

    for idx, pkg in enumerate(packages, start=1):
        name = pkg.get("name", f"item_{idx}")
        logger.info("--- [%d/%d] Processing package '%s' ---", idx, len(packages), name)

        report, symbols = collect_package_documentation(pkg, symbol_types)
        reports.append(report)

        for sym in symbols:
            key = (sym.package, sym.canonical_name)
            if key not in deduped_symbols:
                deduped_symbols[key] = sym

        if delay_seconds > 0 and idx < len(packages):
            time.sleep(delay_seconds)

    # Compute statistics
    status_counts = {}
    for r in reports:
        status_counts[r.status.value] = status_counts.get(r.status.value, 0) + 1

    logger.info("=== Documentation Collection Summary ===")
    for status, count in status_counts.items():
        logger.info("  %s: %d", status, count)
    logger.info("Total unique API symbols collected: %d", len(deduped_symbols))

    # Also build a lookup-ready dictionary compatible with learn_functions.py DOCS
    lookup_dict = {
        sym.canonical_name: (sym.package, sym.url)
        for sym in deduped_symbols.values()
    }

    output_payload = {
        "metadata": {
            "total_packages_processed": len(packages),
            "status_summary": status_counts,
            "total_symbols_collected": len(deduped_symbols),
        },
        "package_reports": [asdict(r) for r in reports],
        "symbols": [asdict(s) for s in deduped_symbols.values()],
        "learn_functions_docs_sample": dict(list(lookup_dict.items())[:25]),
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    logger.info("Saved collection results and symbols to %s", output_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect API documentation and Sphinx inventories for discovered packages."
    )
    parser.add_argument(
        "-i", "--input",
        type=Path,
        default=Path("data/discovered_packages.json"),
        help="Input JSON file containing discovered package records (default: data/discovered_packages.json).",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("data/collected_documentation.json"),
        help="Output JSON file for documentation symbols and reports (default: data/collected_documentation.json).",
    )
    parser.add_argument(
        "--symbol-types",
        nargs="+",
        default=["function", "method", "class", "module"],
        help="Types of symbols to extract (default: function method class module).",
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
    symbol_types = set(args.symbol_types)
    run_documentation_collection(
        input_file=args.input,
        output_file=args.output,
        symbol_types=symbol_types,
        delay_seconds=args.delay,
    )


if __name__ == "__main__":
    main()
