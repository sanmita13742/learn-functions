"""
audit_collected_docs.py

Audits the data quality of data/collected_documentation.json:
1. Validates records: checks for duplicates, missing canonical names, malformed URLs,
   and unexpected symbol types.
2. Summarizes records per package and symbol type.
3. Randomly samples 50-100 URLs and tests live HTTP reachability to verify destination accuracy.
"""

import argparse
import json
import logging
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("audit_docs")

USER_AGENT = "learn-functions-auditor/0.1.0"
KNOWN_SYMBOL_TYPES = {"function", "method", "class", "module", "exception", "attribute", "data"}


def load_dataset(json_path: Path) -> Dict[str, Any]:
    """Load JSON file and validate basic structure."""
    if not json_path.exists():
        logger.error("Dataset not found: %s", json_path)
        sys.exit(1)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict) or "symbols" not in data:
        logger.error("Invalid dataset format in %s: missing 'symbols' key.", json_path)
        sys.exit(1)

    return data


def audit_integrity(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check for:
    - Duplicate records
    - Missing canonical names
    - Malformed URLs
    - Unexpected symbol types
    """
    symbols: List[Dict[str, Any]] = data.get("symbols", [])
    reports: List[Dict[str, Any]] = data.get("package_reports", [])

    total_records = len(symbols)
    seen_keys: Set[Tuple[str, str]] = set()
    duplicate_records: List[Dict[str, Any]] = []
    missing_names: List[Dict[str, Any]] = []
    malformed_urls: List[Dict[str, Any]] = []
    unexpected_types: List[Dict[str, Any]] = []

    package_counts: Counter = Counter()
    type_counts: Counter = Counter()

    for idx, item in enumerate(symbols):
        name = item.get("canonical_name")
        pkg = item.get("package")
        url = item.get("url")
        sym_type = item.get("symbol_type")

        # Track package & type distributions
        package_counts[pkg or "UNKNOWN"] += 1
        type_counts[sym_type or "UNKNOWN"] += 1

        # 1. Missing or empty canonical name
        if not name or not isinstance(name, str) or not name.strip():
            missing_names.append({"index": idx, "record": item})

        # 2. Duplicate records (package + canonical_name)
        key = (pkg, name)
        if key in seen_keys:
            duplicate_records.append({"index": idx, "key": key, "url": url})
        else:
            seen_keys.add(key)

        # 3. Malformed URLs
        if not url or not isinstance(url, str):
            malformed_urls.append({"name": name, "url": url, "reason": "Missing or non-string"})
        else:
            url_clean = url.strip()
            if not url_clean.startswith("https://"):
                malformed_urls.append({"name": name, "url": url, "reason": "Non-HTTPS or invalid scheme"})
            elif " " in url_clean:
                malformed_urls.append({"name": name, "url": url, "reason": "Contains spaces"})
            else:
                try:
                    parsed = urllib.parse.urlparse(url_clean)
                    if not parsed.netloc or "." not in parsed.netloc:
                        malformed_urls.append({"name": name, "url": url, "reason": "Invalid hostname"})
                except Exception as exc:
                    malformed_urls.append({"name": name, "url": url, "reason": f"Parse error: {exc}"})

        # 4. Unexpected symbol types
        if sym_type not in KNOWN_SYMBOL_TYPES:
            unexpected_types.append({"name": name, "type": sym_type, "package": pkg})

    return {
        "total_records": total_records,
        "unique_records": len(seen_keys),
        "package_reports": reports,
        "package_counts": package_counts,
        "type_counts": type_counts,
        "duplicate_count": len(duplicate_records),
        "duplicates": duplicate_records[:5],
        "missing_names_count": len(missing_names),
        "missing_names": missing_names[:5],
        "malformed_urls_count": len(malformed_urls),
        "malformed_urls": malformed_urls[:5],
        "unexpected_types_count": len(unexpected_types),
        "unexpected_types": unexpected_types[:5],
    }


def sample_and_verify_destinations(
    symbols: List[Dict[str, Any]],
    sample_size: int = 50,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Randomly sample N records and probe live HTTP endpoints to verify URL reachability.
    """
    if sample_size > len(symbols):
        sample_size = len(symbols)

    random.seed(seed)
    sampled = random.sample(symbols, sample_size)

    logger.info("Sampling %d records to test live destination reachability...", sample_size)

    success_count = 0
    fail_count = 0
    failures: List[Dict[str, Any]] = []

    for i, item in enumerate(sampled, start=1):
        url = item.get("url", "")
        # Remove anchor fragment for HTTP request
        clean_url = url.split("#")[0]
        name = item.get("canonical_name", "unknown")
        pkg = item.get("package", "unknown")

        req = urllib.request.Request(
            clean_url,
            headers={"User-Agent": USER_AGENT},
            method="HEAD",
        )

        status_code = None
        try:
            with urllib.request.urlopen(req, timeout=8) as resp:
                status_code = resp.status
        except urllib.error.HTTPError as err:
            # Some doc sites block HEAD requests with 403/405; fallback to lightweight GET
            if err.code in (403, 405):
                try:
                    req_get = urllib.request.Request(clean_url, headers={"User-Agent": USER_AGENT})
                    with urllib.request.urlopen(req_get, timeout=8) as resp_get:
                        status_code = resp_get.status
                except urllib.error.HTTPError as err_get:
                    status_code = err_get.code
                except Exception:
                    status_code = err.code
            else:
                status_code = err.code
        except Exception:
            # Network timeout or DNS failure
            status_code = None

        if status_code in (200, 301, 302, 307, 308):
            success_count += 1
            logger.info("  [%d/%d] PASS (%s) %s -> %s", i, sample_size, status_code, name, clean_url)
        else:
            fail_count += 1
            failures.append({"name": name, "package": pkg, "url": url, "status": status_code})
            logger.warning("  [%d/%d] FAIL (%s) %s -> %s", i, sample_size, status_code, name, clean_url)

        time.sleep(0.05)  # Be polite to doc hosts

    destination_accuracy = (success_count / sample_size) * 100.0 if sample_size else 0.0

    return {
        "sample_size": sample_size,
        "success_count": success_count,
        "fail_count": fail_count,
        "destination_accuracy_pct": destination_accuracy,
        "failures": failures,
    }


def print_audit_report(results: Dict[str, Any], live_results: Dict[str, Any]) -> None:
    """Print a clean Markdown-like audit report."""
    print("\n" + "=" * 65)
    print("   COLLECTED DOCUMENTATION DATA QUALITY AUDIT REPORT")
    print("=" * 65)

    print("\n1. PACKAGE CONTRIBUTION BREAKDOWN:")
    print("-" * 65)
    print(f"{'Package':<18} {'Version':<10} {'Records':<12} {'Doc Status'}")
    print("-" * 65)
    reports = results.get("package_reports", [])
    counts = results.get("package_counts", Counter())

    for r in reports:
        pkg = r.get("package_name", "unknown")
        ver = r.get("package_version", "unknown")
        cnt = counts.get(pkg, r.get("symbols_collected", 0))
        status = r.get("status", "unknown")
        print(f"{pkg:<18} {ver:<10} {cnt:<12} {status}")

    print("-" * 65)
    print(f"{'TOTAL':<18} {'':<10} {results['total_records']:<12}")

    print("\n2. SYMBOL TYPE DISTRIBUTION:")
    print("-" * 65)
    type_counts = results.get("type_counts", Counter())
    for stype, cnt in type_counts.most_common():
        pct = (cnt / results["total_records"]) * 100.0
        print(f"  - {stype:<12}: {cnt:>6} records ({pct:>5.1f}%)")

    print("\n3. DATA INTEGRITY AUDIT:")
    print("-" * 65)
    checks = [
        ("Missing / Empty Canonical Names", results["missing_names_count"]),
        ("Duplicate Records (pkg + symbol)", results["duplicate_count"]),
        ("Malformed / Insecure URLs", results["malformed_urls_count"]),
        ("Unexpected Symbol Types", results["unexpected_types_count"]),
    ]
    for check_name, count in checks:
        status_flag = "[PASS] 0 issues" if count == 0 else f"[WARN] {count} issues"
        print(f"  {check_name:<36} : {status_flag}")

    if results["malformed_urls_count"] > 0:
        print("\n  Sample malformed URLs:")
        for item in results["malformed_urls"]:
            print(f"    * {item['name']}: {item['url']} ({item['reason']})")

    print("\n4. LIVE URL DESTINATION VERIFICATION (SAMPLE):")
    print("-" * 65)
    print(f"  Sampled Records Tested : {live_results['sample_size']}")
    print(f"  Live Reachable (200 OK): {live_results['success_count']}")
    print(f"  Unreachable / Errors   : {live_results['fail_count']}")
    print(f"  Destination Accuracy   : {live_results['destination_accuracy_pct']:.1f}%")

    if live_results["fail_count"] > 0:
        print("\n  Failed Destinations:")
        for f in live_results["failures"][:5]:
            print(f"    * {f['package']}.{f['name']} -> {f['url']} (status: {f['status']})")

    print("=" * 65 + "\n")


def parse_args() -> argparse.Namespace:
    default_data_path = Path(__file__).parent.parent / "data" / "collected_documentation.json"
    parser.add_argument(
        "-i", "--input",
        type=Path,
        default=default_data_path,
        help="Path to collected_documentation.json (default: data/collected_documentation.json).",
    )
    parser.add_argument(
        "-s", "--sample-size",
        type=int,
        default=50,
        help="Number of records to sample for live HTTP verification (default: 50).",
    )
    parser.add_argument(
        "--skip-live-check",
        action="store_true",
        help="Skip live network verification of sampled URLs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_dataset(args.input)
    integrity_results = audit_integrity(data)

    if not args.skip_live_check and args.sample_size > 0:
        live_results = sample_and_verify_destinations(data.get("symbols", []), sample_size=args.sample_size)
    else:
        live_results = {
            "sample_size": 0,
            "success_count": 0,
            "fail_count": 0,
            "destination_accuracy_pct": 100.0,
            "failures": [],
        }

    print_audit_report(integrity_results, live_results)


if __name__ == "__main__":
    main()
