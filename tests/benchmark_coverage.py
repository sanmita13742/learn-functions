"""
benchmark_coverage.py

Benchmarking and Coverage Evaluation Suite for learn_functions.
Tests 100 representative code snippets across standard libraries, PyPI packages,
and edge cases to calculate:

Coverage = (Documented Calls Resolved Correctly / Documented Calls Tested) * 100%

Also measures:
- Exact symbol resolution accuracy
- False match rate (attributing a call to the wrong symbol or library)
- URL validity rate
- Resolution latency per call
"""

import argparse
import json
import logging
import sys
import time
import urllib.parse
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import learn_functions as lf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmark")


@dataclass
class TestCase:
    """A benchmark test case with code snippet and expected resolution."""
    case_id: int
    category: str
    description: str
    code: str
    expected_symbol: Optional[str] = None
    expected_library: Optional[str] = None
    expected_status: str = "found"  # 'found', 'unresolved', 'not_indexed', 'candidates'


@dataclass
class CaseResult:
    """Result of running a single benchmark test case."""
    case_id: int
    category: str
    description: str
    expected_symbol: Optional[str]
    expected_library: Optional[str]
    expected_status: str
    actual_symbol: Optional[str] = None
    actual_library: Optional[str] = None
    actual_status: Optional[str] = None
    actual_url: Optional[str] = None
    is_correct: bool = False
    is_false_match: bool = False
    is_url_valid: bool = False
    latency_ms: float = 0.0
    failure_reason: Optional[str] = None


def generate_benchmark_suite() -> List[TestCase]:
    """Generate 100 representative code snippets."""
    cases: List[TestCase] = []

    def add(category: str, desc: str, code: str, exp_sym: Optional[str], exp_lib: Optional[str], exp_st: str = "found"):
        cases.append(TestCase(
            case_id=len(cases) + 1,
            category=category,
            description=desc,
            code=code,
            expected_symbol=exp_sym,
            expected_library=exp_lib,
            expected_status=exp_st,
        ))

    # ---------------------------------------------------------
    # Core User-Requested Common Calls (Cases 1 - 5)
    # ---------------------------------------------------------
    add("Common", "np.reshape()", "import numpy as np\narr = np.reshape(data, (2, 3))", "numpy.reshape", "NumPy")
    add("Common", "pd.read_csv()", "import pandas as pd\ndf = pd.read_csv('data.csv')", "pandas.read_csv", "pandas")
    add("Common", "plt.plot()", "import matplotlib.pyplot as plt\nplt.plot([1, 2], [3, 4])", "matplotlib.pyplot.plot", "matplotlib")
    add("Common", "torch.tensor()", "import torch\nt = torch.tensor([1, 2, 3])", "torch.tensor", "PyTorch")
    add("Common", "requests.get()", "import requests\nr = requests.get('https://example.com')", "requests.get", "requests")

    # ---------------------------------------------------------
    # NumPy (Cases 6 - 20)
    # ---------------------------------------------------------
    add("NumPy", "np.array", "import numpy as np\nx = np.array([1, 2, 3])", "numpy.array", "NumPy")
    add("NumPy", "np.zeros", "import numpy as np\nx = np.zeros((3, 3))", "numpy.zeros", "NumPy")
    add("NumPy", "np.ones", "import numpy as np\nx = np.ones(5)", "numpy.ones", "NumPy")
    add("NumPy", "np.arange", "import numpy as np\nx = np.arange(10)", "numpy.arange", "NumPy")
    add("NumPy", "np.linspace", "import numpy as np\nx = np.linspace(0, 1, 100)", "numpy.linspace", "NumPy")
    add("NumPy", "np.mean", "import numpy as np\nx = np.mean(data)", "numpy.mean", "NumPy")
    add("NumPy", "np.dot", "import numpy as np\nz = np.dot(a, b)", "numpy.dot", "NumPy")
    add("NumPy", "np.concatenate", "import numpy as np\nz = np.concatenate([a, b])", "numpy.concatenate", "NumPy")
    add("NumPy", "np.argmax", "import numpy as np\ni = np.argmax(arr)", "numpy.argmax", "NumPy")
    add("NumPy", "np.clip", "import numpy as np\nx = np.clip(arr, 0, 1)", "numpy.clip", "NumPy")
    add("NumPy", "np.where", "import numpy as np\nx = np.where(arr > 0, 1, 0)", "numpy.where", "NumPy")
    add("NumPy", "np.unique", "import numpy as np\nu = np.unique(arr)", "numpy.unique", "NumPy")
    add("NumPy", "np.transpose", "import numpy as np\nt = np.transpose(arr)", "numpy.transpose", "NumPy")
    add("NumPy", "numpy.eye direct import", "import numpy\nx = numpy.eye(4)", "numpy.eye", "numpy")
    add("NumPy", "from numpy import pad", "from numpy import pad\np = pad(arr, 1)", "numpy.pad", "numpy")

    # ---------------------------------------------------------
    # Pandas (Cases 21 - 35)
    # ---------------------------------------------------------
    add("Pandas", "pd.DataFrame", "import pandas as pd\ndf = pd.DataFrame({'a': [1]})", "pandas.DataFrame", "pandas")
    add("Pandas", "pd.Series", "import pandas as pd\ns = pd.Series([1, 2, 3])", "pandas.Series", "pandas")
    add("Pandas", "pd.concat", "import pandas as pd\ndf = pd.concat([d1, d2])", "pandas.concat", "pandas")
    add("Pandas", "pd.merge", "import pandas as pd\ndf = pd.merge(d1, d2, on='id')", "pandas.merge", "pandas")
    add("Pandas", "pd.to_datetime", "import pandas as pd\ndt = pd.to_datetime('2026-01-01')", "pandas.to_datetime", "pandas")
    add("Pandas", "pd.to_numeric", "import pandas as pd\nn = pd.to_numeric(s)", "pandas.to_numeric", "pandas")
    add("Pandas", "pd.isna", "import pandas as pd\nm = pd.isna(df)", "pandas.isna", "pandas")
    add("Pandas", "pd.notna", "import pandas as pd\nm = pd.notna(df)", "pandas.notna", "pandas")
    add("Pandas", "pd.read_json", "import pandas as pd\ndf = pd.read_json('file.json')", "pandas.read_json", "pandas")
    add("Pandas", "pd.read_sql", "import pandas as pd\ndf = pd.read_sql('SELECT *', conn)", "pandas.read_sql", "pandas")
    add("Pandas", "pd.pivot_table", "import pandas as pd\nt = pd.pivot_table(df, values='x')", "pandas.pivot_table", "pandas")
    add("Pandas", "pd.date_range", "import pandas as pd\nr = pd.date_range('2026-01-01', periods=10)", "pandas.date_range", "pandas")
    add("Pandas", "pd.get_dummies", "import pandas as pd\nd = pd.get_dummies(df['cat'])", "pandas.get_dummies", "pandas")
    add("Pandas", "pd.qcut", "import pandas as pd\nq = pd.qcut(df['a'], q=4)", "pandas.qcut", "pandas")
    add("Pandas", "from pandas import read_parquet", "from pandas import read_parquet\ndf = read_parquet('data.pqt')", "pandas.read_parquet", "pandas")

    # ---------------------------------------------------------
    # Matplotlib (Cases 36 - 50)
    # ---------------------------------------------------------
    add("Matplotlib", "plt.figure", "import matplotlib.pyplot as plt\nfig = plt.figure()", "matplotlib.pyplot.figure", "matplotlib")
    add("Matplotlib", "plt.subplots", "import matplotlib.pyplot as plt\nfig, ax = plt.subplots(1, 2)", "matplotlib.pyplot.subplots", "matplotlib")
    add("Matplotlib", "plt.scatter", "import matplotlib.pyplot as plt\nplt.scatter(x, y)", "matplotlib.pyplot.scatter", "matplotlib")
    add("Matplotlib", "plt.bar", "import matplotlib.pyplot as plt\nplt.bar(names, vals)", "matplotlib.pyplot.bar", "matplotlib")
    add("Matplotlib", "plt.hist", "import matplotlib.pyplot as plt\nplt.hist(data, bins=20)", "matplotlib.pyplot.hist", "matplotlib")
    add("Matplotlib", "plt.title", "import matplotlib.pyplot as plt\nplt.title('My Title')", "matplotlib.pyplot.title", "matplotlib")
    add("Matplotlib", "plt.xlabel", "import matplotlib.pyplot as plt\nplt.xlabel('X')", "matplotlib.pyplot.xlabel", "matplotlib")
    add("Matplotlib", "plt.ylabel", "import matplotlib.pyplot as plt\nplt.ylabel('Y')", "matplotlib.pyplot.ylabel", "matplotlib")
    add("Matplotlib", "plt.legend", "import matplotlib.pyplot as plt\nplt.legend()", "matplotlib.pyplot.legend", "matplotlib")
    add("Matplotlib", "plt.grid", "import matplotlib.pyplot as plt\nplt.grid(True)", "matplotlib.pyplot.grid", "matplotlib")
    add("Matplotlib", "plt.show", "import matplotlib.pyplot as plt\nplt.show()", "matplotlib.pyplot.show", "matplotlib")
    add("Matplotlib", "plt.savefig", "import matplotlib.pyplot as plt\nplt.savefig('fig.png')", "matplotlib.pyplot.savefig", "matplotlib")
    add("Matplotlib", "plt.xlim", "import matplotlib.pyplot as plt\nplt.xlim(0, 10)", "matplotlib.pyplot.xlim", "matplotlib")
    add("Matplotlib", "plt.ylim", "import matplotlib.pyplot as plt\nplt.ylim(0, 10)", "matplotlib.pyplot.ylim", "matplotlib")
    add("Matplotlib", "plt.subplot", "import matplotlib.pyplot as plt\nplt.subplot(2, 1, 1)", "matplotlib.pyplot.subplot", "matplotlib")

    # ---------------------------------------------------------
    # Requests (Cases 51 - 62)
    # ---------------------------------------------------------
    add("Requests", "requests.post", "import requests\nr = requests.post('https://api', json={})", "requests.post", "requests")
    add("Requests", "requests.put", "import requests\nr = requests.put('https://api', data={})", "requests.put", "requests")
    add("Requests", "requests.delete", "import requests\nr = requests.delete('https://api')", "requests.delete", "requests")
    add("Requests", "requests.head", "import requests\nr = requests.head('https://api')", "requests.head", "requests")
    add("Requests", "requests.patch", "import requests\nr = requests.patch('https://api', json={})", "requests.patch", "requests")
    add("Requests", "requests.Session", "import requests\ns = requests.Session()", "requests.Session", "requests")
    add("Requests", "requests.request", "import requests\nr = requests.request('GET', url)", "requests.request", "requests")
    add("Requests", "from requests import get", "from requests import get\nr = get('https://api')", "requests.get", "requests")
    add("Requests", "from requests import post", "from requests import post\nr = post('https://api')", "requests.post", "requests")
    add("Requests", "from requests import Session", "from requests import Session\ns = Session()", "requests.Session", "requests")
    add("Requests", "alias req.get", "import requests as req\nr = req.get('https://api')", "requests.get", "requests")
    add("Requests", "alias req.post", "import requests as req\nr = req.post('https://api')", "requests.post", "requests")

    # ---------------------------------------------------------
    # Flask (Cases 63 - 72)
    # ---------------------------------------------------------
    add("Flask", "flask.Flask", "import flask\napp = flask.Flask(__name__)", "flask.Flask", "Flask")
    add("Flask", "flask.Blueprint", "import flask\nbp = flask.Blueprint('admin', __name__)", "flask.Blueprint", "Flask")
    add("Flask", "flask.render_template", "import flask\nhtml = flask.render_template('index.html')", "flask.render_template", "Flask")
    add("Flask", "flask.redirect", "import flask\nr = flask.redirect('/home')", "flask.redirect", "Flask")
    add("Flask", "flask.url_for", "import flask\nu = flask.url_for('index')", "flask.url_for", "Flask")
    add("Flask", "flask.abort", "import flask\nflask.abort(404)", "flask.abort", "Flask")
    add("Flask", "flask.flash", "import flask\nflask.flash('Logged in!')", "flask.flash", "Flask")
    add("Flask", "from flask import Flask", "from flask import Flask\napp = Flask(__name__)", "flask.Flask", "Flask")
    add("Flask", "from flask import Blueprint", "from flask import Blueprint\nbp = Blueprint('api', __name__)", "flask.Blueprint", "Flask")
    add("Flask", "from flask import render_template", "from flask import render_template\nh = render_template('a.html')", "flask.render_template", "Flask")

    # ---------------------------------------------------------
    # SciPy (Cases 73 - 82)
    # ---------------------------------------------------------
    add("SciPy", "scipy.sparse.csr_matrix", "import scipy.sparse\nm = scipy.sparse.csr_matrix([1, 2])", "scipy.sparse.csr_matrix", "scipy")
    add("SciPy", "scipy.optimize.minimize", "import scipy.optimize\nres = scipy.optimize.minimize(cost_func, x0)", "scipy.optimize.minimize", "scipy")
    add("SciPy", "scipy.spatial.distance.cdist", "import scipy.spatial.distance\nd = scipy.spatial.distance.cdist(xa, xb)", "scipy.spatial.distance.cdist", "scipy")
    add("SciPy", "scipy.linalg.inv", "import scipy.linalg\ninv = scipy.linalg.inv(matrix)", "scipy.linalg.inv", "scipy")
    add("SciPy", "scipy.linalg.eig", "import scipy.linalg\nw, v = scipy.linalg.eig(matrix)", "scipy.linalg.eig", "scipy")
    add("SciPy", "from scipy.sparse import csc_matrix", "from scipy.sparse import csc_matrix\nm = csc_matrix([1, 2])", "scipy.sparse.csc_matrix", "scipy")
    add("SciPy", "from scipy.optimize import minimize", "from scipy.optimize import minimize\nr = minimize(f, 0)", "scipy.optimize.minimize", "scipy")
    add("SciPy", "from scipy.linalg import det", "from scipy.linalg import det\nd = det(m)", "scipy.linalg.det", "scipy")
    add("SciPy", "from scipy.spatial import KDTree", "from scipy.spatial import KDTree\nt = KDTree(data)", "scipy.spatial.KDTree", "scipy")
    add("SciPy", "scipy.integrate.quad", "import scipy.integrate\nres = scipy.integrate.quad(f, 0, 1)", "scipy.integrate.quad", "scipy")

    # ---------------------------------------------------------
    # PyTorch & Builtins (Cases 83 - 92)
    # ---------------------------------------------------------
    add("PyTorch", "torch.utils.data.DataLoader", "import torch.utils.data\ndl = torch.utils.data.DataLoader(ds)", "torch.utils.data.DataLoader", "PyTorch")
    add("PyTorch", "from torchvision.datasets import CIFAR10", "from torchvision.datasets import CIFAR10\nds = CIFAR10('./data')", "torchvision.datasets.CIFAR10", "torchvision")
    add("Builtins", "len()", "x = len([1, 2, 3])", "builtins.len", "Python")
    add("Builtins", "print()", "print('hello')", "builtins.print", "Python")
    add("Builtins", "range()", "for i in range(5): pass", "builtins.range", "Python")
    add("Builtins", "enumerate()", "for i, v in enumerate([1]): pass", "builtins.enumerate", "Python")
    add("Builtins", "zip()", "z = zip([1], [2])", "builtins.zip", "Python")
    add("Builtins", "sum()", "s = sum([1, 2, 3])", "builtins.sum", "Python")
    add("Builtins", "max()", "m = max([1, 2, 3])", "builtins.max", "Python")
    add("Builtins", "min()", "m = min([1, 2, 3])", "builtins.min", "Python")

    # ---------------------------------------------------------
    # Negative Cases & Edge Cases (Cases 93 - 100)
    # Must correctly NOT report false matches!
    # ---------------------------------------------------------
    add("Edge Cases", "Unimported alias np.reshape", "np.reshape(x)", "np.reshape", None, exp_st="unresolved")
    add("Edge Cases", "Generic method images.flatten", "images.flatten()", "images.flatten", None, exp_st="unresolved")
    add("Edge Cases", "Custom user function", "def compute(): pass\ncompute()", "compute", None, exp_st="not_indexed")
    add("Edge Cases", "Unknown 3rd party lib", "import foobar\nfoobar.run()", "foobar.run", None, exp_st="not_indexed")
    add("Edge Cases", "Reassigned alias", "import numpy as np\nnp = 'text'\nnp.reshape(x)", "numpy.reshape", "numpy")
    add("Edge Cases", "Timeit magic call", "import numpy as np\n%timeit np.reshape(x, (2, 2))", "numpy.reshape", "numpy")
    add("Edge Cases", "Import without calls", "import numpy as np\nimport pandas as pd", None, None, exp_st="empty")
    add("Edge Cases", "Subscript receiver call", "items[0].reshape(-1)", "reshape", None, exp_st="candidates")

    return cases


def run_benchmark(cases: List[TestCase]) -> List[CaseResult]:
    """Execute benchmark test suite against learn_functions."""
    results: List[CaseResult] = []

    for case in cases:
        t0 = time.perf_counter()
        analysis = lf.analyze(case.code)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        if case.expected_status == "empty":
            is_correct = len(analysis.results) == 0
            results.append(CaseResult(
                case_id=case.case_id,
                category=case.category,
                description=case.description,
                expected_symbol=None,
                expected_library=None,
                expected_status=case.expected_status,
                actual_symbol=None,
                actual_status="empty" if is_correct else "has_results",
                is_correct=is_correct,
                is_false_match=not is_correct,
                is_url_valid=True,
                latency_ms=latency_ms,
            ))
            continue

        if not analysis.results:
            results.append(CaseResult(
                case_id=case.case_id,
                category=case.category,
                description=case.description,
                expected_symbol=case.expected_symbol,
                expected_library=case.expected_library,
                expected_status=case.expected_status,
                actual_symbol=None,
                actual_status="no_calls_detected",
                is_correct=False,
                is_false_match=False,
                is_url_valid=False,
                latency_ms=latency_ms,
                failure_reason="No call site was extracted from snippet",
            ))
            continue

        # Inspect the primary result
        res = analysis.results[0]
        actual_symbol = res.symbol
        actual_library = res.library
        actual_status = res.status
        actual_url = res.url

        # Check URL validity if present
        is_url_valid = True
        if actual_url:
            if not actual_url.startswith("https://") or " " in actual_url:
                is_url_valid = False
            else:
                parsed = urllib.parse.urlparse(actual_url)
                if not parsed.netloc:
                    is_url_valid = False

        # Determine correctness & false match
        is_correct = False
        is_false_match = False
        failure_reason = None

        if case.expected_status == "found":
            if actual_status == "found":
                # Check symbol & library match (case-insensitive library check)
                symbol_ok = (actual_symbol == case.expected_symbol)
                lib_ok = (
                    case.expected_library is None
                    or (actual_library is not None and actual_library.lower() == case.expected_library.lower())
                )
                if symbol_ok and lib_ok and is_url_valid and actual_url:
                    is_correct = True
                else:
                    if not symbol_ok or not lib_ok:
                        is_false_match = True
                        failure_reason = f"Mismatch: expected ({case.expected_symbol}, {case.expected_library}), got ({actual_symbol}, {actual_library})"
                    elif not is_url_valid or not actual_url:
                        failure_reason = f"Invalid URL returned: {actual_url}"
            else:
                failure_reason = f"Status was '{actual_status}' instead of 'found'"

        elif case.expected_status in ("unresolved", "not_indexed", "candidates"):
            if actual_status == case.expected_status:
                is_correct = True
            else:
                if actual_status == "found":
                    is_false_match = True
                    failure_reason = f"False match! Unexpectedly resolved as 'found' with {actual_symbol}"
                else:
                    failure_reason = f"Status was '{actual_status}', expected '{case.expected_status}'"

        results.append(CaseResult(
            case_id=case.case_id,
            category=case.category,
            description=case.description,
            expected_symbol=case.expected_symbol,
            expected_library=case.expected_library,
            expected_status=case.expected_status,
            actual_symbol=actual_symbol,
            actual_library=actual_library,
            actual_status=actual_status,
            actual_url=actual_url,
            is_correct=is_correct,
            is_false_match=is_false_match,
            is_url_valid=is_url_valid,
            latency_ms=latency_ms,
            failure_reason=failure_reason,
        ))

    return results


def print_benchmark_report(results: List[CaseResult]) -> None:
    """Print complete coverage, precision, and false-match report."""
    total_cases = len(results)
    correct_count = sum(1 for r in results if r.is_correct)
    false_match_count = sum(1 for r in results if r.is_false_match)
    avg_latency = sum(r.latency_ms for r in results) / total_cases if total_cases else 0.0

    # Specifically calculate user formula:
    # Coverage = (documented calls resolved correctly / documented calls tested) * 100%
    documented_cases = [r for r in results if r.expected_status == "found"]
    doc_total = len(documented_cases)
    doc_correct = sum(1 for r in documented_cases if r.is_correct)
    coverage_pct = (doc_correct / doc_total) * 100.0 if doc_total else 0.0

    overall_accuracy = (correct_count / total_cases) * 100.0
    false_match_rate = (false_match_count / total_cases) * 100.0

    print("\n" + "=" * 75)
    print("      LEARN-FUNCTIONS BENCHMARK & LOOKUP COVERAGE REPORT (100 SNIPPETS)")
    print("=" * 75)

    # 1. Five Common Calls Section
    print("\n1. FIVE CRITICAL COMMON CALLS TEST:")
    print("-" * 75)
    print(f"{'Call Tested':<22} {'Expected Target':<25} {'Status':<10} {'Result'}")
    print("-" * 75)
    for r in results[:5]:
        status_str = "PASS [200]" if r.is_correct else "FAIL"
        print(f"{r.description:<22} {str(r.expected_symbol):<25} {str(r.actual_status):<10} {status_str}")

    # 2. Category Breakdown Table
    categories: Dict[str, List[CaseResult]] = {}
    for r in results:
        categories.setdefault(r.category, []).append(r)

    print("\n2. CATEGORY BREAKDOWN:")
    print("-" * 75)
    print(f"{'Category':<16} {'Tested':<8} {'Correct':<9} {'False Match':<13} {'Category Coverage'}")
    print("-" * 75)
    for cat, cat_results in categories.items():
        c_tot = len(cat_results)
        c_cor = sum(1 for r in cat_results if r.is_correct)
        c_fm = sum(1 for r in cat_results if r.is_false_match)
        c_pct = (c_cor / c_tot) * 100.0
        print(f"{cat:<16} {c_tot:<8} {c_cor:<9} {c_fm:<13} {c_pct:>5.1f}%")

    print("-" * 75)

    # 3. Overall Summary & User Formula
    print("\n3. CORE COVERAGE & PRECISION METRICS:")
    print("-" * 75)
    print(f"  Formula: Coverage = (resolved correctly / calls tested) * 100%")
    print(f"  Documented Calls Tested      : {doc_total}")
    print(f"  Documented Calls Correct     : {doc_correct}")
    print(f"  --> LOOKUP COVERAGE          : {coverage_pct:.2f}%")
    print(f"  False Match Rate             : {false_match_rate:.2f}% ({false_match_count} false matches)")
    print(f"  Overall Suite Accuracy       : {overall_accuracy:.2f}% ({correct_count}/{total_cases})")
    print(f"  Average Resolution Latency   : {avg_latency:.2f} ms per call")
    print("=" * 75)

    # Failures list if any
    failures = [r for r in results if not r.is_correct]
    if failures:
        print(f"\nDiscrepancies ({len(failures)}):")
        for f in failures:
            print(f"  Case #{f.case_id} [{f.category}] {f.description}: {f.failure_reason}")
    else:
        print("\nAll 100 benchmark test snippets passed with 0 false matches!")

    print("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 100-snippet benchmark coverage suite.")
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Optional path to save JSON results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Explicitly ensure collected documentation corpus is loaded
    lf.load_documentation()
    cases = generate_benchmark_suite()
    logger.info("Executing benchmark with %d test cases...", len(cases))
    results = run_benchmark(cases)
    print_benchmark_report(results)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = [asdict(r) for r in results]
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info("Saved benchmark report to %s", args.output)


if __name__ == "__main__":
    main()
