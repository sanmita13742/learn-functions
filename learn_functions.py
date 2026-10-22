"""
learn_functions.py

Analyzes Python code snippets (inside Jupyter notebooks or scripts), detects
function/method calls via AST, and looks up official documentation links
for recognized API symbols.

Integrates with data/collected_documentation.json to provide comprehensive
documentation links across libraries discovered and indexed from PyPI.
"""

import ast
import html
import json
import logging
import re
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from IPython.display import HTML, display
except ImportError:
    HTML = None
    display = None

__version__ = "0.1.0"

logger = logging.getLogger("learn_functions")

MAX_ROWS = 50

# Core baseline documentation dictionary
BASELINE_DOCS: Dict[str, Tuple[str, str]] = {
    "builtins.len": (
        "Python",
        "https://docs.python.org/3/library/functions.html#len",
    ),
    "builtins.print": (
        "Python",
        "https://docs.python.org/3/library/functions.html#print",
    ),
    "builtins.range": (
        "Python",
        "https://docs.python.org/3/library/functions.html#func-range",
    ),
    "builtins.enumerate": (
        "Python",
        "https://docs.python.org/3/library/functions.html#enumerate",
    ),
    "builtins.zip": (
        "Python",
        "https://docs.python.org/3/library/functions.html#zip",
    ),
    "builtins.sum": (
        "Python",
        "https://docs.python.org/3/library/functions.html#sum",
    ),
    "builtins.max": (
        "Python",
        "https://docs.python.org/3/library/functions.html#max",
    ),
    "builtins.min": (
        "Python",
        "https://docs.python.org/3/library/functions.html#min",
    ),
    "builtins.list": (
        "Python",
        "https://docs.python.org/3/library/functions.html#func-list",
    ),
    "builtins.dict": (
        "Python",
        "https://docs.python.org/3/library/functions.html#func-dict",
    ),
    "builtins.set": (
        "Python",
        "https://docs.python.org/3/library/functions.html#func-set",
    ),
    "builtins.exec": (
        "Python",
        "https://docs.python.org/3/library/functions.html#exec",
    ),
    "numpy.reshape": (
        "NumPy",
        "https://numpy.org/doc/stable/reference/generated/numpy.reshape.html",
    ),
    "numpy.ndarray.flatten": (
        "NumPy",
        "https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flatten.html",
    ),
    "torchvision.datasets.CIFAR10": (
        "torchvision",
        "https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html",
    ),
    "torch.utils.data.DataLoader": (
        "PyTorch",
        "https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader",
    ),
    "torch.tensor": (
        "PyTorch",
        "https://docs.pytorch.org/docs/stable/generated/torch.tensor.html",
    ),
}

DOCS: Dict[str, Tuple[str, str]] = dict(BASELINE_DOCS)

DEFAULT_COLLECTED_DOCS_PATH = Path(__file__).parent / "data" / "collected_documentation.json"


def load_documentation(json_path: Path = DEFAULT_COLLECTED_DOCS_PATH) -> int:
    """
    Load symbols from collected_documentation.json into DOCS.
    Returns the number of new symbols loaded.
    """
    global DOCS
    if not json_path.exists():
        return 0

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        symbols = data.get("symbols", [])
        loaded_count = 0
        for item in symbols:
            canonical_name = item.get("canonical_name")
            package = item.get("package")
            url = item.get("url")
            if canonical_name and package and url:
                # Validate URL before adding
                if url.startswith("https://") and " " not in url:
                    DOCS[canonical_name] = (package, url)
                    loaded_count += 1
        return loaded_count
    except Exception as exc:
        logger.warning("Could not load documentation from %s: %s", json_path, exc)
        return 0


# Load collected documentation automatically into the index
load_documentation()


@dataclass
class Candidate:
    """A possible documentation match for an unresolved or bare symbol."""
    symbol: str
    library: str
    url: str


@dataclass
class Result:
    """Analysis result for a single identified call site."""
    symbol: str
    status: str  # 'found', 'candidates', 'unresolved', 'not_indexed', 'empty'
    library: Optional[str] = None
    url: Optional[str] = None
    candidates: Tuple[Candidate, ...] = ()
    lines: Tuple[int, ...] = ()
    note: Optional[str] = None
    categories: Tuple[str, ...] = ()


@dataclass
class Analysis:
    """Outcome of analyzing a code snippet."""
    results: List[Result] = field(default_factory=list)
    imports: Dict[str, str] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class CallSite:
    """An AST function/method invocation located at a specific line."""
    name: str
    line: int


# Package-to-category metadata mapping
DEFAULT_PACKAGE_CATEGORIES: Dict[str, Set[str]] = {
    "numpy": {"aiml", "data", "math"},
    "pandas": {"data", "analytics"},
    "scipy": {"aiml", "data", "math", "scientific"},
    "matplotlib": {"visualization", "viz", "data"},
    "torch": {"aiml", "deep_learning"},
    "torchvision": {"aiml", "computer_vision"},
    "requests": {"web", "networking"},
    "flask": {"web", "backend", "api"},
    "builtins": {"core", "builtins"},
    "python": {"core", "builtins"},
}


def get_symbol_categories(symbol: str, library: Optional[str] = None) -> Tuple[str, ...]:
    """Retrieve all categories associated with a symbol or its library/package."""
    cats: Set[str] = set()
    root_pkg = symbol.split(".")[0].lower() if symbol else ""
    lib_name = library.lower() if library else ""

    if root_pkg in DEFAULT_PACKAGE_CATEGORIES:
        cats.update(DEFAULT_PACKAGE_CATEGORIES[root_pkg])
    if lib_name in DEFAULT_PACKAGE_CATEGORIES:
        cats.update(DEFAULT_PACKAGE_CATEGORIES[lib_name])

    # Prefix heuristics
    if root_pkg in ("torch", "torchvision") or symbol.startswith("torch.") or symbol.startswith("torchvision."):
        cats.add("aiml")
    if symbol.startswith("scipy.optimize") or symbol.startswith("scipy.stats"):
        cats.add("aiml")

    return tuple(sorted(cats))


def _parse_ambient_aliases(aliases_input: Optional[Any]) -> Dict[str, str]:
    """Parse external aliases passed either as an import string or as a dict."""
    if not aliases_input:
        return {}
    if isinstance(aliases_input, dict):
        return {str(k): str(v) for k, v in aliases_input.items()}
    if isinstance(aliases_input, str):
        try:
            tree = ast.parse(aliases_input)
            return get_import_aliases(tree)
        except Exception:
            return {}
    return {}


def _filter_results(
    results: List[Result],
    ignore: Optional[Any] = None,
    category: Optional[Any] = None,
) -> List[Result]:
    """Filter results by ignored packages/libraries and/or categories."""
    if not ignore and not category:
        return results

    ignore_set: Set[str] = set()
    if ignore:
        if isinstance(ignore, str):
            ignore_set = {ignore.strip().lower()}
        else:
            ignore_set = {str(x).strip().lower() for x in ignore}

    cat_set: Set[str] = set()
    if category:
        if isinstance(category, str):
            cat_set = {category.strip().lower()}
        else:
            cat_set = {str(x).strip().lower() for x in category}

    filtered: List[Result] = []
    for r in results:
        root_pkg = r.symbol.split(".")[0].lower()
        lib_name = r.library.lower() if r.library else ""

        # Ignore filter
        if ignore_set and (root_pkg in ignore_set or lib_name in ignore_set):
            continue

        # Category filter
        if cat_set:
            r_cats = set(r.categories) or set(get_symbol_categories(r.symbol, r.library))
            if not r_cats.intersection(cat_set):
                continue

        filtered.append(r)

    return filtered


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

def get_name(node: ast.AST) -> Optional[str]:
    """Convert an AST Name or Attribute into a dotted identifier string."""
    if isinstance(node, ast.Name):
        return node.id

    if isinstance(node, ast.Attribute):
        parent = get_name(node.value)
        if parent:
            return f"{parent}.{node.attr}"
        # If receiver is a subscript (e.g. items[0].reshape(-1)), extract attribute
        if isinstance(node.value, ast.Subscript):
            return node.attr
        return None

    return None


def get_import_aliases(tree: ast.AST) -> Dict[str, str]:
    """Extract explicit import mappings from an AST."""
    aliases: Dict[str, str] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                # If dotted import e.g. import torch.utils.data
                # the top-level module is bound to local scope
                top_level = item.name.split(".")[0]
                local_name = item.asname or top_level
                full_name = item.name if item.asname else top_level
                aliases[local_name] = full_name

        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                # Relative import: do not resolve
                continue

            module = node.module or ""
            for item in node.names:
                if item.name == "*":
                    continue
                local_name = item.asname or item.name
                full_name = f"{module}.{item.name}" if module else item.name
                aliases[local_name] = full_name

    return aliases


def _find_assigned_names(tree: ast.AST) -> Set[str]:
    """Find all variable names assigned/reassigned in the snippet."""
    assigned = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assigned.add(target.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            assigned.add(node.name)
    return assigned


def _find_locally_defined_functions(tree: ast.AST) -> Set[str]:
    """Find function and class names defined locally in the snippet."""
    defined = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
    return defined


# ---------------------------------------------------------------------------
# Preprocessing: Notebook magics & whitespace handling
# ---------------------------------------------------------------------------

NON_PYTHON_CELL_MAGICS = {
    "bash", "sh", "zsh", "cmd", "powershell", "perl", "ruby", "R",
    "latex", "markdown", "svg", "html", "js", "javascript",
}


def _preprocess_code(raw_code: str) -> Tuple[str, List[str]]:
    """
    Handle IPython magics and dedent while preserving original line numbers.
    Returns (cleaned_python_code, notes).
    """
    notes: List[str] = []
    lines = raw_code.splitlines()

    if not lines:
        return "", notes

    first_line = lines[0].strip()

    # Check cell magic
    if first_line.startswith("%%"):
        magic_name = first_line[2:].split()[0].lower()
        if magic_name in NON_PYTHON_CELL_MAGICS:
            notes.append(f"Skipped non-Python cell magic '%%{magic_name}'.")
            return "", notes
        # Python cell magic (e.g. %%time, %%capture, %%writefile)
        # Blank out first line so line numbering for subsequent lines stays 1-indexed and exact
        lines[0] = ""

    processed_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("!"):
            # Shell command - replace with blank line
            processed_lines.append("")
        elif stripped.startswith("%") and not stripped.startswith("%%"):
            # Line magic
            tokens = stripped[1:].split()
            magic_cmd = tokens[0] if tokens else ""
            if magic_cmd in ("timeit", "time"):
                # Extract the code after the flags
                # e.g. %timeit -n 10 -r 3 model(x)
                rest_tokens = tokens[1:]
                idx = 0
                while idx < len(rest_tokens):
                    tok = rest_tokens[idx]
                    if tok.startswith("-"):
                        # Flag may take an argument if in ('-n', '-r', '-t', '-c', '-p')
                        if tok in ("-n", "-r", "-t", "-c", "-p") and idx + 1 < len(rest_tokens):
                            idx += 2
                        else:
                            idx += 1
                    else:
                        break
                code_part = " ".join(rest_tokens[idx:])
                processed_lines.append(code_part)
            else:
                processed_lines.append("")
        else:
            processed_lines.append(line)

    code = "\n".join(processed_lines)
    # Dedent if indented snippet
    try:
        dedented = textwrap.dedent(code)
    except Exception:
        dedented = code

    return dedented, notes


# ---------------------------------------------------------------------------
# Extraction & Call Resolution
# ---------------------------------------------------------------------------

BUILTIN_FUNCTIONS = {
    "len", "print", "range", "enumerate", "zip", "sum", "max", "min",
    "list", "dict", "set", "tuple", "int", "str", "float", "bool",
    "open", "type", "isinstance", "issubclass", "super", "abs", "any",
    "all", "map", "filter", "reversed", "sorted", "repr", "iter",
    "next", "hash", "exec", "eval",
}


def extract_calls(code: str) -> List[CallSite]:
    """Parse snippet and return list of CallSites with exact line numbers."""
    if code is None:
        raise TypeError("Code must be a string, not None.")
    if not isinstance(code, str):
        raise TypeError(f"Code must be a string, got {type(code).__name__}.")
    if "\x00" in code:
        raise SyntaxError("Null byte in source code.")

    cleaned_code, _notes = _preprocess_code(code)
    if not cleaned_code.strip():
        return []

    tree = ast.parse(cleaned_code)
    calls: List[CallSite] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = get_name(node.func)
            if name:
                calls.append(CallSite(name=name, line=getattr(node, "lineno", 1)))

    return calls


def find_calls(code: str) -> List[str]:
    """
    Return unique, resolved symbol call strings in source appearance order.
    Raises SyntaxError on invalid Python syntax.
    """
    if code is None:
        raise TypeError("Code must be a string, not None.")
    if not isinstance(code, str):
        raise TypeError(f"Code must be a string, got {type(code).__name__}.")
    if "\x00" in code:
        raise SyntaxError("Null byte in source code.")

    cleaned_code, _ = _preprocess_code(code)
    if not cleaned_code.strip():
        return []

    tree = ast.parse(cleaned_code)
    aliases = get_import_aliases(tree)
    local_defs = _find_locally_defined_functions(tree)

    # Collect calls in source order
    call_nodes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            call_nodes.append(node)

    # Sort by line and column to guarantee deterministic source order
    call_nodes.sort(key=lambda n: (getattr(n, "lineno", 0), getattr(n, "col_offset", 0)))

    ordered_calls: List[str] = []
    seen: Set[str] = set()

    for node in call_nodes:
        name = get_name(node.func)
        if not name:
            continue

        resolved = _resolve_call_name(name, aliases, local_defs)
        if resolved and resolved not in seen:
            seen.add(resolved)
            ordered_calls.append(resolved)

    return ordered_calls


def _resolve_call_name(name: str, aliases: Dict[str, str], local_defs: Set[str]) -> str:
    """Resolve an AST call name using imported aliases and builtin rules."""
    parts = name.split(".")
    first = parts[0]

    if first in aliases:
        resolved_first = aliases[first].split(".")
        parts = resolved_first + parts[1:]

    resolved = ".".join(parts)

    # Check builtins if single token and not locally defined
    if "." not in resolved and resolved in BUILTIN_FUNCTIONS:
        if resolved not in local_defs:
            return f"builtins.{resolved}"

    return resolved


# ---------------------------------------------------------------------------
# Documentation Lookup
# ---------------------------------------------------------------------------

def _is_valid_index(docs: Any) -> bool:
    """Ensure the DOCS index is a valid dict."""
    return isinstance(docs, dict)


def _validate_entry(entry: Any) -> Optional[Tuple[str, str]]:
    """Validate a single DOCS entry: must be (library_name, https_url_without_spaces)."""
    if not isinstance(entry, (tuple, list)) or len(entry) < 2:
        return None
    library, url = entry[0], entry[1]
    if not isinstance(library, str) or not isinstance(url, str):
        return None
    if not url.startswith("https://") or " " in url:
        return None
    return library, url


def lookup_symbol(name: Optional[str]) -> Result:
    """Look up documentation for a symbol or bare name in DOCS."""
    if not name or not isinstance(name, str) or not name.strip():
        return Result(symbol=name or "", status="empty")

    cleaned = name.strip()

    if not _is_valid_index(DOCS):
        return Result(symbol=cleaned, status="not_indexed")

    # 1. Exact match for fully-qualified name
    if cleaned in DOCS:
        valid = _validate_entry(DOCS[cleaned])
        if valid:
            lib, url = valid
            cats = get_symbol_categories(cleaned, lib)
            return Result(symbol=cleaned, status="found", library=lib, url=url, categories=cats)

    # 2. Builtin bare name
    if "." not in cleaned and f"builtins.{cleaned}" in DOCS:
        valid = _validate_entry(DOCS[f"builtins.{cleaned}"])
        if valid:
            lib, url = valid
            cats = get_symbol_categories(f"builtins.{cleaned}", lib)
            return Result(symbol=f"builtins.{cleaned}", status="found", library=lib, url=url, categories=cats)

    # If it's a qualified name that wasn't found in DOCS, do not fuzzy match
    if "." in cleaned:
        return Result(symbol=cleaned, status="not_indexed")

    # 3. Bare name candidate search (find all symbols in DOCS ending with .name)
    candidates: List[Candidate] = []
    target_suffix = f".{cleaned}"

    for sym, entry in DOCS.items():
        valid = _validate_entry(entry)
        if not valid:
            continue
        lib, url = valid
        if sym.endswith(target_suffix):
            candidates.append(Candidate(symbol=sym, library=lib, url=url))

    if candidates:
        # Sort for deterministic order
        candidates.sort(key=lambda c: c.symbol)
        return Result(
            symbol=cleaned,
            status="candidates",
            candidates=tuple(candidates),
        )

    return Result(symbol=cleaned, status="not_indexed")


# ---------------------------------------------------------------------------
# Code Snippet Analysis
# ---------------------------------------------------------------------------

def analyze(
    code: str,
    aliases: Optional[Any] = None,
    ignore: Optional[Any] = None,
    category: Optional[Any] = None,
) -> Analysis:
    """
    Analyze a Python code snippet, resolving imports and documenting calls.
    Supports optional ambient aliases, ignore filters, and category filters.
    """
    if not isinstance(code, str) or not code.strip():
        return Analysis(results=[], imports={}, notes=["No function or method calls found."])

    cleaned_code, notes = _preprocess_code(code)

    try:
        tree = ast.parse(cleaned_code)
    except SyntaxError as exc:
        return Analysis(error=f"Invalid Python syntax: {exc}", notes=notes)

    # Check for non-expanded star imports
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for item in node.names:
                if item.name == "*":
                    notes.append(f"Star import 'from {node.module} import *' not expanded.")

    # Combine ambient/passed aliases and internal code aliases
    ambient_aliases = _parse_ambient_aliases(aliases)
    internal_aliases = get_import_aliases(tree)
    combined_aliases = {**ambient_aliases, **internal_aliases}

    assigned_names = _find_assigned_names(tree)
    local_defs = _find_locally_defined_functions(tree)

    # Extract call nodes preserving line numbers
    call_nodes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            call_nodes.append(node)

    if not call_nodes:
        notes.append("No function or method calls found.")
        return Analysis(results=[], imports=combined_aliases, notes=notes)

    call_nodes.sort(key=lambda n: (getattr(n, "lineno", 0), getattr(n, "col_offset", 0)))

    # Group by resolved symbol name
    symbol_lines: Dict[str, List[int]] = {}
    symbol_raw_names: Dict[str, str] = {}
    ordered_symbol_keys: List[str] = []

    for node in call_nodes:
        raw_name = get_name(node.func)
        if not raw_name:
            continue

        resolved = _resolve_call_name(raw_name, combined_aliases, local_defs)
        if resolved not in symbol_lines:
            symbol_lines[resolved] = []
            symbol_raw_names[resolved] = raw_name
            ordered_symbol_keys.append(resolved)

        symbol_lines[resolved].append(getattr(node, "lineno", 1))

    results: List[Result] = []

    for sym_key in ordered_symbol_keys:
        raw_name = symbol_raw_names[sym_key]
        lines = tuple(sorted(symbol_lines[sym_key]))
        note = None

        first_token = raw_name.split(".")[0]
        # Check if an imported alias was reassigned
        if first_token in combined_aliases and first_token in assigned_names:
            note = f"Variable '{first_token}' reassigned in snippet."

        # Case 1: unimported alias or receiver (e.g. np.reshape without import numpy as np)
        if "." in raw_name and first_token not in combined_aliases:
            attr = raw_name.split(".")[-1]
            candidates = []
            if _is_valid_index(DOCS):
                for d_sym, d_entry in DOCS.items():
                    valid = _validate_entry(d_entry)
                    if valid and d_sym.endswith(f".{attr}"):
                        candidates.append(Candidate(symbol=d_sym, library=valid[0], url=valid[1]))
            candidates.sort(key=lambda c: c.symbol)

            results.append(
                Result(
                    symbol=raw_name,
                    status="unresolved",
                    library=None,
                    url=None,
                    candidates=tuple(candidates),
                    lines=lines,
                    note=note,
                )
            )
            continue

        # Case 2: lookup in DOCS
        res = lookup_symbol(sym_key)
        cats = res.categories or get_symbol_categories(res.symbol, res.library)
        results.append(
            Result(
                symbol=res.symbol,
                status=res.status,
                library=res.library,
                url=res.url,
                candidates=res.candidates,
                lines=lines,
                note=note or res.note,
                categories=cats,
            )
        )

    # Apply filters if requested
    filtered_results = _filter_results(results, ignore=ignore, category=category)
    return Analysis(results=filtered_results, imports=combined_aliases, notes=notes)


# ---------------------------------------------------------------------------
# Rendering: HTML and Plain Text
# ---------------------------------------------------------------------------

def render_html(analysis: Analysis) -> str:
    """Render analysis results as a beautiful, modern, safely-escaped HTML table."""
    if analysis.error:
        return f"<div style='color:#dc2626;padding:10px;background:#fef2f2;border-radius:8px;font-family:sans-serif;font-size:13px;border:1px solid #fecaca;'><strong>Error:</strong> {html.escape(analysis.error)}</div>"

    if not analysis.results:
        msg = " ".join(analysis.notes) if analysis.notes else "No function or method calls found."
        return f"<p style='color:#64748b;font-family:-apple-system,BlinkMacSystemFont,\"Segoe UI\",Roboto,sans-serif;font-size:13px;margin:8px 0;'>{html.escape(msg)}</p>"

    is_truncated = len(analysis.results) > MAX_ROWS
    display_results = analysis.results[:MAX_ROWS - 1] if is_truncated else analysis.results[:MAX_ROWS]

    rows = []
    for r in display_results:
        sym_escaped = html.escape(r.symbol)
        lib_name = r.library or "Unknown"
        lib_escaped = html.escape(lib_name)

        sym_cell = f"<code style='font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12.5px;color:#0f172a;background:#f1f5f9;padding:2px 6px;border-radius:4px;'>{sym_escaped}</code>"
        if r.note:
            sym_cell += f"<div style='font-size:11px;color:#94a3b8;margin-top:2px;'>{html.escape(r.note)}</div>"
        lib_cell = f"<span style='display:inline-block;padding:2px 8px;border-radius:12px;font-size:11.5px;font-weight:600;background:#e0f2fe;color:#0369a1;'>{lib_escaped}</span>"

        if r.status == "found" and r.url:
            safe_url = html.escape(r.url, quote=True)
            ref_cell = (
                f'<a href="{safe_url}" target="_blank" rel="noopener noreferrer" '
                f'style="color:#2563eb;text-decoration:none;font-weight:500;display:inline-flex;align-items:center;gap:4px;">'
                f'Official documentation <span style="font-size:11px;">&#8599;</span></a>'
            )
        elif r.status == "candidates" and r.candidates:
            c_links = [
                f'<a href="{html.escape(c.url, quote=True)}" target="_blank" rel="noopener noreferrer" '
                f'style="color:#2563eb;text-decoration:none;font-size:11.5px;">{html.escape(c.symbol)} ({html.escape(c.library)})</a>'
                for c in r.candidates[:3]
            ]
            ref_cell = f"<span style='color:#64748b;font-size:12px;'>Candidates: {', '.join(c_links)}</span>"
        elif r.status == "unresolved":
            ref_cell = "<span style='color:#94a3b8;font-style:italic;font-size:12px;'>Unresolved receiver</span>"
        else:
            ref_cell = "<span style='color:#94a3b8;font-style:italic;font-size:12px;'>Not indexed yet</span>"

        rows.append(
            f"<tr style='border-bottom:1px solid #f1f5f9;'>"
            f"<td style='padding:9px 14px;'>{sym_cell}</td>"
            f"<td style='padding:9px 14px;'>{lib_cell}</td>"
            f"<td style='padding:9px 14px;'>{ref_cell}</td>"
            f"</tr>"
        )

    if is_truncated:
        remaining = len(analysis.results) - len(display_results)
        rows.append(
            f"<tr><td colspan='3' style='text-align:center;padding:8px 14px;color:#64748b;background:#f8fafc;font-size:12px;border-top:1px solid #e2e8f0;'>"
            f"... {remaining} more not shown</td></tr>"
        )

    total_count = len(analysis.results)
    table = (
        '<div style="max-width:100%;margin:12px 0;border:1px solid #e2e8f0;border-radius:10px;overflow:hidden;box-shadow:0 1px 4px rgba(0,0,0,0.05);font-family:-apple-system,BlinkMacSystemFont,\'Segoe UI\',Roboto,sans-serif;">'
        '<div style="background:#f8fafc;padding:9px 14px;border-bottom:1px solid #e2e8f0;display:flex;align-items:center;justify-content:space-between;">'
        '<span style="font-size:12.5px;font-weight:600;color:#334155;">learn-functions &bull; API Documentation</span>'
        f'<span style="font-size:11.5px;color:#64748b;background:#e2e8f0;padding:1px 7px;border-radius:10px;font-weight:500;">{total_count} calls</span>'
        '</div>'
        '<table style="width:100%;border-collapse:collapse;font-size:13px;text-align:left;background:#ffffff;">'
        '<thead><tr style="background:#fcfcfd;border-bottom:1px solid #e2e8f0;">'
        '<th align="left" style="padding:9px 14px;color:#475569;font-weight:600;font-size:11.5px;text-transform:uppercase;letter-spacing:0.04em;">Symbol</th>'
        '<th align="left" style="padding:9px 14px;color:#475569;font-weight:600;font-size:11.5px;text-transform:uppercase;letter-spacing:0.04em;">Library</th>'
        '<th align="left" style="padding:9px 14px;color:#475569;font-weight:600;font-size:11.5px;text-transform:uppercase;letter-spacing:0.04em;">Reference</th>'
        '</tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )
    return table


def render_text(analysis: Analysis) -> str:
    """Render analysis results as plain text."""
    if analysis.error:
        return f"Error: {analysis.error}"
    if not analysis.results:
        return "\n".join(analysis.notes) if analysis.notes else "No calls found."

    lines = []
    for r in analysis.results:
        if r.status == "found":
            lines.append(f"{r.symbol} -> {r.library}: {r.url}")
        elif r.status == "candidates":
            cand_str = ", ".join(f"{c.symbol} ({c.url})" for c in r.candidates)
            lines.append(f"{r.symbol} (candidates: {cand_str})")
        else:
            lines.append(f"{r.symbol} [{r.status}]")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# High-Level Entry Point
# ---------------------------------------------------------------------------

def _is_in_notebook() -> bool:
    """Check if code is running inside an interactive notebook (Jupyter/Colab)."""
    try:
        from IPython import get_ipython
        shell = get_ipython()
        if shell is None:
            return False
        shell_name = shell.__class__.__name__
        return "ZMQInteractiveShell" in shell_name or "Shell" in shell_name
    except Exception:
        return False


def _looks_like_name(text: Optional[str]) -> bool:
    """Check if text is a single identifier or dotted attribute without call syntax."""
    if not text or not isinstance(text, str):
        return False
    stripped = text.strip()
    if not stripped or "(" in stripped or ")" in stripped or " " in stripped:
        return False
    ident_pattern = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)*$")
    return bool(ident_pattern.match(stripped))


def lookup(
    query: str,
    mode: str = "auto",
    aliases: Optional[Any] = None,
    ignore: Optional[Any] = None,
    category: Optional[Any] = None,
) -> None:
    """
    Main notebook and CLI helper function.
    Displays rich HTML in Jupyter Notebooks or clean text in terminal environments.

    Arguments:
    - query: Code snippet string or bare/qualified function name.
    - mode: 'auto', 'code', or 'symbol' (default: 'auto').
    - aliases: External import string (e.g. 'import pandas as pd') or dict ({'pd': 'pandas'}).
    - ignore: Package or library name(s) to omit (e.g. 'builtins' or ['builtins', 'requests']).
    - category: Filter to specific category (e.g. 'aiml', 'data', 'web', 'visualization').
    """
    if mode not in ("auto", "code", "symbol"):
        raise ValueError(f"Invalid mode '{mode}'. Expected 'auto', 'code', or 'symbol'.")

    if not isinstance(query, str):
        query = str(query) if query is not None else ""

    if mode == "symbol" or (mode == "auto" and _looks_like_name(query)):
        res = lookup_symbol(query)
        cats = res.categories or get_symbol_categories(res.symbol, res.library)
        res = Result(
            symbol=res.symbol,
            status=res.status,
            library=res.library,
            url=res.url,
            candidates=res.candidates,
            lines=res.lines,
            note=res.note,
            categories=cats,
        )
        filtered_results = _filter_results([res], ignore=ignore, category=category)
        analysis = Analysis(results=filtered_results)
    else:
        analysis = analyze(query, aliases=aliases, ignore=ignore, category=category)

    if _is_in_notebook() and display and HTML:
        display(HTML(render_html(analysis)))
    else:
        print(render_text(analysis))
