%%writefile learn_functions.py

# Move the imports, DOCS dictionary,
# helper functions, and lookup() here.
import ast
import html
from IPython.display import display, HTML

DOCS = {
    "builtins.len": (
        "Python",
        "https://docs.python.org/3/library/functions.html#len"
    ),
    "numpy.reshape": (
        "NumPy",
        "https://numpy.org/doc/stable/reference/generated/numpy.reshape.html"
    ),
    "numpy.ndarray.flatten": (
        "NumPy",
        "https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flatten.html"
    ),
    "torchvision.datasets.CIFAR10": (
        "torchvision",
        "https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html"
    ),
    "torch.utils.data.DataLoader": (
        "PyTorch",
        "https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader"
    ),
}


def get_name(node):
    """Convert an AST name or attribute into a dotted name."""
    if isinstance(node, ast.Name):
        return node.id

    if isinstance(node, ast.Attribute):
        parent = get_name(node.value)
        return f"{parent}.{node.attr}" if parent else None

    return None


def get_import_aliases(tree):
    """Map imported aliases to their fully qualified names."""
    aliases = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                local_name = item.asname or item.name.split(".")[0]
                full_name = item.name if item.asname else local_name
                aliases[local_name] = full_name

        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""

            for item in node.names:
                if item.name == "*":
                    continue

                local_name = item.asname or item.name
                aliases[local_name] = (
                    f"{module}.{item.name}" if module else item.name
                )

    return aliases


def _resolve(name, aliases):
    parts = name.split(".")
    first = parts[0]

    if first in aliases:
        parts = aliases[first].split(".") + parts[1:]

    resolved = ".".join(parts)

    if "." not in resolved and resolved in {
        "len", "print", "range", "enumerate", "zip", "sum", "max", "min"
    }:
        return f"builtins.{resolved}"

    return resolved

def find_calls(code):
    """Display official documentation links for recognized calls."""
    try:
        tree = ast.parse(code)
    except SyntaxError as error:
        print(f"Invalid Python syntax: {error}")
        return

    aliases = get_import_aliases(tree)
    calls = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = get_name(node.func)

            if name:
                calls.add(_resolve(name, aliases))

    if not calls:
        print("No function or method calls found.")
        return

    rows = []

    for name in sorted(calls):
        if name in DOCS:
            library, url = DOCS[name]
            rows.append(
                f"<tr><td>{html.escape(name)}</td>"
                f"<td>{html.escape(library)}</td>"
                f'<td><a href="{html.escape(url, quote=True)}" '
                f'target="_blank" rel="noopener noreferrer">'
                f"Official documentation</a></td></tr>"
            )
        else:
            rows.append(
                f"<tr><td>{html.escape(name)}</td>"
                f"<td>Unknown</td><td>Not indexed yet</td></tr>"
            )

    output = """
    <table style="border-collapse:collapse;width:100%">
      <thead>
        <tr>
          <th align="left">Symbol</th>
          <th align="left">Library</th>
          <th align="left">Reference</th>
        </tr>
      </thead>
      <tbody>
    """ + "".join(rows) + "</tbody></table>"

    display(HTML(output))

