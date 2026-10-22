import ast
import contextlib
import io
import os
import tempfile
import unittest
from unittest import mock

import learn_functions as lf
from learn_functions import (
    find_calls,
    get_import_aliases,
    get_name,
)


# ----------------------------------------------------------------------------
# Section B / C / parsing: extraction  (checklist suite)
# ----------------------------------------------------------------------------
class TestBasicExtraction(unittest.TestCase):

    def test_builtin_call(self):
        self.assertIn("builtins.len", find_calls("len(data)"))

    def test_numpy_alias(self):
        code = "import numpy as np\nnp.reshape(data, shape)"
        self.assertIn("numpy.reshape", find_calls(code))

    def test_torchvision_class_alias(self):
        code = (
            "from torchvision.datasets import CIFAR10 as Dataset\n"
            "Dataset(root='./data')"
        )
        self.assertIn("torchvision.datasets.CIFAR10", find_calls(code))

    def test_nested_calls(self):
        calls = set(find_calls("print(len(list(data)))"))
        self.assertTrue(
            {"builtins.print", "builtins.len", "builtins.list"}.issubset(calls)
        )

    def test_method_preserves_receiver(self):
        self.assertIn("images.flatten", find_calls("images.flatten()"))

    def test_multiple_calls(self):
        calls = find_calls("len(data)\nmax(data)")
        self.assertIn("builtins.len", calls)
        self.assertIn("builtins.max", calls)

    def test_repeated_calls_are_deduplicated(self):
        calls = find_calls("len(a)\nlen(b)")
        self.assertEqual(calls.count("builtins.len"), 1)

    def test_calls_in_comprehensions(self):
        self.assertIn("transform", find_calls("[transform(x) for x in data]"))

    def test_call_inside_decorator(self):
        self.assertIn("decorator", find_calls("@decorator()\ndef f():\n    pass"))

    def test_async_call(self):
        calls = find_calls("async def f():\n    await fetch_data()")
        self.assertIn("fetch_data", calls)

    def test_nested_callable(self):
        self.assertIn("factory", find_calls("factory()()"))

    def test_subscript_receiver(self):
        self.assertIn("reshape", find_calls("items[0].reshape(-1)"))

    def test_dynamic_receiver(self):
        self.assertIn("get_model", find_calls("get_model().predict(data)"))

    def test_unicode_identifiers(self):
        self.assertEqual(find_calls("données.résumé()"), ["données.résumé"])

    def test_order_is_source_order_and_deterministic(self):
        code = "b()\na()\nb()"
        self.assertEqual(find_calls(code), ["b", "a"])
        self.assertEqual(find_calls(code), find_calls(code))


class TestImports(unittest.TestCase):

    def test_module_alias(self):
        aliases = get_import_aliases(ast.parse("import numpy as np"))
        self.assertEqual(aliases["np"], "numpy")

    def test_from_import_alias(self):
        tree = ast.parse("from torchvision.datasets import CIFAR10 as Dataset")
        self.assertEqual(
            get_import_aliases(tree)["Dataset"], "torchvision.datasets.CIFAR10"
        )

    def test_import_only_has_no_calls(self):
        self.assertEqual(find_calls("import pandas as pd"), [])

    def test_multiple_aliases(self):
        aliases = get_import_aliases(
            ast.parse("import numpy as np\nimport pandas as pd")
        )
        self.assertEqual(aliases["np"], "numpy")
        self.assertEqual(aliases["pd"], "pandas")

    def test_star_import_is_not_expanded(self):
        aliases = get_import_aliases(ast.parse("from numpy import *"))
        self.assertNotIn("*", aliases)

    def test_dotted_import_keeps_top_level(self):
        aliases = get_import_aliases(ast.parse("import torch.utils.data"))
        self.assertEqual(aliases["torch"], "torch")
        self.assertIn(
            "torch.utils.data.DataLoader",
            find_calls("import torch.utils.data\ntorch.utils.data.DataLoader(d)"),
        )

    def test_relative_import_is_not_resolved(self):
        self.assertEqual(get_import_aliases(ast.parse("from . import helper")), {})

    def test_unimported_alias_stays_unresolved(self):
        self.assertEqual(find_calls("np.reshape(x)"), ["np.reshape"])

    def test_local_definition_shadows_builtin(self):
        self.assertEqual(find_calls("def sum(a):\n    pass\nsum(x)"), ["sum"])


class TestParsingErrors(unittest.TestCase):

    def test_invalid_syntax(self):
        with self.assertRaises(SyntaxError):
            find_calls("def broken(")

    def test_empty_input(self):
        self.assertEqual(find_calls(""), [])

    def test_whitespace_input(self):
        self.assertEqual(find_calls("   \n  "), [])

    def test_comments_are_not_calls(self):
        self.assertEqual(find_calls("# model.predict(data)"), [])

    def test_strings_are_not_calls(self):
        self.assertEqual(find_calls('"model.predict(data)"'), [])

    def test_multiline_call(self):
        calls = find_calls("model.predict(\n    data\n)")
        self.assertIn("model.predict", calls)

    def test_indented_snippet_is_dedented(self):
        self.assertEqual(find_calls("    len(x)"), ["builtins.len"])

    def test_null_byte_is_a_syntax_error(self):
        with self.assertRaises(SyntaxError):
            find_calls("x\x00")

    def test_non_string_is_a_type_error(self):
        with self.assertRaises(TypeError):
            find_calls(None)


class TestNameExtraction(unittest.TestCase):

    def test_simple_name(self):
        node = ast.parse("reshape", mode="eval").body
        self.assertEqual(get_name(node), "reshape")

    def test_dotted_name(self):
        node = ast.parse("torch.utils.data.DataLoader", mode="eval").body
        self.assertEqual(get_name(node), "torch.utils.data.DataLoader")

    def test_call_is_not_a_simple_name(self):
        node = ast.parse("get_model()", mode="eval").body
        self.assertIsNone(get_name(node))


# ----------------------------------------------------------------------------
# Section E: notebook syntax
# ----------------------------------------------------------------------------
class TestNotebookSyntax(unittest.TestCase):

    def test_timeit_line_magic(self):
        self.assertEqual(find_calls("%timeit model(x)"), ["model"])

    def test_timeit_with_flags(self):
        self.assertEqual(find_calls("%timeit -n 10 -r 3 model(x)"), ["model"])

    def test_shell_command_is_not_python(self):
        self.assertEqual(find_calls("!pip install numpy"), [])

    def test_other_line_magic_is_skipped(self):
        self.assertEqual(
            find_calls("%matplotlib inline\nlen(x)"), ["builtins.len"]
        )

    def test_python_cell_magic_body_is_analysed(self):
        self.assertEqual(find_calls("%%capture out\nlen(x)"), ["builtins.len"])

    def test_non_python_cell_magic_is_skipped_with_note(self):
        analysis = lf.analyze("%%bash\nls -l\nlen(x)")
        self.assertEqual(analysis.results, [])
        self.assertTrue(any("bash" in n for n in analysis.notes))

    def test_line_numbers_survive_preprocessing(self):
        sites = lf.extract_calls("%matplotlib inline\n\nlen(x)")
        self.assertEqual(sites[0].line, 3)


# ----------------------------------------------------------------------------
# Section D: lookup_symbol
# ----------------------------------------------------------------------------
class TestLookupSymbol(unittest.TestCase):

    def test_qualified_exact_match(self):
        result = lf.lookup_symbol("numpy.reshape")
        self.assertEqual(result.status, "found")
        self.assertTrue(result.url.startswith("https://numpy.org"))

    def test_bare_name_returns_candidates(self):
        result = lf.lookup_symbol("reshape")
        self.assertEqual(result.status, "candidates")
        self.assertIn("numpy.reshape", [c.symbol for c in result.candidates])

    def test_class_short_name(self):
        result = lf.lookup_symbol("CIFAR10")
        self.assertIn(
            "torchvision.datasets.CIFAR10", [c.symbol for c in result.candidates]
        )

    def test_builtin_bare_name_is_exact(self):
        self.assertEqual(lf.lookup_symbol("len").status, "found")

    def test_unknown_symbol(self):
        result = lf.lookup_symbol("totally_unknown")
        self.assertEqual(result.status, "not_indexed")
        self.assertIsNone(result.url)

    def test_qualified_name_is_never_fuzzy_matched(self):
        self.assertEqual(lf.lookup_symbol("np.reshape").status, "not_indexed")

    def test_empty_and_whitespace(self):
        self.assertEqual(lf.lookup_symbol("").status, "empty")
        self.assertEqual(lf.lookup_symbol("   ").status, "empty")
        self.assertEqual(lf.lookup_symbol(None).status, "empty")


# ----------------------------------------------------------------------------
# Section A / C / D: analyze (snippet lookup)
# ----------------------------------------------------------------------------
class TestAnalyze(unittest.TestCase):

    def one(self, code):
        results = lf.analyze(code).results
        self.assertEqual(len(results), 1, results)
        return results[0]

    def test_import_only_reports_import_not_call(self):
        analysis = lf.analyze("import pandas as pd")
        self.assertEqual(analysis.results, [])
        self.assertEqual(analysis.imports, {"pd": "pandas"})

    def test_alias_without_import_does_not_claim_numpy(self):
        result = self.one("np.reshape(x)")
        self.assertEqual(result.status, "unresolved")
        self.assertIsNone(result.url)
        self.assertEqual(result.library, None)

    def test_alias_with_import_resolves(self):
        result = self.one("import numpy as np\nnp.reshape(x)")
        self.assertEqual(result.status, "found")
        self.assertEqual(result.symbol, "numpy.reshape")

    def test_generic_method_does_not_assume_library(self):
        result = self.one("images.flatten()")
        self.assertEqual(result.status, "unresolved")
        self.assertIsNone(result.library)
        self.assertIn("numpy.ndarray.flatten", [c.symbol for c in result.candidates])

    def test_unknown_custom_library_has_no_link(self):
        result = self.one("import mylib\nmylib.go()")
        self.assertEqual(result.status, "not_indexed")
        self.assertIsNone(result.url)

    def test_line_numbers_are_kept_per_symbol(self):
        result = self.one("len(a)\n\nlen(b)")
        self.assertEqual(result.lines, (1, 3))

    def test_reassigned_alias_is_flagged(self):
        result = self.one("import numpy as np\nnp = 3\nnp.reshape(x)")
        self.assertIn("reassigned", result.note)

    def test_star_import_is_noted_not_expanded(self):
        analysis = lf.analyze("from numpy import *\nreshape(x)")
        self.assertTrue(any("not expanded" in n for n in analysis.notes))
        self.assertEqual(analysis.results[0].status, "candidates")

    def test_syntax_error_is_friendly(self):
        analysis = lf.analyze("model.forward(")
        self.assertTrue(analysis.error.startswith("Invalid Python syntax"))

    def test_empty_input(self):
        analysis = lf.analyze("")
        self.assertIsNone(analysis.error)
        self.assertTrue(analysis.notes)

    def test_prose_does_not_crash(self):
        self.assertIsNotNone(lf.analyze("how do I reshape an array?").error)

    def test_no_calls(self):
        analysis = lf.analyze("x = 1")
        self.assertEqual(analysis.results, [])
        self.assertTrue(any("No function" in n for n in analysis.notes))

    def test_failure_does_not_corrupt_later_calls(self):
        lf.analyze("def broken(")
        self.assertEqual(self.one("len(x)").status, "found")


# ----------------------------------------------------------------------------
# Index validation
# ----------------------------------------------------------------------------
class TestIndexValidation(unittest.TestCase):

    def test_malformed_entries_are_ignored(self):
        bad = {
            "none.entry": None,
            "short.tuple": ("Lib",),
            "http.only": ("Lib", "http://insecure.example/doc"),
            "space.url": ("Lib", "https://example.com/a b"),
            "numpy.reshape": ("NumPy", "https://numpy.org/reshape"),
        }
        with mock.patch.object(lf, "DOCS", bad):
            self.assertEqual(lf.lookup_symbol("numpy.reshape").status, "found")
            for key in ("none.entry", "short.tuple", "http.only", "space.url"):
                self.assertEqual(lf.lookup_symbol(key).status, "not_indexed")

    def test_empty_index(self):
        with mock.patch.object(lf, "DOCS", {}):
            self.assertEqual(lf.lookup_symbol("len").status, "not_indexed")
            self.assertEqual(lf.analyze("len(x)").results[0].status, "not_indexed")

    def test_index_that_is_not_a_dict(self):
        with mock.patch.object(lf, "DOCS", None):
            self.assertEqual(lf.lookup_symbol("len").status, "not_indexed")


# ----------------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------------
class TestRendering(unittest.TestCase):

    def test_html_special_characters_are_escaped(self):
        analysis = lf.Analysis(
            results=[lf.Result("<b>x</b>", "not_indexed", note="<script>x</script>")]
        )
        output = lf.render_html(analysis)
        self.assertNotIn("<b>x</b>", output)
        self.assertNotIn("<script>", output)
        self.assertIn("&lt;b&gt;", output)

    def test_malformed_html_input_is_never_injected(self):
        output = lf.render_html(lf.analyze("<script>alert(1)</script>"))
        self.assertNotIn("<script>", output)

    def test_many_results_are_truncated(self):
        code = "\n".join(f"f{i}(x)" for i in range(120))
        output = lf.render_html(lf.analyze(code))
        self.assertLessEqual(output.count("<tr>"), lf.MAX_ROWS + 1)
        self.assertIn("more not shown", output)

    def test_no_results_renders_message_not_empty_table(self):
        output = lf.render_html(lf.analyze("x = 1"))
        self.assertNotIn("<table", output)
        self.assertIn("No function", output)

    def test_found_result_renders_a_safe_link(self):
        output = lf.render_html(lf.analyze("len(x)"))
        self.assertIn('rel="noopener noreferrer"', output)
        self.assertIn("https://docs.python.org", output)

    def test_text_fallback(self):
        text = lf.render_text(lf.analyze("len(x)"))
        self.assertIn("builtins.len", text)


class TestLookupEntryPoint(unittest.TestCase):

    def run_lookup(self, *args, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return lf.lookup(*args, **kwargs)

    def test_returns_none_and_never_raises(self):
        for text in ("len", "import numpy as np\nnp.reshape(x)", "", "def broken(", "!ls"):
            self.assertIsNone(self.run_lookup(text))

    def test_modes(self):
        self.assertIsNone(self.run_lookup("len", mode="symbol"))
        self.assertIsNone(self.run_lookup("len(x)", mode="code"))
        with self.assertRaises(ValueError):
            self.run_lookup("len", mode="nope")

    def test_auto_mode_picks_symbol_for_names(self):
        self.assertTrue(lf._looks_like_name("numpy.reshape"))
        self.assertFalse(lf._looks_like_name("np.reshape()"))
        self.assertFalse(lf._looks_like_name(""))


# ----------------------------------------------------------------------------
# Safety: submitted code is parsed, never executed
# ----------------------------------------------------------------------------
class TestNeverExecutes(unittest.TestCase):

    def test_side_effects_do_not_happen(self):
        with tempfile.NamedTemporaryFile(delete=False) as handle:
            path = handle.name
        try:
            lf.analyze(f"__import__('os').remove({path!r})")
            find_calls(f"__import__('os').remove({path!r})")
            self.assertTrue(os.path.exists(path))
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_exit_is_not_run(self):
        self.assertEqual(find_calls("exec('raise SystemExit')"), ["builtins.exec"])
        lf.analyze("import sys\nsys.exit(1)")


if __name__ == "__main__":
    unittest.main(verbosity=2)