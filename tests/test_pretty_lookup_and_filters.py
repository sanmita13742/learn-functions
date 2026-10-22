"""
tests/test_pretty_lookup_and_filters.py

Unit test suite verifying:
1. 'aliases' argument integration with AST resolver (string and dict formats).
2. 'ignore' argument filtering by package / library name.
3. 'category' argument filtering (e.g. 'aiml', 'web', 'data').
4. Pretty HTML table rendering and safe HTML escaping.
5. Full backward compatibility of lookup() and analyze() without extra arguments.
"""

import contextlib
import io
import unittest

import learn_functions as lf


class TestAmbientAliases(unittest.TestCase):
    """Test passing external imports to analyze() and lookup()."""

    @classmethod
    def setUpClass(cls):
        cls.orig_docs = dict(lf.DOCS)
        lf.load_documentation()

    @classmethod
    def tearDownClass(cls):
        lf.DOCS.clear()
        lf.DOCS.update(cls.orig_docs)

    def test_ambient_aliases_as_import_string(self):
        code = "df = pd.read_csv('data.csv')"
        # Without aliases, pd is unimported receiver
        res_unimported = lf.analyze(code).results[0]
        self.assertEqual(res_unimported.status, "unresolved")

        # With aliases as string
        res = lf.analyze(code, aliases="import pandas as pd").results[0]
        self.assertEqual(res.status, "found")
        self.assertEqual(res.symbol, "pandas.read_csv")
        self.assertEqual(res.library, "pandas")

    def test_ambient_aliases_as_dict(self):
        code = "arr = np.reshape(data, (2, 3))"
        res = lf.analyze(code, aliases={"np": "numpy"}).results[0]
        self.assertEqual(res.status, "found")
        self.assertEqual(res.symbol, "numpy.reshape")

    def test_multiple_aliases_in_string(self):
        code = "df = pd.read_csv('x')\nr = requests.get('url')"
        analysis = lf.analyze(
            code,
            aliases="import pandas as pd\nimport requests",
        )
        symbols = [r.symbol for r in analysis.results]
        self.assertIn("pandas.read_csv", symbols)
        self.assertIn("requests.get", symbols)

    def test_internal_import_overrides_external_alias(self):
        # Code explicitly imports custom module with same alias
        code = "import my_custom_numpy as np\nnp.func()"
        res = lf.analyze(code, aliases={"np": "numpy"}).results[0]
        self.assertEqual(res.symbol, "my_custom_numpy.func")


class TestIgnoreFilter(unittest.TestCase):
    """Test filtering out unwanted packages / libraries."""

    def setUp(self):
        self.code = """
import numpy as np
x = len([1, 2, 3])
arr = np.reshape(data, (2, 2))
print('done')
"""

    def test_ignore_single_string(self):
        analysis = lf.analyze(self.code, ignore="builtins")
        symbols = [r.symbol for r in analysis.results]
        self.assertNotIn("builtins.len", symbols)
        self.assertNotIn("builtins.print", symbols)
        self.assertIn("numpy.reshape", symbols)

    def test_ignore_by_library_name_case_insensitive(self):
        analysis = lf.analyze(self.code, ignore="PYTHON")
        symbols = [r.symbol for r in analysis.results]
        self.assertNotIn("builtins.len", symbols)
        self.assertIn("numpy.reshape", symbols)

    def test_ignore_list_of_packages(self):
        analysis = lf.analyze(self.code, ignore=["builtins", "numpy"])
        self.assertEqual(len(analysis.results), 0)

    def test_ignore_none_preserves_all(self):
        analysis = lf.analyze(self.code, ignore=None)
        symbols = [r.symbol for r in analysis.results]
        self.assertIn("builtins.len", symbols)
        self.assertIn("numpy.reshape", symbols)


class TestCategoryFilter(unittest.TestCase):
    """Test filtering by domain category (aiml, web, data, visualization, core)."""

    @classmethod
    def setUpClass(cls):
        cls.orig_docs = dict(lf.DOCS)
        lf.load_documentation()

    @classmethod
    def tearDownClass(cls):
        lf.DOCS.clear()
        lf.DOCS.update(cls.orig_docs)

    def setUp(self):
        self.code = """
import numpy as np
import requests
import matplotlib.pyplot as plt
x = len([1, 2])
arr = np.reshape(data, (2, 2))
r = requests.get('https://example.com')
plt.plot([1], [2])
"""

    def test_filter_aiml_category(self):
        analysis = lf.analyze(self.code, category="aiml")
        symbols = [r.symbol for r in analysis.results]
        self.assertIn("numpy.reshape", symbols)
        self.assertNotIn("requests.get", symbols)
        self.assertNotIn("builtins.len", symbols)

    def test_filter_web_category(self):
        analysis = lf.analyze(self.code, category="web")
        symbols = [r.symbol for r in analysis.results]
        self.assertIn("requests.get", symbols)
        self.assertNotIn("numpy.reshape", symbols)
        self.assertNotIn("matplotlib.pyplot.plot", symbols)

    def test_filter_visualization_category(self):
        analysis = lf.analyze(self.code, category="visualization")
        symbols = [r.symbol for r in analysis.results]
        self.assertIn("matplotlib.pyplot.plot", symbols)
        self.assertNotIn("requests.get", symbols)
        self.assertNotIn("numpy.reshape", symbols)

    def test_filter_multiple_categories(self):
        analysis = lf.analyze(self.code, category=["aiml", "web"])
        symbols = [r.symbol for r in analysis.results]
        self.assertIn("numpy.reshape", symbols)
        self.assertIn("requests.get", symbols)
        self.assertNotIn("builtins.len", symbols)


class TestCombinedFilters(unittest.TestCase):
    """Test combining aliases, ignore, and category simultaneously."""

    def test_aliases_plus_ignore_plus_category(self):
        code = "arr = np.reshape(x, (2, 2))\nr = requests.get('url')\nx = len([1])"
        analysis = lf.analyze(
            code,
            aliases="import numpy as np\nimport requests",
            ignore="builtins",
            category="aiml",
        )
        symbols = [r.symbol for r in analysis.results]
        self.assertEqual(symbols, ["numpy.reshape"])


class TestPrettyRenderingAndSafety(unittest.TestCase):
    """Test visual card rendering and HTML safety."""

    @classmethod
    def setUpClass(cls):
        cls.orig_docs = dict(lf.DOCS)
        lf.load_documentation()

    @classmethod
    def tearDownClass(cls):
        lf.DOCS.clear()
        lf.DOCS.update(cls.orig_docs)

    def test_html_is_safely_escaped(self):
        evil_name = "<script>alert('xss')</script>"
        evil_note = "<img src=x onerror=alert(1)>"
        analysis = lf.Analysis(
            results=[lf.Result(symbol=evil_name, status="not_indexed", note=evil_note)]
        )
        rendered = lf.render_html(analysis)
        self.assertNotIn("<script>", rendered)
        self.assertNotIn("<img", rendered)
        self.assertIn("&lt;script&gt;", rendered)
        self.assertIn("&lt;img", rendered)

    def test_pretty_card_structure(self):
        analysis = lf.analyze("import requests\nrequests.get('https://example.com')")
        rendered = lf.render_html(analysis)
        self.assertIn("learn-functions", rendered)
        self.assertIn("requests.get", rendered)
        self.assertIn('rel="noopener noreferrer"', rendered)
        self.assertIn("Official documentation", rendered)

    def test_empty_results_no_table(self):
        analysis = lf.analyze("x = 10")
        rendered = lf.render_html(analysis)
        self.assertNotIn("<table", rendered)
        self.assertIn("No function", rendered)


class TestBackwardCompatibility(unittest.TestCase):
    """Ensure existing lookup() and analyze() signatures and behaviors remain intact."""

    def test_lookup_default_signature_returns_none(self):
        with contextlib.redirect_stdout(io.StringIO()):
            res = lf.lookup("len(x)")
            self.assertIsNone(res)

    def test_lookup_with_aliases_and_filters(self):
        with contextlib.redirect_stdout(io.StringIO()):
            res = lf.lookup(
                "df = pd.read_csv('x')",
                aliases="import pandas as pd",
                ignore="builtins",
                category="data",
            )
            self.assertIsNone(res)

    def test_analyze_without_new_args_preserves_default(self):
        analysis = lf.analyze("x = len([1])\nimport requests\nr = requests.get('u')")
        self.assertEqual(len(analysis.results), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
