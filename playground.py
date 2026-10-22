import learn_functions as lf

# Ensure collected documentation index is loaded
lf.load_documentation()

# A snippet where pd, np, and requests are called without importing them inside the snippet
code = """
resp = requests.get('https://example.com')
df = pd.DataFrame({'a': [1, 2]})
arr = np.reshape([1, 2, 3, 4], (2, 2))
n = len([1, 2, 3])
"""

print("=== 1. Standard Lookup with Ambient Aliases ===")
lf.lookup(code, aliases="import pandas as pd\nimport requests\nimport numpy as np")

print("\n=== 2. Filtering: Ignore builtins and requests ===")
lf.lookup(code, aliases="import pandas as pd\nimport requests\nimport numpy as np", ignore=["builtins", "requests"])

print("\n=== 3. Filtering by Category: AIML only ===")
lf.lookup(code, aliases="import pandas as pd\nimport requests\nimport numpy as np", category="aiml")

print("\n=== 4. Filtering by Category: Web only ===")
lf.lookup(code, aliases="import pandas as pd\nimport requests\nimport numpy as np", category="web")
