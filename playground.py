import learn_functions as lf

code = """
import requests
import pandas as pd
import flask
from scipy import sparse

resp = requests.get('https://example.com')
df = pd.DataFrame({'a': [1, 2]})
app = flask.Flask(__name__)
m = sparse.csr_matrix([1, 2])
"""


print(lf.lookup(code))
# for r in analysis.results:
#     print(f"{r.symbol} ({r.status}) -> {r.library}: {r.url}")
