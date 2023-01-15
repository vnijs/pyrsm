import pandas as pd
import numpy as np
from pyrsm.regression import sim_prediction
from pyrsm.regression import reg_dashboard

np.random.seed(1234)
nr = 100
df = pd.DataFrame()
df["x1"] = np.random.uniform(0, 1, nr)
df["x2"] = 1 - df["x1"]
df["x3"] = np.random.choice(["a", "b", "c"], nr)
df["response_prob"] = np.random.uniform(0, 1, nr)
df["response"] = np.where((df["x1"] > 0.5) & (df["response_prob"] > 0.5), "yes", "no")

# df.iloc[0, 0] = 1000
# sim_prediction(df, vary="x1", minq=0, maxq=1)
# sim_prediction(df, vary="x1")


def test_sim_prediction():
    ret = sim_prediction(df)
    assert ret.loc[0, "x3"] == "b", "Incorrectly generated simulated data"
