from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

project_root = Path('./').absolute().parent.parent

eval_score = pd.read_pickle(project_root / "genedisco_output" / "eval_score.pickle")
print(eval_score)

test_score = pd.read_pickle(project_root / "genedisco_output" / "test_score.pickle")
print(test_score)

results = pd.DataFrame(
    pd.read_pickle(project_root / "genedisco_output" / "results.pickle")
)

plot_df = results[
    ['MeanAbsoluteError', 'RootMeanSquaredError', "SymmetricMeanAbsolutePercentageError", 'SpearmanRho', ]
].rename({
    'MeanAbsoluteError': "MAE",
    'RootMeanSquaredError': "RMSE",
    "SymmetricMeanAbsolutePercentageError": "sMAPE",
    'SpearmanRho': "Spearman",
}, axis=1)

plot_df["MAE"].plot()
plt.grid()
plt.show()

plot_df["RMSE"].plot()
plt.grid()
plt.show()

plot_df["sMAPE"].plot()
plt.grid()
plt.show()

plot_df["Spearman"].plot()
plt.grid()
plt.show()


