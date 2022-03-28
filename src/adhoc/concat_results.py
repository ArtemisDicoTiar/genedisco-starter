from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

filter_essential = lambda df: df[
    ['MeanAbsoluteError', 'RootMeanSquaredError', "SymmetricMeanAbsolutePercentageError", 'SpearmanRho', ]
].rename({
    'MeanAbsoluteError': "MAE",
    'RootMeanSquaredError': "RMSE",
    "SymmetricMeanAbsolutePercentageError": "sMAPE",
    'SpearmanRho': "Spearman",
}, axis=1)

bald_top = filter_essential(pd.read_csv('./bald_top.csv'))
badge = filter_essential(pd.read_csv('./badge.csv'))
random = filter_essential(pd.read_csv('./random.csv'))
ensemble = filter_essential(pd.read_csv('./ensemble.csv'))

mae = pd.DataFrame({
    "bald_top": bald_top['MAE'].to_list(),
    "badge": badge['MAE'].to_list(),
    "random": random['MAE'].to_list(),
    "ensemble": ensemble['MAE'].to_list(),
})

rmse = pd.DataFrame({
    "bald_top": bald_top['RMSE'].to_list(),
    "badge": badge['RMSE'].to_list(),
    "random": random['RMSE'].to_list(),
    "ensemble": ensemble['RMSE'].to_list(),
})

smape = pd.DataFrame({
    "bald_top": bald_top['sMAPE'].to_list(),
    "badge": badge['sMAPE'].to_list(),
    "random": random['sMAPE'].to_list(),
    "ensemble": ensemble['sMAPE'].to_list(),
})

spearman = pd.DataFrame({
    "bald_top": bald_top['Spearman'].to_list(),
    "badge": badge['Spearman'].to_list(),
    "random": random['Spearman'].to_list(),
    "ensemble": ensemble['Spearman'].to_list(),
})

mae.plot()
plt.title("MeanAbsoluteError")
plt.xlabel("cycles")
plt.ylabel("mae")
plt.grid()
plt.show()

rmse.plot()
plt.title("RootMeanSquaredError")
plt.xlabel("cycles")
plt.ylabel("rmse")
plt.grid()
plt.show()

smape.plot()
plt.title("SymmetricMeanAbsolutePercentageError")
plt.xlabel("cycles")
plt.ylabel("smape")
plt.grid()
plt.show()

spearman.plot()
plt.title("SpearmanRho")
plt.xlabel("cycles")
plt.ylabel("spearman_rho")
plt.grid()
plt.show()

print("a")



