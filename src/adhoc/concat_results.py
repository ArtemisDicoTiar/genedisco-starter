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
bald_top_teacher = filter_essential(pd.read_csv('./bald_top_teacher.csv'))
bald_top_teacher_multi = filter_essential(pd.read_csv('./bald_teacher_multi.csv'))
bald_top_teacher_vote = filter_essential(pd.read_csv('./bald_teacher_vote.csv'))
bald_top_teacher_vote2 = filter_essential(pd.read_csv('./bald_teacher_vote2.csv'))
badge = filter_essential(pd.read_csv('./badge.csv'))
badge_teacher = filter_essential(pd.read_csv('./badge_teacher.csv'))
badge_teacher_multi = filter_essential(pd.read_csv('./badge_teacher_multi.csv'))
badge_teacher_vote = filter_essential(pd.read_csv('./badge_teacher_vote.csv'))
badge_teacher_vote2 = filter_essential(pd.read_csv('./badge_teacher_vote2.csv'))
random = filter_essential(pd.read_csv('./random.csv'))
ensemble = filter_essential(pd.read_csv('./ensemble.csv'))
ensemble_teacher = filter_essential(pd.read_csv('./ensemble_teacher.csv'))
ensemble_teacher_bestonly = filter_essential(pd.read_csv('./ensemble_teacher_bestonly.csv'))

total_data = {
    # "bald_top": bald_top,
    # "bald_top_teacher": bald_top_teacher,
    "bald_top_teacher_vote": bald_top_teacher_vote,
    # "bald_top_teacher_vote2": bald_top_teacher_vote2,
    # "bald_top_teacher_multi": bald_top_teacher_multi,
    # "badge": badge,
    # "badge_teacher": badge_teacher,
    "badge_teacher_vote": badge_teacher_vote,
    # "badge_teacher_vote2": badge_teacher_vote2,
    # "badge_teacher_multi": badge_teacher_multi,
    "random": random,
    # "ensemble": ensemble,
    # "ensemble_teacher": ensemble_teacher,
    # "ensemble_teacher_bestonly": ensemble_teacher_bestonly,
}

get_df_of = lambda which: pd.DataFrame({
    i[0]: i[1][which].to_list()
    for i in total_data.items()
})

mae = get_df_of("MAE")
rmse = get_df_of("RMSE")
smape = get_df_of("sMAPE")
spearman = get_df_of("Spearman")


mae.plot()
plt.title("MeanAbsoluteError")
plt.xlabel("cycles")
plt.ylabel("mae")
plt.grid()
plt.savefig("./mae.png")
plt.show()

rmse.plot()
plt.title("RootMeanSquaredError")
plt.xlabel("cycles")
plt.ylabel("rmse")
plt.grid()
plt.savefig("./rmse.png")
plt.show()

smape.plot()
plt.title("SymmetricMeanAbsolutePercentageError")
plt.xlabel("cycles")
plt.ylabel("smape")
plt.grid()
plt.savefig("./smape.png")
plt.show()

spearman.plot()
plt.title("SpearmanRho")
plt.xlabel("cycles")
plt.ylabel("spearman_rho")
plt.grid()
plt.savefig("./spearman.png")
plt.show()

print("a")



