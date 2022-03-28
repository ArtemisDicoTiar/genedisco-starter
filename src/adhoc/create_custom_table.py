from pathlib import Path

import pandas as pd

model = "ensemble"

project_root = Path('./').absolute().parent.parent

eval_score = pd.read_pickle(project_root / "genedisco_output" / "eval_score.pickle")
print(eval_score)

test_score = pd.read_pickle(project_root / "genedisco_output" / "test_score.pickle")
print(test_score)

results = pd.DataFrame(
    pd.read_pickle(project_root / "genedisco_output" / "results.pickle")
)

results.to_csv(f'./{model}.csv', index=False)
