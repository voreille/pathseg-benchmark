# %%
import pandas as pd

# %%
df = pd.read_csv("/home/valentin/workspaces/pathseg-benchmark/data/fss/anorak/split_test_fold0.csv")
# %%
df.loc[df["validation_fold"] == "fold0", "split"] = "test"

# %%
df.head()

# %%
df.to_csv("/home/valentin/workspaces/pathseg-benchmark/data/fss/anorak/split_test_fold0.csv", index=False)

# %%
