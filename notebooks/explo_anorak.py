# %%
import pandas as pd
import numpy as np

# %%
split_df = pd.read_csv(
    "/home/valentin/workspaces/pathseg-benchmark/data/ANORAK_20x/split_df.csv"
)
class_ratios = pd.read_csv(
    "/home/valentin/workspaces/pathseg-benchmark/data/ANORAK_20x/class_ratios.csv"
)

# %%
split_df

# %%
out = (
    split_df.loc[split_df["is_val"]]
      .assign(
          sample_id=split_df["image_id"],
          validation_fold=lambda x: "fold" + x["fold"].astype(str),
          split="train",  # usually "train" because val is part of training CV, not held-out test
      )[["sample_id", "validation_fold", "split"]]
      .drop_duplicates(subset=["sample_id"])  # should already be unique if CV is consistent
      .reset_index(drop=True)
)
# %%

out["validation_fold"].unique()
# %%
out

# %%
out["split"] = "train"
out.to_csv(
    "/home/valentin/workspaces/pathseg-benchmark/data/fss/anorak/split.csv", index=False
)


# %%
class_ratios

# %%
df = split_df.merge(class_ratios, on="image_id", how="left")

# %%

df_train = df[df["is_train"]]
# %%
df_train[df_train["label4_ratio"] > 0.5]

# %%
