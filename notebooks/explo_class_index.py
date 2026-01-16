# %%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# %%
# Load CSV
df = pd.read_csv(
    "/home/valentin/workspaces/pathseg-benchmark/data/fss/class_index_preview.csv"
)

# Filter small areas
AREA_THRESHOLD = 5000  # adjust
df = df[df["area_um2"] >= AREA_THRESHOLD]
df["area_px"] = df["area_um2"] / (df["mpp"] ** 2)


# %%
g = sns.catplot(
    data=df,
    x="dataset_class_id",
    y="area_px",
    col="dataset_id",
    kind="violin",
    inner="quartile",
    scale="width",
    sharey=False,
    col_wrap=2,
    height=4,
    aspect=1.2,
)

g.set_axis_labels("Dataset class ID", "Area (µm²)")
g.set_titles("Dataset: {col_name}")

plt.tight_layout()
plt.show()

# %%
counts = (
    df
    .groupby(["dataset_id", "dataset_class_id"])
    .size()
    .reset_index(name="count")
)
sns.set_theme(style="whitegrid")

g = sns.catplot(
    data=counts,
    x="dataset_class_id",
    y="count",
    col="dataset_id",
    kind="bar",
    col_wrap=2,
    height=4,
    aspect=1.2,
    sharey=False
)

g.set_axis_labels("Dataset class ID", "Number of occurrences")
g.set_titles("Dataset: {col_name}")

plt.tight_layout()
plt.show()


# %%
# --- merge (what you already did) ---
df_ignite_metadata = pd.read_csv("/home/valentin/workspaces/pathseg-benchmark/data/fss/ignite/metadata.csv")
df_ignite = df[df["dataset_id"] == "ignite"].copy()

# %%

df_merged = pd.merge(
    df_ignite,
    df_ignite_metadata[["sample_id", "split", "validation_fold"]],
    on="sample_id",
    how="left"
)

# %%
# sanity checks
print("rows:", len(df_merged))
print("missing fold:", df_merged["validation_fold"].isna().sum())
print(df_merged["split"].value_counts(dropna=False))

# %%
df_merged["validation_fold"].unique()
# %%

# Keep only train (where folds exist)
df_train = df_merged[df_merged["split"] == "train"].copy()

# Optional: enforce fold ordering fold0..fold4
fold_order = [f"fold{i}" for i in range(5)]
df_train["validation_fold"] = pd.Categorical(df_train["validation_fold"], categories=fold_order, ordered=True)

# Count occurrences per class per fold
counts_fold = (
    df_train
    .groupby(["validation_fold", "dataset_class_id"])
    .size()
    .reset_index(name="count")
)

# Plot: one facet per fold (histogram-like bars)
sns.set_theme(style="whitegrid")
g = sns.catplot(
    data=counts_fold,
    x="dataset_class_id",
    y="count",
    col="validation_fold",
    kind="bar",
    col_wrap=3,
    sharey=False,
    height=4,
    aspect=1.3,
    order=sorted(df_train["dataset_class_id"].unique()),
)
g.set_axis_labels("Class ID", "Occurrences (>= area threshold)")
g.set_titles("{col_name}")
for ax in g.axes.flatten():
    ax.tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.show()
# %%
m = counts_fold["dataset_class_id"] == 13
print(counts_fold[m])
# %%
df_train[(df_train["dataset_class_id"] == 13)]

# %%
