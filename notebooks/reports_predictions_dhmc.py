# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# %%
predictions_dir = Path(
    "/home/valentin/workspaces/pathseg-benchmark/data/wsi/DHMC/conformal_predictions/fragrant-music-139"
    # "/home/valentin/workspaces/pathseg-benchmark/data/wsi/DHMC/conformal_predictions/giddy-spaceship-137"
)

dhmc_metadata_df = pd.read_csv(
    "/mnt/nas7/data/Personal/Valentin/DartmouthLungCancerHistologyDataset/MetaData_Release_1.0.csv"
)
dhmc_metadata_df["wsi_id"] = dhmc_metadata_df["File Name"].str.split(".").str[0]

predictions_files = list(predictions_dir.glob("*stats.csv"))

predictions_df = pd.concat(
    [pd.read_csv(f) for f in predictions_files],
    ignore_index=True,
)
# %%
GT_MAP = {
    "micropapillary": "micropapillary",
    "solid": "solid",
    "papillary": "papillary",
    "acinar": "acinar",
    "lepidic": "lepidic",
}

gt_df = dhmc_metadata_df[["wsi_id", "Class"]].copy()
gt_df["gt_pattern"] = gt_df["Class"].map(GT_MAP)

gt_df = gt_df.dropna(subset=["gt_pattern"])
gt_df.head()
# %%
# %%
PATTERN_ORDER = [
    "cribriform",
    "micropapillary",
    "solid",
    "papillary",
    "acinar",
    "lepidic",
]


# %%
def dominant_pattern_from_rows(
    df: pd.DataFrame,
    area_col: str = "area_px",
    drop_zero: bool = True,
) -> pd.DataFrame:
    df = df.copy()

    invalid = {
        "background",
        "",
        "__fg_union__",
        "singleton_conformal_set",
        "empty_conformal_set",
        "multi_class_conformal_set",
    }
    df = df[~df["pattern_name"].isin(invalid)]

    df["pattern_name"] = pd.Categorical(
        df["pattern_name"],
        categories=PATTERN_ORDER,
        ordered=True,
    )

    df = df.sort_values(
        ["wsi_id", area_col, "pattern_name"],
        ascending=[True, False, True],
    )

    out = (
        df.groupby("wsi_id", as_index=False)
        .first()[["wsi_id", "pattern_name", area_col]]
        .rename(columns={"pattern_name": "pred_pattern", area_col: "pred_area"})
    )

    if drop_zero:
        out = out[out["pred_area"] > 0].copy()

    return out


# %%
SCENARIO_CONFIGS = {
    "head_b_argmax": {
        "stat_type": "head_b",
        "prediction_type": "argmax",
        "compartment_name": None,
    },
    "tumor_x_head_b_argmax": {
        "stat_type": "a_by_b",
        "prediction_type": "argmax",
        "compartment_name": "Tumor epithelium",
    },
    "head_b_safe": {
        "stat_type": "head_b",
        "prediction_type": "safe",
        "compartment_name": None,
    },
    "tumor_x_head_b_safe": {
        "stat_type": "a_by_b",
        "prediction_type": "safe",
        "compartment_name": "Tumor epithelium",
    },
}


# %%
def build_scenario(predictions_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    mask = predictions_df["stat_type"].eq(cfg["stat_type"])
    mask &= predictions_df["prediction_type"].eq(cfg["prediction_type"])

    if cfg["compartment_name"] is not None:
        mask &= predictions_df["compartment_name"].eq(cfg["compartment_name"])

    return dominant_pattern_from_rows(predictions_df[mask], area_col="area_px")


# %%
scenario_tables = {
    name: build_scenario(predictions_df, cfg) for name, cfg in SCENARIO_CONFIGS.items()
}


# %%


def evaluate_scenario(
    scenario_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    scenario_name: str,
):
    eval_df = gt_df.merge(scenario_df, on="wsi_id", how="inner").copy()

    labels = [
        p
        for p in PATTERN_ORDER
        if p in set(eval_df["gt_pattern"]) | set(eval_df["pred_pattern"])
    ]

    y_true = eval_df["gt_pattern"]
    y_pred = eval_df["pred_pattern"]

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    summary_df = pd.DataFrame(
        [
            {
                "scenario": scenario_name,
                "n_slides": len(eval_df),
                "accuracy": accuracy_score(y_true, y_pred),
                "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
                "macro_f1": f1_score(
                    y_true, y_pred, labels=labels, average="macro", zero_division=0
                ),
                "weighted_f1": f1_score(
                    y_true, y_pred, labels=labels, average="weighted", zero_division=0
                ),
                "macro_precision": precision_score(
                    y_true, y_pred, labels=labels, average="macro", zero_division=0
                ),
                "macro_recall": recall_score(
                    y_true, y_pred, labels=labels, average="macro", zero_division=0
                ),
            }
        ]
    )

    report_df = pd.DataFrame(
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            output_dict=True,
            zero_division=0,
        )
    ).T

    return eval_df, cm_df, summary_df, report_df


# %%


def plot_confusion_matrix(
    cm_df,
    title="Confusion matrix",
    normalize=False,
    figsize=(6, 5),
):
    cm = cm_df.copy()

    if normalize:
        cm = cm.div(cm.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm.values)

    ax.set_xticks(range(len(cm.columns)))
    ax.set_yticks(range(len(cm.index)))
    ax.set_xticklabels(cm.columns, rotation=45, ha="right")
    ax.set_yticklabels(cm.index)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_title(title)

    # annotate cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm.iloc[i, j]
            text = f"{val:.2f}" if normalize else f"{int(val)}"
            ax.text(j, i, text, ha="center", va="center")

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()


# %%

all_summaries = []
all_eval = {}
all_cm = {}
all_reports = {}

for name, sdf in scenario_tables.items():
    eval_df, cm_df, summary_df, report_df = evaluate_scenario(sdf, gt_df, name)
    all_eval[name] = eval_df
    all_cm[name] = cm_df
    all_reports[name] = report_df
    all_summaries.append(summary_df)

summary_df = pd.concat(all_summaries, ignore_index=True)
summary_df

# %%
plot_confusion_matrix(
    all_cm["head_b_argmax"],
    title="Normalized confusion matrix - H-optimus-1",
    normalize=True,
)
# %%
plot_confusion_matrix(
    all_cm["tumor_x_head_b_argmax"],
    title="Normalized confusion matrix - H-optimus-1 (only tumor)",
    normalize=True,
)



# %%
plot_confusion_matrix(
    all_cm["tumor_x_head_b_safe"],
    title="Normalized confusion matrix - tumor_x_head_b_safe",
    normalize=True,
)