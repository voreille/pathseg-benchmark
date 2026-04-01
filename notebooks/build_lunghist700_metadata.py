# %%
import random

import pandas as pd

# %%
df = pd.read_csv(
    "/home/valentin/workspaces/pathseg-benchmark/data/external_val/LungHist700/metadata.csv"
)
# %%
df.head()
# %%
df.info()

# %%
patient_ids = df["patient_id"].unique()

class_names = df["class_name"].unique()
# %%
class_names
# %%
class_by_patient = {}
for patient_id in patient_ids:
    class_labels = df[df["patient_id"] == patient_id]["class_name"].unique().tolist()
    class_by_patient[patient_id] = class_labels


# %%
class_by_patient

# %%
patient_by_class = {}
for class_name in class_names:
    patient_ids = df[df["class_name"] == class_name]["patient_id"].unique().tolist()
    patient_by_class[class_name] = patient_ids
# %%
patient_by_class

# %%
class_by_patient[1]
# %%
support_patient_ids = []
for class_name in class_names:
    patient_id = random.choice(patient_by_class[class_name])
    support_patient_ids.append(patient_id)

# %%
support_patient_ids

# %%
class_id_mapping = {
    "nor": 0,
    "aca_bd": 1,
    "aca_md": 2,
    "aca_pd": 3,
    "scc_bd": 4,
    "scc_md": 5,
    "scc_pd": 6,
}
df["label_id"] = df["class_name"].map(class_id_mapping)

# %%
df["sample_id"] = df["filename"]
df["image_relpath"] = df["filename"].apply(lambda x: f"{x}.png")

# %%
df["split"] = "test"
df.loc[df["patient_id"].isin(support_patient_ids), "split"] = "train"

# %%
df.to_csv(
    "/home/valentin/workspaces/pathseg-benchmark/data/external_val/LungHist700/metadata_formatted.csv",
    index=False,
)

# %%
