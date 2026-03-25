# %%
import random

import pandas as pd

# %%
df = pd.read_csv("/home/valentin/workspaces/pathseg-benchmark/data/external_val/LungHist700/metadata.csv")
# %%
df.head()
# %%
df.info()

# %%
patient_ids = df["patient_id"].unique()

class_names = df["class_name"].unique()
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
