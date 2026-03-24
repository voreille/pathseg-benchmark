# %%
from pathlib import Path

import pandas as pd


# %%
cptac_slides_dir = Path("/mnt/nas6/data/CPTAC/CPTAC-LUAD_v12/LUAD")
clinical_csv = Path("/mnt/nas6/data/CPTAC/TCIA_CPTAC_LUAD_Pathology_Data_Table.csv")

df_cptac = pd.read_csv(clinical_csv)
# %%

pattern_map_cptac = {
    "papillary": "Lung Papillary Adenocarcinoma",
    "acinar": "Lung Acinar Adenocarcinoma",
    "solid": "Lung Solid Pattern Predominant Adenocarcinoma",
    "micropapillary": "Lung Micropapillary Adenocarcinoma",
}

# %%
histo_types_list = df_cptac["Tumor_Histological_Type"].unique()


# %%
histo_types_list = [h for h in histo_types_list if isinstance(h, str)]

pattern_map_cptac = {
    "lepidic": [
        "Lepidic adenocarcinoma",
    ],
    "acinar": [
        "Acinar adenocarcinoma",
        "Adenocarcinoma, acinar predominant",
        "Adenocarcinoma, acinic subtype",
        "Adenocarcinoma, acinar and papillary predominant.",
    ],
    "papillary": [
        "Papillary adenocarcinoma",
    ],
    "micropapillary": [
        "Micropapillary adenocarcinoma",
    ],
    "solid": [
        "Solid adenocarcinoma",
    ],
}
