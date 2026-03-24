# %%
import pandas as pd

# %%
df = pd.read_parquet("/home/valentin/workspaces/pathseg-benchmark/data/index/anorak/tile_index_t896_s896.parquet")
# %%
df.head()
# %%
df.info()

# %%
