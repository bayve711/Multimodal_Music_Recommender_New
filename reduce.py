import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD

# 1) Load the original data
df = pd.read_csv('data/id_vgg19_mmsr.tsv', sep='\t')
ids = df['id'].copy()
features = df.drop(columns=['id']).astype(np.float32)

# 2) Apply TruncatedSVD (or PCA)
svd = TruncatedSVD(n_components=256, random_state=42)
reduced_features = svd.fit_transform(features)

# 3) Create a new DataFrame
reduced_df = pd.DataFrame(reduced_features, columns=[f'vgg19_{i}' for i in range(256)])
reduced_df.insert(0, 'id', ids)

# 4) Save the result to a new file
reduced_df.to_csv('data/id_vgg19_mmsr_reduced.tsv', sep='\t', index=False)


df_res = pd.read_csv('data/id_resnet_mmsr.tsv', sep='\t')
ids_res = df_res['id'].copy()
features_res = df_res.drop(columns=['id']).astype(np.float32)

svd_res = TruncatedSVD(n_components=256, random_state=42)
reduced_features_res = svd_res.fit_transform(features_res)

reduced_df_res = pd.DataFrame(reduced_features_res, columns=[f'resnet_{i}' for i in range(256)])
reduced_df_res.insert(0, 'id', ids_res)

reduced_df_res.to_csv('data/id_resnet_mmsr_reduced.tsv', sep='\t', index=False)

