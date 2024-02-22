import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def impute_data(df):
    is_missing = df[[col for col in df.columns
                     if any(pd.isnull(df[col])) and not all(pd.isnull(df[col]))]]
    imputer = IterativeImputer(missing_values=np.nan)
    imputed = imputer.fit_transform(is_missing)
    imputed_df = pd.DataFrame(imputed,columns=is_missing.columns)
    merged_df = df.combine_first(imputed_df)

    return merged_df