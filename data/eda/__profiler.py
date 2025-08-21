import pandas as pd
import numpy as np

class DataFrameProfiler:
    def __init__(self, df, cat_thresh=10):
        self.df = df
        self.cat_thresh = cat_thresh
    #
    def missing_counts(self):
        missings = self.df.isnull().sum()
        missings.columns = ['n_missings']
        return missings
    #
    def distinct_counts(self):
        distinct = self.df.nunique(dropna=False)
        distinct.columns = ['n_unique']
        return distinct
    #
    def column_types(self):
        types = {}
        for nm, xf in self.df.items():
            if pd.api.types.is_categorical_dtype(xf):
                cat_dtype = xf.cat.categories.dtype
                ctype = 'cat:'
                #
                if pd.api.types.is_numeric_dtype(cat_dtype):
                    ctype += 'numeric' 
                elif pd.api.types.is_datetime64_any_dtype(cat_dtype):
                    ctype += 'datetime'
                elif pd.api.types.is_bool_dtype(cat_dtype):
                    ctype += 'bool'
                else:
                    ctype += 'object' 
                #
                types[nm] = ctype
            #
            else:            
                if pd.api.types.is_bool_dtype(xf):
                    ctype = 'bool'
                elif pd.api.types.is_numeric_dtype(xf):
                    ctype = 'numeric'
                elif pd.api.types.is_datetime64_any_dtype(xf):
                    ctype = 'datetime'
                elif pd.api.types.is_string_dtype(xf):
                    ctype = 'string'
                else:
                    ctype = 'object'
                #
                types[nm] = ctype
        #
        return types
    #
    def proccess_cat(self, df):
        nrows = len(self.df)
        query = df.query("distinct<@self.cat_thresh")
        category_counts = []
        if len(query)>0:
            for col, xf in self.df[query.index].items():
                cat_counts = xf.value_counts().reset_index()
                cat_counts.columns = ['categories', 'count']
                cat_counts['column'] = col
                #
                cat_counts = cat_counts.assign(category_ratio=cat_counts['count'] / nrows)
                #
                p_obs = cat_counts['category_ratio'].values
                p_uni = np.ones_like(p_obs) / len(p_obs)
                #
                kld = np.sum(p_obs * np.log(p_obs / p_uni))
                #
                cat_counts = cat_counts.assign(
                    **{'Kullback-Leibler': kld,
                       'entropy': -np.sum(p_obs * np.log(p_obs)),
                       'gini': 1 - np.sum(p_obs**2)
                       }
                )
                #
                category_counts.append(cat_counts)
            #
            return pd.concat(category_counts)
        else:
            return 
    #
    @classmethod
    def run(cls, df, cat_threa):
        inst = cls(df, cat_threa)
        missing = inst.missing_counts()
        distinct = inst.distinct_counts()
        types = inst.column_types()
        #
        profiling =  pd.DataFrame({
            'ctype': types,
            'missing': missing,
            'missing_ratio': missing / len(inst.df),
            'distinct': distinct
         })
        #
        cat_counts = inst.proccess_cat(profiling)
        #
        return profiling, cat_counts