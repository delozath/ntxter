import pandas as pd
import numpy as np

class DataProfiler:   
    def missing_counts(self, df):
        missings = df.isnull().sum()
        missings.columns = ['n_missings']
        return missings
    
    def distinct_counts(self, df):
        distinct = df.nunique(dropna=False)
        distinct.columns = ['n_unique']
        return distinct
    
    def column_types(self, df):
        types = {}
        for nm, xf in df.items():
            if pd.api.types.is_categorical_dtype(xf):
                cat_dtype = xf.cat.categories.dtype
                ctype = 'cat:'
                
                if pd.api.types.is_numeric_dtype(cat_dtype):
                    ctype += 'numeric' 
                elif pd.api.types.is_datetime64_any_dtype(cat_dtype):
                    ctype += 'datetime'
                elif pd.api.types.is_bool_dtype(cat_dtype):
                    ctype += 'bool'
                else:
                    ctype += 'object' 
                
                types[nm] = ctype
            
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
                
                types[nm] = ctype
        
        return types
    
    def proccess_cat(self, df, profile, min_categories):
        nrows = len(df)
        query = profile.query("distinct<@min_categories")
        category_counts = []
        if len(query)>0:
            for col, xf in df[query.index].items():
                cat_counts = xf.value_counts().reset_index()
                cat_counts.columns = ['categories', 'count']
                cat_counts['column'] = col
                
                cat_counts = cat_counts.assign(category_ratio=cat_counts['count'] / nrows)
                
                p_obs = cat_counts['category_ratio'].values
                p_uni = np.ones_like(p_obs) / len(p_obs)
                
                kld = np.sum(p_obs * np.log(p_obs / p_uni))
                
                cat_counts = cat_counts.assign(
                    **{'Kullback-Leibler': kld,
                       'entropy': -np.sum(p_obs * np.log(p_obs)),
                       'gini': 1 - np.sum(p_obs**2)
                       }
                )
                
                category_counts.append(cat_counts)
            
            return pd.concat(category_counts)
        else:
            return 

    def run(self, df, min_categories=10):
        missing = self.missing_counts(df)
        distinct = self.distinct_counts(df)
        types = self.column_types(df)
        
        profiling =  pd.DataFrame({
            'ctype': types,
            'missing': missing,
            'missing_ratio': missing / len(df),
            'distinct': distinct
         })
        
        cat_counts = self.proccess_cat(df, profiling, min_categories)
        return profiling, cat_counts