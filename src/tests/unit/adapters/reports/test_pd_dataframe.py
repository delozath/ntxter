from pathlib import Path


import pandas as pd
import numpy as np

ROWS = 200
COLNAMES = ['A', 'B', 'C', 'D', 'E']
BINS = 5


from ntxter.adapters.reports.pd_dataframe import ReportByGroupIntoMD

def gen_data():
    data = np.random.normal(0, 1, size=(ROWS, len(COLNAMES)))
    df = pd.DataFrame(data, columns=COLNAMES)

    grouping = COLNAMES[-1]
    df[grouping] = pd.cut(df[grouping], bins=BINS, labels=[f'group_{i}' for i in range(BINS)])

    return df, grouping

def test_PandasDFtoMDReport_build():
    df, group = gen_data()

    cls = ReportByGroupIntoMD()
    report = cls.build(df, grouping=group)
    print(report)
    assert True #[PASSED] md watched on printing output in CLI

def test_PandasDFtoMDReport_save():
    df, group = gen_data()
    
    pfname = Path('.') / 'random_name_4213/report.md'
    cls = ReportByGroupIntoMD()
    md_report = cls.save(pfname, df=df, grouping=group)

    assert pfname.exists()

    with open(pfname, 'r') as f:
        file_content = f.read()
    
    assert file_content == md_report