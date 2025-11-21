from typing import List
from pathlib import Path

import pandas as pd


from ntxter.core.reports.dataframes import BaseDataFrameReport

def base_save(pfname: Path, md_report: str) -> None:
    if not isinstance(pfname, Path):
        pfname = Path(pfname)
    
    pfname.parent.mkdir(parents=True, exist_ok=True)
    pfname.write_text(md_report)
    

class ReportIntoMD(BaseDataFrameReport[pd.DataFrame, str]):
    def build(self, df: pd.DataFrame, /, **kwargs) -> str:
        report = f"# Pandas DataFrame Report\n\n"
        report += df.to_markdown()
        return report

    def save(self, pfname: Path, /, *, df: pd.DataFrame, **kwargs) -> None | str:
        md_report = self.build(df)
        base_save(pfname, md_report)
        return md_report


class ReportByGroupIntoMD(BaseDataFrameReport[pd.DataFrame, str]):
    def build(self, df: pd.DataFrame, /, *, grouping: str | List[str], **kwargs) -> str:
        report = f"# Pandas DataFrame Reporting by Groups: {grouping}\n\n"
        for name, grp in df.groupby(grouping):
            report += f"## Group: {name}\n"
            report += grp.drop(columns=grouping).to_markdown() + "\n\n"
    
        return report

    def save(self, pfname: Path, /, *, df: pd.DataFrame, grouping: str | List[str], **kwargs) -> None | str:
        md_report = self.build(df, grouping=grouping)
        base_save(pfname, md_report)
        return md_report
