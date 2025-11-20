from typing import List
from pathlib import Path

import pandas as pd


from ntxter.core.reports.dataframes import BaseDataFrameReport


class ReportByGroupIntoMD(BaseDataFrameReport[pd.DataFrame, str]):
    def build(self, df: pd.DataFrame, /, *, grouping: str | List[str], **kwargs) -> str:
        report = f"# Grouping Report {grouping}\n\n"
        for name, grp in df.groupby(grouping):
            report += f"## Group: {name}\n"
            report += grp.drop(columns=grouping).to_markdown() + "\n\n"
    
        return report

    def save(self, pfname: Path, /, *, df: pd.DataFrame, grouping: str | List[str], **kwargs) -> None | str:
        md_report = self.build(df, grouping=grouping)
        if not isinstance(pfname, Path):
            pfname = Path(pfname)
        else:
            pfname.parent.mkdir(parents=True, exist_ok=True)
            pfname.write_text(md_report)
        
        return md_report
