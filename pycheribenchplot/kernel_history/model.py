import pandas as pd
from pandera import Field, SchemaModel, check, dataframe_check
from pandera.typing import Index, Series

from pycheribenchplot.core.model import DataModel, GlobalModel
from pycheribenchplot.core.util import new_logger


class CommonFileChangesModel(SchemaModel):
    """
    Common fields that record changes to cheribsd files
    """
    filename: Index[str] = Field(alias="file", check_name=True)
    target_type: Series[str] = Field(isin=["header", "kernel", "lib", "prog"])
    updated: Series[str]
    changes: Series[object]
    changes_purecap: Series[object]
    hybrid_specific: Series[bool]
    change_comment: Series[str] = Field(nullable=True)

    @check("target_type")
    def check_warn_target_null(cls, target_type: Series[str]):
        logger = new_logger("cheribsd-changes-model")
        nulls = target_type.isna()
        offending_files = target_type.index.get_level_values("file")[nulls]
        for fname in offending_files:
            logger.warning("%s has CHERI changes annotation but misses target_type field", fname)
        return pd.Series(True, index=target_type.index)

    @dataframe_check
    def changes_nonnull(cls, df: pd.DataFrame) -> Series[bool]:
        """
        Check that the changes + changes_purecap columns are collectively non-null.
        """
        logger = new_logger("cheribsd-changes-model")
        nulls = (df["changes"].isna() & df["changes_purecap"].isna())
        offending_files = df.index.get_level_values("file")[nulls]
        for fname in offending_files:
            logger.error("%s has CHERI changes annotation but have no changes category", fname)
        return ~nulls


class RawFileChangesModel(DataModel, CommonFileChangesModel):
    """
    Cheribsd file changes for a specific kernel configuration
    """
    pass


class AllFileChangesModel(GlobalModel, CommonFileChangesModel):
    """
    Aggregate cheribsd file changes model.
    This is the union of all the :class:`RawFileChangesModel`s.
    """
    pass


class CompilationDBModel(DataModel):
    """
    The list of files touched during the compilation process.
    This is specific to a 'benchmark' run.
    """
    files: Series[str]


class AllCompilationDBModel(GlobalModel):
    """
    Union of all the files touched during the compilation process.
    Aggregated union of all benchmark runs.
    """
    # Must at least have an index
    index: Index[int]
    files: Series[str]


class LoCCountModel(GlobalModel):
    """
    SLoC count by file
    """
    filename: Index[str] = Field(alias="file", check_name=True)
    code: Series[int]
    comment: Series[int]
    blank: Series[int]


class LoCDiffModel(GlobalModel):
    """
    SLoC changes by file
    """
    filename: Index[str] = Field(alias="file", check_name=True)
    how: Index[str] = Field(isin=["added", "same", "modified", "removed"])
    code: Series[int]
    comment: Series[int]
    blank: Series[int]
