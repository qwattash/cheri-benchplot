import pandas as pd


def generalized_xs(df: pd.DataFrame,
                   match: list | None = None,
                   levels: list | None = None,
                   complement=False,
                   droplevels=False,
                   **kwargs):
    """
    Generalized cross section that allows slicing on multiple named levels.
    Example:
    Given a dataframe, generaized_xs(df, [0, 1], levels=["k0", "k1"]) gives:

     k0 | k1 | k2 || V
     0  | 0  | 0  || 1
     0  | 0  | 1  || 2
     0  | 1  | 0  || 3  generalized_xs()   k0 | k1 | k2 || V
     0  | 1  | 1  || 4 ==================> 0  | 1  | 0  || 3
     1  | 0  | 0  || 5                     0  | 1  | 1  || 4
     1  | 0  | 1  || 6
     1  | 1  | 0  || 7
     1  | 1  | 1  || 8

    :param df: The dataframe to cross section
    :param match: Optional list of match values
    :param levels: Optional list of match levels, must have the same number of
    entries as ``match``
    :param complement: Return the complement of the matching rows, default False.
    :param droplevels: Whether to remove the match levels from the result, default False.
    :param \**kwargs: level=<match-value> pairs, behave as ``match`` and ``levels``
    """
    if match is None:
        match = []
    if levels is None:
        levels = []
    levels.extend(kwargs.keys())
    match.extend(kwargs.values())

    assert len(match) == len(levels)
    nlevels = len(df.index.names)
    slicer = [slice(None)] * nlevels
    for m, level_name in zip(match, levels):
        level_idx = df.index.names.index(level_name)
        slicer[level_idx] = m
    sel = pd.Series(False, index=df.index)
    sel.loc[tuple(slicer)] = True
    if complement:
        sel = ~sel
    result = df[sel]
    if droplevels:
        if nlevels > len(levels):
            result = result.droplevel(levels)
        else:
            result = result.reset_index()
    return result
