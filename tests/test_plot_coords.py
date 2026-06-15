from unittest.mock import MagicMock
import pytest
import polars as pl

from pycheribenchplot.core.plot_grid.coords import (
    CoordGenerator,
    CoordGenConfig,
)


@pytest.fixture
def mock_axes():
    return MagicMock()


def test_coord_generator_init_validation(mock_axes):
    # Valid orientations
    assert CoordGenerator(mock_axes, orient="x").orient == "x"
    assert CoordGenerator(mock_axes, orient="y").orient == "y"

    # Invalid orientation
    with pytest.raises(ValueError, match="Invalid orient value"):
        CoordGenerator(mock_axes, orient="invalid")


def test_coord_generator_dependent_vars_validation(mock_axes):
    generator = CoordGenerator(mock_axes, orient="x")
    df = pl.DataFrame({"category": ["A", "B"], "val": [1, 2]})

    # Empty dependent variables list should raise ValueError
    with pytest.raises(
        ValueError, match="At least one dependent_var column must be given"
    ):
        generator.compute_coordinates(df, independent_var="category", dependent_vars=[])

    # Multiple dependent variables should raise AssertionError (current limitation)
    with pytest.raises(AssertionError, match="need to lift this limitation"):
        generator.compute_coordinates(
            df, independent_var="category", dependent_vars=["val", "val2"]
        )


def test_coord_generator_shift_alignment_validation(mock_axes):
    generator = CoordGenerator(mock_axes, orient="x")
    config = CoordGenConfig(shift_by="group")

    # Unaligned shift groups (group A has 2 rows, group B has 1 row)
    df_unaligned = pl.DataFrame(
        {"category": ["X", "Y", "X"], "group": ["A", "A", "B"], "val": [10, 20, 30]}
    )
    with pytest.raises(
        ValueError, match="The input dataframe is not aligned at the shift_by level"
    ):
        generator.compute_coordinates(df_unaligned, "category", ["val"], config=config)

    # Aligned shift groups (both have 2 rows)
    df_aligned = pl.DataFrame(
        {
            "category": ["X", "Y", "X", "Y"],
            "group": ["A", "A", "B", "B"],
            "val": [10, 20, 30, 40],
        }
    )
    res = generator.compute_coordinates(df_aligned, "category", ["val"], config=config)
    assert res is not None


def test_coord_generator_stack_alignment_validation(mock_axes):
    generator = CoordGenerator(mock_axes, orient="x")
    config = CoordGenConfig(stack_by="stack")

    # Unaligned stack groups
    df_unaligned = pl.DataFrame(
        {"category": ["X", "Y", "X"], "stack": ["S1", "S1", "S2"], "val": [10, 20, 30]}
    )
    with pytest.raises(
        ValueError, match="The input dataframe is not aligned at the stack_by level"
    ):
        generator.compute_coordinates(df_unaligned, "category", ["val"], config=config)

    # Aligned stack groups
    df_aligned = pl.DataFrame(
        {
            "category": ["X", "Y", "X", "Y"],
            "stack": ["S1", "S1", "S2", "S2"],
            "val": [10, 20, 30, 40],
        }
    )
    res = generator.compute_coordinates(df_aligned, "category", ["val"], config=config)
    assert res is not None


def test_coord_generator_basic_mapping(mock_axes):
    generator = CoordGenerator(mock_axes, orient="x")
    df = pl.DataFrame(
        {
            "category": ["catB", "catA", "catB", "catC", "catA"],
            "val": [10, 20, 30, 40, 50],
        }
    )

    res = generator.compute_coordinates(df, "category", ["val"])

    # Output should preserve original row ordering
    assert res["__gen_index"].to_list() == [0, 1, 2, 3, 4]

    # Categories should be mapped based on first-appearance order: catB -> 0.0, catA -> 1.0, catC -> 2.0
    assert res["__gen_coord"].to_list() == [0.0, 1.0, 0.0, 2.0, 1.0]

    # With shift_by=None and stack_by=None, offset is centered, and stack is 0
    # ngroups = 1, gap_ratio = 0.9.
    # width = 1.0 * 0.9 / 1 = 0.9
    # group_adjust = -0.45, align_adjust = 0.45
    # offset = 0 * 0.9 - 0.45 + 0.45 = 0.0
    assert res["__gen_offset"].to_list() == [0.0] * 5
    assert res["__gen_stack"].to_list() == [0] * 5


def test_coord_generator_shift_offsets(mock_axes):
    generator = CoordGenerator(mock_axes, orient="x")

    # 2 categories, 2 shift groups -> aligned
    df = pl.DataFrame(
        {
            "category": ["cat1", "cat1", "cat2", "cat2"],
            "group": ["G1", "G2", "G1", "G2"],
            "val": [10, 20, 30, 40],
        }
    )

    # Test center alignment (default)
    config_center = CoordGenConfig(
        shift_by="group", gap_ratio=0.8, pad_ratio=0.9, align="center"
    )
    res_center = generator.compute_coordinates(
        df, "category", ["val"], config=config_center
    )

    # ngroups = 2, gap_ratio = 0.8. width = 0.8 / 2 = 0.4.
    # group_adjust = -0.4, align_adjust = 0.2
    # G1 offset = 0 * 0.4 - 0.4 + 0.2 = -0.2
    # G2 offset = 1 * 0.4 - 0.4 + 0.2 = 0.2
    # width with padding = 0.4 * 0.9 = 0.36
    assert res_center.filter(pl.col("group") == "G1")[
        "__gen_offset"
    ].to_list() == pytest.approx([-0.2, -0.2])
    assert res_center.filter(pl.col("group") == "G2")[
        "__gen_offset"
    ].to_list() == pytest.approx([0.2, 0.2])
    assert res_center["__gen_width"].to_list() == pytest.approx([0.36] * 4)

    # Test left alignment
    config_left = CoordGenConfig(
        shift_by="group", gap_ratio=0.8, pad_ratio=1.0, align="left"
    )
    res_left = generator.compute_coordinates(
        df, "category", ["val"], config=config_left
    )

    # G1 offset = 0 * 0.4 - 0.4 + 0.0 = -0.4
    # G2 offset = 1 * 0.4 - 0.4 + 0.0 = 0.0
    assert res_left.filter(pl.col("group") == "G1")[
        "__gen_offset"
    ].to_list() == pytest.approx([-0.4, -0.4])
    assert res_left.filter(pl.col("group") == "G2")[
        "__gen_offset"
    ].to_list() == pytest.approx([0.0, 0.0])


def test_coord_generator_stack_offsets(mock_axes):
    generator = CoordGenerator(mock_axes, orient="x")

    # 1 category, 3 stack groups
    df = pl.DataFrame(
        {
            "category": ["cat1", "cat1", "cat1"],
            "stack": ["S1", "S2", "S3"],
            "val": [10, 25, 40],
        }
    )

    config = CoordGenConfig(stack_by="stack")
    res = generator.compute_coordinates(df, "category", ["val"], config=config)

    # Stack offsets should be [0, val_0, val_0 + val_1] -> [0, 10, 35]
    assert res["__gen_stack"].to_list() == [0, 10, 35]


def test_coord_generator_complex_shift_and_stack(mock_axes):
    generator = CoordGenerator(mock_axes, orient="x")

    # 2 categories, 2 shift groups, 2 stack groups (total 8 rows, aligned)
    df = pl.DataFrame(
        {
            "category": [
                "cat1",
                "cat1",
                "cat1",
                "cat1",
                "cat2",
                "cat2",
                "cat2",
                "cat2",
            ],
            "group": ["G1", "G1", "G2", "G2", "G1", "G1", "G2", "G2"],
            "stack": ["S1", "S2", "S1", "S2", "S1", "S2", "S1", "S2"],
            "val": [10, 20, 30, 40, 15, 25, 35, 45],
        }
    )

    config = CoordGenConfig(
        shift_by="group", stack_by="stack", gap_ratio=0.8, align="center"
    )
    res = generator.compute_coordinates(df, "category", ["val"], config=config)

    # Verify shift offsets
    # ngroups = 2, gap_ratio = 0.8 -> width = 0.4
    # G1 offset -> -0.2
    # G2 offset -> 0.2
    assert res.filter(pl.col("group") == "G1")[
        "__gen_offset"
    ].to_list() == pytest.approx([-0.2, -0.2, -0.2, -0.2])
    assert res.filter(pl.col("group") == "G2")[
        "__gen_offset"
    ].to_list() == pytest.approx([0.2, 0.2, 0.2, 0.2])

    # Verify stack offsets:
    # cat1 + G1: S1 -> 0, S2 -> 10
    # cat1 + G2: S1 -> 0, S2 -> 30
    # cat2 + G1: S1 -> 0, S2 -> 15
    # cat2 + G2: S1 -> 0, S2 -> 35
    assert res.filter((pl.col("category") == "cat1") & (pl.col("group") == "G1"))[
        "__gen_stack"
    ].to_list() == [0, 10]
    assert res.filter((pl.col("category") == "cat1") & (pl.col("group") == "G2"))[
        "__gen_stack"
    ].to_list() == [0, 30]
    assert res.filter((pl.col("category") == "cat2") & (pl.col("group") == "G1"))[
        "__gen_stack"
    ].to_list() == [0, 15]
    assert res.filter((pl.col("category") == "cat2") & (pl.col("group") == "G2"))[
        "__gen_stack"
    ].to_list() == [0, 35]
