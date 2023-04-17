import json
import uuid
from unittest.mock import Mock, PropertyMock

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from pycheribenchplot.core.analysis import AnalysisTask
from pycheribenchplot.core.config import AnalysisConfig
from pycheribenchplot.kernel_history.cheribsd_annotations import *
from pycheribenchplot.kernel_history.model import *


@pytest.fixture
def fake_kernel_changes(fake_simple_benchmark):
    """
    Produce a dataframe mimicking the output of CheriBSDKernelFileChanges.

    XXX this may possibly be randomised from the model
    """
    data = {
        "schema": {
            "fields": [{
                "name": "dataset_id",
                "type": "string"
            }, {
                "name": "dataset_gid",
                "type": "string"
            }, {
                "name": "iteration",
                "type": "integer"
            }, {
                "name": "file",
                "type": "string"
            }, {
                "name": "updated",
                "type": "string"
            }, {
                "name": "target_type",
                "type": "string"
            }, {
                "name": "changes",
                "type": "string"
            }, {
                "name": "changes_purecap",
                "type": "string"
            }, {
                "name": "change_comment",
                "type": "string"
            }, {
                "name": "hybrid_specific",
                "type": "boolean"
            }],
            "primaryKey": ["file", "dataset_id", "dataset_gid", "iteration"]
        },
        "data": [
            {
                "dataset_id": str(fake_simple_benchmark.config.uuid),
                "dataset_gid": str(fake_simple_benchmark.config.g_uuid),
                "iteration": -1,
                "file": "cheribsd/sys/fs/smbfs/smbfs_vfsops.c",
                "updated": "20221205",
                "target_type": "kernel",
                "changes": ["user_capabilities", "integer_provenance"],
                "changes_purecap": [],
                "change_comment": None,
                "hybrid_specific": False
            },
            {
                "dataset_id": str(fake_simple_benchmark.config.uuid),
                "dataset_gid": str(fake_simple_benchmark.config.g_uuid),
                "iteration": -1,
                "file": "cheribsd/sys/netinet/udp_usrreq.c",
                "updated": "20221205",
                "target_type": "kernel",
                "changes": ["user_capabilities"],
                "changes_purecap": [],
                "change_comment": None,
                "hybrid_specific": True
            },
            {
                "dataset_id": str(fake_simple_benchmark.config.uuid),
                "dataset_gid": str(fake_simple_benchmark.config.g_uuid),
                "iteration": -1,
                "file": "cheribsd/sys/kern/subr_pctrie.c",
                "updated": "20221205",
                "target_type": "kernel",
                "changes": [],
                "changes_purecap": ["kdb"],
                "change_comment": None,
                "hybrid_specific": False
            },
        ]
    }

    df = pd.read_json(json.dumps(data), orient="table")
    return df.sort_index()


@pytest.fixture
def other_kernel_changes(fake_kernel_changes):
    """
    Prepare another set of changes with one different entry and one
    different changes tag. This is used to verify that we are doing the union
    of the changes correctly.
    """
    df = fake_kernel_changes.reset_index()
    df.iat[0, df.columns.get_loc("file")] = "cheribsd/sys/aaa_file.c"
    df.iat[1, df.columns.get_loc("changes_purecap")] += ["support"]
    return df.set_index(fake_kernel_changes.index.names)


@pytest.fixture
def fake_cdb(fake_simple_benchmark):
    """
    Produce a fake compilation DB consistent with the fake_kernel_changes
    """
    data = {
        "files": [
            "cheribsd/sys/fs/smbfs/smbfs_vfsops.c",
            "cheribsd/sys/netinet/udp_usrreq.c",
            "cheribsd/sys/kern/subr_pctrie.c",
            "cheribsd/sys/kern/subr_vmem.c",
        ]
    }
    return pd.read_json(json.dumps(data))


@pytest.fixture
def other_cdb(fake_cdb):
    df = fake_cdb.copy()
    df.iat[0, df.columns.get_loc("files")] = "cheribsd/sys/aaa_file.c"
    return df


def test_raw_dataset_validate(fake_kernel_changes, fake_simple_benchmark):
    """
    Check the kernel changes dataframe model
    """

    expect = fake_kernel_changes.reset_index()
    expect["dataset_id"] = expect["dataset_id"].map(uuid.UUID)
    expect["dataset_gid"] = expect["dataset_gid"].map(uuid.UUID)
    expect["change_comment"] = expect["change_comment"].astype(object)
    expect = expect.set_index(fake_kernel_changes.index.names)

    schema = RawFileChangesModel.to_schema(fake_simple_benchmark.session)
    df = schema.validate(fake_kernel_changes)

    assert_frame_equal(df, expect)


def test_kernel_file_changes_union(mocker, fake_session, fake_kernel_changes, other_kernel_changes, fake_cdb,
                                   other_cdb):
    """
    Verify that CheriBSDAllFileChanges produces the union of the annotations and
    compilation DB datasets.
    """
    # Prepare two sets of changes as if they were loaded by two different loader tasks
    mock_files_dep = mocker.patch.object(CheriBSDAllFileChanges, "load_files", new_callable=PropertyMock)
    mock_load_task_0 = Mock()
    mock_load_task_1 = Mock()
    mock_load_task_0.df.get.return_value = fake_kernel_changes.copy()
    mock_load_task_1.df.get.return_value = other_kernel_changes.copy()
    mock_files_dep.return_value = [mock_load_task_0, mock_load_task_1]
    mock_cdb_dep = mocker.patch.object(CheriBSDAllFileChanges, "load_cdb", new_callable=PropertyMock)
    mock_cdb_task_0 = Mock()
    mock_cdb_task_1 = Mock()
    mock_cdb_task_0.df.get.return_value = fake_cdb.copy()
    mock_cdb_task_1.df.get.return_value = other_cdb.copy()
    mock_cdb_dep.return_value = [mock_cdb_task_0, mock_cdb_task_1]

    task = CheriBSDAllFileChanges(fake_session, AnalysisConfig())
    task.run()

    annotations = task.df.get()
    expect_files = {
        "cheribsd/sys/fs/smbfs/smbfs_vfsops.c", "cheribsd/sys/netinet/udp_usrreq.c", "cheribsd/sys/kern/subr_pctrie.c",
        "cheribsd/sys/aaa_file.c"
    }
    assert set(annotations.index.unique("file")) == expect_files
    changes = annotations.loc["cheribsd/sys/kern/subr_pctrie.c"]["changes_purecap"]
    assert len(changes) == 2
    assert set(changes) == set(["kdb", "support"])

    cdb = task.compilation_db.get()
    expect_cdb_files = {
        "cheribsd/sys/fs/smbfs/smbfs_vfsops.c", "cheribsd/sys/netinet/udp_usrreq.c", "cheribsd/sys/kern/subr_pctrie.c",
        "cheribsd/sys/kern/subr_vmem.c", "cheribsd/sys/aaa_file.c"
    }
    assert set(cdb["files"]) == expect_cdb_files
    assert len(cdb) == len(expect_cdb_files)
