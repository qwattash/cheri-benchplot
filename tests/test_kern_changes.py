import json
import uuid

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from pycheribenchplot.core.analysis import AnalysisTask
from pycheribenchplot.kernel_history.analysis import *
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
                "file": "/cheribsd/sys/fs/smbfs/smbfs_vfsops.c",
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
                "file": "/cheribsd/sys/netinet/udp_usrreq.c",
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
                "file": "/cheribsd/sys/kern/subr_pctrie.c",
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


def test_kernel_file_changes_union(mocker, mock_scheduler, fake_session, fake_kernel_changes, fake_benchmark_factory):
    """
    Try to load the test dataset and produce the union of the changes
    """
    # b1 = fake_benchmark_factory(randomize_uuid=True)
    # b2 = fake_benchmark_factory(randomize_uuid=True)

    # raw_data = RawFileChangesModel.to_schema(fake_session).validate(fake_kernel_changes)
    # tmp = raw_data.reset_index()
    # b1_data = tmp.assign(dataset_id=b1.uuid).set_index(raw_data.index.names)
    # b2_data = tmp.assign(dataset_id=b2.uuid).set_index(raw_data.index.names)

    # loader_1 =

    # mocker.patch.object(b1, "run", new=lambda self_: self_.output_map["df"].assign(b1_data))
    # mocker.patch.object(b2, "run", new=lambda self_: self_.output_map["df"].assign(b2_data))

    # union_task = CheriBSDAllFileChanges()

    # mock_scheduler.add_task()
    pass
