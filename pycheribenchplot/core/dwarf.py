import json
import re
from collections import namedtuple
from dataclasses import dataclass, field, replace
from enum import Flag, IntFlag, auto
from functools import reduce
from pathlib import Path

import networkx as nx
import pandas as pd
from pandera import Field
from pandera.typing import DataFrame, Index, Series

from ..ext import pydwarf
from .model import DataModel
from .util import gzopen, new_logger


class DWARFStructLayoutModel(DataModel):
    """
    Contains a dump of structure and union data layouts.

    Note that not all information is dumped here. This data model should be sufficient
    to determine padding, alignment and sub-object representablity.

    There is one row for each member, nested structures are unrolled and the
    member name is prefixed by the parent names with the C-style '.' scope operator.
    Offsets are adjusted to reflect the parent structure offset.

    Note that the type ID is not really an itentifier for our purposes. There may be
    multiple DWARF entries that refer to the same structure. This may happen if the same
    structure imported and defined in multiple compilation units.
    The index of this model should be unique, any duplicates must be removed as they
    are expected to refer to the same structure.
    The members are uniquely identified by the combination of (offset, name) which should
    be sufficient for linearized union members that share the same offset.
    """
    file: Index[str]
    line: Index[int]
    base_name: Index[str]
    member_name: Index[str]
    size: Index[int]
    member_size: Index[float] = Field(nullable=True)

    type_id: Series[int]
    member_type_name: Series[str]
    member_offset: Series[float]


class StructLayoutGraph:
    """
    Represent a graph containing structure layouts.

    This provides helper methods to serialize/deserialize the internal graph,
    as well as build structure layout trees.

    The graph nodes are identified by the NodeID tuple below.
    Each node has the following additional attributes:
    - type_id: the original DWARF Die offset
    - type_name: the full type name, including cv-qualifiers
    - size: the size of the structure or field
    """
    #: Type of the graph nodes
    # NodeID = namedtuple("NodeID", ["file", "line", "base_name", "size", "member_name", "member_offset"])
    NodeID = namedtuple("NodeID", ["file", "line", "member", "size", "member_size"])
    Attrsv2 = ["base_name", "member_name", "member_type_name", "member_offset"]

    @dataclass
    class AddLayoutContext:
        """
        Helper context for traversing a structure layout
        """
        #: Normalized "base" name of the top-level structure
        layout_type_name: str
        #: Normalized source file path of the top-level structure
        layout_file: str
        #: Top-level type info
        root_info: pydwarf.TypeInfo
        #: Root node ID
        root_node: "NodeID" = None
        #: Stack of parent member objects
        parent_stack: "list[NodeID]" = field(default_factory=list)
        #: Cumulative structure member offset
        offset: int = 0

        @property
        def parent(self):
            return self.parent_stack[-1] if self.parent_stack else self.root_node

        def __post_init__(self):
            if self.root_node is None:
                self.root_node = StructLayoutGraph.NodeID(file=self.layout_file,
                                                          line=self.root_info.line,
                                                          member=self.layout_type_name,
                                                          size=self.root_info.size,
                                                          member_size=None)
                self.parent_stack = [self.root_node]

    @classmethod
    def load(cls, benchmark, path: Path) -> "StructLayoutGraph":
        """
        Load a GML representation of the structure layout graph
        """
        with gzopen(path, "r") as fd:
            data = json.load(fd)
        g = nx.relabel_nodes(nx.node_link_graph(data), lambda r: cls.NodeID(*r))
        g.graph["roots"] = [cls.NodeID(*r) for r in g.graph["roots"]]

        layout_graph = StructLayoutGraph(benchmark)
        layout_graph.layouts = g
        return layout_graph

    def __init__(self, benchmark):
        self.benchmark = benchmark
        self.logger = benchmark.logger
        self.layouts = nx.DiGraph()
        self.layouts.graph["roots"] = []

    def _normalize_anon(self, name: str) -> str:
        """
        Fixup anonymous structure names.

        These include the full path of the compilation unit, make this relative
        to the benchmark src_path if possible. This will result in more legible
        data.
        """
        if name.find("<anon>") == -1:
            return name
        match = re.match(r"(.*)<anon>\.(.*)\.([0-9]+)", name)
        if not match:
            return name
        path = Path(match.group(2))
        relpath = self._normalize_path(path)
        return match.group(1) + "<anon>." + str(relpath) + "." + match.group(3)

    def _normalize_path(self, strpath: str) -> str:
        """
        Fixup a path string to make it relative to the benchmark src_path.

        This will result in more legible data.
        """
        src_root = self.benchmark.user_config.src_path
        path = Path(strpath)
        if not path.is_relative_to(src_root):
            # No changes
            return strpath
        return str(path.relative_to(src_root))

    def _add_member(self, ctx: "AddLayoutContext", member: pydwarf.Member):
        """
        Create a graph node corresponding to the given member and attach it to the last parent.

        Note that bit offsets are represented as a floating point value for the offset field.
        The same is done for sizes.
        """
        float_offset = (member.offset * 8 + member.bit_offset) / 8
        float_size = (member.size * 8 + member.bit_size) / 8
        member_node = self.NodeID(file=ctx.layout_file,
                                  line=ctx.root_info.line,
                                  member=ctx.parent.member + "." + member.name,
                                  size=ctx.root_info.size,
                                  member_size=float_size)
        if member_node in self.layouts.nodes:
            # Just link the nodes here
            self.layouts.add_edge(ctx.parent, member_node)
            return member_node

        attrs = {
            "type_id": member.type_info.handle,
            "base_name": self._normalize_anon(ctx.root_info.base_name),
            "type_name": self._normalize_anon(member.type_info.type_name),
            "offset": ctx.offset + float_offset,
            "member_line": member.line
        }
        if attrs["offset"] + float_size > ctx.root_info.size:
            self.logger.error(
                "Inconsistent data for %s (Die @ %s %#x), member %s of type "
                "%s @ %#x with size %#x exceeds total structure size %#x", ctx.root_info.type_name, ctx.root_info.file,
                ctx.root_info.handle, member_node.member, member.type_info.type_name, attrs["offset"],
                member_node.member_size, ctx.root_info.size)
            raise RuntimeError("Inconsistent structure layout")

        self.layouts.add_node(member_node, **attrs)
        self.layouts.add_edge(ctx.parent, member_node)
        return member_node

    def _do_add_layout(self, ctx: "AddLayoutContext", type_info: pydwarf.TypeInfo):
        """
        Visit nested members and create the corresponding tree structure in the graph.
        """
        records = []

        for index, member in enumerate(type_info.layout):
            member_node = self._add_member(ctx, member)
            flags = member.type_info.flags
            # Only recurse into composite types
            if (flags & (pydwarf.TypeInfoFlags.kIsStruct | pydwarf.TypeInfoFlags.kIsUnion)) == 0:
                continue
            # Do not recurse into composite type pointers or arrays
            if flags & (pydwarf.TypeInfoFlags.kIsPtr | pydwarf.TypeInfoFlags.kIsArray):
                continue

            nested_ctx = replace(ctx,
                                 parent_stack=ctx.parent_stack + [member_node],
                                 offset=self.layouts.nodes[member_node]["offset"])
            self._do_add_layout(nested_ctx, member.type_info)

    def dump(self, path: Path):
        """
        Write a GML representation of the graph.
        """
        with gzopen(path, "w") as fd:
            json.dump(self.layouts, fd, default=nx.node_link_data)

    def add_layout(self, type_info: pydwarf.TypeInfo):
        """
        Given a TypeInfo object describing a structure union or class,
        produce a tree that represents is hierarchical structure.
        The resulting tree is inserted in the graph.
        """
        ctx = self.AddLayoutContext(layout_type_name=self._normalize_anon(type_info.base_name),
                                    layout_file=self._normalize_path(type_info.file),
                                    root_info=type_info)
        if ctx.root_node in self.layouts.graph["roots"]:
            # We already have this struct, check the attributes
            node_type_name = self.layouts.nodes[ctx.root_node]["type_name"]
            if node_type_name != type_info.type_name:
                self.logger.error("Duplicate node %s, invalid type_name expect: %s found %s", ctx.root_node,
                                  node_type_name, type_info.type_name)
                raise RuntimeError("Invalid duplicate node")
            if ctx.root_node.size != type_info.size:
                self.logger.error("Duplicate node %s, invalid size expect: %s found %s", ctx.root_node,
                                  ctx.root_node.size, type_info.size)
                raise RuntimeError("Invalid duplicate node")
            return

        self.layouts.add_node(ctx.root_node,
                              type_id=type_info.handle,
                              base_name=ctx.layout_type_name,
                              type_name=type_info.type_name,
                              offset=0,
                              member_line=None)
        self._do_add_layout(ctx, type_info)
        self.layouts.graph["roots"] += [ctx.root_node]


class DWARFInfoSource:
    """
    Manage DWARF information for a single object file.
    """
    def __init__(self, benchmark, path: Path):
        self.benchmark = benchmark
        self.logger = benchmark.logger
        self.path = path
        self._dwi = pydwarf.DWARFInspector.load(str(path))

    def build_struct_layout_graph(self, graph: StructLayoutGraph | None = None) -> StructLayoutGraph:
        """
        Produce a networkx graph that represents the structure layouts contained
        by the target object.

        Note that if a graph is provided, nodes will be appended to the existing graph.
        """
        if graph is None:
            graph = StructLayoutGraph(self.benchmark)

        self._dwi.visit_struct_layouts(lambda type_info: graph.add_layout(type_info))
        return graph

    def build_struct_layout_table(self) -> DataFrame[DWARFStructLayoutModel]:
        """
        Produce a flattened representation of the structure layout graph.
        """
        graph = self.build_struct_layout_graph()
        Record = namedtuple("Record", [
            "file", "line", "member_name", "size", "member_size", "type_id", "base_name", "member_type_name",
            "member_offset"
        ])
        records = []

        gl = graph.layouts
        # dfs_visit preorder
        for root_node in graph.layouts.graph["roots"]:
            for node in nx.dfs_preorder_nodes(graph.layouts, root_node):
                record = Record(
                    # Index
                    file=root_node.file,
                    line=root_node.line,
                    base_name=gl.nodes[node]["base_name"],
                    member_name=node.member,
                    size=node.size,
                    member_size=node.member_size,
                    # Data
                    type_id=gl.nodes[node]["type_id"],
                    member_type_name=gl.nodes[node]["type_name"],
                    member_offset=gl.nodes[node]["offset"])
                records.append(record)

        df = pd.DataFrame.from_records(records, columns=Record._fields)
        df.set_index(["file", "line", "base_name", "member_name", "size", "member_size"], inplace=True)
        df = df[~df.index.duplicated(keep="first")]
        return df


class DWARFManager:
    """
    Manages DWARF information for all object files imported by a
    benchmark instance.
    Each source object file is registered using a unique name
    (akin to what is done with the symbolizer). At some point
    we may need to integrate this with the symbolizer in some way.
    """
    def __init__(self, benchmark: "BenchmarkBase"):
        self.logger = new_logger(f"{benchmark.uuid}.DWARFHelper")
        self.benchmark = benchmark
        self.objects = {}

    def register_object(self, obj_key: str, path: Path) -> DWARFInfoSource:
        if obj_key in self.objects:
            # Check that the path also matches
            if self.objects[obj_key].path == path:
                return
        assert obj_key not in self.objects, "Duplicate DWARF object source"
        self.objects[obj_key] = DWARFInfoSource(self.benchmark, path)
        return self.objects[obj_key]

    def get_object(self, obj_key: str) -> DWARFInfoSource | None:
        return self.objects.get(obj_key, None)
