class LLVMSubobjectSizeDistribution(PlotTask, StructLayoutAnalysisMixin):
    """
    Generate plots showing the distribution of subobject bounds sizes.

    This was originally designed to work with compiler diagnostics
    we should be able to adapt it to use both diagnostics and static data.
    """
    public = True
    task_namespace = "subobject"
    task_name = "subobject-size-distribution-plot"

    @output
    def size_distribution(self):
        return PlotTarget(self, "size-distribution-kern")

    def load_one_dataset(self, gen_task: ExtractImpreciseSubobject) -> pl.DataFrame:
        # The requested top for the member, this rounds up the size to the next byte
        # boundary for bit fields.
        req_top = (MemberBounds.offset + StructMember.size + func.min(func.coalesce(StructMember.bit_size, 0), 1))
        # yapf: disable
        all_members = (
            select(
                StructType.file, StructType.line,
                StructType.name, MemberBounds.name.label("member_name"),
                MemberBounds.offset, StructMember.size,
                req_top.label("req_top"), MemberBounds.base, MemberBounds.top
            )
            .join(MemberBounds.member_entry)
            .join(MemberBounds.owner_entry)
            .where(
                (MemberBounds.base < MemberBounds.offset) |
                (MemberBounds.top > MemberBounds.offset + StructMember.size +
                 func.coalesce(func.min(StructMember.bit_size, 0), 1))
            )
        )
        # yapf: enable
        df = pl.read_database(imprecise_members, gen_task.struct_layout_db.sql_engine)
        return df

    def _plot_size_distribution(self, df: pl.DataFrame, target: PlotTarget):
        """
        Helper to plot the distribution of subobject bounds sizes
        """
        # Determine buckets we are going to use
        min_size = max(df["size"].min(), 1)
        max_size = max(df["size"].max(), 1)
        log_buckets = range(int(np.log2(min_size)), int(np.log2(max_size)) + 1)
        buckets = [2**i for i in log_buckets]

        with new_figure(target.paths()) as fig:
            ax = fig.subplots()
            sns.histplot(df, x="size", stat="count", bins=buckets, ax=ax)
            ax.set_yscale("log", base=10)
            ax.set_xscale("log", base=2)
            ax.set_xlabel("size (bytes)")
            ax.set_ylabel("# of csetbounds")

    def run_plot(self):
        containers = self.data.all_layouts.get()
        ## XXX Port this
        # Note that it would be nice if we could also read stats
        # from the compiler-emitted setbounds and see how these distributions
        # compare.

        # # Filter only setbounds that are marked kind=subobject
        # data_df = df.loc[df["kind"] == SetboundsKind.SUBOBJECT.value]

        # sns.set_theme()

        # show_df = data_df.loc[data_df["src_module"] == "kernel"]
        # self._plot_size_distribution(show_df, self.size_distribution_kernel)

        # show_df = data_df.loc[data_df["src_module"] != "kernel"]
        # self._plot_size_distribution(show_df, self.size_distribution_modules)

        # self._plot_size_distribution(data_df, self.size_distribution_all)
