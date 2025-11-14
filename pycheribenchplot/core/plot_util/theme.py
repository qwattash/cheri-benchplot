import matplotlib.pyplot as plt
from matplotlib import colormaps as cmap_registry
from matplotlib.colors import ListedColormap

# Default color cycle, lifted from Seaborn default color cycle
benchplot_tab_colors = [
    "#4c72b0",  # blue
    "#dd8452",  # orange
    "#55a868",  # green
    "#c44e52",  # red
    "#8172b3",  # violet
    "#937860",  # brown
    "#da8bc3",  # purple
    "#8c8c8c",  # gray
    "#ccb974",  # lime green
    "#64b5cd",  # cyan
]

default_color_cycle = plt.cycler("color", benchplot_tab_colors)

# Create and register color maps
benchplot_tab = ListedColormap(benchplot_tab_colors, name="benchplot_tab")

cmap_registry.register(benchplot_tab)

# Default theme configuration, this is applied to the rc_context in each PlotTask.
default_theme = {
    # Axes settings
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.labelcolor": ".15",
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.prop_cycle": default_color_cycle,

    # Grid settings
    "axes.grid.axis": "both",
    "grid.color": ".8",
    "grid.linewidth": 1,
    "grid.linestyle": "-",

    # Ticks
    "xtick.color": ".15",
    "ytick.color": ".15",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "xtick.minor.size": 0,
    "ytick.minor.size": 0,
    "xtick.major.pad": 7,
    "ytick.major.pad": 7,

    # Lines
    "lines.solid_capstyle": "round",
    "lines.linewidth": 1.5,

    # Patches (bars, etc.)
    "patch.linewidth": 0.5,
    "patch.facecolor": "#1f77b4",
    "patch.edgecolor": "white",
    "patch.force_edgecolor": True,

    # Text
    "text.color": ".15",
    "font.family": ["sans-serif"],
    "font.sans-serif": ["Liberation Sans", "Bitstream Vera Sans", "sans-serif"],
    "font.size": 11,
    "axes.labelsize": "medium",
    "axes.titlesize": "large",
    "xtick.labelsize": "medium",
    "ytick.labelsize": "medium",
    "legend.fontsize": "medium",
    "legend.title_fontsize": None,

    # Legend
    "legend.frameon": False,
    "legend.numpoints": 1,
    "legend.scatterpoints": 1,
}
