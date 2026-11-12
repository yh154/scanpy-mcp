"""Tools wrapping the scanpy.pl public plotting API (scanpy commit e099d9bcfdfc5ef065355c168ba2f639349939e7)."""
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Literal

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; set before pyplot is used
import matplotlib.pyplot as plt  # noqa: E402
import scanpy as sc  # noqa: E402
from fastmcp import FastMCP  # noqa: E402
from matplotlib import rc_context  # noqa: E402

REFERENCE = "https://github.com/scverse/scanpy/tree/e099d9bcfdfc5ef065355c168ba2f639349939e7/src/scanpy/plotting"
scanpy_pl_mcp = FastMCP(name="scanpy_pl")


def _read(h5ad_path):
    path = Path(h5ad_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"h5ad file not found: {h5ad_path}")
    return sc.read_h5ad(path)


def _require_obs(adata, key, what):
    if key not in adata.obs:
        raise ValueError(f"{what} '{key}' not found in adata.obs (columns: {list(adata.obs.columns)})")


def _out_dir(output_dir, prefix):
    base = Path(output_dir).expanduser().resolve() if output_dir else None
    if base:
        base.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix + "_", dir=base)).resolve()


def _save_open_figures(out_dir, stem, description):
    """Save every open matplotlib figure as PNG, close them, return artifact entries."""
    artifacts = []
    for i, num in enumerate(plt.get_fignums(), start=1):
        path = out_dir / f"{stem}_{i}.png"
        plt.figure(num).savefig(path, dpi=150, bbox_inches="tight")
        artifacts.append({"description": description, "path": str(path.resolve())})
    plt.close("all")
    if not artifacts:
        raise RuntimeError("scanpy produced no figure")
    return artifacts


@contextmanager
def _fresh_figures():
    """Start from no open figures and always close them, so a failed call cannot leak into the next artifact."""
    plt.close("all")
    try:
        yield
    finally:
        plt.close("all")


def _result(message, artifacts, **extra):
    return {"message": message, "reference": REFERENCE, "artifacts": artifacts, **extra}


@scanpy_pl_mcp.tool()
def scanpy_pl_umap(
    h5ad_path: Annotated[str, "h5ad file with an X_umap embedding in adata.obsm"],
    color: Annotated[list[str], "Genes (var_names) and/or obs columns to color cells by; one panel each"],
    size: Annotated[float | None, "Dot size; None uses scanpy's automatic size"] = None,
    frameon: Annotated[bool | None, "Draw the axes frame; None keeps scanpy default"] = None,
    ncols: Annotated[int, "Number of panel columns when several colors are given"] = 4,
    vmax: Annotated[str | float | None, "Upper color limit, number or percentile string such as 'p99'"] = None,
    add_outline: Annotated[bool, "Draw an outline around each dot"] = False,
    legend_loc: Annotated[str | None, "Categorical legend placement, e.g. 'right margin', 'on data', 'none'"] = "right margin",
    legend_fontsize: Annotated[float | None, "Legend font size"] = None,
    legend_fontoutline: Annotated[int | None, "Outline width of on-data legend text"] = None,
    title: Annotated[str | None, "Panel title; None uses the color key"] = None,
    palette: Annotated[str | None, "Matplotlib palette/colormap name for categorical colors"] = None,
    use_raw: Annotated[bool | None, "Use adata.raw for gene expression; None keeps scanpy default"] = None,
    figsize: Annotated[list[float] | None, "Per-panel figure size [width, height] in inches"] = None,
    output_dir: Annotated[str | None, "Base output directory; a fresh subdirectory is created"] = None,
) -> dict:
    """Plot cells on the stored UMAP embedding colored by genes or annotations (scanpy.pl.umap).
    Input is an h5ad with X_umap plus color keys; output is a PNG of the UMAP panel(s).
    """
    adata = _read(h5ad_path)
    if "X_umap" not in adata.obsm:
        raise ValueError("adata.obsm['X_umap'] is missing; compute a UMAP before plotting")
    out_dir = _out_dir(output_dir, "scanpy_pl_umap")
    kwargs = dict(
        color=color, size=size, frameon=frameon, ncols=ncols, vmax=vmax, add_outline=add_outline,
        legend_loc=legend_loc, legend_fontsize=legend_fontsize, legend_fontoutline=legend_fontoutline,
        title=title, palette=palette, use_raw=use_raw,
    )
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    with _fresh_figures():
        with rc_context({"figure.figsize": tuple(figsize)} if figsize else {}):
            sc.pl.umap(adata, show=False, **kwargs)
        artifacts = _save_open_figures(out_dir, "umap", "UMAP colored by " + ", ".join(color))
    return _result("UMAP plot saved", artifacts, n_cells=int(adata.n_obs), color=color)


@scanpy_pl_mcp.tool()
def scanpy_pl_violin(
    h5ad_path: Annotated[str, "h5ad file with expression and/or obs annotations"],
    keys: Annotated[list[str], "Genes (var_names) or numeric obs columns to plot"],
    groupby: Annotated[str | None, "obs column that defines the violin groups; None plots all cells together"] = None,
    stripplot: Annotated[bool, "Overlay individual cells as dots"] = True,
    log: Annotated[bool, "Plot on a log scale"] = False,
    use_raw: Annotated[bool | None, "Use adata.raw for gene expression; None keeps scanpy default"] = None,
    multi_panel: Annotated[bool, "Draw one panel per key"] = False,
    rotation: Annotated[float | None, "Rotation of x tick labels in degrees"] = None,
    inner: Annotated[str | None, "Seaborn violin interior, e.g. 'box', 'quartile', 'point'; None keeps default"] = None,
    figsize: Annotated[list[float] | None, "Figure size [width, height] in inches"] = None,
    output_dir: Annotated[str | None, "Base output directory; a fresh subdirectory is created"] = None,
) -> dict:
    """Draw violin plots of genes or numeric annotations, optionally grouped (scanpy.pl.violin).
    Input is an h5ad plus keys and an optional obs grouping column; output is a PNG.
    """
    adata = _read(h5ad_path)
    if groupby is not None:
        _require_obs(adata, groupby, "groupby column")
    out_dir = _out_dir(output_dir, "scanpy_pl_violin")
    kwargs = dict(rotation=rotation, use_raw=use_raw)
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    if inner is not None:
        kwargs["inner"] = inner
    with _fresh_figures():
        with rc_context({"figure.figsize": tuple(figsize)} if figsize else {}):
            sc.pl.violin(adata, keys, groupby=groupby, stripplot=stripplot, log=log,
                         multi_panel=multi_panel, show=False, **kwargs)
        artifacts = _save_open_figures(out_dir, "violin", "Violin plot of " + ", ".join(keys))
    return _result("Violin plot saved", artifacts, keys=keys, groupby=groupby)


@scanpy_pl_mcp.tool()
def scanpy_pl_dotplot(
    h5ad_path: Annotated[str, "h5ad file with expression and the groupby annotation"],
    var_names: Annotated[
        list[str] | dict[str, list[str]],
        "Genes to show: a list, or a dict mapping group label to its marker genes (e.g. cell type -> genes)",
    ],
    groupby: Annotated[str, "obs column that defines the rows (cell groups)"],
    dendrogram: Annotated[bool, "Add a dendrogram (computed by scanpy if absent)"] = False,
    standard_scale: Annotated[Literal["var", "group"] | None, "Scale expression to 0-1 per gene or per group"] = None,
    use_raw: Annotated[bool | None, "Use adata.raw for expression; None keeps scanpy default"] = None,
    log: Annotated[bool, "Log-transform expression before averaging"] = False,
    swap_axes: Annotated[bool, "Put genes on the y axis and groups on the x axis"] = False,
    cmap: Annotated[str, "Matplotlib colormap for mean expression"] = "Reds",
    title: Annotated[str | None, "Figure title"] = None,
    figsize: Annotated[list[float] | None, "Figure size [width, height] in inches"] = None,
    output_dir: Annotated[str | None, "Base output directory; a fresh subdirectory is created"] = None,
) -> dict:
    """Draw a dot plot of mean expression and fraction of expressing cells per group (scanpy.pl.dotplot).
    Input is an h5ad, genes (or marker dict) and a groupby column; output is a PNG.
    """
    adata = _read(h5ad_path)
    _require_obs(adata, groupby, "groupby column")
    out_dir = _out_dir(output_dir, "scanpy_pl_dotplot")
    kwargs = dict(standard_scale=standard_scale, use_raw=use_raw, title=title,
                  figsize=tuple(figsize) if figsize else None)
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    with _fresh_figures():
        sc.pl.dotplot(adata, var_names, groupby, dendrogram=dendrogram, log=log,
                      swap_axes=swap_axes, cmap=cmap, show=False, **kwargs)
        artifacts = _save_open_figures(out_dir, "dotplot", f"Dot plot grouped by {groupby}")
    return _result("Dot plot saved", artifacts, groupby=groupby,
                   n_groups=int(adata.obs[groupby].nunique()))


@scanpy_pl_mcp.tool()
def scanpy_pl_rank_genes_groups_dotplot(
    h5ad_path: Annotated[str, "h5ad file with sc.tl.rank_genes_groups results in adata.uns"],
    n_genes: Annotated[int | None, "Top-ranked genes shown per group; None keeps scanpy default (10)"] = None,
    groups: Annotated[list[str] | None, "Subset of groups to plot; None plots all"] = None,
    values_to_plot: Annotated[
        Literal["scores", "logfoldchanges", "pvals", "pvals_adj", "log10_pvals", "log10_pvals_adj"] | None,
        "Statistic used to color the dots instead of mean expression",
    ] = None,
    min_logfoldchange: Annotated[float | None, "Minimum log fold change for a gene to be shown"] = None,
    key: Annotated[str | None, "uns key of the ranking results; None uses 'rank_genes_groups'"] = None,
    vmin: Annotated[float | None, "Lower color limit"] = None,
    vmax: Annotated[float | None, "Upper color limit"] = None,
    cmap: Annotated[str | None, "Matplotlib colormap name"] = None,
    output_dir: Annotated[str | None, "Base output directory; a fresh subdirectory is created"] = None,
) -> dict:
    """Plot top differentially expressed genes per group as a dot plot (scanpy.pl.rank_genes_groups_dotplot).
    Input is an h5ad containing rank_genes_groups results; output is a PNG.
    """
    adata = _read(h5ad_path)
    uns_key = key or "rank_genes_groups"
    if uns_key not in adata.uns:
        raise ValueError(f"adata.uns['{uns_key}'] is missing; run sc.tl.rank_genes_groups first")
    out_dir = _out_dir(output_dir, "scanpy_pl_rank_genes_groups_dotplot")
    kwargs = dict(n_genes=n_genes, groups=groups, values_to_plot=values_to_plot,
                  min_logfoldchange=min_logfoldchange, key=key, vmin=vmin, vmax=vmax, cmap=cmap)
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    with _fresh_figures():
        sc.pl.rank_genes_groups_dotplot(adata, show=False, **kwargs)
        artifacts = _save_open_figures(out_dir, "rank_genes_groups_dotplot", "Ranked-genes dot plot")
    return _result("Rank-genes-groups dot plot saved", artifacts, key=uns_key, groups=groups)


@scanpy_pl_mcp.tool()
def scanpy_pl_paga(
    h5ad_path: Annotated[str, "h5ad file with a computed PAGA graph (adata.uns['paga'], from sc.tl.paga)"],
    threshold: Annotated[float | None, "Hide edges with connectivity below this value; None keeps scanpy default"] = None,
    color: Annotated[str | None, "obs column or gene used to color nodes; None keeps scanpy default"] = None,
    layout: Annotated[Literal["fa", "fr", "drl", "eq_tree"] | None, "Graph layout (rt/rt_circular fail in pinned scanpy); None keeps default"] = None,
    node_size_scale: Annotated[float, "Scale factor for node sizes"] = 1.0,
    edge_width_scale: Annotated[float, "Scale factor for edge widths"] = 1.0,
    title: Annotated[str | None, "Plot title"] = None,
    rng: Annotated[int | None, "Integer seed for the layout random generator"] = None,
    output_dir: Annotated[str | None, "Base output directory; a fresh subdirectory is created"] = None,
) -> dict:
    """Plot the PAGA abstracted cluster graph (scanpy.pl.paga).
    Input is an h5ad with a precomputed PAGA graph; output is a PNG of the graph.
    """
    adata = _read(h5ad_path)
    if "paga" not in adata.uns:
        raise ValueError("adata.uns['paga'] is missing; run sc.tl.paga(adata, groups=...) first")
    if color is not None and color not in adata.obs and color not in adata.var_names:
        raise ValueError(f"color '{color}' is neither an obs column nor a gene")
    out_dir = _out_dir(output_dir, "scanpy_pl_paga")
    kwargs = dict(threshold=threshold, color=color, layout=layout, title=title, rng=rng)
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    with _fresh_figures():
        sc.pl.paga(adata, node_size_scale=node_size_scale, edge_width_scale=edge_width_scale,
                   show=False, **kwargs)
        artifacts = _save_open_figures(out_dir, "paga", "PAGA graph")
    return _result("PAGA plot saved", artifacts, groups_key=str(adata.uns["paga"].get("groups")))
