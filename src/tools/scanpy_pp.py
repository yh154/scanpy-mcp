"""Tools wrapping scanpy.pp / scanpy.experimental.pp (scanpy commit e099d9bcfdfc5ef065355c168ba2f639349939e7).

Each tool reads an h5ad file, calls the public scanpy preprocessing function directly, and writes the
resulting AnnData (plus native tables/figures) into a fresh per-call directory.
"""

import contextlib
import sys
import tempfile
from pathlib import Path
from typing import Annotated, Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import anndata as ad
import scanpy as sc
from fastmcp import FastMCP

_BLOB = "https://github.com/scverse/scanpy/blob/e099d9bcfdfc5ef065355c168ba2f639349939e7"
REFERENCE = f"{_BLOB}/docs/tutorials/basics/clustering-2017.ipynb"
_REFS = {
    "calculate_qc_metrics": f"{_BLOB}/src/scanpy/preprocessing/_qc.py",
    "filter_cells": f"{_BLOB}/src/scanpy/preprocessing/_simple.py",
    "filter_genes": f"{_BLOB}/src/scanpy/preprocessing/_simple.py",
    "normalize_total": f"{_BLOB}/src/scanpy/preprocessing/_normalization.py",
    "log1p": f"{_BLOB}/src/scanpy/preprocessing/_simple.py",
    "highly_variable_genes": f"{_BLOB}/src/scanpy/preprocessing/_highly_variable_genes.py",
    "scale": f"{_BLOB}/src/scanpy/preprocessing/_scale.py",
    "regress_out": f"{_BLOB}/src/scanpy/preprocessing/_simple.py",
    "pca": f"{_BLOB}/src/scanpy/preprocessing/_pca/__init__.py",
    "neighbors": f"{_BLOB}/src/scanpy/neighbors/__init__.py",
    "scrublet": f"{_BLOB}/docs/tutorials/basics/clustering.ipynb",
    "normalize_pearson_residuals": f"{_BLOB}/docs/tutorials/experimental/pearson_residuals.ipynb",
    "highly_variable_genes_pearson_residuals": f"{_BLOB}/docs/tutorials/experimental/pearson_residuals.ipynb",
}

scanpy_pp_mcp = FastMCP(name="scanpy_pp")


# ---------------------------------------------------------------- internal helpers
def _load(path: str):
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"Input h5ad not found: {path}")
    return ad.read_h5ad(p), p


def _require_layer(adata, layer):
    if layer is not None and layer not in adata.layers:
        raise ValueError(f"layer '{layer}' not in adata.layers {list(adata.layers)}")


def _require_obs(adata, keys):
    missing = [k for k in keys if k not in adata.obs.columns]
    if missing:
        raise ValueError(f"obs columns not found: {missing}")


def _fresh_dir(output_dir, tool):
    base = Path(output_dir).expanduser() if output_dir else Path(tempfile.gettempdir()) / "scanpy_pp_mcp"
    base.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f"{tool}_", dir=base)).resolve()


@contextlib.contextmanager
def _quiet():
    """Keep upstream prints off stdout (the MCP stdio channel)."""
    with contextlib.redirect_stdout(sys.stderr):
        yield


def _save(adata, src: Path, tool: str, output_dir):
    out = _fresh_dir(output_dir, tool)
    h5 = out / f"{src.stem}_{tool}.h5ad"
    adata.write_h5ad(h5)
    return out, h5


def _result(tool, message, artifacts, **summary):
    return {"message": message, "reference": _REFS[tool], "artifacts": artifacts, **summary}


def _art(description, path):
    return {"description": description, "path": str(Path(path).resolve())}


def _save_current_figure(path):
    fig = plt.gcf()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")


# ---------------------------------------------------------------- tools
@scanpy_pp_mcp.tool()
def scanpy_pp_calculate_qc_metrics(
    data_path: Annotated[str, "h5ad file with raw counts (cells x genes)"],
    qc_vars: Annotated[list[str], "Existing boolean adata.var columns (e.g. 'mt') for which pct_counts_<var> is computed"] = [],
    var_prefix_flags: Annotated[dict[str, str] | None, "Optional new boolean var columns to create from gene-name prefixes, e.g. {'mt': 'MT-'}; must not already exist"] = None,
    percent_top: Annotated[list[int] | None, "Ranks for pct_counts_in_top_N_genes; null to skip (scanpy default 50,100,200,500)"] = [50, 100, 200, 500],
    log1p: Annotated[bool, "Also store log1p-transformed QC metrics"] = True,
    layer: Annotated[str | None, "Layer to use instead of X"] = None,
    output_dir: Annotated[str | None, "Base output directory; a fresh subdirectory is created"] = None,
) -> dict:
    """Compute per-cell and per-gene QC metrics (counts, genes detected, qc_vars percentages) with scanpy.pp.calculate_qc_metrics.
    Input is an h5ad of counts; output is an h5ad with QC columns in obs/var plus obs/var QC CSV tables.
    """
    adata, src = _load(data_path)
    _require_layer(adata, layer)
    for col, prefix in (var_prefix_flags or {}).items():
        if col in adata.var.columns:
            raise ValueError(f"var column '{col}' already exists; refusing to overwrite")
        adata.var[col] = adata.var_names.str.startswith(prefix)
    missing = [v for v in qc_vars if v not in adata.var.columns]
    if missing:
        raise ValueError(f"qc_vars not in adata.var: {missing}")
    with _quiet():
        sc.pp.calculate_qc_metrics(
            adata, qc_vars=list(qc_vars) + [k for k in (var_prefix_flags or {}) if k not in qc_vars],
            percent_top=None if percent_top is None else tuple(percent_top),
            log1p=log1p, layer=layer, out=False,
        )
    out, h5 = _save(adata, src, "calculate_qc_metrics", output_dir)
    obs_csv, var_csv = out / "qc_obs.csv", out / "qc_var.csv"
    adata.obs.to_csv(obs_csv)
    adata.var.to_csv(var_csv)
    return _result(
        "calculate_qc_metrics", "QC metrics computed",
        [_art("AnnData with QC metrics", h5), _art("per-cell QC table", obs_csv), _art("per-gene QC table", var_csv)],
        n_obs=int(adata.n_obs), n_vars=int(adata.n_vars),
    )


@scanpy_pp_mcp.tool()
def scanpy_pp_filter_cells(
    data_path: Annotated[str, "h5ad file (cells x genes)"],
    min_genes: Annotated[int | None, "Minimum number of genes expressed per cell"] = None,
    min_counts: Annotated[int | None, "Minimum total counts per cell"] = None,
    max_genes: Annotated[int | None, "Maximum number of genes expressed per cell"] = None,
    max_counts: Annotated[int | None, "Maximum total counts per cell"] = None,
    output_dir: Annotated[str | None, "Base output directory; a fresh subdirectory is created"] = None,
) -> dict:
    """Filter cells by gene/count thresholds with scanpy.pp.filter_cells (exactly one threshold per call, as upstream).
    Input is an h5ad; output is the filtered h5ad.
    """
    adata, src = _load(data_path)
    n0 = int(adata.n_obs)
    with _quiet():
        sc.pp.filter_cells(adata, min_genes=min_genes, min_counts=min_counts, max_genes=max_genes, max_counts=max_counts)
    out, h5 = _save(adata, src, "filter_cells", output_dir)
    return _result("filter_cells", "Cells filtered", [_art("filtered AnnData", h5)],
                   n_obs_before=n0, n_obs_after=int(adata.n_obs), n_vars=int(adata.n_vars))


@scanpy_pp_mcp.tool()
def scanpy_pp_filter_genes(
    data_path: Annotated[str, "h5ad file (cells x genes)"],
    min_cells: Annotated[int | None, "Minimum number of cells expressing the gene"] = None,
    min_counts: Annotated[int | None, "Minimum total counts of the gene"] = None,
    max_cells: Annotated[int | None, "Maximum number of cells expressing the gene"] = None,
    max_counts: Annotated[int | None, "Maximum total counts of the gene"] = None,
    output_dir: Annotated[str | None, "Base output directory; a fresh subdirectory is created"] = None,
) -> dict:
    """Filter genes by cell/count thresholds with scanpy.pp.filter_genes (exactly one threshold per call, as upstream).
    Input is an h5ad; output is the filtered h5ad.
    """
    adata, src = _load(data_path)
    n0 = int(adata.n_vars)
    with _quiet():
        sc.pp.filter_genes(adata, min_cells=min_cells, min_counts=min_counts, max_cells=max_cells, max_counts=max_counts)
    out, h5 = _save(adata, src, "filter_genes", output_dir)
    return _result("filter_genes", "Genes filtered", [_art("filtered AnnData", h5)],
                   n_vars_before=n0, n_vars_after=int(adata.n_vars), n_obs=int(adata.n_obs))


@scanpy_pp_mcp.tool()
def scanpy_pp_normalize_total(
    data_path: Annotated[str, "h5ad file with raw counts"],
    target_sum: Annotated[float | None, "Target total counts per cell (e.g. 1e4); null uses the median of total counts"] = None,
    exclude_highly_expressed: Annotated[bool, "Exclude highly expressed genes when computing size factors"] = False,
    max_fraction: Annotated[float, "With exclude_highly_expressed: genes above this fraction of a cell's counts are excluded"] = 0.05,
    layer: Annotated[str | None, "Layer to normalize instead of X"] = None,
    key_added: Annotated[str | None, "obs key to store the per-cell normalization factor"] = None,
    output_dir: Annotated[str | None, "Base output directory; a fresh subdirectory is created"] = None,
) -> dict:
    """Library-size normalize each cell to target_sum with scanpy.pp.normalize_total.
    Input is an h5ad of counts; output is the normalized h5ad.
    """
    adata, src = _load(data_path)
    _require_layer(adata, layer)
    with _quiet():
        sc.pp.normalize_total(adata, target_sum=target_sum, exclude_highly_expressed=exclude_highly_expressed,
                              max_fraction=max_fraction, layer=layer, key_added=key_added)
    out, h5 = _save(adata, src, "normalize_total", output_dir)
    return _result("normalize_total", "Normalized to target sum", [_art("normalized AnnData", h5)],
                   n_obs=int(adata.n_obs), n_vars=int(adata.n_vars), target_sum=target_sum)


@scanpy_pp_mcp.tool()
def scanpy_pp_log1p(
    data_path: Annotated[str, "h5ad file with (normalized) non-negative data"],
    base: Annotated[float | None, "Logarithm base; null is the natural log"] = None,
    layer: Annotated[str | None, "Layer to transform instead of X"] = None,
    output_dir: Annotated[str | None, "Base output directory; a fresh subdirectory is created"] = None,
) -> dict:
    """Apply log(1+x) with scanpy.pp.log1p (records uns['log1p']).
    Input is an h5ad; output is the log-transformed h5ad.
    """
    adata, src = _load(data_path)
    _require_layer(adata, layer)
    with _quiet():
        sc.pp.log1p(adata, base=base, layer=layer)
    out, h5 = _save(adata, src, "log1p", output_dir)
    return _result("log1p", "log1p applied", [_art("log-transformed AnnData", h5)],
                   n_obs=int(adata.n_obs), n_vars=int(adata.n_vars))


@scanpy_pp_mcp.tool()
def scanpy_pp_highly_variable_genes(
    data_path: Annotated[str, "h5ad file; log-normalized in X for seurat/cell_ranger, raw counts (via layer) for seurat_v3"],
    flavor: Annotated[Literal["seurat", "cell_ranger", "seurat_v3", "seurat_v3_paper"], "HVG method"] = "seurat",
    n_top_genes: Annotated[int | None, "Number of HVGs to keep; null uses the upstream default (all cutoff-passing genes for seurat/cell_ranger, 2000 for seurat_v3)"] = None,
    min_mean: Annotated[float, "Minimum mean expression cutoff (seurat/cell_ranger)"] = 0.0125,
    max_mean: Annotated[float, "Maximum mean expression cutoff (seurat/cell_ranger)"] = 3,
    min_disp: Annotated[float, "Minimum normalized dispersion cutoff (seurat/cell_ranger)"] = 0.5,
    max_disp: Annotated[float, "Maximum normalized dispersion cutoff (seurat/cell_ranger)"] = float("inf"),
    span: Annotated[float, "LOESS span (seurat_v3)"] = 0.3,
    n_bins: Annotated[int, "Number of mean-expression bins (seurat/cell_ranger)"] = 20,
    layer: Annotated[str | None, "Layer to use instead of X (e.g. 'counts' for seurat_v3)"] = None,
    batch_key: Annotated[str | None, "obs column for batch-aware HVG selection"] = None,
    subset: Annotated[bool, "Subset the output to HVGs instead of only annotating var['highly_variable']"] = False,
    output_dir: Annotated[str | None, "Base output directory; a fresh subdirectory is created"] = None,
) -> dict:
    """Annotate (optionally subset to) highly variable genes with scanpy.pp.highly_variable_genes.
    Input is an h5ad; output is the annotated h5ad, an HVG table CSV and the HVG dispersion plot.
    """
    adata, src = _load(data_path)
    _require_layer(adata, layer)
    if batch_key is not None:
        _require_obs(adata, [batch_key])
    with _quiet():
        sc.pp.highly_variable_genes(
            adata, layer=layer, n_top_genes=n_top_genes, min_disp=min_disp, max_disp=max_disp, min_mean=min_mean,
            max_mean=max_mean, span=span, n_bins=n_bins, flavor=flavor, subset=subset, batch_key=batch_key,
        )
    out, h5 = _save(adata, src, "highly_variable_genes", output_dir)
    table = out / "highly_variable_genes.csv"
    adata.var.to_csv(table)
    artifacts = [_art("AnnData with HVG annotation", h5), _art("var table with HVG columns", table)]
    if not subset:
        with _quiet():
            sc.pl.highly_variable_genes(adata, show=False)
        fig = out / "highly_variable_genes.png"
        _save_current_figure(fig)
        artifacts.append(_art("HVG dispersion plot", fig))
    return _result("highly_variable_genes", "Highly variable genes annotated", artifacts,
                   n_highly_variable=int(adata.var["highly_variable"].sum()), n_obs=int(adata.n_obs), n_vars=int(adata.n_vars), flavor=flavor)


@scanpy_pp_mcp.tool()
def scanpy_pp_scale(
    data_path: Annotated[str, "h5ad file with (log-normalized) data"],
    zero_center: Annotated[bool, "Zero-center the data (densifies sparse data)"] = True,
    max_value: Annotated[float | None, "Clip scaled values to this absolute value; null for no clipping"] = None,
    layer: Annotated[str | None, "Layer to scale instead of X"] = None,
    output_dir: Annotated[str | None, "Base output directory; a fresh subdirectory is created"] = None,
) -> dict:
    """Scale each gene to unit variance (optionally zero-centered and clipped) with scanpy.pp.scale.
    Input is an h5ad; output is the scaled h5ad.
    """
    adata, src = _load(data_path)
    _require_layer(adata, layer)
    with _quiet():
        sc.pp.scale(adata, zero_center=zero_center, max_value=max_value, layer=layer)
    out, h5 = _save(adata, src, "scale", output_dir)
    return _result("scale", "Data scaled", [_art("scaled AnnData", h5)],
                   n_obs=int(adata.n_obs), n_vars=int(adata.n_vars), max_value=max_value)


@scanpy_pp_mcp.tool()
def scanpy_pp_regress_out(
    data_path: Annotated[str, "h5ad file whose obs contains the covariates"],
    keys: Annotated[list[str], "obs columns to regress out (e.g. total_counts, pct_counts_mt)"],
    layer: Annotated[str | None, "Layer to regress instead of X"] = None,
    n_jobs: Annotated[int | None, "Number of parallel jobs"] = None,
    output_dir: Annotated[str | None, "Base output directory; a fresh subdirectory is created"] = None,
) -> dict:
    """Regress out unwanted obs covariates from each gene with scanpy.pp.regress_out.
    Input is an h5ad with the covariate columns; output is the corrected h5ad.
    """
    adata, src = _load(data_path)
    _require_layer(adata, layer)
    _require_obs(adata, keys)
    with _quiet():
        sc.pp.regress_out(adata, list(keys), layer=layer, n_jobs=n_jobs)
    out, h5 = _save(adata, src, "regress_out", output_dir)
    return _result("regress_out", "Covariates regressed out", [_art("corrected AnnData", h5)],
                   n_obs=int(adata.n_obs), n_vars=int(adata.n_vars), keys=list(keys))


@scanpy_pp_mcp.tool()
def scanpy_pp_pca(
    data_path: Annotated[str, "h5ad file with (scaled) data"],
    n_comps: Annotated[int | None, "Number of principal components (scanpy default 50)"] = None,
    layer: Annotated[str | None, "Layer to use instead of X"] = None,
    svd_solver: Annotated[Literal["arpack", "randomized", "auto", "lobpcg", "tsqr"] | None, "SVD solver; null uses the scanpy default"] = None,
    zero_center: Annotated[bool, "Zero-center the data before PCA"] = True,
    mask_var: Annotated[str | None, "var boolean column selecting genes; null uses 'highly_variable' if present (scanpy default)"] = None,
    rng: Annotated[int | None, "Random seed; null uses the scanpy default"] = None,
    output_dir: Annotated[str | None, "Base output directory; a fresh subdirectory is created"] = None,
) -> dict:
    """Compute principal components with scanpy.pp.pca (X_pca, PCs, variance ratio).
    Input is an h5ad; output is the h5ad with PCA results.
    """
    adata, src = _load(data_path)
    _require_layer(adata, layer)
    kwargs = {}
    if mask_var is not None:
        if mask_var not in adata.var.columns:
            raise ValueError(f"mask_var '{mask_var}' not in adata.var")
        kwargs["mask_var"] = mask_var
    if rng is not None:
        kwargs["rng"] = rng
    with _quiet():
        sc.pp.pca(adata, n_comps=n_comps, layer=layer, svd_solver=svd_solver, zero_center=zero_center, **kwargs)
    out, h5 = _save(adata, src, "pca", output_dir)
    return _result("pca", "PCA computed", [_art("AnnData with PCA", h5)],
                   n_obs=int(adata.n_obs), n_pcs=int(adata.obsm["X_pca"].shape[1]),
                   variance_ratio_first5=[float(x) for x in adata.uns["pca"]["variance_ratio"][:5]])


@scanpy_pp_mcp.tool()
def scanpy_pp_neighbors(
    data_path: Annotated[str, "h5ad file with a representation (X_pca by default)"],
    n_neighbors: Annotated[int, "Neighborhood size"] = 15,
    n_pcs: Annotated[int | None, "Number of PCs of the representation to use; null uses all"] = None,
    use_rep: Annotated[str | None, "obsm key of the representation; null uses X_pca (or X if absent)"] = None,
    method: Annotated[Literal["umap", "gauss"], "Connectivity kernel"] = "umap",
    metric: Annotated[str | None, "Distance metric; null uses euclidean"] = None,
    knn: Annotated[bool, "Restrict to a hard k-nearest-neighbor graph"] = True,
    key_added: Annotated[str | None, "Store the graph under this key instead of the default"] = None,
    rng: Annotated[int | None, "Random seed; null uses the scanpy default"] = None,
    output_dir: Annotated[str | None, "Base output directory; a fresh subdirectory is created"] = None,
) -> dict:
    """Build the k-nearest-neighbor graph and connectivities with scanpy.pp.neighbors.
    Input is an h5ad with PCA; output is the h5ad with distances/connectivities.
    """
    adata, src = _load(data_path)
    if use_rep is not None and use_rep not in adata.obsm:
        raise ValueError(f"use_rep '{use_rep}' not in adata.obsm {list(adata.obsm)}")
    kwargs = {} if metric is None else {"metric": metric}
    if rng is not None:
        kwargs["rng"] = rng
    with _quiet():
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, use_rep=use_rep, method=method,
                        knn=knn, key_added=key_added, **kwargs)
    out, h5 = _save(adata, src, "neighbors", output_dir)
    return _result("neighbors", "Neighbor graph computed", [_art("AnnData with neighbor graph", h5)],
                   n_obs=int(adata.n_obs), n_neighbors=n_neighbors, graph_keys=sorted(k for k in adata.obsp.keys()))


@scanpy_pp_mcp.tool()
def scanpy_pp_scrublet(
    data_path: Annotated[str, "h5ad file with raw counts in X"],
    batch_key: Annotated[str | None, "obs column of batches; doublets are simulated and scored per batch"] = None,
    sim_doublet_ratio: Annotated[float, "Number of simulated doublets relative to observed cells"] = 2.0,
    expected_doublet_rate: Annotated[float, "Expected doublet rate for the experiment"] = 0.05,
    stdev_doublet_rate: Annotated[float, "Uncertainty in the expected doublet rate"] = 0.02,
    n_prin_comps: Annotated[int, "Number of PCs used to embed cells and simulated doublets"] = 30,
    n_neighbors: Annotated[int | None, "kNN size; null uses round(0.5*sqrt(n_obs))"] = None,
    threshold: Annotated[float | None, "Doublet score threshold; null selects it automatically"] = None,
    rng: Annotated[int | None, "Random seed; null uses the scanpy default"] = None,
    output_dir: Annotated[str | None, "Base output directory; a fresh subdirectory is created"] = None,
) -> dict:
    """Score and predict doublets with scanpy.pp.scrublet (obs doublet_score, predicted_doublet).
    Input is an h5ad of raw counts; output is the annotated h5ad and a score-distribution plot.
    """
    adata, src = _load(data_path)
    if batch_key is not None:
        _require_obs(adata, [batch_key])
    kwargs = {} if rng is None else {"rng": rng}
    with _quiet():
        sc.pp.scrublet(
            adata, batch_key=batch_key, sim_doublet_ratio=sim_doublet_ratio, expected_doublet_rate=expected_doublet_rate,
            stdev_doublet_rate=stdev_doublet_rate, n_prin_comps=n_prin_comps, n_neighbors=n_neighbors,
            threshold=threshold, **kwargs,
        )
    out, h5 = _save(adata, src, "scrublet", output_dir)
    artifacts = [_art("AnnData with doublet scores", h5)]
    with _quiet():
        sc.pl.scrublet_score_distribution(adata, show=False)
    fig = out / "scrublet_score_distribution.png"
    _save_current_figure(fig)
    artifacts.append(_art("doublet score distribution", fig))
    return _result("scrublet", "Doublets scored", artifacts,
                   n_obs=int(adata.n_obs), n_predicted_doublets=int(adata.obs["predicted_doublet"].sum()))


@scanpy_pp_mcp.tool()
def scanpy_pp_normalize_pearson_residuals(
    data_path: Annotated[str, "h5ad file with raw (integer) counts, typically already subset to HVGs"],
    theta: Annotated[float, "Negative-binomial overdispersion (theta=np.inf gives Poisson)"] = 100,
    clip: Annotated[float | None, "Residual clipping bound; null uses sqrt(n_obs)"] = None,
    layer: Annotated[str | None, "Layer to use instead of X"] = None,
    check_values: Annotated[bool, "Warn if the data does not look like raw counts"] = True,
    output_dir: Annotated[str | None, "Base output directory; a fresh subdirectory is created"] = None,
) -> dict:
    """Replace counts by analytic Pearson residuals with scanpy.experimental.pp.normalize_pearson_residuals.
    Input is an h5ad of raw counts; output is the h5ad whose X holds Pearson residuals.
    """
    adata, src = _load(data_path)
    _require_layer(adata, layer)
    with _quiet():
        sc.experimental.pp.normalize_pearson_residuals(adata, theta=theta, clip=clip, layer=layer, check_values=check_values)
    out, h5 = _save(adata, src, "normalize_pearson_residuals", output_dir)
    return _result("normalize_pearson_residuals", "Pearson residuals computed", [_art("AnnData with Pearson residuals", h5)],
                   n_obs=int(adata.n_obs), n_vars=int(adata.n_vars), theta=theta)


@scanpy_pp_mcp.tool()
def scanpy_pp_highly_variable_genes_pearson_residuals(
    data_path: Annotated[str, "h5ad file with raw (integer) counts"],
    n_top_genes: Annotated[int, "Number of genes to select by residual variance"] = 2000,
    theta: Annotated[float, "Negative-binomial overdispersion"] = 100,
    clip: Annotated[float | None, "Residual clipping bound; null uses sqrt(n_obs)"] = None,
    batch_key: Annotated[str | None, "obs column for batch-aware selection"] = None,
    layer: Annotated[str | None, "Layer to use instead of X"] = None,
    subset: Annotated[bool, "Subset the output to the selected genes"] = False,
    check_values: Annotated[bool, "Warn if the data does not look like raw counts"] = True,
    chunksize: Annotated[int, "Number of genes processed per chunk"] = 1000,
    output_dir: Annotated[str | None, "Base output directory; a fresh subdirectory is created"] = None,
) -> dict:
    """Select highly variable genes by Pearson residual variance with scanpy.experimental.pp.highly_variable_genes.
    Input is an h5ad of raw counts; output is the annotated h5ad, an HVG table CSV and a residual-variance plot.
    """
    adata, src = _load(data_path)
    _require_layer(adata, layer)
    if batch_key is not None:
        _require_obs(adata, [batch_key])
    with _quiet():
        sc.experimental.pp.highly_variable_genes(
            adata, flavor="pearson_residuals", n_top_genes=n_top_genes, theta=theta, clip=clip, batch_key=batch_key,
            layer=layer, subset=subset, check_values=check_values, chunksize=chunksize,
        )
    out, h5 = _save(adata, src, "highly_variable_genes_pearson_residuals", output_dir)
    table = out / "highly_variable_genes_pearson_residuals.csv"
    adata.var.to_csv(table)
    artifacts = [_art("AnnData with HVG annotation", h5), _art("var table with residual variances", table)]
    if not subset:
        # Residual variance vs mean plot, adapted from docs/tutorials/experimental/pearson_residuals.ipynb
        hvgs = adata.var["highly_variable"]
        fig_, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(adata.var["mean_counts"], adata.var["residual_variances"], s=3, edgecolor="none")
        ax.scatter(adata.var["mean_counts"][hvgs], adata.var["residual_variances"][hvgs], c="tab:red", label="selected genes", s=3, edgecolor="none")
        ax.set_xscale("log")
        ax.set_xlabel("mean expression")
        ax.set_yscale("log")
        ax.set_ylabel("residual variance")
        ax.legend()
        fig = out / "highly_variable_genes_pearson_residuals.png"
        _save_current_figure(fig)
        artifacts.append(_art("residual variance vs mean plot", fig))
    return _result("highly_variable_genes_pearson_residuals", "Pearson-residual HVGs selected", artifacts,
                   n_highly_variable=int(adata.var["highly_variable"].sum()), n_obs=int(adata.n_obs), n_vars=int(adata.n_vars))
