"""Scanpy tools (scanpy.tl): clustering, embedding, marker genes, ingest, trajectories, gene scoring.

Thin wrappers over the public scanpy API pinned at scverse/scanpy commit
e099d9bcfdfc5ef065355c168ba2f639349939e7. Each tool reads h5ad file(s), calls the
scanpy function directly on the in-memory AnnData, and writes a new h5ad (plus tables
where natural) into a fresh per-call directory. Parameters left as None are not passed
so that the pinned scanpy defaults apply.

Reference executions: docs/tutorials/basics/clustering.ipynb, clustering-2017.ipynb,
integrating-data-using-ingest.ipynb, docs/how-to/cell-cycle.ipynb,
docs/tutorials/trajectories/paga-paul15.ipynb, docs/tutorials/plotting/advanced.ipynb.
"""

import os
import tempfile
from pathlib import Path
from typing import Annotated, Literal

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from fastmcp import FastMCP

REFERENCE = "https://github.com/scverse/scanpy/blob/e099d9bcfdfc5ef065355c168ba2f639349939e7/src/scanpy/tools/"
REFERENCE_NOTEBOOKS = (
    "https://github.com/scverse/scanpy/blob/e099d9bcfdfc5ef065355c168ba2f639349939e7/docs/tutorials/"
)

scanpy_tl_mcp = FastMCP(name="scanpy_tl")


# ----------------------------------------------------------------------------- helpers
def _ref(name: str) -> str:
    return REFERENCE + name


def _kw(**kwargs) -> dict:
    """Keep only parameters the user set so the pinned scanpy defaults apply to the rest."""
    return {k: v for k, v in kwargs.items() if v is not None}


def _new_dir(output_dir: str | None, prefix: str) -> Path:
    base = output_dir or os.environ.get("SCANPY_MCP_OUTPUT_DIR") or tempfile.gettempdir()
    Path(base).mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix + "_", dir=base)).resolve()


def _read(path: str, what: str = "h5ad_path") -> ad.AnnData:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"{what}: file not found: {path}")
    return sc.read_h5ad(p)


def _need_obs(adata: ad.AnnData, key: str, what: str) -> None:
    if key not in adata.obs.columns:
        raise ValueError(f"{what} '{key}' is not a column of adata.obs; available: {list(adata.obs.columns)}")


def _write(adata: ad.AnnData, out: Path, name: str) -> str:
    path = out / name
    adata.write_h5ad(path)
    return str(path.resolve())


def _result(message: str, reference: str, artifacts: list[tuple[str, str]], **summary) -> dict:
    return {
        "message": message,
        "reference": reference,
        "artifacts": [{"description": d, "path": p} for d, p in artifacts],
        **summary,
    }


# ----------------------------------------------------------------------------- tools
@scanpy_tl_mcp.tool()
def scanpy_tl_leiden(
    h5ad_path: Annotated[str, "h5ad with a neighbors graph (run sc.pp.neighbors first)"],
    resolution: Annotated[float, "Resolution; higher gives more clusters"] = 1.0,
    key_added: Annotated[str | None, "obs column for the labels (scanpy default 'leiden'); use different keys for multiple resolutions"] = None,
    flavor: Annotated[Literal["leidenalg", "igraph"] | None, "Backend; None uses the scanpy default (leidenalg); the 2017 and paga tutorials use igraph"] = None,
    n_iterations: Annotated[int, "Leiden iterations; -1 runs until convergence, 2 as in the tutorials with igraph"] = -1,
    directed: Annotated[bool | None, "Treat the graph as directed; None uses the scanpy default (igraph flavor requires False)"] = None,
    use_weights: Annotated[bool, "Use edge weights from the connectivities"] = True,
    random_state: Annotated[int, "Seed for the optimisation"] = 0,
    restrict_to_key: Annotated[str | None, "obs column to restrict clustering to (restrict_to); requires restrict_to_groups"] = None,
    restrict_to_groups: Annotated[list[str] | None, "Categories of restrict_to_key to sub-cluster"] = None,
    neighbors_key: Annotated[str | None, "Key in adata.uns holding the neighbors graph if not the default"] = None,
    output_dir: Annotated[str | None, "Base output directory; a fresh subdirectory is created"] = None,
) -> dict:
    """Cluster cells with the Leiden algorithm on the kNN graph (scanpy.tl.leiden).
    Input is an h5ad with neighbors; output is an h5ad with the cluster labels in obs plus a label CSV.
    """
    adata = _read(h5ad_path)
    restrict_to = None
    if (restrict_to_key is None) != (restrict_to_groups is None):
        raise ValueError("restrict_to_key and restrict_to_groups must be given together")
    if restrict_to_key is not None:
        _need_obs(adata, restrict_to_key, "restrict_to_key")
        restrict_to = (restrict_to_key, list(restrict_to_groups))
    key = key_added or "leiden"
    sc.tl.leiden(
        adata,
        resolution=resolution,
        random_state=random_state,
        directed=directed,
        use_weights=use_weights,
        n_iterations=n_iterations,
        **_kw(flavor=flavor, restrict_to=restrict_to, neighbors_key=neighbors_key, key_added=key_added),
    )
    out = _new_dir(output_dir, "scanpy_tl_leiden")
    h5 = _write(adata, out, "leiden.h5ad")
    csv = out / "leiden_labels.csv"
    adata.obs[[key]].to_csv(csv)
    counts = adata.obs[key].value_counts().sort_index()
    return _result(f"Leiden clustering written to obs['{key}'] ({counts.size} clusters).",
                   _ref("_leiden.py"),
                   [("h5ad with Leiden labels", h5), ("cluster labels per cell", str(csv.resolve()))],
                   key_added=key, n_clusters=int(counts.size),
                   cluster_sizes={str(k): int(v) for k, v in counts.items()}, n_obs=int(adata.n_obs))


@scanpy_tl_mcp.tool()
def scanpy_tl_umap(
    h5ad_path: Annotated[str, "h5ad with a neighbors graph (run sc.pp.neighbors first)"],
    min_dist: Annotated[float, "Effective minimum distance between embedded points"] = 0.5,
    spread: Annotated[float, "Effective scale of embedded points; with min_dist sets embedding clumpiness"] = 1.0,
    n_components: Annotated[int, "Dimensions of the embedding"] = 2,
    maxiter: Annotated[int | None, "Optimisation epochs; None lets UMAP choose by dataset size"] = None,
    alpha: Annotated[float, "Initial learning rate"] = 1.0,
    gamma: Annotated[float, "Weight of negative samples"] = 1.0,
    negative_sample_rate: Annotated[int, "Negative samples per positive sample"] = 5,
    init_pos: Annotated[str | None, "Initialisation: 'auto', 'spectral', 'random', or an obsm key; None uses the scanpy default ('auto')"] = None,
    random_state: Annotated[int, "Seed for the embedding"] = 0,
    neighbors_key: Annotated[str | None, "Key in adata.uns holding the neighbors graph if not the default"] = None,
    key_added: Annotated[str | None, "Store the embedding in obsm[key_added] (pinned scanpy; no 'X_' prefix) instead of 'X_umap'"] = None,
    output_dir: Annotated[str | None, "Base output directory; a fresh subdirectory is created"] = None,
) -> dict:
    """Embed the neighbor graph with UMAP (scanpy.tl.umap).
    Input is an h5ad with neighbors; output is an h5ad with obsm['X_umap'].
    """
    adata = _read(h5ad_path)
    sc.tl.umap(
        adata, min_dist=min_dist, spread=spread, n_components=n_components, alpha=alpha, gamma=gamma,
        negative_sample_rate=negative_sample_rate, random_state=random_state,
        **_kw(maxiter=maxiter, init_pos=init_pos, neighbors_key=neighbors_key, key_added=key_added),
    )
    obsm_key = key_added or "X_umap"  # pinned scanpy stores obsm[key_added] verbatim
    out = _new_dir(output_dir, "scanpy_tl_umap")
    h5 = _write(adata, out, "umap.h5ad")
    return _result(f"UMAP embedding stored in obsm['{obsm_key}'].", _ref("_umap.py"),
                   [("h5ad with UMAP embedding", h5)],
                   obsm_key=obsm_key, embedding_shape=list(adata.obsm[obsm_key].shape))


@scanpy_tl_mcp.tool()
def scanpy_tl_rank_genes_groups(
    h5ad_path: Annotated[str, "h5ad with normalised/log-transformed expression and a grouping column in obs"],
    groupby: Annotated[str, "obs column defining the groups to compare (e.g. a Leiden key)"],
    method: Annotated[Literal["t-test", "t-test_overestim_var", "wilcoxon", "logreg"] | None, "Test; None uses the scanpy default (t-test)"] = None,
    groups: Annotated[list[str] | None, "Subset of groups to test; None tests all groups"] = None,
    reference: Annotated[str, "Group to compare against, or 'rest'"] = "rest",
    n_genes: Annotated[int | None, "Number of genes to rank; None ranks all genes"] = None,
    corr_method: Annotated[Literal["benjamini-hochberg", "bonferroni"], "Multiple-testing correction (not for logreg)"] = "benjamini-hochberg",
    rankby_abs: Annotated[bool, "Rank by absolute score instead of score"] = False,
    pts: Annotated[bool, "Also compute the fraction of cells expressing each gene per group"] = False,
    tie_correct: Annotated[bool | None, "Tie correction for wilcoxon; None uses the scanpy default"] = None,
    mask: Annotated[str | None, "Gene mask such as 'var.highly_variable' (a boolean var column) restricting the genes tested, as in the clustering tutorials; None tests all genes"] = None,
    max_iter: Annotated[int | None, "Maximum iterations for method 'logreg' (tutorial uses 1000); None uses the scikit-learn default"] = None,
    use_raw: Annotated[bool | None, "Use adata.raw; None uses raw if present (scanpy default)"] = None,
    layer: Annotated[str | None, "Layer to use instead of X"] = None,
    key_added: Annotated[str | None, "Key in adata.uns for the results (default 'rank_genes_groups')"] = None,
    output_dir: Annotated[str | None, "Base output directory; a fresh subdirectory is created"] = None,
) -> dict:
    """Rank marker genes for each group of an obs column (scanpy.tl.rank_genes_groups).
    Input is an h5ad plus groupby key; output is an h5ad with uns['rank_genes_groups'] and a CSV of all ranked genes.
    """
    adata = _read(h5ad_path)
    _need_obs(adata, groupby, "groupby")
    present = set(adata.obs[groupby].astype(str))
    if groups is not None:
        missing = [g for g in groups if g not in present]
        if missing:
            raise ValueError(f"groups {missing} not found in obs['{groupby}']; available: {sorted(present)}")
    if reference != "rest" and reference not in present:
        raise ValueError(f"reference '{reference}' not found in obs['{groupby}']; available: {sorted(present)}")
    sc.tl.rank_genes_groups(
        adata, groupby, groups=groups if groups is not None else "all", reference=reference,
        corr_method=corr_method, rankby_abs=rankby_abs, pts=pts,
        **_kw(method=method, n_genes=n_genes, tie_correct=tie_correct, mask=mask, max_iter=max_iter,
              use_raw=use_raw, layer=layer, key_added=key_added),
    )
    key = key_added or "rank_genes_groups"
    df = sc.get.rank_genes_groups_df(adata, group=None, key=key)
    out = _new_dir(output_dir, "scanpy_tl_rank_genes_groups")
    h5 = _write(adata, out, "rank_genes_groups.h5ad")
    csv = out / "rank_genes_groups_all.csv"
    df.to_csv(csv, index=False)
    return _result(f"Ranked genes for groups of obs['{groupby}'] stored in uns['{key}'].",
                   _ref("_rank_genes_groups.py"),
                   [("h5ad with rank_genes_groups results", h5), ("ranked genes table, all groups", str(csv.resolve()))],
                   uns_key=key, method=str(adata.uns[key]["params"]["method"]), n_rows=int(len(df)),
                   groups=[str(g) for g in adata.uns[key]["names"].dtype.names])


@scanpy_tl_mcp.tool()
def scanpy_tl_get_rank_genes_groups_df(
    h5ad_path: Annotated[str, "h5ad that already contains rank_genes_groups results"],
    group: Annotated[list[str] | None, "Group(s) to export; None exports all groups"] = None,
    key: Annotated[str, "uns key of the rank_genes_groups results"] = "rank_genes_groups",
    pval_cutoff: Annotated[float | None, "Keep genes with adjusted p-value below this"] = None,
    log2fc_min: Annotated[float | None, "Keep genes with log2 fold change above this"] = None,
    log2fc_max: Annotated[float | None, "Keep genes with log2 fold change below this"] = None,
    gene_symbols: Annotated[str | None, "var column with gene symbols to add as 'symbol'"] = None,
    output_dir: Annotated[str | None, "Base output directory; a fresh subdirectory is created"] = None,
) -> dict:
    """Export rank_genes_groups results as a filtered table (scanpy.get.rank_genes_groups_df).
    Input is an h5ad with ranking results; output is a CSV (names, scores, logfoldchanges, pvals, pvals_adj).
    """
    adata = _read(h5ad_path)
    if key not in adata.uns:
        raise ValueError(f"uns['{key}'] not found; run scanpy_tl_rank_genes_groups first")
    df = sc.get.rank_genes_groups_df(
        adata, group=group, key=key,
        **_kw(pval_cutoff=pval_cutoff, log2fc_min=log2fc_min, log2fc_max=log2fc_max, gene_symbols=gene_symbols),
    )
    out = _new_dir(output_dir, "scanpy_tl_get_rank_genes_groups_df")
    csv = out / "rank_genes_groups_df.csv"
    df.to_csv(csv, index=False)
    return _result(f"Exported {len(df)} rows.", "https://github.com/scverse/scanpy/blob/e099d9bcfdfc5ef065355c168ba2f639349939e7/src/scanpy/get/get.py",
                   [("marker gene table", str(csv.resolve()))],
                   n_rows=int(len(df)), columns=[str(c) for c in df.columns])


@scanpy_tl_mcp.tool()
def scanpy_tl_ingest(
    query_h5ad_path: Annotated[str, "Query h5ad to map onto the reference (same var_names as reference)"],
    reference_h5ad_path: Annotated[str, "Reference h5ad with PCA, neighbors and UMAP computed (and obs labels to transfer)"],
    obs: Annotated[list[str] | None, "Reference obs column(s) to transfer to the query; None transfers none"] = None,
    embedding_method: Annotated[list[str], "Embeddings to map: 'umap' and/or 'pca'"] = ["umap", "pca"],
    labeling_method: Annotated[str, "Label transfer method, e.g. 'knn'"] = "knn",
    neighbors_key: Annotated[str | None, "Key in reference adata.uns of the neighbors graph if not default"] = None,
    output_dir: Annotated[str | None, "Base output directory; a fresh subdirectory is created"] = None,
) -> dict:
    """Map a query dataset onto a reference embedding and transfer labels (scanpy.tl.ingest).
    Inputs are query and reference h5ad; output is the query h5ad with obsm/obs entries (UMAP from ingest is not bit-reproducible).
    """
    query = _read(query_h5ad_path, "query_h5ad_path")
    ref = _read(reference_h5ad_path, "reference_h5ad_path")
    obs_keys = list(obs) if obs else None
    for k in obs_keys or []:
        _need_obs(ref, k, "obs (reference)")
    sc.tl.ingest(query, ref, obs=obs_keys, embedding_method=list(embedding_method),
                 labeling_method=labeling_method, **_kw(neighbors_key=neighbors_key))
    out = _new_dir(output_dir, "scanpy_tl_ingest")
    h5 = _write(query, out, "ingested_query.h5ad")
    artifacts = [("query h5ad with ingested embeddings/labels", h5)]
    if obs_keys:
        csv = out / "ingest_labels.csv"
        query.obs[obs_keys].to_csv(csv)
        artifacts.append(("transferred labels per cell", str(csv.resolve())))
    return _result("Query mapped onto the reference.", _ref("_ingest.py"), artifacts,
                   transferred_obs=obs_keys or [], obsm_keys=list(query.obsm.keys()), n_obs=int(query.n_obs))


@scanpy_tl_mcp.tool()
def scanpy_tl_paga(
    h5ad_path: Annotated[str, "h5ad with a neighbors graph and a categorical grouping column"],
    groups: Annotated[str, "obs column of categorical groups (e.g. Leiden/annotation labels) used as PAGA nodes"],
    model: Annotated[Literal["v1.2", "v1.0"], "PAGA connectivity model"] = "v1.2",
    use_rna_velocity: Annotated[bool, "Use RNA velocity to orient edges (needs velocity graph)"] = False,
    neighbors_key: Annotated[str | None, "Key in adata.uns holding the neighbors graph if not default"] = None,
    output_dir: Annotated[str | None, "Base output directory; a fresh subdirectory is created"] = None,
) -> dict:
    """Compute the PAGA abstracted graph of cell groups (scanpy.tl.paga).
    Input is an h5ad with neighbors plus a groups key; output is an h5ad with uns['paga'] and a connectivities CSV.
    """
    adata = _read(h5ad_path)
    _need_obs(adata, groups, "groups")
    sc.tl.paga(adata, groups=groups, model=model, use_rna_velocity=use_rna_velocity,
               **_kw(neighbors_key=neighbors_key))
    cats = [str(c) for c in adata.obs[groups].astype("category").cat.categories]
    conn = pd.DataFrame(adata.uns["paga"]["connectivities"].toarray(), index=cats, columns=cats)
    out = _new_dir(output_dir, "scanpy_tl_paga")
    h5 = _write(adata, out, "paga.h5ad")
    csv = out / "paga_connectivities.csv"
    conn.to_csv(csv)
    return _result(f"PAGA computed for {len(cats)} groups of obs['{groups}'].", _ref("_paga.py"),
                   [("h5ad with uns['paga']", h5), ("group connectivity matrix", str(csv.resolve()))],
                   groups_key=groups, n_groups=len(cats))


@scanpy_tl_mcp.tool()
def scanpy_tl_diffmap(
    h5ad_path: Annotated[str, "h5ad with a neighbors graph (run sc.pp.neighbors first)"],
    n_comps: Annotated[int, "Number of diffusion components"] = 15,
    random_state: Annotated[int, "Seed for the eigensolver"] = 0,
    neighbors_key: Annotated[str | None, "Key in adata.uns holding the neighbors graph if not default"] = None,
    output_dir: Annotated[str | None, "Base output directory; a fresh subdirectory is created"] = None,
) -> dict:
    """Compute a diffusion map embedding (scanpy.tl.diffmap).
    Input is an h5ad with neighbors; output is an h5ad with obsm['X_diffmap'] and uns['diffmap_evals'].
    """
    adata = _read(h5ad_path)
    sc.tl.diffmap(adata, n_comps=n_comps, random_state=random_state, **_kw(neighbors_key=neighbors_key))
    out = _new_dir(output_dir, "scanpy_tl_diffmap")
    h5 = _write(adata, out, "diffmap.h5ad")
    return _result("Diffusion map stored in obsm['X_diffmap'].", _ref("_diffmap.py"),
                   [("h5ad with diffusion map", h5)],
                   embedding_shape=list(adata.obsm["X_diffmap"].shape),
                   eigenvalues=[float(x) for x in np.asarray(adata.uns["diffmap_evals"])[:n_comps]])


@scanpy_tl_mcp.tool()
def scanpy_tl_dpt(
    h5ad_path: Annotated[str, "h5ad with a neighbors graph and diffusion map (run scanpy_tl_diffmap first)"],
    root_cell_index: Annotated[int | None, "0-based index of the root cell; sets uns['iroot']"] = None,
    root_obs_key: Annotated[str | None, "obs column used to pick the root cell (first cell whose value equals root_obs_value)"] = None,
    root_obs_value: Annotated[str | None, "Value of root_obs_key identifying the root population"] = None,
    n_dcs: Annotated[int, "Number of diffusion components to use"] = 10,
    n_branchings: Annotated[int, "Number of branchings to detect (0 = pseudotime only)"] = 0,
    min_group_size: Annotated[float, "Minimum fraction of cells in a branching group"] = 0.01,
    allow_kendall_tau_shift: Annotated[bool, "Allow a shift in the Kendall tau branching criterion"] = True,
    neighbors_key: Annotated[str | None, "Key in adata.uns holding the neighbors graph if not default"] = None,
    output_dir: Annotated[str | None, "Base output directory; a fresh subdirectory is created"] = None,
) -> dict:
    """Compute diffusion pseudotime from a root cell (scanpy.tl.dpt).
    Input is an h5ad with diffmap and a root (index or obs key/value, or existing uns['iroot']); output is an h5ad with obs['dpt_pseudotime'] and a CSV.
    """
    adata = _read(h5ad_path)
    if root_cell_index is not None and root_obs_key is not None:
        raise ValueError("give either root_cell_index or root_obs_key/root_obs_value, not both")
    if root_obs_key is not None:
        _need_obs(adata, root_obs_key, "root_obs_key")
        if root_obs_value is None:
            raise ValueError("root_obs_value is required with root_obs_key")
        hits = np.flatnonzero(adata.obs[root_obs_key].astype(str).values == str(root_obs_value))
        if hits.size == 0:
            raise ValueError(f"no cell has obs['{root_obs_key}'] == '{root_obs_value}'")
        adata.uns["iroot"] = int(hits[0])
    elif root_cell_index is not None:
        if not 0 <= root_cell_index < adata.n_obs:
            raise ValueError(f"root_cell_index {root_cell_index} out of range for {adata.n_obs} cells")
        adata.uns["iroot"] = int(root_cell_index)
    if "iroot" not in adata.uns:
        raise ValueError("no root: pass root_cell_index or root_obs_key/root_obs_value, or provide uns['iroot']")
    sc.tl.dpt(adata, n_dcs=n_dcs, n_branchings=n_branchings, min_group_size=min_group_size,
              allow_kendall_tau_shift=allow_kendall_tau_shift, **_kw(neighbors_key=neighbors_key))
    cols = [c for c in ("dpt_pseudotime", "dpt_groups") if c in adata.obs.columns]
    out = _new_dir(output_dir, "scanpy_tl_dpt")
    h5 = _write(adata, out, "dpt.h5ad")
    csv = out / "dpt_pseudotime.csv"
    adata.obs[cols].to_csv(csv)
    pt = adata.obs["dpt_pseudotime"].values
    finite = pt[np.isfinite(pt)]  # upstream marks unreachable cells as inf
    return _result("Diffusion pseudotime stored in obs['dpt_pseudotime'].", _ref("_dpt.py"),
                   [("h5ad with DPT", h5), ("pseudotime per cell", str(csv.resolve()))],
                   iroot=int(adata.uns["iroot"]), n_finite=int(np.isfinite(pt).sum()),
                   n_nonfinite=int(pt.size - finite.size), pseudotime_range=[float(finite.min()), float(finite.max())])


@scanpy_tl_mcp.tool()
def scanpy_tl_score_genes(
    h5ad_path: Annotated[str, "h5ad with normalised/log-transformed expression"],
    gene_list: Annotated[list[str], "Genes whose average expression defines the score (must be in var_names)"],
    score_name: Annotated[str, "obs column to store the score"] = "score",
    ctrl_size: Annotated[int, "Number of control genes per gene-list gene's expression bin"] = 50,
    gene_pool: Annotated[list[str] | None, "Genes to draw controls from; None uses all genes"] = None,
    n_bins: Annotated[int, "Number of expression-level bins for control selection"] = 25,
    use_raw: Annotated[bool | None, "Use adata.raw; None uses raw if present (scanpy default)"] = None,
    layer: Annotated[str | None, "Layer to use instead of X"] = None,
    random_state: Annotated[int, "Seed for control gene sampling"] = 0,
    output_dir: Annotated[str | None, "Base output directory; a fresh subdirectory is created"] = None,
) -> dict:
    """Score cells for a gene set against expression-matched control genes (scanpy.tl.score_genes).
    Input is an h5ad plus gene list; output is an h5ad with the score in obs and a score CSV.
    """
    adata = _read(h5ad_path)
    pool = adata.raw.var_names if (use_raw if use_raw is not None else adata.raw is not None) else adata.var_names
    missing = [g for g in gene_list if g not in pool]
    if len(missing) == len(gene_list):
        raise ValueError("none of gene_list is present in the data's var_names")
    sc.tl.score_genes(adata, gene_list, ctrl_size=ctrl_size, n_bins=n_bins, score_name=score_name,
                      random_state=random_state,
                      **_kw(gene_pool=gene_pool, use_raw=use_raw, layer=layer))
    out = _new_dir(output_dir, "scanpy_tl_score_genes")
    h5 = _write(adata, out, "score_genes.h5ad")
    csv = out / "gene_scores.csv"
    adata.obs[[score_name]].to_csv(csv)
    return _result(f"Gene score stored in obs['{score_name}'].", _ref("_score_genes.py"),
                   [("h5ad with gene score", h5), ("score per cell", str(csv.resolve()))],
                   score_name=score_name, n_genes_requested=len(gene_list), genes_not_in_data=missing,
                   score_mean=float(adata.obs[score_name].mean()))


@scanpy_tl_mcp.tool()
def scanpy_tl_score_genes_cell_cycle(
    h5ad_path: Annotated[str, "h5ad with normalised/log-transformed expression"],
    s_genes: Annotated[list[str], "Genes marking S phase"],
    g2m_genes: Annotated[list[str], "Genes marking G2/M phase"],
    n_bins: Annotated[int, "Number of expression-level bins for control selection"] = 25,
    use_raw: Annotated[bool | None, "Use adata.raw; None uses raw if present (scanpy default)"] = None,
    layer: Annotated[str | None, "Layer to use instead of X"] = None,
    random_state: Annotated[int, "Seed for control gene sampling"] = 0,
    output_dir: Annotated[str | None, "Base output directory; a fresh subdirectory is created"] = None,
) -> dict:
    """Assign S/G2M/G1 cell-cycle phase from marker gene scores (scanpy.tl.score_genes_cell_cycle).
    Input is an h5ad plus S and G2M gene lists; output is an h5ad with obs S_score, G2M_score, phase and a CSV.
    """
    adata = _read(h5ad_path)
    pool = adata.raw.var_names if (use_raw if use_raw is not None else adata.raw is not None) else adata.var_names
    missing_s = [g for g in s_genes if g not in pool]
    missing_g = [g for g in g2m_genes if g not in pool]
    sc.tl.score_genes_cell_cycle(adata, s_genes=list(s_genes), g2m_genes=list(g2m_genes), n_bins=n_bins,
                                 random_state=random_state, **_kw(use_raw=use_raw, layer=layer))
    out = _new_dir(output_dir, "scanpy_tl_score_genes_cell_cycle")
    h5 = _write(adata, out, "cell_cycle.h5ad")
    csv = out / "cell_cycle_scores.csv"
    adata.obs[["S_score", "G2M_score", "phase"]].to_csv(csv)
    return _result("Cell-cycle scores and phase written to obs.", _ref("_score_genes.py"),
                   [("h5ad with cell-cycle scores", h5), ("scores and phase per cell", str(csv.resolve()))],
                   phase_counts={str(k): int(v) for k, v in adata.obs["phase"].value_counts().items()},
                   s_genes_not_in_data=missing_s, g2m_genes_not_in_data=missing_g)


if __name__ == "__main__":
    scanpy_tl_mcp.run()
