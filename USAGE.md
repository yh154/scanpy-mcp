# Scanpy MCP server

MCP server exposing 28 tools from [scverse/scanpy](https://github.com/scverse/scanpy) (pinned commit
`e099d9bcfdfc5ef065355c168ba2f639349939e7`) for single-cell RNA-seq preprocessing, clustering,
marker genes, label transfer, trajectories and plotting, plus one workflow prompt
(`scanpy_clustering_tutorial`). Every tool is a thin wrapper over a public scanpy function; no
scientific method is reimplemented.

## Scope and limits

- Verified against the scanpy tutorials `clustering`, `clustering-2017`, `pearson_residuals`,
  `cell-cycle`, `integrating-data-using-ingest`, `paga-paul15`, `plotting/core` and
  `plotting/advanced`, using new direct scanpy calls on the same inputs.
- **Deviations from the tutorials:**
  - `paga-paul15`: `sc.tl.louvain` could not be installed on this platform (no Python 3.13 wheel,
    source build fails), so the notebook was run with Leiden clustering instead. The PAGA, diffusion
    map and DPT references for that tutorial therefore use Leiden groups and are not comparable to
    the published Louvain figures.
  - `integrating-data-using-ingest`: the BBKNN comparison cells were skipped (`bbknn` not installed).
    `ingest` UMAP coordinates are not reproducible between runs; labels and `X_pca` were compared.
- **Not included (deferred):** `aggregate`, `load_dataset`, `combat`, `tsne`, `draw_graph`,
  `louvain`, `embedding_density`, and the external `bbknn`, `harmony`, `scanorama`. The tutorial
  steps `sc.pl.scatter`, the highly-variable-genes plot and the PCA plots have no tool.
- Tools take **h5ad file paths** and write new h5ad/PNG/CSV files into a fresh sub-directory per call
  (under `output_dir` when given). Input files are never modified. You must supply the data; there
  is no dataset-loading tool.
- Correctness here covers the selected tools on the tutorial inputs. It does not establish
  whole-repository correctness or reproduce any paper's results.

## Tested environment

- CPython 3.13.11, macOS x86_64. Linux and Windows were not tested.
- fastmcp 4.0.3; scanpy installed from the pinned upstream commit (see `src/requirements.txt`,
  122 pinned packages). Scanpy is not bundled; its BSD-3-Clause license applies to the upstream code.
- No GPU, credentials or data downloads are needed to run the tools. Installing scanpy from git needs
  `git` and network access.
- numba may print a warning that it runs in serial mode; installing `tbb` can speed up some steps.
  Doublet scoring on ~17k cells is slow in serial mode.

## Install

From the extracted folder:

```bash
uv venv .venv --python 3.13
uv pip install --python .venv/bin/python -r src/requirements.txt
.venv/bin/python -c "import scanpy, fastmcp"
```

## Start the server

```bash
.venv/bin/python src/scanpy_mcp.py        # stdio transport
```

Client registration (syntax from `fastmcp install claude-code --help`; not run by this package):

```bash
.venv/bin/fastmcp install claude-code src/scanpy_mcp.py --name scanpy --with-requirements src/requirements.txt
```

## Tools

Chain tools by passing the h5ad path one tool returns into the next. Each tool returns
`message`, `reference` and `artifacts` (absolute paths) plus a few summary fields.
Parameters left unset use the scanpy defaults. Use each tool's input schema for the full
optional-parameter list and defaults.

### Preprocessing (`scanpy.pp`, `scanpy.experimental.pp`) — 13 tools

| Tool | Use | Required inputs |
|---|---|---|
| `scanpy_pp_calculate_qc_metrics` | Compute per-cell and per-gene QC metrics (counts, genes detected, qc_vars percentages) with scanpy.pp.calculate_qc_metrics. | `data_path` |
| `scanpy_pp_filter_cells` | Filter cells by gene/count thresholds with scanpy.pp.filter_cells (exactly one threshold per call, as upstream). | `data_path` |
| `scanpy_pp_filter_genes` | Filter genes by cell/count thresholds with scanpy.pp.filter_genes (exactly one threshold per call, as upstream). | `data_path` |
| `scanpy_pp_highly_variable_genes` | Annotate (optionally subset to) highly variable genes with scanpy.pp.highly_variable_genes. | `data_path` |
| `scanpy_pp_highly_variable_genes_pearson_residuals` | Select highly variable genes by Pearson residual variance with scanpy.experimental.pp.highly_variable_genes. | `data_path` |
| `scanpy_pp_log1p` | Apply log(1+x) with scanpy.pp.log1p (records uns['log1p']). | `data_path` |
| `scanpy_pp_neighbors` | Build the k-nearest-neighbor graph and connectivities with scanpy.pp.neighbors. | `data_path` |
| `scanpy_pp_normalize_pearson_residuals` | Replace counts by analytic Pearson residuals with scanpy.experimental.pp.normalize_pearson_residuals. | `data_path` |
| `scanpy_pp_normalize_total` | Library-size normalize each cell to target_sum with scanpy.pp.normalize_total. | `data_path` |
| `scanpy_pp_pca` | Compute principal components with scanpy.pp.pca (X_pca, PCs, variance ratio). | `data_path` |
| `scanpy_pp_regress_out` | Regress out unwanted obs covariates from each gene with scanpy.pp.regress_out. | `data_path`, `keys` |
| `scanpy_pp_scale` | Scale each gene to unit variance (optionally zero-centered and clipped) with scanpy.pp.scale. | `data_path` |
| `scanpy_pp_scrublet` | Score and predict doublets with scanpy.pp.scrublet (obs doublet_score, predicted_doublet). | `data_path` |

### Analysis (`scanpy.tl`, `scanpy.get`) — 10 tools

| Tool | Use | Required inputs |
|---|---|---|
| `scanpy_tl_diffmap` | Compute a diffusion map embedding (scanpy.tl.diffmap). | `h5ad_path` |
| `scanpy_tl_dpt` | Compute diffusion pseudotime from a root cell (scanpy.tl.dpt). | `h5ad_path` |
| `scanpy_tl_get_rank_genes_groups_df` | Export rank_genes_groups results as a filtered table (scanpy.get.rank_genes_groups_df). | `h5ad_path` |
| `scanpy_tl_ingest` | Map a query dataset onto a reference embedding and transfer labels (scanpy.tl.ingest). | `query_h5ad_path`, `reference_h5ad_path` |
| `scanpy_tl_leiden` | Cluster cells with the Leiden algorithm on the kNN graph (scanpy.tl.leiden). | `h5ad_path` |
| `scanpy_tl_paga` | Compute the PAGA abstracted graph of cell groups (scanpy.tl.paga). | `h5ad_path`, `groups` |
| `scanpy_tl_rank_genes_groups` | Rank marker genes for each group of an obs column (scanpy.tl.rank_genes_groups). | `h5ad_path`, `groupby` |
| `scanpy_tl_score_genes` | Score cells for a gene set against expression-matched control genes (scanpy.tl.score_genes). | `h5ad_path`, `gene_list` |
| `scanpy_tl_score_genes_cell_cycle` | Assign S/G2M/G1 cell-cycle phase from marker gene scores (scanpy.tl.score_genes_cell_cycle). | `h5ad_path`, `s_genes`, `g2m_genes` |
| `scanpy_tl_umap` | Embed the neighbor graph with UMAP (scanpy.tl.umap). | `h5ad_path` |

### Plotting (`scanpy.pl`) — 5 tools

| Tool | Use | Required inputs |
|---|---|---|
| `scanpy_pl_dotplot` | Draw a dot plot of mean expression and fraction of expressing cells per group (scanpy.pl.dotplot). | `h5ad_path`, `var_names`, `groupby` |
| `scanpy_pl_paga` | Plot the PAGA abstracted cluster graph (scanpy.pl.paga). | `h5ad_path` |
| `scanpy_pl_rank_genes_groups_dotplot` | Plot top differentially expressed genes per group as a dot plot (scanpy.pl.rank_genes_groups_dotplot). | `h5ad_path` |
| `scanpy_pl_umap` | Plot cells on the stored UMAP embedding colored by genes or annotations (scanpy.pl.umap). | `h5ad_path`, `color` |
| `scanpy_pl_violin` | Draw violin plots of genes or numeric annotations, optionally grouped (scanpy.pl.violin). | `h5ad_path`, `keys` |

## Workflow prompt

`scanpy_clustering_tutorial(h5ad_path, out_dir, marker_genes, target_cluster="7", min_cells=5)`
returns step-by-step instructions that replicate the scverse clustering tutorial in order
(QC, filtering, doublets, normalization, highly variable genes, PCA, neighbors, UMAP, Leiden at
several resolutions, marker genes). It only returns text; it does not run the analysis. The input
h5ad needs `obs["sample"]` and boolean `var` columns `mt`, `ribo`, `hb`. Note `min_cells` defaults to 5
here, while the tutorial uses 3.

## Notes on specific tools

- `scanpy_tl_ingest`: the reference must already have PCA, neighbors and UMAP computed and share genes
  with the query.
- `scanpy_tl_score_genes_cell_cycle`: does not drop genes absent from the data (upstream behavior);
  missing genes are reported. Pass gene lists already filtered to the data to match the tutorial.
- `scanpy_tl_dpt`: needs a root cell (index, or an obs key and value); unreachable cells are reported
  as non-finite.
- `scanpy_pl_paga`: layouts `rt` and `rt_circular` are not offered (they fail in the pinned scanpy).
- Several scanpy arguments used here (`layer`, `batch_key`, `key_added`, ...) are deprecated upstream and
  print `FutureWarning` on stderr; they still work in the pinned version.

## Validation performed

- 194 module tests passed (60 preprocessing, 96 analysis, 38 plotting) against the tutorial checkpoints,
  new direct scanpy calls, changed parameters, error cases and repeated-call isolation.
- 57 real-stdio acceptance cases (every tool, error cases, repeated calls) passed against the full
  server in the development environment and in a fresh environment built from `src/requirements.txt`.
- Test fixtures and reports are not shipped in this package.
