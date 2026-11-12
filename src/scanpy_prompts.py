"""MCP workflow prompts for the Scanpy server.

Prompts return instructions only; they do not run the analysis.
Workflow follows docs/tutorials/basics/clustering.ipynb of scverse/scanpy.
"""
from typing import Annotated

from fastmcp import FastMCP

scanpy_prompts_mcp = FastMCP(name="scanpy_prompts")


@scanpy_prompts_mcp.prompt(
    name="scanpy_clustering_tutorial",
    description="Replicate the scverse Scanpy preprocessing and clustering tutorial with the scanpy_* tools.",
)
def scanpy_clustering_tutorial(
    h5ad_path: Annotated[str, "Concatenated h5ad with obs['sample'] and boolean var columns mt, ribo, hb"],
    out_dir: Annotated[str, "Directory for outputs"],
    marker_genes: Annotated[str, "Marker genes per cell type for the dotplots, e.g. {'T': ['CD3D'], 'B': ['MS4A1']}"],
    target_cluster: Annotated[str, "Cluster whose top marker genes are shown on the UMAP (tutorial uses '7')"] = "7",
    min_cells: Annotated[int, "min_cells for gene filtering (tutorial default is 3)"] = 5,
) -> str:
    return f"""You have the Scanpy MCP tools (scanpy_pp_*, scanpy_tl_*, scanpy_pl_*). Replicate the scverse
"Preprocessing and clustering" tutorial on the h5ad I provide.

Input: {h5ad_path}
Marker genes for the dotplots: {marker_genes}
Output directory: {out_dir}

RULES
1. Inspect before every step. Before the first tool call, and before each later one, read what
   the previous tool returned: cell/gene counts, obs/var columns, layers, and which keys exist.
   Confirm the next call's inputs (obs keys, batch_key, groupby, var flags) exist. If a required
   key is missing, stop and tell me; never create or guess one.
2. Use each tool's default settings. Change a default only where the values below require it, or
   where a default would give a wrong result. State the reason for any other change first.
3. Chain tools by passing the h5ad path each tool returns into the next. Never modify an input
   file. Report the artifact path after each step.
4. If a step fails or a result looks implausible (for example, zero cells left after filtering),
   stop and report. Do not substitute a different method.

STEPS, IN THIS ORDER
 1. scanpy_pp_calculate_qc_metrics: qc_vars=["mt","ribo","hb"], log1p=True.
 2. scanpy_pl_violin on n_genes_by_counts, total_counts and pct_counts_mt (multi-panel, jitter 0.4).
    Summarize the QC distributions. Don't filter on mt yet.
 3. scanpy_pp_filter_cells: min_genes=100.
 4. scanpy_pp_filter_genes: min_cells={min_cells}.
 5. scanpy_pp_scrublet: batch_key="sample".
 6. scanpy_pp_normalize_total (defaults), then scanpy_pp_log1p (defaults).
 7. scanpy_pp_highly_variable_genes: n_top_genes=2000, batch_key="sample".
 8. scanpy_pp_pca (defaults).
 9. scanpy_pp_neighbors (defaults).
10. scanpy_tl_umap (defaults), then scanpy_pl_umap colored by "sample".
11. scanpy_tl_leiden: flavor="igraph", n_iterations=2. Then scanpy_pl_umap colored by "leiden",
    then by ["leiden","predicted_doublet","doublet_score"].
12. Multi-resolution: scanpy_tl_leiden three times with flavor="igraph" and resolution 0.02, 0.5
    and 2.0, key_added "leiden_res_0.02", "leiden_res_0.50", "leiden_res_2.00". Plot them on the
    UMAP (legend_loc="on data").
13. scanpy_pl_dotplot with my marker genes, groupby="leiden_res_0.02", standard_scale="var";
    repeat for "leiden_res_0.50".
14. scanpy_tl_rank_genes_groups: groupby="leiden_res_0.50", method="wilcoxon". Then
    scanpy_pl_rank_genes_groups_dotplot (groupby="leiden_res_0.50", standard_scale="var", n_genes=5).
15. scanpy_tl_get_rank_genes_groups_df for group "{target_cluster}". Then scanpy_pl_umap colored by
    its top 5 genes plus "leiden_res_0.50".

FINISH
Give me a table of every step: tool, parameters actually used, any default you changed and why,
and output path. List the figures. Then give the cluster counts at each resolution and the top 5
marker genes per cluster.
"""
