"""Scanpy MCP server: preprocessing, analysis and plotting tools plus a workflow prompt.

Tools wrap the public scanpy API (pinned: scverse/scanpy e099d9bcfdfc5ef065355c168ba2f639349939e7).
"""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_SRC / "tools"))

from fastmcp import FastMCP  # noqa: E402

from scanpy_pl import scanpy_pl_mcp  # noqa: E402
from scanpy_pp import scanpy_pp_mcp  # noqa: E402
from scanpy_prompts import scanpy_prompts_mcp  # noqa: E402
from scanpy_tl import scanpy_tl_mcp  # noqa: E402

mcp = FastMCP(name="scanpy")
# Tool names already carry their module prefix, so mount without an extra namespace.
for _server in (scanpy_pp_mcp, scanpy_tl_mcp, scanpy_pl_mcp, scanpy_prompts_mcp):
    mcp.mount(_server)

if __name__ == "__main__":
    mcp.run()
