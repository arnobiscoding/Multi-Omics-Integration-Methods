"""
VGAT-PoE-DEC Benchmark Pipeline Entry Point
===========================================

Exposes the pipeline workflow for running multi-dataset multi-seed benchmarking.
"""

import sys
import os

curr_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(curr_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from vgat_poe_dec_pipeline import main as run_pipeline_main, run_vgat_poe_dec_workflow
except ImportError:
    from ..vgat_poe_dec_pipeline import main as run_pipeline_main, run_vgat_poe_dec_workflow


if __name__ == "__main__":
    run_pipeline_main()
