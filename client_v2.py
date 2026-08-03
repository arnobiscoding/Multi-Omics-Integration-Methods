"""
SpatialAblate Benchmark Client V2 Entrypoint Alias
===================================================

This module imports and exposes SpatialAblateClientV2 and main CLI entrypoint
from spatialablate_client_v2.py.
"""

from spatialablate_client_v2 import SpatialAblateClientV2, main, get_cluster_count

if __name__ == "__main__":
    main()
