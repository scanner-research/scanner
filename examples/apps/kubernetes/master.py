import scannerpy
import os

scannerpy.start_master(
    port='8080',
    block=True,
    watchdog=False,
    prefetch_table_metadata=True,
    no_workers_timeout=180)
