import scannerpy
import os

scannerpy.start_master(
    port='8080',
    block=True,
    watchdog=False,
    no_workers_timeout=180)
