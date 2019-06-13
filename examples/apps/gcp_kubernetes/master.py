import scannerpy
import os

if __name__ == "__main__":
    scannerpy.start_master(port='8080',
                           block=True,
                           watchdog=False,
                           no_workers_timeout=180)
