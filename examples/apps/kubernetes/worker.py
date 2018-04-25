import scannerpy
import os

scannerpy.start_worker(
    '{}:{}'.format(os.environ['SCANNER_MASTER_SERVICE_HOST'],
                   os.environ['SCANNER_MASTER_SERVICE_PORT']),
    block=True,
    watchdog=False,
    prefetch_table_metadata=True,
    port=5002)
