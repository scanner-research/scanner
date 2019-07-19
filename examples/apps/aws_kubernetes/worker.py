import scannerpy
import os

if __name__ == "__main__":
    scannerpy.start_worker('{}:{}'.format(
        os.environ['SCANNER_MASTER_SERVICE_HOST'],
        os.environ['SCANNER_MASTER_SERVICE_PORT']),
                           block=True,
                           watchdog=False,
                           port=5002)
