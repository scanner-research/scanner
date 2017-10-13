class BulkJob:
    """
    Specifies a set of jobs that will share the same execution DAG.
    """
    def __init__(self, dag, jobs):
        self._dag = dag
        self._jobs = jobs

    def dag(self):
        return self._dag

    def jobs(self):
        return self._jobs
