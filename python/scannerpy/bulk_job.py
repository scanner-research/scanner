class BulkJob:
    """
    Specifies a set of jobs that will share the same execution DAG.
    """
    def __init__(self, output, jobs):
        self._output = output
        self._jobs = jobs

    def output(self):
        return self._output

    def jobs(self):
        return self._jobs
