

class Job(object):
    """
    A specification of a table to produce as output of a bulk job.
    """
    def __init__(self, op_args):
        self._op_args = op_args

    def op_args(self):
        return self._op_args
