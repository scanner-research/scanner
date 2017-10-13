class Job:
    """
    A specification of a table to produce as output of a bulk job.
    """
    def __init__(self, output_table_name, op_args):
        self._output_table_name = output_table_name
        self._op_args = op_args

    def output_table_name(self):
        return self._output_table_name

    def op_args(self):
        return self._op_args
