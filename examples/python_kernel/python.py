from scannerpy import Database, Job
import marshal

with Database(debug=True) as db:
    MyOp = db.register_python_op('my_kernel.py')

    frame = db.table('example').as_op().range(1000, 1010)
    test = MyOp(frame)
    job = Job(columns = [test], name = 'example_py')
    db.run(job, force=True, pipeline_instances_per_node=1)
