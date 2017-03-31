from scannerpy import Database
import marshal

def my_kernel(columns):
    print len(columns[0]), len(columns[1])
    return ['1']

with Database(debug=True) as db:
    op = db.ops.Python(
        kernel=marshal.dumps(my_kernel.__code__))

    db.run(db.sampler().all([('example', 'example_py')]), op, force=True)
