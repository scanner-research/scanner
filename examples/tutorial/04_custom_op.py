from scannerpy import Database

db = Database()

# To load a custom op into the Scanner runtime, we use db.load_op to open the
# shared library we compiled. If the op takes arguments, it also optionally
# takes a path to the generated python file for the arg protobuf.
db.load_op('resize_op/resize_op.so', 'resize_op/args_pb2.py')

# Then we use our op just like in the other examples.
resize = db.ops.Resize(scale=0.5)

sampler = db.sampler()
tasks = sampler.all([('example', 'example_resized')])
db.run(tasks, resize)
