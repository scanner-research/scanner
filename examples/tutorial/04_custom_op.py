from scannerpy import Database
import os.path

db = Database()

if not os.path.isfile('resize_op/build/resize_op.so'):
    print('You need to build the custom op first: \n'
          '$ cd resize_op; mkdir build && cd build; cmake ..; make')
    exit()

# To load a custom op into the Scanner runtime, we use db.load_op to open the
# shared library we compiled. If the op takes arguments, it also optionally
# takes a path to the generated python file for the arg protobuf.
db.load_op('resize_op/build/resize_op.so', 'resize_op/build/args_pb2.py')

# Then we use our op just like in the other examples.
resize = db.ops.Resize(width=200, height=300)

sampler = db.sampler()
tasks = sampler.all([('example', 'example_resized')])
db.run(tasks, resize, force=True)
