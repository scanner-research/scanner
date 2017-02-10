from scannerpy import Database

db = Database()
db.load_op('resize_op/resize_op.so', 'resize_op/args_pb2.py')

resize = db.ops.Resize(scale=0.5)
