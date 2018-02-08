import grpc
from scanner.engine import rpc_pb2, rpc_pb2_grpc
from concurrent import futures
import time
import threading
from Queue import Queue

# Assume only one machine / one worker
# The python master is by default at localhost:5000
class MasterServicer(rpc_pb2_grpc.MasterServicer):
  """Provides methods that implement functionality of master service."""

  def __init__(self):
    self._worker = None
    self._worker_active = False
    self._lock = threading.Lock()
    self._input_queue = Queue()
    self._output_queue = Queue()
    self._finished = False
    self._count = 0

  # rpc Shutdown (Empty) returns (Result) {}
  # Essentially do nothing but we need it to make worker happy
  def Shutdown(self, request, context):
    print("Received shutdown signal!")
    result = rpc_pb2.Result(success=True)
    empty = rpc_pb2.Empty()
    self._worker.Shutdown(empty)
    return result

  # rpc PushRow (ElementDescriptor) returns (Empty) {}
  def PushRow(self, request, context):
    if request.row_id == -1:
      self._finished = True
    else:
      self._input_queue.put(request)
      print("Pushed a row into input queue.")
    empty = rpc_pb2.Empty()
    return empty

  # rpc PullRow (Empty) returns (ElementDescriptor) {}
  def PullRow(self, request, context):
    print("Pulled a row from input queue.")
    while self._output_queue.empty():
      time.sleep(1)
    element_descriptor = self._output_queue.get()
    return element_descriptor

  # Called after a new worker spawns to register with the master
  # rpc RegisterWorker (WorkerParams) returns (Registration) {}
  def RegisterWorker(self, request, context):
    self._lock.acquire()

    port = request.port
    channel = grpc.insecure_channel('localhost:'+port)
    self._worker = rpc_pb2_grpc.WorkerStub(channel)
    self._worker_active = True
    registration = rpc_pb2.Registration(node_id=0)

    # Skip LoadOp because we have already done this in C++ master
    self._lock.release()
    print("{} Worker registered in python master.".format(time.asctime( time.localtime(time.time()) )))
    return registration

  # Called when a worker is removed
  # rpc UnregisterWorker (NodeInfo) returns (Empty) {}
  def UnregisterWorker(self, request, context):
    self._lock.acquire()

    self._worker_active = False

    self._lock.release()
    print("Worker unregistered in python master.")
    empty = rpc_pb2.Empty()
    return empty


  # Internal
  # rpc NextWork (NodeInfo) returns (NewWork) {}
  def NextWork(self, request, context):
    self._lock.acquire()

    new_work = rpc_pb2.NewWork()
    new_work.table_id = 0
    new_work.job_index = 0
    new_work.task_index = self._count
    self._count += 1
    print("Asking for next work, job_index={}, task_index={}.".format(new_work.job_index, new_work.task_index))
    new_work.output_rows.append(0)
    if not self._worker_active:
      print("Asking for next work, but worker is inactive.")
      new_work.no_more_work = True

    else:
      if self._input_queue.empty():
        print("Asking for next work, but the work queue is empty.")
        if self._finished:
          print("Asking for next work, but all works are already finished.")
          new_work.no_more_work = True
        else:
          print("Asking for next work, waiting for work.")
          new_work.wait_for_work = True

      else:
        element_descriptor = self._input_queue.get()  # type of rpc_pb2.ElementDescriptor()
        new_work.rows.extend([element_descriptor])
        print("Asking for next work, pulled row_id={} from input queue.".format(element_descriptor.row_id))
        print("The length of buffer of pulled row is: {}".format(len(element_descriptor.buffer)))

    self._lock.release()
    return new_work

  # rpc FinishedWork (FinishedWorkParameters) returns (Empty) {}
  def FinishedWork(self, request, context):
    element_descriptor = request.rows[0]
    self._output_queue.put(element_descriptor)
    print("Pushed row_id={} back to output queue.".format(element_descriptor.row_id))
    print("The length of row buffer is: {}".format(len(element_descriptor.buffer)))
    empty = rpc_pb2.Empty()
    return empty

  # rpc FinishedJob (FinishedJobParams) returns (Empty) {}
  def FinishedJob(self, request, context):
    result = request.result
    print("Finished job with result: {}".format(result))
    empty = rpc_pb2.Empty()
    return empty

  # rpc NewJob (BulkJobParameters) returns (Result) {}
  def NewJob(self, request, context):
    result = rpc_pb2.Result(success=True)
    job_params = request
    self._worker.NewJob(job_params)
    return result

if __name__ == "__main__":
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
  rpc_pb2_grpc.add_MasterServicer_to_server(MasterServicer(), server)
  server.add_insecure_port('localhost:5000')
  server.start()
  print("{} Python master started.".format(time.asctime( time.localtime(time.time()) )))
  try:
    while True:
      time.sleep(60 * 60 * 24)
  except KeyboardInterrupt:
    print('error!!!!!')
    server.stop(0)
