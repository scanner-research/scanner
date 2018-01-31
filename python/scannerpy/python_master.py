import grpc
import rpc_pb2
import rpc_pb2_grpc
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
    print("Python master created.")

  # rpc PushRow (ElementDescriptor) returns (Empty) {}
  def PushRow(self, request, context):
    self._input_queue.put(request)
    empty = rpc_pb2.Empty()
    return empty

  # rpc PullRow (Empty) returns (ElementDescriptor) {}
  def PullRow(self, request, context):
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
    print("Worker registered in python master.")
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
        new_work.rows.append(element_descriptor)
        print("Asking for next work, pulled row_id={} from input queue.".format(element_descriptor.row_id))

    self._lock.release()
    return new_work

  # rpc FinishedWork (FinishedWorkParameters) returns (Empty) {}
  def FinishedWork(self, request, context):
    self._lock.acquire()

    element_descriptor = request.rows
    self._output_queue.put(element_descriptor)

    self._lock.release()
    print("Pushed row_id={} back to output queue.".format(element_descriptor.row_id))
    empty = rpc_pb2.Empty()
    return empty

  # rpc FinishedJob (FinishedJobParams) returns (Empty) {}
  def FinishedJob(self, request, context):
    self._lock.acquire()

    result = request.result

    self._lock.release()
    print("Finished job with result: {}".format(result))
    empty = rpc_pb2.Empty()
    return empty