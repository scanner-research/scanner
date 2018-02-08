from scannerpy import (Database, Config, DeviceType, ColumnType, BulkJob, Job,
                       ProtobufGenerator, ScannerException)
import os
import time
import numpy as np

cwd = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':

  frameList = []
  for i in range(5):
    frameList.append(np.random.random((3, 3)))

  with Database(stream_mode=True) as db:
    db.register_op('TestRealtime', ['frame'], ['dummy'])
    db.register_python_kernel('TestRealtime', DeviceType.CPU,
                              cwd + '/test_realtime_kernel.py')

    input = db.ops.MemoryInput()

    test_out = db.ops.TestRealtime(frame=input)
    output = db.ops.MemoryOutput(columns=[test_out])
    job = Job(
      op_args={
        input: db.table('dummy_input').column('col1'),
        output: "dummy_output"
      }
    )
    bulk_job = BulkJob(output=output, jobs=[job])

    job_result = db.run(bulk_job, force=True)
    for frame in frameList:
      input.push(frame.tobytes())
    print("Now pull frame from scanner!")
    for i in range(5):
      print('pull {:d}'.format(i))
      outframe = output.pull()
      result = (frameList[i].tobytes() == outframe)
      print("Row id " + str(i) + " passed test? " + str(result))
    input.close()
    print('sleep for 10')
    time.sleep(10)
    print('done sleeping')
    # db_result.wait_until_done()
