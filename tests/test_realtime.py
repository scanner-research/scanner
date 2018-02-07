from scannerpy import (Database, Config, DeviceType, ColumnType, BulkJob, Job,
                       ProtobufGenerator, ScannerException)
import os
import pickle
import time

cwd = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
  # cap = cv2.VideoCapture(0)

  with open('data.pkl', 'rb') as pkl_file:
    frameList = pickle.load(pkl_file)
    frameList = frameList[:5]

  # Capture frame-by-frame
  # ret, frame = cap.read()

  with Database(stream_mode=True) as db:
    db.register_op('TestRealtime', [('frame', ColumnType.Stream)], ['dummy'])
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
    input.close()
    print('sleep for 10')
    time.sleep(10)
    print('done sleeping')
    #ob_result.wait_until_done()


  print("Now save pulled frame to file!")
  with open('output.pkl', 'wb') as output_pkl:
    pickle.dump(outframe, output_pkl)

  # Our operations on the frame come here
  # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  # Display the resulting frame
  # cv2.imshow('frame',outframe)
  # if cv2.waitKey(1) & 0xFF == ord('q'):
  #   break

  # When everything done, release the capture
  # cap.release()
  # cv2.destroyAllWindows()
