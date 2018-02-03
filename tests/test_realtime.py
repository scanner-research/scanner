from scannerpy import (Database, Config, DeviceType, ColumnType, BulkJob, Job,
                       ProtobufGenerator, ScannerException)
import numpy as np
import cv2
import os
import pickle

if __name__ == '__main__':
  # cap = cv2.VideoCapture(0)

  with open('data.pkl', 'rb') as pkl_file:
    frameList = pickle.load(pkl_file)
    frameList = frameList[:5]

  # Capture frame-by-frame
  # ret, frame = cap.read()

  with Database(stream_mode=True) as db:
    input = db.ops.MemoryInput()
    for frame in frameList:
      input.push(frame.tobytes())

    # hist = db.ops.Histogram(frame=frame)
    output = db.ops.MemoryOutput(columns=[input])
    job = Job(
      op_args={
        input: db.table('dummy').column(''),
        output: "dummy_output"
      }
    )
    bulk_job = BulkJob(output=output, jobs=[job])
    db.run(bulk_job, force=True)

    outframe = output.pull()

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
