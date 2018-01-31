import numpy as np
import cv2
import os

if __name__ == '__main__':
  cap = cv2.VideoCapture(0)

  while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    input = db.ops.MemoryInput()
    input.push(frame)
    # hist = db.ops.Histogram(frame=frame)
    output = db.ops.MemoryOutput(columns=[input])
    job = Job(
        op_args={}
    )
    bulk_job = BulkJob(output=output, jobs=[job])
    db.run(bulk_job, force=True)
    
    outframe = output.pull()

    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',outframe)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  # When everything done, release the capture
  cap.release()
  cv2.destroyAllWindows()