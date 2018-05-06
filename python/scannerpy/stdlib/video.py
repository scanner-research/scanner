import cv2

def write_video(path, frames, fps=24.0):
    assert len(frames) > 0

    output = cv2.VideoWriter(
        path,
        cv2.VideoWriter_fourcc(*'X264'),
        fps,
        (frames[0].shape[1], frames[0].shape[0]))

    for frame in frames:
        output.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
