import numpy as np
import scannerpy as sp
import scannertools.imgproc

@sp.register_python_op()
def CloneChannels(config, frame: sp.FrameType) -> sp.FrameType:
    return np.dstack([frame for _ in range(config.args['replications'])])

def main():
    cl = sp.Client()

    # Create a stored stream to represent the input video
    input_stream = sp.NamedVideoStream(cl, 'sample-clip', path='sample-clip.mp4')

    # Define a Computation Graph
    frames = cl.io.Input([input_stream])
    sampled_frames = cl.streams.Stride(frames, [2]) # Select every other frame
    resized_frames = cl.ops.Resize(frame=sampled_frames, width=[640], height=[480]) # Resize input frame
    grayscale_frames = cl.ops.ConvertColor(frame=resized_frames, conversion=['COLOR_RGB2GRAY']) 
    grayscale3_frames = cl.ops.CloneChannels(frame=grayscale_frames, replications=3) 

    # Create a stored stream to represent the output video
    output_stream = sp.NamedVideoStream(cl, 'sample-grayscale')
    output = cl.io.Output(grayscale3_frames, [output_stream])

    # Execute the computation graph
    cl.run(output, sp.PerfParams.manual(50, 250))

    # Save the resized video as an mp4 file
    output_stream.save_mp4('sample-grayscale')

    input_stream.delete(cl)
    output_stream.delete(cl)


if __name__ == "__main__":
    main()
