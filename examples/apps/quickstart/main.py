from scannerpy import Client

def main():
    sc = Client()
    
    # Create a stored stream to represent the input video
    input_stream = NamedVideoStream(sc, 'sample-clip', path='sample-clip.mp4')
    
    # Define a Computation Graph
    frame = sc.io.Input([input_stream])
    sampled_frame = sc.streams.Stride(frame, 3) # Select every third frame
    resized = sc.ops.Resize(frame=sampled_frame, width=640, height=480) # Resize input frame
    
    # Create a stored stream to represent the output video
    output_stream = NamedVideoStream(sc, 'sample-clip-resized')
    output = sc.io.Output(resized, [output_stream])
    
    # Execute the computation graph 
    sc.run(output)
    
    # Save the resized video as an mp4 file
    output_stream.save_mp4('sample-clip-resized')


if __name__ == "__main__":
    main()
