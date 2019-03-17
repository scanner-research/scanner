from scannerpy import Client

# The following performs the same computation as main.py, but now on a collection
# of videos instead of just one
def main():
    sc = Client()
    
    # Create a stored stream to represent the input video
    videos_to_process = [
        ('sample-clip-1', 'sample-clip-1.mp4'),
        ('sample-clip-2', 'sample-clip-2.mp4'),
        ('sample-clip-3', 'sample-clip-3.mp4')
       ]
    input_streams = [NamedVideoStream(sc, info[0], path=info[1])
                     for info in videos_to_process]
    
    # Define a Computation Graph
    frame = sc.io.Input(input_streams)
    sampled_frame = sc.streams.Stride(frame, 3) # Select every third frame
    resized = sc.ops.Resize(frame=sampled_frame, width=640, height=480) # Resize input frame
    
    # Create stored streams to represent the output videos
    output_streams = [NamedVideoStream(sc, info[0] + 'resized')
                     for info in videos_to_process]
    output = sc.io.Output(resized, output_streams)
    
    # Execute the computation graph 
    sc.run(output)
    
    # Save the resized video as an mp4 file
    for stream in output_streams:
        stream.save_mp4(stream.name())


if __name__ == "__main__":
    main()
