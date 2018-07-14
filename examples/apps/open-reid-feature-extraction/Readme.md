Open-ReID on Scanner
====================

# Steps:
## Build the image
- Build a docker image using the Dockerfile in this folder.
```
nvidia-docker build -t scanner-openreid .
```

## Download the pre-trained model for Open-ReID
### Option 1: Download the pre-trained model on VIPeR dataset
Link: [model_best.pth.tar](https://drive.google.com/open?id=1LDiX4AyuJbhPzVPJF6BOOjuOaN81F6IZ)

### Option 2: Train a model using the original Open-ReID repository.
Link: Follow the instruction on the [Open-ReID repository](https://github.com/Cysu/open-reid)

## Run a new docker container
- In this folder, run the following command to create a new docker container and map the current directory to the same path so that we can access the `model_best.pth.tar` from inside the container.
```
docker run --runtime=nvidia -it -v $(pwd):/opt/scanner/examples/apps/open-reid-feature-extraction/ scanner-openreid /bin/bash
```

## Run the example
- Change directory to the `open-reid-feature-extraction` example
```
cd /opt/scanner/examples/apps/open-reid-feature-extraction/
```
- Get an example video
```
wget https://storage.googleapis.com/scanner-data/public/sample-clip.mp4
```
- Run the `extract_features.py` to extract the Open-ReID for every frame. In practice, you should perform the human detection on each frame. Then, provide the frame and the bounding boxes for each person into the Open-ReID kernel. We want to extract the Open-ReID feature for each bounding boxes. In this example, we will skip the human detection part.
```
python3 extract_features.py sample-clip.mp4 model_best.pth.tar
```
- The results are saved in the `reid_features.npy` which can be loaded using `np.load("reid_features.npy")`