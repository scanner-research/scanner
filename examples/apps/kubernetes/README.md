1. Create a Google Cloud project.

2. Create a bucket. Put the name into `storage.bucket` in `config.toml`.

3. In Google Cloud Console, go to **Storage > Settings > Interoperability**. Enable interoperable access. Create a new Access Key/Secret pair.

4. Install the [gcloud SDK](https://cloud.google.com/sdk/downloads).

5. Install the kubernetes command-line management tool `kubectl` by running:
```
gcloud components install kubectl
```

6. Start a Kubernetes cluster with:
```
export ZONE=us-west1-b

gcloud container clusters create example-cluster \
       --zone "$ZONE" \
       --machine-type "n1-standard-2" \
       --num-nodes 1

gcloud container clusters get-credentials example-cluster --zone "$ZONE"

gcloud container node-pools create workers \
       --zone "$ZONE" \
       --cluster example-cluster \
       --machine-type "n1-standard-2" \
       --num-nodes 1 \
       --enable-autoscaling \
       --min-nodes 0 \
       --max-nodes 5 \
       --preemptible
```

This creates a cluster with one master node and a scalable pool of worker nodes (min 0, max 5) with 2 CPU cores each. You can also run `./start_cluster.sh`.


7. Save your storage keys from step 3 by running this (replacing `YOUR_ACCESS_KEY` and `YOUR_SECRET`):
```
kubectl create secret generic aws-storage-key \
    --from-literal=AWS_ACCESS_KEY_ID=YOUR_ACCESS_KEY \
    --from-literal=AWS_SECRET_ACCESS_KEY=YOUR_SECRET
```

8. Expose the Scanner port from your master node by running:
```
kubectl expose deploy/scanner-master --type=NodePort --port=8080
```

8. Deploy Scanner to your cluster by running:
```
#!/bin/bash

export PROJECT=$(gcloud config get-value project)

docker build -f Dockerfile.master -t gcr.io/$PROJECT/scanner-master:cpu .
docker build -f Dockerfile.worker -t gcr.io/$PROJECT/scanner-worker:cpu .

gcloud docker -- push gcr.io/$PROJECT/scanner-master:cpu
gcloud docker -- push gcr.io/$PROJECT/scanner-worker:cpu

kubectl delete deploy --all
kubectl create -f master.yml
kubectl create -f worker.yml
```

This builds Docker images for the master and worker (from `Dockerfile.master` and `Dockerfile.worker`), pushes them to the Google Container Registry, and then sends configuration (from `master.yml` and `worker.yml`) to your Kubernetes cluster. You can also run `./deploy.sh`.

9. Copy an example video into your bucket (replacing `YOUR_BUCKET` with your bucket name from step 2):
```
gsutil cp gs://scanner-data/public/sample-clip.mp4 gs://YOUR_BUCKET/sample.mp4
```

10. Test our example histogram pipeline by running:
```
python3 example.py
```

And that's it! You have Scanner successfully running on Kubernetes.
