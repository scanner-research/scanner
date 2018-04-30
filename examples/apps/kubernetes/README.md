1. Create a Google Cloud project.

2. Create a bucket on Google Cloud Storage (GCS). Put the name into `storage.bucket` in `config.toml`.

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


7. Save your storage keys from step 3 by running this (replacing `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`):
```
AWS_ACCESS_KEY_ID=<...>
AWS_SECRET_ACCESS_KEY=<...>

kubectl create secret generic aws-storage-key \
    --from-literal=AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
    --from-literal=AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
```

8. Get your project ID by running:
```
export PROJECT=$(gcloud config get-value project)
echo $PROJECT
```

9. Replace `YOUR_PROJECT_ID` in `master.yml.template` and `worker.yml.template`:
```
sed "s/YOUR_PROJECT_ID/$PROJECT/g" master.yml.template > master.yml
sed "s/YOUR_PROJECT_ID/$PROJECT/g" worker.yml.template > worker.yml
```

9. Deploy Scanner to your cluster by running:
```
docker pull scannerresearch/scanner:cpu
docker build -f Dockerfile.master -t gcr.io/$PROJECT/scanner-master:cpu .
docker build -f Dockerfile.worker -t gcr.io/$PROJECT/scanner-worker:cpu .

gcloud docker -- push gcr.io/$PROJECT/scanner-master:cpu
gcloud docker -- push gcr.io/$PROJECT/scanner-worker:cpu

kubectl delete deploy --all
kubectl create -f master.yml
kubectl create -f worker.yml
```

This builds Docker images for the master and worker (from `Dockerfile.master` and `Dockerfile.worker`), pushes them to the Google Container Registry, and then sends configuration (from `master.yml` and `worker.yml`) to your Kubernetes cluster. You can also run `./deploy.sh`.

10. Expose the Scanner port from your master node by running:
```
kubectl expose deploy/scanner-master --type=NodePort --port=8080
```

11. Copy an example video into your bucket (replacing `YOUR_BUCKET` with your bucket name from step 2):
```
gsutil cp gs://scanner-data/public/sample-clip.mp4 gs://YOUR_BUCKET/sample.mp4
```

12. Test our example histogram pipeline by running:
```
apt-get install jq
python3 example.py
```

And that's it! You have Scanner successfully running on Kubernetes.

13. Shutdown your Kubernetes cluster

```
gcloud container clusters delete example-cluster
```
