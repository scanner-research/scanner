#!/bin/bash

export PROJECT=$(gcloud config get-value project)

docker build -f Dockerfile.master -t gcr.io/$PROJECT/scanner-master:cpu .
docker build -f Dockerfile.worker -t gcr.io/$PROJECT/scanner-worker:cpu .

gcloud docker -- push gcr.io/$PROJECT/scanner-master:cpu
gcloud docker -- push gcr.io/$PROJECT/scanner-worker:cpu

kubectl delete deploy --all
kubectl create -f master.yml
kubectl create -f worker.yml
