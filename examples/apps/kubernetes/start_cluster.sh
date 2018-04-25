#!/bin/bash

ZONE=us-west1-b

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
