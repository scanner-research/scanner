cd ~/capture

### 1. Check if container repo exists
aws ecr describe-repositories --repository-names scanner
REG_EXISTS=$?
if [ $REG_EXISTS -ne 0 ]; then
    # Create container repo
    aws ecr create-repository --repository-name scanner
fi

# Get container repo URI
REPO_URI=$(aws ecr describe-repositories --repository-names scanner | jq -r '.repositories[0].repositoryUri')
echo $REPO_URI

### 2. Build master and worker docker images
docker pull scannerresearch/scanner:cpu-latest

docker build -t $REPO_URI:scanner-master . \
       -f Dockerfile.master

docker build -t $REPO_URI:scanner-worker . \
       -f Dockerfile.worker

aws configure set default.region us-west-2

# Provides an auth token to enable pushing to container repo
LOGIN_CMD=$(aws ecr get-login --no-include-email)
eval $LOGIN_CMD

# Push master and worker images
docker push $REPO_URI:scanner-master
docker push $REPO_URI:scanner-worker

### 2. Deploy master and worker services

# Create secret for sharing AWS credentials with instances
kubectl create secret generic aws-storage-key \
        --from-literal=AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
        --from-literal=AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY

# Replace REPO_NAME with the location of the docker image
sed "s|<REPO_NAME>|$REPO_URI:scanner-master|g" master.yml.template > master.yml
sed "s|<REPO_NAME>|$REPO_URI:scanner-worker|g" worker.yml.template > worker.yml

# Record existing replicas for worker so we can scale the service after deleting
REPLICAS=$(kubectl get deployments scanner-worker -o json | jq '.spec.replicas' -r)

# Delete and then redeploy the master and worker services
kubectl delete deploy --all
kubectl create -f master.yml
kubectl create -f worker.yml

# If there was an existing service, scale the new one back up to the same size
if [[ "$REPLICAS" ]]; then
    kubectl scale deployment/scanner-worker --replicas=$REPLICAS
fi

### 3. Expose the master port for the workers to connect to
kubectl expose -f master.yml --type=LoadBalancer --target-port=8080
