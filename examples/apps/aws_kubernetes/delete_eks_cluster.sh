programname=$0

function usage {
    echo "usage: $programname name"
    echo "  name    name of the cluster"
    exit 1
}

if [ $# == 0 ]; then
    usage
fi

NAME=$1

CLUSTER_NAME=$NAME
ROLE_ARN=arn:aws:iam::459065735846:role/eksServiceRole

### 1. Delete worker nodes
aws cloudformation delete-stack --stack-name $CLUSTER_NAME-workers

### 2. Delete kubectl config for connecting to cluster
rm ~/.kube/config-$CLUSTER_NAME

### 3. Delete the EKS cluster

aws eks delete-cluster --name $CLUSTER_NAME
