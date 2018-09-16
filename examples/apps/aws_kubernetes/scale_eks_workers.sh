programname=$0

function usage {
    echo "usage: $programname name nodes"
    echo "  name    name to give the cluster"
    echo "  nodes   number of machines to scale to"
    exit 1
}

if [ $# != 2 ]; then
    usage
fi

NAME=$1
NODES=$(($2 + 1))

CLUSTER_NAME=$NAME

# 1. Get VPC info
VPC_STACK_NAME=eks-vpc

# Get VPC ID
VPC_ID=$(aws cloudformation describe-stacks --stack-name $VPC_STACK_NAME \
             | jq -r '.Stacks[0].Outputs[] | select(.OutputKey=="VpcId") | .OutputValue')

# Get security group ids
SECURITY_GROUP_IDS=$(aws cloudformation describe-stacks --stack-name $VPC_STACK_NAME \
                         | jq -r '.Stacks[0].Outputs[] | select(.OutputKey=="SecurityGroups") | .OutputValue')

# Get subnet outputs
SUBNET_IDS=$(aws cloudformation describe-stacks --stack-name $VPC_STACK_NAME \
                 | jq -r '.Stacks[0].Outputs[] | select(.OutputKey=="SubnetIds") | .OutputValue')

# 2. Change the autoscaling group to spawn more nodes
aws cloudformation update-stack --stack-name $CLUSTER_NAME-workers \
    --use-previous-template \
    --capabilities CAPABILITY_IAM \
    --parameters \
    ParameterKey=ClusterName,ParameterValue=$CLUSTER_NAME \
    ParameterKey=ClusterControlPlaneSecurityGroup,ParameterValue=$SECURITY_GROUP_IDS \
    ParameterKey=NodeGroupName,ParameterValue=$CLUSTER_NAME-workers-node-group \
    ParameterKey=NodeAutoScalingGroupMinSize,ParameterValue=1 \
    ParameterKey=NodeAutoScalingGroupMaxSize,ParameterValue=$NODES \
    ParameterKey=NodeInstanceType,ParameterValue=c4.8xlarge \
    ParameterKey=NodeImageId,ParameterValue=ami-73a6e20b \
    ParameterKey=KeyName,ParameterValue=devenv-key \
    ParameterKey=VpcId,ParameterValue=$VPC_ID \
    ParameterKey=Subnets,ParameterValue=\"$SUBNET_IDS\"

echo "Waiting for EKS worker node group to be updated... (may take a while)"
aws cloudformation wait stack-update-complete --stack-name $CLUSTER_NAME-workers
echo "EKS worker node group updated."

# 2. Tell kubernetes to start up more pods
kubectl scale deployment/scanner-worker --replicas=$(($NODES - 1))

