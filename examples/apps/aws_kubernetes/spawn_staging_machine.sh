#!/bin/bash

AMI=<AMI_ID>

INSTANCE_ID=$(
    aws ec2 run-instances \
        --image-id $AMI \
        --security-group-ids sg-a6558ed8 \
        --count 1 \
        --instance-type m5.12xlarge \
        --block-device-mappings "[{\"DeviceName\": \"/dev/sda1\",\"Ebs\":{\"VolumeSize\":128}}]" \
        --key-name ec2-key \
        --query 'Instances[0].InstanceId' \
        --output text)

TEMP=$(aws ec2 describe-instances \
           --instance-ids $INSTANCE_ID \
           --query 'Reservations[0].Instances[0].PublicIpAddress' \
           --output text)

echo $TEMP
