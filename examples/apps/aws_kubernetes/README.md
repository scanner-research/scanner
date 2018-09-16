Scanner + Kubernetes on Amazon EKS
==================================

This document will guide you through setting up a
[kubernetes](https://kubernetes.io/docs/concepts/overview/what-is-kubernetes/)
cluster on AWS that is ready to process scanner jobs.

Pre-requisites:
---------------

1. Install Scanner via [installation](http://scanner.run/installation.html)

2. Install [Docker](https://docs.docker.com/install/)

3. Create a [Google Cloud Platform account](https://cloud.google.com/) and then
   create a Google Cloud project.

4. Install the [gcloud SDK](https://cloud.google.com/sdk/downloads) and then
   install the kubernetes command-line management tool `kubectl` by running:
```
gcloud components install kubectl
```

5. Install the `jq` tool for parsing JSON (used in this example):

Ubuntu 16.04
```
apt-get install jq
```

macOS
```
brew install jq
```

6. Create a bucket on AWS S3. Put the name into `storage.bucket` in `scanner-config.yaml.template`.

There are three components required to get started running jobs on AWS:

1. Get setup to connect to AWS machines

2. Optional: create a staging machine on AWS for setting up the cluster and running jobs.

3. Create a kubernetes cluster.

4. Run a job on the cluster.

## Connecting to AWS machines

To connect to the AWS cluster, you need to acquire authentication keys. This will 
take the form of an `access key id` and a `secret key`.

Now, install the AWS command line interface and configure it using your
`access key id` and `secret key`, making sure to set your default region to
`us-west-2`:
```
pip3 install awscli
aws configure
```

Finally, we will generate a key pair that can be used to sign into our EC2
machines:
```
aws ec2 create-key-pair --key-name ec2-key --query 'KeyMaterial' --output text > ec2-key.pem
chmod 600 ec2-key.pem
```
This commands saves a private key to the file `ec2-key.pem`. Now you're ready to
connect to an EC2 machine.

## Optional: Creating a staging machine

Our first act will be to create a "staging" machine on AWS. The purpose of this
machine is to serve as the "staging" ground for managing our cluster and
executing long-running jobs. This is preferable to using your own local machine
because the bandwidth from this machine to AWS services (such as S3) will be
much higher and the connectivity will be more stable.

### Building from scratch

If there is no existing AMI, you can build the image yourself:

1. Find the current AMI id for Ubuntu 16.04 by going to
   https://cloud-images.ubuntu.com/locator/ec2/ in your browser, typing
   `us-west-2 hvm xenial ebs` into the search box, and then copying the `AMI-ID`
   field.
2. Open the `spawn_staging_machine.sh` script and change the `AMI=...` variable
   to the Ubuntu AMI ID you copied from step 1.
3. Run `bash ./spawn_staging_machine.sh` again to spawn a staging machine. This
   will take a few moments. Once it's complete, it will output the public IP of
   the machine which you can use to access it.
4. Connect to the remote machine by running:
   ```
   ssh -i path/to/your/ec2-key.pem ubuntu@<ip-address>
   ```
5. Setup your AWS keys in the environment by running (replacing the <...>):
   ```
   echo "export AWS_ACCESS_KEY_ID=<your-access-key>" >> ~/.bashrc
   echo "export AWS_SECRET_ACCESS_KEY=<your-secret-key>" >> ~/.bashrc
   ```
6. Run `exec $SHELL` to reload your bash configuration.
7. Run the following script to install the dependencies required for the staging
   machine:
   ```
   bash ./build_staging_machine.sh
   ```
   This process will query you for your `access key id` and `secret key`, since
   it setups the AWS cli.

### Using a pre-built AMI

Once you've built the image from scratch, you can create an AMI out of it and resuse
it. You can then create a new instance by replacing <AMI_ID> in `spawn_staging_machine.sh` 
with the AMID ID. Then run:
```
bash ./spawn_staging_machine.sh
```

If the command succeeds, it will return an IP address which you can use to login
to the machine using the following command:
```
ssh -i ec2-key.pem ubuntu@<ip-address>
```

## Create a kubernetes cluster

To create a kubernetes cluster, ssh into the staging machine and simply run:
```
cd capture
bash ./create_eks_cluster.sh <cluster-name> <num-workers>
exec $SHELL # To update environment variables
```
where `<cluster-name` is the name you will use to identify the cluster, and
`<num-workers>` is the number of worker nodes to create.

NOTE: The default worker machines are c4.8xlarge instances. If you'd like to try
out different configurations, you need to modify:

1. `create_eks_cluster.sh`: Change the `NodeInstanceType` value from c4.8xlarge
   to a machine of your choice.
2. `worker.yml.template`: Change the `cpu` value from `35.0` to the number of cores
   on your machine type less 1 (kubernetes spawns some services on the machine that
   ask for ~ 1 core).

## Update the cluster with new code

Before running a job on the cluster, you must build and deploy the code.

```
cd capture
bash ./build_and_deploy.sh
```

## Run a job on the cluster


Copy `scanner_cli_script.sh.template` and modify it with the correct parameters 
(they are the same as the DerpCLI command). Then, simply run the script.

## Scaling the cluster up or down

```
cd capture
bash ./scale_eks_workers.sh <cluster-name> <num-workers>
```

## Deleting the cluster

```
cd capture
bash ./delete_eks_cluster.sh <cluster-name>
```
