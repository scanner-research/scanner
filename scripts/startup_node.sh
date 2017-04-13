if [ -z "$1" ]
  then
    echo "Usage: startup_node.sh <id_for_node>"
    exit
fi

num_gpus=$2
if [ -z "$2" ]
  then
  	num_gpus=1
    echo "num_gpus not specified. Defaulting to 1"
fi

gcloud compute --project "visualdb-1046" disks create "hackinstance-$1" --size "20" --zone "us-east1-d" --source-snapshot "hacksnapshot" --type "pd-standard"
gcloud beta compute --project "visualdb-1046" instances create "hackinstance-$1" --zone "us-east1-d" --machine-type "n1-standard-4" --network "default" --metadata "ssh-keys=ubuntu:ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQDXJ3JrrWKc0TAM5KBXYmuTVAG06DyA8F1hHbqUULCNp767bDNN1dTF9zTo+ZDWdCuHm49XWrpRK552G8U0A55HvBEjOj4eEUSuAibd0uDAYMZr3dJNTzXNU/KfgnbJYGbRboBk3fu47D4bhKPmjX5ZDsSN++BuUYpf1bH829invPBzlGeBb/QRe3Jk9DMK/swIqFc4j6PWeOItj4/1flXFFruR/bT0p2/MIxTTAMAWlhHRYqhtia1YYMbfdv38eqZMH1GY+n7GQJTuKBDvz0qPxCus86xaE4vCawD+iQJFuD8XxppsHbc1+oCAmi5AtbUeHXjXirN95itMBi7S2evd ubuntu,node_id=$1" --maintenance-policy "TERMINATE" --service-account "50518136478-compute@developer.gserviceaccount.com" --scopes "https://www.googleapis.com/auth/cloud-platform" --accelerator type=nvidia-tesla-k80,count=$num_gpus --tags "http-server","https-server" --disk "name=hackinstance-$1,device-name=hackinstance-$1,mode=rw,boot=yes,auto-delete=yes"

