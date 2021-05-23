#!/usr/bin/env bash
# example command to run
# bash create_container_aws_wandb.sh -g 0 -p 8001
# -g: GPU number, for a non-GPU machine, pass -1
# -p: port number (this is needed if you want to start jupyter lab on a remote machine)

# group ID for `cotton` group

# get inputs
while getopts "g:p:" OPTION; do
	case $OPTION in
		g) gpu=$OPTARG;;
		p) port=$OPTARG;;
		*) exit 1 ;;
	esac
done

image=shenoynikhil/pest-monitoring-detectron2:v0
name=pm-detectron2-container-aws

if [[ -z $port ]] ; then
	port=8888
fi

if [[ -z $WANDB_API_KEY ]] ; then
echo "ERROR: set the environment variable WANDB_API_KEY"
	exit 0
fi

NV_GPU=$gpu nvidia-docker run --rm -it \
	--name "$gpu"_"$name" \
	-p $port:$port \
	-v /home/ubuntu/projects/pest-monitoring-new/:/workspace/pest-monitoring-new \
	-v /home/ubuntu/projects/detectron2_repo/:/workspace/detectron2_repo \
	-p $port:$port \
	--ipc host \
	$image bash

