#!/usr/bin/env bash
# example command to run
# bash create_container.sh -g 0 -c 1-10 -n pm-detectron2-container -u shenoy -p 8001
# -g: GPU number, for a non-GPU machine, pass -1
# -n: name of the container
# -u: username (this is the name of folder you created inside outputs/ folder)
# -p: port number (this is needed if you want to start jupyter lab on a remote machine)

# group ID for `cotton` group
GID=3002

# get inputs
while getopts "g:n:u:p:c:" OPTION; do
	case $OPTION in
		g) gpu=$OPTARG;;
		n) name=$OPTARG;;
		u) user=$OPTARG;;
		p) port=$OPTARG;;
		c) cpulist=$OPTARG;;
		*) exit 1 ;;
	esac
done

image=wadhwaniai/pest-monitoring-detectron2:v0

name=pm-detectron2-container
# workdir="/workspace/"

if [[ -z $WANDB_API_KEY ]] ; then
echo "ERROR: set the environment variable WANDB_API_KEY"
	exit 0
fi

NV_GPU=$gpu taskset --cpu-list $cpulist nvidia-docker run --rm -it \
	--shm-size 16G \
	--name "$gpu"_"$name" \
	-v /scratchh:/scratchh \
	-v /home/users/"$user"/projects/pest-monitoring-new/:/workspace/pest-monitoring-new \
	-v /home/users/"$user"/projects/detectron2_repo/:/workspace/detectron2_repo \
	-v /home/users/:/users/ \
	-v /scratchh/home/cotton-common/data:/data \
	-v /scratchh/home/"$user"/:/output \
	-v ~/.gitconfig:/etc/gitconfig \
	-w /workspace/pest-monitoring-new \
	-p $port:$port \
	--ipc host \
	--env WANDB_DOCKER=$image \
	--env WANDB_API_KEY=$WANDB_API_KEY \
	$image $exp
