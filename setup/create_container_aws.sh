#!/usr/bin/env bash
# example command to run
# bash create_container_aws.sh -g 0 -p 8001
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


docker run --gpus $gpu --rm -it \
        --name "$gpu"_"$name" \
        --env NVIDIA_DISABLE_REQUIRE=1 \
        -p $port:$port \
        -v <path_of_folder>/:/wokspace/<folder_name> \
        # eg: -v /home/sachin/projects/pest-monitoring-new/:/workspace/pest-monitoring-new \
        --ipc host \
        $image bash
