## Setup
#### 1. Clone the Repository
Clone the repository and detectron2 code for reference and we will be mounting it in our docker container too.

```bash
cd ~/
mkdir projects; cd projects;
git clone https://github.com/WadhwaniAI/pest-monitoring-new.git
git clone https://github.com/facebookresearch/detectron2.git detectron2_repo
```

#### 2. Setting up Docker Image (skip if on sif or thor)
If the docker image is not present,
```bash
cd ~/projects/pest-monitoring-new/setup/

docker pull wadhwaniai/pest-monitoring-detectron2:v0
```

OR if you want to build the dockerimage from the Dockerfile
```bash
docker build -t wadhwaniai/pest-monitoring-detectron2:v0 .
```

#### 3. Creating a Container
After this, create container using the `create_container.sh` script,
```bash
cd ~/projects/pest-monitoring-new/setup/

# for a GPU machine (For Wadhwani Users)
bash create_container.sh -g 0 -c 1-10 -n pm-detectron2-container -u shenoy -p 8001

>>> Explanation
-g: GPU number, pass -1 for a non-GPU machine
-c: CPU list, to be assigned
-n: name of the container
-e: path to the folder where data and outputs are to be stored
-u: username (this is the name of folder you created inside outputs/ folder)
-p: port number (this is needed if you want to start jupyter lab on a remote machine)
```

#### 4. Jupyter Notebook
To start a jupyter notebook,
```bash
cd ~/projects/pest-monitoring-new/setup/

# for a GPU machine (For Wadhwani Users)
bash jupyter.sh 8001

>>> Explanation
NOTE: Use the same port as the one used to start the container
```
