"""
Downloads data version (split) files.
"""
from os import makedirs
from os.path import basename, exists, isdir, join
from subprocess import call

from termcolor import colored

DATA_DIR = "/data/coco/raw"
makedirs(DATA_DIR, exist_ok=True)

URLs = [
    "http://images.cocodataset.org/zips/val2014.zip",
    "http://images.cocodataset.org/zips/train2014.zip",
    "http://images.cocodataset.org/zips/test2014.zip",
    "http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
    "http://images.cocodataset.org/annotations/image_info_test2014.zip",
]

for url in URLs:
    # download and unzip files at relevant location
    fname = basename(url)
    print(colored(f"=> Downloading {fname}", "yellow"))

    zip_fpath = join(DATA_DIR, fname)
    unzip_folder = zip_fpath.split(".zip")[0]

    # download and unzip
    if not exists(zip_fpath):
        call(f"wget {url} -O {zip_fpath} -q --show-progress", shell=True)

    if not isdir(unzip_folder):
        call(f"unzip {zip_fpath} -d {DATA_DIR}", shell=True)
