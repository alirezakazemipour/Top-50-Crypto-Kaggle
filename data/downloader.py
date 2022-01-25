"""
A general module designed for automated dataset-downloading form Kaggle.
"""
import argparse
import os
import zipfile
import shutil


def download_dataset(args):
    print("***If you live in Iran, turn on your VPN!***")
    os.environ["KAGGLE_CONFIG_DIR"] = os.path.dirname(os.path.realpath(__file__))
    if not os.path.exists("kaggle.json"):
        os.environ["KAGGLE_USERNAME"] = args.kaggle_username
        os.environ["KAGGLE_KEY"] = args.kaggle_key
    os.system("kaggle datasets download -d " + args.dataset_name)


def prepare_dataset(dataset_name):
    if os.getcwd().split(os.sep)[-1] == "data":
        path = "../datasets/" + dataset_name
    else:
        path = os.path.join(os.getcwd(), "datasets", dataset_name)
    os.makedirs(path, exist_ok=True)
    with zipfile.ZipFile(dataset_name + ".zip", mode='r') as f:
        f.extractall(path)
    os.remove(dataset_name + ".zip")

    for subpath in os.listdir(path):
        folder = os.path.join(path, subpath)
        if os.path.isdir(folder):
            for dir in os.listdir(folder):
                file = os.path.join(path, subpath, dir)
                shutil.move(file, path)
            os.removedirs(folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enter your kaggle's API token (if not provided in a kaggle.json in this directory) and dataset's "
                    "id to be downloaded from kaggle.")
    parser.add_argument("--kaggle_username", type=str, default="", help="Your Kaggle's API token username")
    parser.add_argument("--kaggle_key", type=str, default="", help="Your Kaggle's API token key")
    parser.add_argument("--dataset_name", type=str, default="", required=True,
                        help="Dataset's url in 'Username/Dataset name' format")
    args = parser.parse_args()
    dataset_name = args.dataset_name.split("/")[-1].strip('\u202c')
    if not os.path.exists(dataset_name + ".zip"):
        download_dataset(args)
    else:
        print("Dataset already exists")

    prepare_dataset(dataset_name)
