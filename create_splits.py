import argparse
import os
import random
import math
import shutil

import pandas as pd

from utils import get_module_logger


def split(source, destination):
    """
    Create three splits from the processed records. The files should be moved to new folders in the
    same directory. This folder should be named train, val and test.

    args:
        - source [str]: source data directory, contains the processed tf records
        - destination [str]: destination data directory, contains 3 sub folders: train / val / test
    """

    # read the statistics that we created during EDA
    df = pd.read_pickle("data_statistics.pkl")

    # generate lists with day and night samples
    files = df.loc[:, "filename"]
    night = df.loc[:, "night"]
    day_files = list(files[~night])
    night_files = list(files[night])

    logger.info(f"Found {len(files)} data files.")

    # shuffle the file lists
    random.shuffle(night_files)
    random.shuffle(day_files)

    # calculate number of files for each split and for day and night
    validation_size = 0.15
    test_size = 0.10

    n_day_val_files = math.ceil(len(day_files) * validation_size)
    n_day_test_files = math.ceil(len(day_files) * test_size)
    n_night_val_files = math.ceil(len(night_files) * validation_size)
    n_night_test_files = math.ceil(len(night_files) * test_size)

    train_files = []
    validation_files = []
    test_files = []

    # pick the smaller validation and test splits first
    for _ in range(n_day_val_files):
        validation_files.append(day_files.pop())
    for _ in range(n_day_test_files):
        test_files.append(day_files.pop())

    for _ in range(n_night_val_files):
        validation_files.append(night_files.pop())
    for _ in range(n_night_test_files):
        test_files.append(night_files.pop())

    # take remainder for training
    for elem in day_files:
        train_files.append(elem)

    for elem in night_files:
        train_files.append(elem)

    logger.info(
        f"Took {len(train_files)} files for training dataset ({100*len(train_files)/len(files):.2f}%)"
    )
    logger.info(
        f"Took {len(validation_files)} files for validation dataset ({100*len(validation_files)/len(files):.2f}%)"
    )
    logger.info(
        f"Took {len(test_files)} files for testing dataset ({100*len(test_files)/len(files):.2f}%)"
    )

    # create folders
    folders = ["train", "val", "test"]
    for folder in folders:
        try:
            os.makedirs(destination + "/" + folder)
        except FileExistsError:
            pass

    # move the files in their target folders
    logger.info("Moving files to target folders")
    try:
        for file in test_files:
            shutil.move(source + "/" + file, destination + "/test" + file)
        for file in validation_files:
            shutil.move(source + "/" + file, destination + "/val")
        for file in train_files:
            shutil.move(source + "/" + file, destination + "/train")
    except FileNotFoundError:
        logger.error("File not found. Check input path")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split data into training / validation / testing"
    )
    parser.add_argument(
        "--source",
        required=False,
        help="source data directory",
        default="data/processed",
    )
    parser.add_argument(
        "--destination",
        required=False,
        help="destination data directory",
        default="data/splitted",
    )
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info("Creating splits...")
    split(args.source, args.destination)
