from pathlib import Path
import csv
import os

from .dataloaders.dataloader import FoldDataLoader

track_file_name = False

HEADER_NAMES = ["i_fold", "j_fold", "type", "epoch", "loss", "sensitivity", "specificity", "sp_value"]

def make_track_file() -> int:
    file_path = Path().cwd() / "log.csv"

    option = None
    invalid_option = True

    if file_path.exists():
        while invalid_option:
            print("\033[93mWarn: 'log.csv' from previous runs already exists, overwrite (y/n)? \033[0m", end="")
            option = input()
            invalid_option = False if option in ["y", "n", "Y", "N"] else True

    elif option in ["y", "Y"] or option == None:
        with open("log.csv", "w", newline='') as track_file:
            writer = csv.DictWriter(track_file, fieldnames = HEADER_NAMES)
            writer.writeheader()

            global track_file_name
            track_file_name = True
        return 1

    else:
        return 0



def write_track_file(i_fold : int, j_fold: int, data_type : str, epoch : int, loss : float, sensitivity : float, specificity : float, sp_value : float) -> int:
    file_path = Path().cwd() / "log.csv"

    option = None
    invalid_option = True

    info_to_write = {"i_fold" : i_fold,
                     "j_fold" : j_fold,
                     "type" : data_type,
                     "epoch" : epoch,
                     "loss" : loss,
                     "sensitivity" : sensitivity,
                     "specificity" : specificity,
                     "sp_value" : sp_value}


    while True:
        if file_path.exists() and track_file_name:
            with open(file_path, "a") as track_file:
                writer = csv.DictWriter(track_file, fieldnames=HEADER_NAMES)

                writer.writerow(info_to_write)

            return 1       

        elif not file_path.exists():
            print("\033[92mStatus: creating 'log.csv'...'\033[0m")
            make_track_file()

        elif file_path.exists() and not track_file_name:
            while invalid_option:
                print("\033[93mWarn: 'log.csv' from previous runs already exists, overwrite (y/n)? \033[0m", end="")
                option = input()
                invalid_option = False if option in ["y", "n", "Y", "N"] else True

            if option in ["y", "Y"]:
                os.remove("log.csv")
            else:
                return 0