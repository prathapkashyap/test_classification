from pathlib import Path
import logging
import argparse

# LOGGER_NAME = "logs"

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default = "bert")
    # print(parser.parse_args()model)
    return parser.parse_args()


def get_target_index_map(target_names):
    target_map = {}
    for i, name in enumerate(target_names):
        target_names[name] = i
    return target_map
# def init_log(file_name = f"{LOGGER})