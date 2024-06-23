import json
import os


def load_config():
    cwd = os.getcwd()
    config_path = os.path.join(cwd, '../config.json')
    with open(config_path, 'r') as file:
        return json.load(file)