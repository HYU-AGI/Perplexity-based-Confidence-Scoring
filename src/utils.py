# This code is based on the original repository by Junyi Ye (MIT License, 2024)
# Source: https://github.com/JunyiYe/CreativeMath


import json
import logging
import os


def load_json(data_path):
    logger = logging.getLogger(__name__)
    with open(data_path, "r", encoding="UTF-8") as file:
        data = json.load(file)
    logger.info(f"JSON data loaded from {os.path.abspath(data_path)}")
    return data


def save_json(data, data_path):
    logger = logging.getLogger(__name__)
    with open(data_path, "w", encoding="UTF-8") as file:
        json.dump(data, file, indent=4)
    logger.info(f"Results saved to {os.path.abspath(data_path)}")


def extract_yes_no(response):
    if "YES" in response:
        return "YES"
    else:
        return "NO"
    
    
def load_api_keys(args=None):
    file_path = "API_KEYS.json"
    with open(file_path, "r") as file:
        return json.load(file)
    