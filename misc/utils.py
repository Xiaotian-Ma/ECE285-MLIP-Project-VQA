import json
import torch.nn as nn


def getopt(opt, key, default_value):
    if default_value is None and (opt is None or opt[key] is None):
        print('ERROR: required key ' + key + ' was not provided in opt')
    if opt is None:
        return default_value
    v = opt[key]
    if v is None:
        v = default_value
    return v


def read_json(path):
    data = None
    with open(path, 'r') as data_file:
        data = json.load(data_file)
    return data


def write_json(path, data):
    with open(path, 'w') as outfile:
        json.dump(data, outfile)


def count_key(data):
    return len(data)

