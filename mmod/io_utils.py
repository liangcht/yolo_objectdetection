import os
try:
    from yaml import danger_load as yaml_load
except ImportError:
    from yaml import load as yaml_load


def load_from_yaml_file(file_name):
    with open(file_name, 'r') as fp:
        return yaml_load(fp)


def read_to_buffer(file_name):
    with open(file_name, 'r') as fp:
        all_line = fp.read()
    return all_line


def ensure_directory(path):
    if path == '' or path == '.':
        return
    if path and len(path) > 0:
        if not os.path.exists(path):
            os.makedirs(path)


def write_to_file(contxt, file_name):
    p = os.path.dirname(file_name)
    ensure_directory(p)
    with open(file_name, 'w') as fp:
        fp.write(contxt)
