import os

def list_files(path):
    if not path[-1] == '/': path += '/'
    files = os.listdir(path)
    return [path + file for file in files]

def load_languages():
    paths = list_files('testing_data/wiki_languages')
    lib = {}
    for path in paths:
        language = path.split('/')[-1].split('.')[0]
        file = open(path, 'r')
        lib[language] = file.read()
    return lib
