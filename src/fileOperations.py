import os
import pickle

def checkIfDirecotryExists(directory):
    return os.path.isdir(directory)

def checkIfFileExists(file):
    return os.path.isfile(file)

def savingPickelFile(file, data):
    try:
        with open(file, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        print ('Error saving file %s' % e)
        exit()