import pickle
import json
import os
import lzma

def pkDump(variable, filePath):
    f = open(filePath, 'wb')
    pickle.dump(variable, f)

def pkLoad(filePath):
    f = open(filePath, 'rb')
    variable = pickle.load(f)
    return variable

def pkDump_lzma(variable, filePath):
    with lzma.open(filePath, "wb") as f:
        pickle.dump(variable, f)

def pkLoad_lzma(filePath):
    with lzma.open(filePath, "rb") as f:
        variable = pickle.load(f)
    return variable

def jsonDump(dictionary, filePath):
    f = open(filePath, 'w')
    json.dump(dictionary, fp=f, indent=4)
    f.close()

def jsonLoad(filePath):
    f = open(filePath, 'r')
    dictionary = json.load(f)
    f.close()
    return dictionary

def mkdir(dirPath):
    if not os.path.exists(dirPath):
        os.mkdir(dirPath)
