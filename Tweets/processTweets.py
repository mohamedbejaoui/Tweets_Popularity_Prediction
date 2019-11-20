# !/usr/bin/python3

import os, sys
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("dir", help="directory of tweets")
args = parser.parse_args()

def processOneFile(filePath):
    df = pd.read_json(filePath, lines=True)
    for index, row in df.iterrows():
        rt = row.retweeted_status
        if(rt and (isinstance(rt, dict)) and ('text' in rt)):
            print('RETWEET ', rt['text'])

def processFiles(dirPath):
    for file in listdir(dirPath):
        path = join(dirPath, file)
        if isfile(path):
            processOneFile(path)

processFiles(args.dir)

                     
