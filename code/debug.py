import os
import argparse
import importlib
from datetime import datetime
import pytz
import re

from classifier_lewidi import *
from params_lewidi import *
from DataManager import *

def main():
        
    filepath1 = '../data/ConvAbuse_dataset/'
    filepath2 = '../data/ArMIS_dataset/'
    filepath3 = '../data/HS-Brexit_dataset/'
    filepath4 = '../data/MD-Agreement_dataset/'

    data = DataManager(filepath1, filepath2, filepath3, filepath4)
    def_params = params_lewidi()

    def_params.task = "multi_task"

    key_train = "conv_train"
    key_test = "conv_dev"

    annotators = list(sorted(set(itertools.chain.from_iterable(data.dataset_groups[key_train]['annotators'].str.findall("\w+")))))
        
    model = ToxicityClassifier(data.dataset_groups[key_train], data.dataset_groups[key_test], annotators=annotators, params=def_params)

    score, results = model.CV()



if __name__ == "__main__":
    main()    