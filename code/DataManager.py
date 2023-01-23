from collections import defaultdict
import pandas as pd
import re
import json
import itertools
import random
import os
import numpy as np

class DataManager():
    
    def __init__(self, filepathConv, filepathArMis, filepathBrex, filepathMD):

        self.dataset_groups= defaultdict(list)
        self.dataset_groups['conv_train'] = (self.open_file(filepathConv + 'ConvAbuse_train.json', 1))
        self.dataset_groups['conv_dev'] = (self.open_file(filepathConv + 'ConvAbuse_dev.json', 1))
        self.dataset_groups['conv_test'] = (self.open_file(filepathConv + 'ConvAbuse_test.json', 1))
        
        self.dataset_groups['ar_train'] = (self.open_file(filepathArMis + 'ArMIS_train.json', 0))
        self.dataset_groups['ar_dev'] = (self.open_file(filepathArMis + 'ArMIS_dev.json', 0))
        self.dataset_groups['ar_test'] = (self.open_file(filepathArMis + 'ArMIS_test.json', 0))
        
        #HS-Brexit
        self.dataset_groups['br_train'] = (self.open_file(filepathBrex + 'HS-Brexit_train.json', 0))
        self.dataset_groups['br_dev'] = (self.open_file(filepathBrex + 'HS-Brexit_dev.json', 0))
        self.dataset_groups['br_test'] = (self.open_file(filepathBrex + 'HS-Brexit_test.json', 0))
        
        #MD
        self.dataset_groups['md_train'] = (self.open_file(filepathMD + 'MD-Agreement_train.json', 0))
        self.dataset_groups['md_dev'] = (self.open_file(filepathMD + 'MD-Agreement_dev.json', 0))
        self.dataset_groups['md_test'] = (self.open_file(filepathMD + 'MD-Agreement_test.json', 0))

        for key in self.dataset_groups:
            if key == 'conv_dev':
                self.dataset_groups[key] = self.fill_the_dev(self.dataset_groups[key], 1)
            elif key in ['ar_dev','br_dev','md_dev']:
                self.dataset_groups[key] = self.fill_the_dev(self.dataset_groups[key], 0)
            else:
                continue
    
        for key in self.dataset_groups:
            print("I'm in with the", key, "dataset")
                    
            self.dataset_groups[key]["hard_label"] = pd.to_numeric(self.dataset_groups[key]["hard_label"], downcast="integer")
                
            if key not in ['md_train', 'md_dev', 'md_test']:
                
                conv_flag, test_flag = 0, 0
                
                if key in ['conv_train','conv_dev']:
                    conv_flag = 1
                elif key in  ['conv_test', 'ar_test', 'br_test']:
                    test_flag = 1

                self.dataset_groups[key] = self.dataset_groups[key].join(pd.DataFrame(
                                    self.annotation_annotator_split(self.dataset_groups[key], conv_flag, test_flag), 
                                    index=self.dataset_groups[key].index))
                
            if key == 'md_test':
                self.dataset_groups[key]['abuse'] = np.nan
                
                
        
    def open_file(self, files, flag):
        

        # Opening JSON file
        f = open(files)

        # returns JSON object as 
        # a dictionary

        data = json.load(f)
        # Iterating through the json

        d = defaultdict(list)

        for i,j in data.items():
            #Saving agent utterances
            x  = j['text']
            start = x.rfind("user") + 8
            if flag == 1:
                start = x.rfind("user") + 8
                d['text'].append(x[start:-2])
            else:
                d['text'].append(x[start:-2])
            d['annotations'].append(j['annotations'])
            d['annotators'].append(j["annotators"])
            d['soft_label_0'].append(j["soft_label"]['0'])
            d['soft_label_1'].append(j["soft_label"]['1'])
            d['hard_label'].append(j["hard_label"])

        # Closing file
        f.close()
        return_pd = pd.DataFrame.from_dict(d)
        return return_pd

    def fill_the_dev(self, panda, flag):

        pd.options.mode.chained_assignment = None  # default='warn'

        for ind in panda.index:

            annotations = [int(d) for d in re.findall(r'-?\d+', panda['annotations'][ind])]
            count_zero, count_one, totalcount = 0,0,0
            
            if flag == 1:
                for element in annotations:
                    if element >= 0:
                        count_zero+= 1 
                    else:
                        count_one +=1
                    totalcount +=1 
            else:         
                for element in annotations:
                    if element == 0:
                        count_zero+= 1 
                    else:
                        count_one +=1
                    totalcount +=1             

            soft_zero, soft_one = count_zero / float(totalcount) , count_one / float(totalcount)

            panda["soft_label_0"][ind] = soft_zero
            panda["soft_label_1"][ind] = soft_one

            if soft_zero > soft_one:
                panda["hard_label"][ind] = 0
            elif soft_zero < soft_one:
                panda["hard_label"][ind] = 1
            else:
                panda["hard_label"][ind] = random.randrange(0, 2)
                
        return panda
    
    def annotation_annotator_split(self, panda, conv_flag, test_flag):
        Ann_dict = defaultdict(list)
        annotators = list(sorted(set(itertools.chain.from_iterable(panda['annotators'].str.findall("\w+")))))


        for ind in panda.index:
            if test_flag != 1:
                annotations = [int(d) for d in re.findall(r'-?\d+', panda['annotations'][ind])]

            Ann_index=0
        
            for person in annotators:
                if person in panda["annotators"].str.findall("\w+")[ind] and test_flag !=1:
                    Ann_dict[person].append(annotations[Ann_index]) 
                    Ann_index=+1
                else: 
                    Ann_dict[person].append("None")
                    
        final = pd.DataFrame(Ann_dict)
        
        if conv_flag == 1:
            vals_to_replace = {-3:1, -2:1, -2:1, -1:1, 0:0, 1:0, 'None': np.nan}
        else:
            vals_to_replace = {1:1, 0:0, 'None': np.nan}    
        for k in final.keys():
            final[k] = final[k].map(vals_to_replace)

        return final
        
