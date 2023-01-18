import os
import numpy as np
import random
import torch

class params_lewidi():
    def __init__(self):
        self.batch_size = 4
        self.learning_rate =  1e-9
        self.max_len = 128
        self.num_epochs = 10
        self.random_state = 9999
        self.num_folds = 2
        self.task = "single"
        self.batch_weight = None
        self.sort_by = None
        self.stratified = True
        self.predict = "label"
        self.mc_passes = 10
        self.set_seed()
        
    def update(self, new):
        for k, v in new.__dict__.items():
            if getattr(new, k) is not None:
                print("Changing the default value of {} from {} to {}".format(k, getattr(self, k), v))
                setattr(self, k, v)

    def update_dict(self, new):
        for k, v in new.items():
              print("Changing the default value of {} from {} to {}".format(k, getattr(self, k), v))
              setattr(self, k, v)
              
    def set_seed(self) -> None: 
        initial_seed_value: int = 12345

        os.environ['PYTHONHASHSEED']=str(initial_seed_value)
        random.seed(initial_seed_value)
        np.random.seed(initial_seed_value)
        torch.manual_seed(initial_seed_value)\
    