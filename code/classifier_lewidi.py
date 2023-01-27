from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, r2_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
import re
from scipy.special import expit, logit

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModel
from arabert.preprocess import ArabertPreprocessor
from transformers import BertTokenizer, BertModel, BertConfig, get_scheduler
from scipy.special import softmax
from tqdm import tqdm, trange

import pandas as pd
import numpy as np
import math

from random import *
from collections import Counter, defaultdict
from rich.progress import track, Progress, BarColumn, MofNCompleteColumn, TimeElapsedColumn, TimeRemainingColumn

training_progress_bar = Progress("{task.description}", BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(), TimeRemainingColumn())


class ClassifierBert(nn.Module):
    def __init__(self, device, tasks=["abuse"], labels=2, flag = 0):
        super(ClassifierBert, self).__init__()

        if flag == 1:
            self.bert = AutoModel.from_pretrained("aubmindlab/bert-base-arabertv2",
                                              return_dict= True)
        else:
            self.bert = BertModel.from_pretrained("bert-base-uncased",
                                              return_dict=True)

        self.tasks = tasks
        self.labels = labels
        self.linear_layer = dict()
        self.sigmoid_layer = nn.Sigmoid()
        for task in tasks:
            self.linear_layer[task] = nn.Linear(BertConfig().hidden_size, labels).to(device)

    def forward(self, ids, attn):
        outputs = self.bert(
            ids,
            attn
        )
        self.task_logits = dict()
        hidden = outputs.last_hidden_state[:, 0, :]
        for task in self.tasks:
            self.task_logits[task] = self.linear_layer[task](hidden)
        return self.extract_outputs()

    def extract_outputs(self):
    

        if self.labels > 2:
            logits = {str(label): self.task_logits["abuse"][:, label] for label in range(self.labels)}
            sig_logits = {str(label): torch.sigmoid(logits[str(label)]) for label in range(self.labels)}
            predictions = {str(label): [1 if x > 0.5 else 0 for x in sig_logits[str(label)]] for label in
                           range(self.labels)}
        elif self.labels == 1:
            logits = self.task_logits
            predictions = {task: [x.item() for x in self.sigmoid_layer(logits[task])] for task in self.tasks}
        else:
            logits = self.task_logits
            predictions = {task: [x.item() for x in torch.argmax(logits[task], dim=-1)] for task in self.tasks}
        return logits, predictions




class AbuseClassifier():
    def __init__(self, data_train, data_dev, data_test, annotators, params, task_labels=["abuse"]):
        
        #if eng: torch.device("cuda:0") if torch.cuda.is_available() else 
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        self.data_train = data_train
        self.data_dev = data_dev
        self.data_test = data_test
        self.annotators = annotators
        
        if params.ar_dat == 1:
            self.tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv2")
            model_name = "aubmindlab/bert-base-arabertv2"
            arabert_prep = ArabertPreprocessor(model_name=model_name)
            self.data_train["text"] =  self.data_train["text"].map(lambda a: arabert_prep.preprocess(a))
            self.data_dev["text"] =  self.data_dev["text"].map(lambda a: arabert_prep.preprocess(a))
            self.data_test["text"] =  self.data_test["text"].map(lambda a: arabert_prep.preprocess(a))
        else:
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            
        self.multi_label, self.multi_task, self.ensemble, self.single, self.log_reg = False, False, False, False, False
        setattr(self, params.task, True)
        if self.single or self.log_reg:
            self.task_labels = task_labels
        else:
            self.task_labels = annotators

        self.majority_vote()
        # self.uncertainty()
        print("Train data shape after majority voting", self.data_train.shape)
        print("Train data shape after majority voting", self.data_dev.shape)
        
        # Setting the parameters
        self.params = params
        print([(k, v) for k, v in self.params.__dict__.items()])

    def majority_vote(self):
        self.data_train["abuse"] = self.data_train['hard_label']
        self.data_dev["abuse"] = self.data_dev['hard_label']

    # def uncertainty(self):
    #     self.data_train["uncertainty"] = (self.data_train[self.annotators].sum(axis=1) \
    #                                 * (self.data_train[self.annotators].count(axis=1) - self.data_train[self.annotators].sum(
    #                 axis=1)) \
    #                                 / (self.data_train[self.annotators].count(axis=1) * self.data_train[self.annotators].count(
    #                 axis=1)))
    #     self.data_dev["uncertainty"] = (self.data_dev[self.annotators].sum(axis=1) \
    #                                 * (self.data_dev[self.annotators].count(axis=1) - self.data_dev[self.annotators].sum(
    #                 axis=1)) \
    #                                 / (self.data_dev[self.annotators].count(axis=1) * self.data_dev[self.annotators].count(
    #                 axis=1)))

    def CV(self):
        if self.ensemble:
            ensemble_results = pd.DataFrame()
            for annotator in self.annotators:
                print("Training model for annotator", annotator)
                self.task_labels = "abuse"
                scores, results = self._CV(self.data.rename(columns={annotator: "abuse", "abuse": "_abuse"}))
                ensemble_results[annotator + "_pred"] = results["abuse_pred"]
                ensemble_results[annotator + "_label"] = results["abuse_label"]
                ensemble_results[annotator + "_masked_pred"] = results["abuse_masked_pred"]
                ensemble_results[annotator + "_masked_label"] = results["abuse_masked_label"]
            self.task_labels = self.annotators
            scores = self.report_results(ensemble_results)
            return scores, ensemble_results
        else:
            return self._CV(self.data_train, self.data_dev, self.data_test)

    def masks(self, df):
        df = df.replace(0, 1)
        df = df.replace(np.nan, 0)
        new_labels = LabelEncoder().fit_transform([''.join(str(l) for l in row) for i, row in df.iterrows()])
        return new_labels

    def _CV(self, train, dev, test):
        
        results = pd.DataFrame()
        """
        if i == 1:
            test.to_csv(os.path.join(self.params.source_dir, "results", "GHC", "test_file.csv"), index=False)
        else:
            test.to_csv(os.path.join(self.params.source_dir, "results", "GHC", "test_file.csv"), index=False, header=False, mode="a") #     results = results.
        """
        train_batches = self.get_batches(train)
        dev_batches = self.get_batches(dev)
        test_batches = self.get_batches(test)

        self.train_model(train_batches, dev_batches)
        if self.params.predict == "label":
            
            if len(self.task_labels) > 1:
                results_cluttered = self.predict(test_batches)
                ann_pred_list = []
                for x in range(len(self.annotators)):
                    ann_pred_list.append(self.annotators[x] + "_pred")

                results = results_cluttered[ann_pred_list]   
                
            else:
                results_cluttered = self.predict(test_batches)
                results = pd.DataFrame()
                results["hard_label"] = results_cluttered["abuse_pred"]
                results["soft_label_0"] = results_cluttered["abuse_logit"].str[0]
                results["soft_label_1"] = results_cluttered["abuse_logit"].str[1]
                
                
            # testing on the validation set
            # if 

            print("Test:")
            
        return results

    def new_model(self):
        if self.multi_task:
            return ClassifierBert(self.device, tasks=self.annotators, flag = self.params.ar_dat)
        elif self.multi_label:
            return ClassifierBert(self.device, labels=len(self.annotators), flag = self.params.ar_dat)
        elif self.log_reg:
            return ClassifierBert(self.device, labels=1, tasks=self.task_labels, flag = self.params.ar_dat)
        else:
            return ClassifierBert(self.device, flag = self.params.ar_dat)

    def create_loss_functions(self):
        losses = dict()
        # self.class_weight = dict()

        for task_label in self.task_labels:
            # flag 
            _labels = [int(x) for x in self.data_train[task_label].dropna().tolist()]
            weight = compute_class_weight(class_weight = 'balanced',
                                          classes = np.unique(_labels),
                                          y = _labels)
            if len(weight) == 1:
                weight = [0.01, 1]
            weight = torch.tensor(weight, dtype=torch.float32).to(self.device)
            if self.multi_label:
                losses[task_label] = nn.BCEWithLogitsLoss(reduction="sum")  # , pos_weight=class_weight)
            elif self.log_reg:
                losses[task_label] = nn.MSELoss()
            else:
                losses[task_label] = nn.CrossEntropyLoss(weight=weight)
                #Todo NOT CROSS ENTROPY

        return losses

    def train_model(self, batches_train, batches_dev):
        self.model = self.new_model()
        self.model = self.model.to(self.device)
        
        self.optimizer = AdamW(self.model.parameters(), lr=self.params.learning_rate)
        
        train_batches = batches_train
        val_batches = batches_dev

        
        # num_training_steps = self.params.num_epochs * len(train_batches)
        
        # lr_scheduler = get_scheduler(name="linear", optimizer=self.optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

        self.loss = self.create_loss_functions()
        with training_progress_bar: 
            task_id = training_progress_bar.add_task("Training", total=self.params.num_epochs)
            batch_task_id = training_progress_bar.add_task("Training batch", total=len(train_batches))
            
            training_progress_bar.console.print("Starting with the training now...")
            
            for epoch in range(self.params.num_epochs):
                
                self.model.train()
                
                loss_val = 0
                for batch in train_batches:
                    X_ids = torch.tensor(batch["inputs"]).to(self.device)
                    X_att = torch.tensor(batch["attentions"]).to(self.device)
                    if len([x for task_label in self.task_labels for x in batch["masks"][task_label]]) == 0:
                        continue

                    logits, _ = self.model(X_ids, attn=X_att)
                    class_loss = dict()
                    weighted_sum = 0
                    for task_label in self.task_labels:

                        masked_logits = logits[task_label][batch["masks"][task_label]]
                        masked_labels = [batch["labels"][task_label][x] for x in batch["masks"][task_label]]
                        # print(masked_labels, batch["masks"][task_label])
                        if self.multi_task or self.ensemble:
                            masked_labels = torch.tensor(masked_labels).type("torch.LongTensor").to(self.device)
                        else:
                            masked_labels = torch.tensor(masked_labels).to(self.device)

                        if len(batch["masks"][task_label]) > 0:
                            ## list of loss values for each batch instance
                            class_loss[task_label] = self.loss[task_label](masked_logits, masked_labels)

                            ## using a column of the data as the weight for loss value of each instance
                            # Batch["weight"] shows the instance weight (based on its certainty), class_weight shows the class weight for positive and negative labels
                            # batch["weights"][batch_i] *
                            """
                            class_loss[task_label] = sum([ batch_loss[mask_i] * self.class_weight[task_label][masked_labels[mask_i]]
                                                        for mask_i, batch_i in enumerate(batch["masks"][task_label])])
                            weighted_sum += sum([self.class_weight[task_label][label] for label in masked_labels])
                            """
                    total_loss = sum(class_loss.values())  # / weighted_sum
                    loss_val += total_loss.item()
                    total_loss.backward()
                    self.optimizer.step()
                    
                    training_progress_bar.advance(batch_task_id)

                print("Epoch", epoch, "-", "Loss", round(loss_val, 3))
                if val_batches:
                    val_results = self.predict(val_batches, self.model)
                    print("Validation")
                    val_score = self.report_results(val_results)
                    print(val_score)
                
                
                training_progress_bar.advance(task_id)
                training_progress_bar.reset(batch_task_id)
                

                
                
    def predict(self, batches, model= None): 
        self.model.eval()
        
        
        results = defaultdict(list)
        for batch in batches:

            X_ids = torch.tensor(batch["inputs"]).to(self.device)
            X_att = torch.tensor(batch["attentions"]).to(self.device)

            logits, predictions = self.model(X_ids, attn=X_att)
            for task_label in self.task_labels:

                # ground truth labels per annotator
                ground_truth = batch['labels'][task_label]
                
                preds = predictions[task_label]
                masked_labels= []
                masked_predictions = []
                
                for idx, item in enumerate(ground_truth):
                    if np.isnan(item):
                        masked_labels.append(np.nan)
                        masked_predictions.append(np.nan)
                    else:    
                        masked_labels.append(ground_truth[idx])
                        masked_predictions.append(preds[idx])
                
                results[task_label + "_masked_pred"].extend(masked_predictions)
                results[task_label + "_masked_label"].extend(masked_labels)
                results[task_label + "_pred"].extend(predictions[task_label])
                results[task_label + "_label"].extend(batch["labels"][task_label])

                
                if self.params.task == "single":
                    results[task_label + "_logit"].extend(
                        softmax(logits[task_label].cpu().detach().numpy(), axis=1)) # [:, 1]
                    results[task_label +"_soft_label_0"]= (batch["soft_label_0"])
                    results[task_label +"_soft_label_1"]= (batch["soft_label_1"])          

            
                
                
        return pd.DataFrame.from_dict(results)    


    def mc_predict(self, batches, model=None):
        results = defaultdict(list)
        soft = nn.Softmax(dim=1)
        num_samples = sum([batch["batch_len"] for batch in batches])
        dropout_predictions = np.empty((0, num_samples, 1))

        for task_label in self.task_labels:
            for mc_pass in range(self.params.mc_passes):
                self.model.eval()
                self.enable_dropout(self.model)
                mc_predictions = np.empty((0, 1))

                for batch in batches:
                    X_ids = torch.tensor(batch["inputs"]).to(self.device)
                    X_att = torch.tensor(batch["attentions"]).to(self.device)
                    logits, predictions = self.model(X_ids, attn=X_att)

                    predictions = np.array(predictions[task_label])
                    mc_predictions = np.vstack((mc_predictions, predictions[:, np.newaxis]))

                dropout_predictions = np.vstack((dropout_predictions,
                                                 mc_predictions[np.newaxis, :]))
            results[task_label + "_mean"] = list(np.squeeze(np.mean(dropout_predictions, axis=0)))
            results[task_label + "_variance"] = list(np.squeeze(np.var(dropout_predictions, axis=0)))

        return pd.DataFrame.from_dict(results)

    def enable_dropout(self, model):
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()





    def report_results(self, results):
                
        def cross_entropy(targets_soft, predictions_soft, epsilon = 1e-12):                                
            predictions = np.clip(predictions_soft, epsilon, 1. - epsilon)                                      
            N = predictions.shape[0]
            ce = -np.sum(targets_soft*np.log(predictions+1e-9))/N
            return ce
        
                    
        def extract_hard_soft(abuse_masked_label, abuse_masked_pred):
            # hard_eval, 
            soft_labels_targets, soft_labels_pred  = [], []
            for ind in abuse_masked_pred.index:
            
                # hard_eval.append(abuse_masked_label["hard_label"][ind])
                soft_labels_targets.append([abuse_masked_label["soft_label_0"][ind],abuse_masked_label["soft_label_1"][ind]])
                soft_labels_pred.append([abuse_masked_pred["soft_label_0"][ind],abuse_masked_pred["soft_label_1"][ind]])

            # hard_eval = np.asarray(hard_eval, dtype=np.int8)
            # hard_eval = np.expand_dims(hard_eval, axis=1)
            soft_labels_targets = np.asarray(soft_labels_targets)
            soft_labels_pred = np.asarray(soft_labels_pred)
            return soft_labels_targets, soft_labels_pred
        
    
        def filter_na(df):
            if "annotations" not in df.columns:
                pd.options.mode.chained_assignment = None  # default='warn'
                df["annotations"] = np.nan
                df["annotations"] = df["annotations"].astype(object)
                df["hard_label"] = np.nan
                df["soft_label_0"] = np.nan
                df["soft_label_1"] = np.nan

                for ind, row in df.iterrows():

                    count_zero, count_one, totalcount = 0,0,0
                    annotations = []
                    for element in row:
                        if not np.isnan(element):
                            annotations.append(int(element))
                            if element == 0:
                                count_zero+= 1 
                            else:
                                count_one +=1
                            totalcount +=1 
                    df.at[ind,"annotations"] = annotations
                    soft_zero, soft_one = count_zero / float(totalcount) , count_one / float(totalcount)

                    df["soft_label_0"][ind] = soft_zero
                    df["soft_label_1"][ind] = soft_one

                    if soft_zero > soft_one:
                        df["hard_label"][ind] = 0
                    elif soft_zero < soft_one:
                        df["hard_label"][ind] = 1
                    else:
                        df["hard_label"][ind] = randrange(0, 2)

            return df

        # check for 1 in all columns
        if self.log_reg:
            label_col = self.task_labels[0] + "_label"
            pred_col = self.task_labels[0] + "_pred"
            r2 = r2_score(results[label_col], results[pred_col])
            scores = {"r2": round(r2, 4)}
            return scores
        
        if len(self.task_labels) > 1:
            
            label_cols = [col + "_label" for col in self.annotators]
            pred_cols = [col + "_pred" for col in self.annotators]

            masked_label_cols = [col + "_masked_label" for col in self.annotators]
            masked_pred_cols = [col + "_masked_pred" for col in self.annotators]
            
            abuse_masked_label = filter_na(results[masked_label_cols])
            abuse_masked_pred = filter_na(results[masked_pred_cols])
            
            f1 = f1_score(abuse_masked_label["hard_label"], abuse_masked_pred["hard_label"], average = 'micro')

            soft_label_masked, soft_pred_masked = extract_hard_soft(abuse_masked_label, abuse_masked_pred)
            
            cr_ent = cross_entropy(soft_label_masked, soft_pred_masked)
            
            abuse_label = filter_na(results[label_cols])
            abuse_pred = filter_na(results[pred_cols])
            
        else:

            abuse_label = results["abuse_label"] 
            abuse_pred = results["abuse_pred"] 
            print("Accuracy of aggregated label")

        
        
        
        if len(self.task_labels) == 1:
            soft_label, soft_pred = [], []
            for ind in results.index:
               soft_label.append([results["abuse_soft_label_0"][ind],results["abuse_soft_label_1"][ind]])
               soft_pred.append([results["abuse_logit"][ind][0],results["abuse_logit"][ind][1]])
            
            soft_label = np.asarray(soft_label)
            soft_pred = np.asarray(soft_pred)
            f1_unmasked = f1_score(results["abuse_label"], results["abuse_pred"], average = 'micro')
            f1 = 0
            cr_ent = 0
            
        else:
            soft_label, soft_pred = extract_hard_soft(abuse_label, abuse_pred)
            f1_unmasked = f1_score(abuse_label["hard_label"], abuse_pred["hard_label"], average = 'micro')
            
        cr_ent_unmasked = cross_entropy(soft_label, soft_pred)
        
        # "F1 masked": round(f1, 4), "Cross entropy masked": round(cr_ent, 4),
        
        scores = {"F1 masked": round(f1, 4),
                  "Cross entropy masked": round(cr_ent, 4),
                  "F1 unmasked": round(f1_unmasked, 4),
                  "Cross entropy unmasked": round(cr_ent_unmasked, 4)}

        return scores
        
    
    

    def get_batches(self, data):
        if isinstance(self.params.sort_by, str):
            data = data.sort_values(by=[self.params.sort_by], ascending=False).reset_index()
        batches = list()

        for s in range(0, len(data), self.params.batch_size):
            e = s + self.params.batch_size if s + self.params.batch_size < len(data) else len(data)
            data_info = self.batch_to_info(data["text"].tolist()[s: e])
            anno_batch = dict()
            mask_batch = dict()
            for task_label in self.task_labels:
                anno_batch[task_label] = data[task_label].tolist()[s: e]
                mask_batch[task_label] = [i for i, h in enumerate(anno_batch[task_label]) \
                                          if not math.isnan(h)]
            data_info["labels"] = anno_batch
            data_info["masks"] = mask_batch
            data_info["soft_label_0"] = data["soft_label_0"]
            data_info["soft_label_1"] = data["soft_label_1"]
            #data_info["majority_vote"] = data["abuse"].tolist()[s: e]
            data_info["batch_len"] = e - s
            
            if isinstance(self.params.batch_weight, str):
                data_info["weights"] = data[self.params.batch_weight].tolist()[s: e]
            else:
                data_info["weights"] = [1 for i in range(e - s)]

            batches.append(data_info)
        return batches

    def batch_to_info(self, batch):
        batch_info = dict()
        if isinstance(self.params.max_len, int):
            tokens = self.tokenizer(batch,
                                    padding="max_length",
                                    max_length=self.params.max_len,
                                    truncation=True)
        else:
            tokens = self.tokenizer(batch,
                                    padding=True,
                                    truncation=True)
        batch_info["inputs"] = tokens["input_ids"]
        batch_info["attentions"] = tokens["attention_mask"]
        return batch_info