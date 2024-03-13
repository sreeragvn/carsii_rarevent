import torch
import numpy as np
from config.configurator import configs
import pandas as pd
import os
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
class Metric(object):

    def __init__(self):
        self.metrics = configs['test']['metrics']
        self.k = configs['test']['k']
        with open(configs['train']['parameter_label_mapping_path'], 'rb') as f:
            _label_mapping = pickle.load(f)
        self._num_classes = len(list(_label_mapping.keys()))

    def eval_new(self, model, test_dataloader, test):
        true_labels = torch.empty(0).to(configs['device'])
        pred_scores = torch.empty(0).to(configs['device'])

        metrics = {'precision': [],
               'recall': [],
               'f1score': [],
               'accuracy': []}

        model.eval()
        y_true = []
        y_pred_probs = []

        for _, tem in enumerate(test_dataloader):
            if not isinstance(tem, list):
                tem = [tem]
            batch_data = list(
                map(lambda x: x.long().to(configs['device']), tem))
            _, _, _, _, _, _, _, common_item = batch_data
            # predict result
            with torch.no_grad():
                outputs = model.full_predict(batch_data)
                preds = torch.sigmoid(outputs)

                true_labels = torch.cat((true_labels, common_item), dim=0).to(configs['device'])
                pred_scores = torch.cat((pred_scores, preds), dim=0).to(configs['device'])
                
                # y_true.extend(common_item.cpu().numpy())
                # y_pred_probs.extend(preds.cpu().numpy())
        y_true = true_labels.cpu().tolist()
        y_pred_probs = pred_scores.cpu().squeeze().tolist() 
        y_pred = (torch.tensor(y_pred_probs) > 0.5).float().cpu().numpy()
        f1 = f1_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)

        metrics["precision"] = [precision]
        metrics["recall"] = [recall]
        metrics["f1score"] = [f1]
        metrics["accuracy"] = [accuracy]
        for metric in metrics:
            metrics[metric] = np.array(metrics[metric])
        print(metrics)
        return metrics

    def eval(self, model, test_dataloader, test=False):
        # for most GNN models, you can have all embeddings ready at one forward
        # if 'eval_at_one_forward' in configs['test'] and configs['test']['eval_at_one_forward']:
        #     return self.eval_at_one_forward(model, test_dataloader)
        
        metrics_data = self.eval_new(model, test_dataloader, test)
        return metrics_data
