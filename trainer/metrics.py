import torch
import numpy as np
from config.configurator import configs
import pandas as pd
import os
import pickle
if configs['train']['conf_mat']:
    from torchmetrics.classification import MulticlassConfusionMatrix
class Metric(object):

    def __init__(self):
        self.metrics = configs['test']['metrics']
        self.k = configs['test']['k']
        with open(configs['train']['parameter_label_mapping_path'], 'rb') as f:
            _label_mapping = pickle.load(f)
        self._num_classes = len(list(_label_mapping.keys()))

    def precision_at_k(output, target, k=3):
        _, indices = torch.topk(output, k, dim=1)
        correct = torch.sum(indices == target.view(-1, 1))
        precision = correct.float() / (k * target.size(0))
        return precision.item()
    
    # def metric_call(self, true_labels, predicted_labels):

    #     path_to_metrics = 'results_metrics'
    #     with open(configs['train']['parameter_label_mapping_path'], 'rb') as f:
    #             _label_mapping = pickle.load(f)
    #     _num_classes = len(list(_label_mapping.keys()))
    #     # metrics per class to dataframe
    #     accuracy = Accuracy(task="multiclass", average=None, num_classes=_num_classes).to(configs['device'])
    #     f1 = F1Score(task="multiclass", average=None, num_classes=_num_classes).to(configs['device'])
    #     precision = Precision(task="multiclass", average=None, num_classes=_num_classes).to(configs['device'])
    #     recall = Recall(task="multiclass", average=None, num_classes=_num_classes).to(configs['device'])
    #     conf_matrix = ConfusionMatrix(task="multiclass", num_classes=_num_classes).to(configs['device'])

    #     # acc_list = accuracy(torch.tensor(true_labels), torch.tensor(predicted_labels)).tolist()
    #     # f1_list = f1(torch.tensor(true_labels), torch.tensor(predicted_labels)).tolist()
    #     # precision_list = precision(torch.tensor(true_labels), torch.tensor(predicted_labels)).tolist()
    #     # recall_list = recall(torch.tensor(true_labels), torch.tensor(predicted_labels)).tolist()
    #     # cm = conf_matrix(torch.tensor(true_labels), torch.tensor(predicted_labels)).tolist()

    #     acc = np.mean(accuracy(true_labels, predicted_labels).cpu().numpy())
    #     f1 = np.mean(f1(true_labels, predicted_labels).cpu().numpy())
    #     precision = np.mean(precision(true_labels, predicted_labels).cpu().numpy())
    #     recall = np.mean(recall(true_labels, predicted_labels).cpu().numpy())
    #     # cm = conf_matrix(true_labels, predicted_labels).tolist()

    #     # metrics_data = [acc_list, f1_list, recall_list, precision_list]
    #     results = {'Accuracy': acc, 'F1Score': f1, 'Recall': recall, "Precision": precision}
    #     # metrics_df = pd.DataFrame(metrics_data, columns=_label_mapping.keys(), index=index_names)
    #     # metrics_df.to_csv(os.path.join(path_to_metrics, "class_metrics.csv"))
    #     # cm_df = pd.DataFrame(cm, columns=_label_mapping.keys(), index=_label_mapping.keys())
    #     # cm_df.to_csv(os.path.join(path_to_metrics, "confusion_matrix.csv"))
    #     return results

    def metrics_calc(self, target, output):

        ks=self.k 
        metrics = {'precision': [],
               'recall': [],
               'f1score': [],
               'accuracy': []}

        for k in ks:
            _, indices = torch.topk(output, k)
            correct = torch.sum(indices == target.view(-1, 1))

            # Precision at k
            precision = correct.float() / (k * target.size(0))
            metrics['precision'].append(round(precision.item(), 2))

            # Recall at k
            relevant_items = target.view(-1, 1).expand_as(indices)
            true_positives = torch.sum(indices == relevant_items)
            recall = true_positives.float() / target.size(0)
            metrics['recall'].append(round(recall.item(), 2))

            try:
                # F1 score at k
                f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                metrics['f1score'].append(round(f1_score.item(), 2))
            except AttributeError:
                print(f1_score)
                metrics['f1score'].append(0)

            # Accuracy at k
            accuracy = correct.float() / target.size(0)
            metrics['accuracy'].append(round(accuracy.item(), 2))

        for metric in metrics:
            metrics[metric] = np.array(metrics[metric])
        # print(metrics)
        return metrics
    
    def eval_new(self, model, test_dataloader, test):
        true_labels = torch.empty(0).to(configs['device'])
        pred_labels = torch.empty(0).to(configs['device'])
        pred_scores = torch.empty(0).to(configs['device'])

        for _, tem in enumerate(test_dataloader):
            if not isinstance(tem, list):
                tem = [tem]
            batch_data = list(
                map(lambda x: x.long().to(configs['device']), tem))
            _, _, batch_last_items, _, _, _, _, _  = batch_data
            # predict result
            with torch.no_grad():
                batch_pred = model.full_predict(batch_data)
            predicted_labels = batch_pred.max(1)[1]
            true_labels = torch.cat((true_labels, batch_last_items), dim=0).to(configs['device'])
            pred_labels = torch.cat((pred_labels, predicted_labels), dim=0).to(configs['device'])
            pred_scores = torch.cat((pred_scores, batch_pred), dim=0).to(configs['device'])
        metrics_data = self.metrics_calc(true_labels, pred_scores)
        if test and not configs['train']['model_test_run'] and configs['train']['conf_mat']:
            self.cm = MulticlassConfusionMatrix(num_classes=self._num_classes+1)
            cm = self.cm(pred_scores, true_labels)
            conf_matrix_np = cm.numpy()
            cm_name = configs['test']['save_path']
            np.savetxt(f'results_metrics/cm_{cm_name}.csv', conf_matrix_np, delimiter=',', fmt='%d')

        # Accuracy based on top three
        # _, top_indices = torch.topk(pred_scores, 3)
        # correct_top = 0
        # for i, true_label in enumerate(true_labels):
        #     if true_label.item() in top_indices[i].cpu().numpy():
        #         correct_top += 1
        # accuracy_top_three.append(correct_top / len(pred_labels))
        # metrics_data['AccTopThree'] = np.mean(accuracy_top_three)
        return metrics_data

    def eval(self, model, test_dataloader, test=False):
        # for most GNN models, you can have all embeddings ready at one forward
        # if 'eval_at_one_forward' in configs['test'] and configs['test']['eval_at_one_forward']:
        #     return self.eval_at_one_forward(model, test_dataloader)
        
        metrics_data = self.eval_new(model, test_dataloader, test)
        return metrics_data
        # result = {}
        # for metric in self.metrics:
        #     result[metric] = np.zeros(len(self.k))

        # batch_ratings = []
        # ground_truths = []
        # test_user_count = 0
        # test_user_num = len(test_dataloader.dataset.test_users)
        # for _, tem in enumerate(test_dataloader):
        #     if not isinstance(tem, list):
        #         tem = [tem]
        #     test_user = tem[0].numpy().tolist()
        #     batch_data = list(
        #         map(lambda x: x.long().to(configs['device']), tem))
        #     # predict result
        #     with torch.no_grad():
        #         batch_pred = model.full_predict(batch_data)
        #     test_user_count += batch_pred.shape[0]
        #     # filter out history items
        #     batch_pred = self._mask_history_pos(
        #         batch_pred, test_user, test_dataloader)
        #     _, batch_rate = torch.topk(batch_pred, k=max(self.k))
        #     batch_ratings.append(batch_rate.cpu())
        #     # ground truth
        #     ground_truth = []
        #     for user_idx in test_user:
        #         ground_truth.append(
        #             list(test_dataloader.dataset.user_pos_lists[user_idx]))
        #     ground_truths.append(ground_truth)
        # assert test_user_count == test_user_num

        # # calculate metrics
        # data_pair = zip(batch_ratings, ground_truths)
        # eval_results = []
        # for _data in data_pair:
        #     eval_results.append(self.eval_batch(_data, self.k))
        # for batch_result in eval_results:
        #     for metric in self.metrics:
        #         result[metric] += batch_result[metric] / test_user_num
        # print(metrics_data)
        # print(result)
        # return result
    
    # def recall(self, test_data, r, k):
    #     right_pred = r[:, :k].sum(1)
    #     recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    #     recall = np.sum(right_pred / recall_n)
    #     return recall

    # def precision(self, r, k):
    #     right_pred = r[:, :k].sum(1)
    #     precis_n = k
    #     precision = np.sum(right_pred) / precis_n
    #     return precision

    # def mrr(self, r, k):
    #     pred_data = r[:, :k]
    #     scores = 1. / np.arange(1, k + 1)
    #     pred_data = pred_data * scores
    #     pred_data = pred_data.sum(1)
    #     return np.sum(pred_data)

    # def ndcg(self, test_data, r, k):
    #     assert len(r) == len(test_data)
    #     pred_data = r[:, :k]

    #     test_matrix = np.zeros((len(pred_data), k))
    #     for i, items in enumerate(test_data):
    #         length = k if k <= len(items) else len(items)
    #         test_matrix[i, :length] = 1
    #     max_r = test_matrix
    #     idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
    #     dcg = pred_data * (1. / np.log2(np.arange(2, k + 2)))
    #     dcg = np.sum(dcg, axis=1)
    #     idcg[idcg == 0.] = 1.
    #     ndcg = dcg / idcg
    #     ndcg[np.isnan(ndcg)] = 0.
    #     return np.sum(ndcg)

    # def get_label(self, test_data, pred_data):
    #     r = []
    #     for i in range(len(test_data)):
    #         ground_true = test_data[i]
    #         predict_topk = pred_data[i]
    #         pred = list(map(lambda x: x in ground_true, predict_topk))
    #         pred = np.array(pred).astype("float")
    #         r.append(pred)
    #     return np.array(r).astype('float')

    # def eval_batch(self, data, topks):
    #     sorted_items = data[0].numpy()
    #     ground_true = data[1]
    #     r = self.get_label(ground_true, sorted_items)

    #     result = {}
    #     for metric in self.metrics:
    #         result[metric] = []

    #     for k in topks:
    #         for metric in result:
    #             if metric == 'recall':
    #                 result[metric].append(self.recall(ground_true, r, k))
    #             if metric == 'ndcg':
    #                 result[metric].append(self.ndcg(ground_true, r, k))
    #             if metric == 'precision':
    #                 result[metric].append(self.precision(r, k))
    #             if metric == 'mrr':
    #                 result[metric].append(self.mrr(r, k))

    #     for metric in result:
    #         result[metric] = np.array(result[metric])

    #     return result

    # def _mask_history_pos(self, batch_rate, test_user, test_dataloader):
    #     if not hasattr(test_dataloader.dataset, 'user_history_lists'):
    #         return batch_rate
    #     for i, user_idx in enumerate(test_user):
    #         pos_list = test_dataloader.dataset.user_history_lists[user_idx]
    #         batch_rate[i, pos_list] = -1e8
    #     return batch_rate
    
    # def eval_at_one_forward(self, model, test_dataloader):
    #     result = {}
    #     for metric in self.metrics:
    #         result[metric] = np.zeros(len(self.k))

    #     batch_ratings = []
    #     ground_truths = []
    #     test_user_count = 0
    #     test_user_num = len(test_dataloader.dataset.test_users)

    #     with torch.no_grad():
    #         user_emb, item_emb = model.generate()

    #     for _, tem in enumerate(test_dataloader):
    #         if not isinstance(tem, list):
    #             tem = [tem]
    #         test_user = tem[0].numpy().tolist()
    #         batch_data = list(
    #             map(lambda x: x.long().to(configs['device']), tem))
    #         # predict result
    #         batch_u = batch_data[0]
    #         batch_u_emb, all_i_emb = user_emb[batch_u], item_emb
    #         with torch.no_grad():
    #             batch_pred = model.rating(batch_u_emb, all_i_emb)
    #         test_user_count += batch_pred.shape[0]
    #         # filter out history items
    #         batch_pred = self._mask_history_pos(
    #             batch_pred, test_user, test_dataloader)
    #         _, batch_rate = torch.topk(batch_pred, k=max(self.k))
    #         batch_ratings.append(batch_rate.cpu())
    #         # ground truth
    #         ground_truth = []
    #         for user_idx in test_user:
    #             ground_truth.append(
    #                 list(test_dataloader.dataset.user_pos_lists[user_idx]))
    #         ground_truths.append(ground_truth)
    #     assert test_user_count == test_user_num

    #     # calculate metrics
    #     data_pair = zip(batch_ratings, ground_truths)
    #     eval_results = []
    #     for _data in data_pair:
    #         eval_results.append(self.eval_batch(_data, self.k))
    #     for batch_result in eval_results:
    #         for metric in self.metrics:
    #             result[metric] += batch_result[metric] / test_user_num

    #     return result
