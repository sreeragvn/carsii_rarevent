import gc
import os
import time
import copy
import torch
import random
import datetime
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from numpy import random
from copy import deepcopy
import torch.optim as optim
from trainer.metrics import Metric
from config.configurator import configs
from models.bulid_model import build_model
from .utils import DisabledSummaryWriter, log_exceptions
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import LambdaLR

configs['test']['save_path'] = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))

if 'tensorboard' in configs['train'] and configs['train']['tensorboard']:
    timestr = configs['test']['save_path']
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=f'runs/{timestr}')
else:
    writer = DisabledSummaryWriter()


def init_seed():
    if 'reproducible' in configs['train']:
        if configs['train']['reproducible']:
            seed = configs['train']['seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

class Trainer(object):
    def __init__(self, data_handler, logger):
        self.data_handler = data_handler
        self.logger = logger
        self.metric = Metric()

    def create_optimizer(self, model):
        optim_config = configs['optimizer']
        initial_lr = optim_config['lr']
        # final_lr = optim_config['final_lr']
        # total_epochs = configs['train']['epoch']
        # gamma = (final_lr / initial_lr) ** (1 / total_epochs)
        gamma = 0.999

        if optim_config['name'] == 'adam':
            self.optimizer = optim.Adam(model.parameters(
            ), lr=initial_lr, weight_decay=optim_config['weight_decay'])
            # self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=5, factor=0.9, min_lr=1e-6, verbose=True)
            # self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=[30, 60, 90, 120, 150, 180], gamma=0.1)
            self.scheduler = ExponentialLR(self.optimizer, gamma=gamma)


    def train_epoch(self, model, epoch_idx):
        # This method encapsulates the training logic for one epoch of a recommender system. It involves iterating over batches, computing and backpropagating the loss, and logging relevant information. The specifics may vary based on the model and data handling mechanisms used in the recommender system.

        # prepare training data
        # Retrieves the training data loader from the data_handler.
        # Calls the sample_negs method on the training dataset, which might be related to negative sampling in recommendation systems.
        train_dataloader = self.data_handler.train_dataloader
        #todo val loss and train loss are different in model test run where you have both dataset the same. check this.
        # train_dataloader.dataset.sample_negs()
        # Initializes dictionaries for tracking loss values (loss_log_dict) and the cumulative epoch loss (ep_loss).
        # steps calculates the number of steps (batches) in the training dataset.
        # for recording loss
        loss_log_dict = {}
        ep_loss = 0
        steps = len(train_dataloader.dataset) // configs['train']['batch_size']
        # start this epoch
        # Sets the model in training mode.
        model.train()

        for i, tem in tqdm(enumerate(train_dataloader), desc='Training Recommender', total=len(train_dataloader)):
        # for _, tem in enumerate(train_dataloader):
            # Iterates over batches in the training data using the train_dataloader.
            # self.optimizer.zero_grad()
            if not configs['train']['gradient_accumulation']: 
                self.optimizer.zero_grad()
            batch_data = list(map(lambda x: x.long().to(configs['device']), tem))
            loss, loss_dict = model.cal_loss(batch_data)
            ep_loss += loss.item()
            loss.backward()
            # self.optimizer.step()
            # Zeroes the gradients, moves batch data to the device specified in the configuration, computes loss, backpropagates, and performs an optimizer step.
            if configs['train']['gradient_accumulation'] and (i + 1) % configs['train']['accumulation_steps'] == 0:
                # Perform gradient clipping
                # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
            elif not configs['train']['gradient_accumulation']:
                # Perform gradient clipping
                # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                self.optimizer.step()
            # record loss
            # Records loss values in loss_log_dict. The loss values are normalized by the length of the training dataloader.
            for loss_name in loss_dict:
                _loss_val = float(loss_dict[loss_name]) / len(train_dataloader)
                if loss_name not in loss_log_dict:
                    loss_log_dict[loss_name] = _loss_val
                else:
                    loss_log_dict[loss_name] += _loss_val

        test_loader = self.data_handler.test_dataloader
        total_val_loss = 0
        with torch.no_grad():
            for i, val_tem in enumerate(test_loader):
                val_batch_data = list(map(lambda x: x.long().to(configs['device']), val_tem))
                val_loss, _ = model.cal_loss(val_batch_data)
                avg_val_loss = val_loss.item() / len(test_loader)
                total_val_loss += avg_val_loss

        # self.scheduler.step(total_val_loss)
        if configs['train']['gradient_accumulation'] and not (i + 1) <= configs['train']['accumulation_steps']:
           self.scheduler.step()
        elif not configs['train']['gradient_accumulation']:
           self.scheduler.step()
        total_val_loss = round(total_val_loss, 2)
        print('val_loss: ', total_val_loss)
        writer.add_scalar('Loss/train', ep_loss / steps, epoch_idx)
        writer.add_scalar('Loss/val', total_val_loss / steps, epoch_idx)
        # Uses a writer (probably a TensorBoard SummaryWriter) to log the training loss for the epoch.

        # log loss
        if configs['train']['log_loss']:
            self.logger.log_loss(epoch_idx, loss_log_dict)
        else:
            self.logger.log_loss(epoch_idx, loss_log_dict, save_to_log=False)

    @log_exceptions
    def train(self, model):
        # The method orchestrates the training process for a recommender system, including the training loop, evaluation, and potential early stopping based on validation metrics. This method is responsible for training a recommender system model. The training process includes multiple epochs, evaluation steps, and potentially early stopping based on a specified patience criteria.
        total_parameters = model.count_parameters()
        print(f"Total number of parameters in the model: {total_parameters}")
        # Initializes the optimizer for the model.
        self.create_optimizer(model)
        train_config = configs['train']

        if not train_config['early_stop']:
            # Iterates over the specified number of epochs (train_config['epoch']).
            # Calls train_epoch method to train the model for each epoch.
            # Calls evaluate method at specified intervals during training.
            for epoch_idx in range(train_config['epoch']):
                # train
                self.train_epoch(model, epoch_idx)
                # evaluate
                if epoch_idx % train_config['test_step'] == 0:
                    self.evaluate(model, epoch_idx)
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch_idx + 1}, Learning Rate: {current_lr}")
            self.test(model)
            self.save_model(model)
            return model
        # If early stopping is enabled (train_config['early_stop'] is True), the training loop includes logic for early stopping.
        # Keeps track of the patience counter (now_patience) and the best epoch, metric, and model state during the training process.
        # Breaks out of the loop if the patience criteria are met.
        elif train_config['early_stop']:
            now_patience = 0
            best_epoch = 0
            best_metric = -1e9
            best_state_dict = None
            for epoch_idx in range(train_config['epoch']):
                # train
                self.train_epoch(model, epoch_idx)
                # evaluate
                if epoch_idx % train_config['test_step'] == 0:
                    eval_result = self.evaluate(model, epoch_idx)

                    if eval_result[configs['test']['metrics'][0]][0] > best_metric:
                        now_patience = 0
                        best_epoch = epoch_idx
                        best_metric = eval_result[configs['test']['metrics'][0]][0]
                        best_state_dict = deepcopy(model.state_dict())
                        self.logger.log("Validation score increased.  Copying the best model ...")
                    else:
                        now_patience += 1
                        self.logger.log(f"Early stop counter: {now_patience} out of {configs['train']['patience']}")

                    # early stop
                    if now_patience == configs['train']['patience']:
                        break

            # re-initialize the model and load the best parameter
            # Re-initializes the model and loads the parameters of the best model based on the early stopping criteria.
            # Evaluates and tests the model on the best epoch.
            # Saves the best model.
            self.logger.log("Best Epoch {}".format(best_epoch))
            model = build_model(self.data_handler).to(configs['device'])
            model.load_state_dict(best_state_dict)
            self.evaluate(model)
            model = build_model(self.data_handler).to(configs['device'])
            model.load_state_dict(best_state_dict)
            self.test(model)
            self.save_model(model)
            return model

    @log_exceptions
    def evaluate(self, model, epoch_idx=None):
        # This method encapsulates the logic for evaluating the recommender system model on either the validation set or the test set. It computes evaluation metrics, logs the results, and can be used during the training process to monitor the model's performance on held-out data.

        # The provided code defines an evaluate method within a class. This method is used to evaluate the performance of a recommender system model on either a validation set or a test set, depending on the availability of the corresponding dataloaders.

        # Sets the model to evaluation mode.
        model.eval()
        if hasattr(self.data_handler, 'valid_dataloader'):
            eval_result = self.metric.eval(model, self.data_handler.valid_dataloader)
            writer.add_scalar('HR/test', eval_result[configs['test']['metrics'][0]][0], epoch_idx)
            self.logger.log_eval(eval_result, configs['test']['k'], data_type='Validation set', epoch_idx=epoch_idx)
        elif hasattr(self.data_handler, 'test_dataloader'):
            eval_result = self.metric.eval(model, self.data_handler.test_dataloader)
            writer.add_scalar('HR/test', eval_result[configs['test']['metrics'][0]][0], epoch_idx)
            self.logger.log_eval(eval_result, configs['test']['k'], data_type='Test set', epoch_idx=epoch_idx)
        else:
            raise NotImplemented
        # Checks if a validation dataloader (valid_dataloader) is available in the data_handler.
        # If available, evaluates the model on the validation set using the metric.eval method.
        # If not, checks if a test dataloader (test_dataloader) is available.
        # If available, evaluates the model on the test set using the metric.eval method.
        # Logs evaluation results and scalar values to TensorBoard (writer).
        return eval_result

    @log_exceptions
    def test(self, model):
        # This method serves the purpose of assessing the performance of the recommender system model on the test set. It evaluates the model using the specified metrics, logs the evaluation results, and can be used to understand how well the model generalizes to unseen data. The test method is typically called after training the model to assess its final performance.

        # The provided code defines a test method within a class. This method is responsible for testing the performance of a recommender system model on a test set

        # Sets the model to evaluation mode.
        model.eval()

        # Checks if a test dataloader (test_dataloader) is available in the data_handler.
        # If available, evaluates the model on the test set using the metric.eval method.
        # Logs evaluation results using the logger.
        if hasattr(self.data_handler, 'test_dataloader'):
            eval_result = self.metric.eval(model, self.data_handler.test_dataloader, test=True)
            self.logger.log_eval(eval_result, configs['test']['k'], data_type='Test set')
        else:
            raise NotImplemented
        return eval_result

    def save_model(self, model):
        # The provided code defines a save_model method within a class. This method is responsible for saving the parameters of a trained recommender system model to a file. 
        if configs['train']['save_model']:
            model_state_dict = model.state_dict()
            model_name = configs['model']['name']
            data_name = configs['data']['name']
            timestr = configs['test']['save_path']
            if not configs['tune']['enable']:
                save_dir_path = './checkpoint/{}'.format(model_name)
                if not os.path.exists(save_dir_path):
                    os.makedirs(save_dir_path)
                # timestamp = int(time.time())
                torch.save(
                    model_state_dict, '{}/{}-{}-{}.pth'.format(save_dir_path, model_name, data_name, timestr))
                self.logger.log("Save model parameters to {}".format(
                    '{}/{}-{}.pth'.format(save_dir_path, model_name, timestr)))
            else:
                save_dir_path = './checkpoint/{}/tune'.format(model_name)
                if not os.path.exists(save_dir_path):
                    os.makedirs(save_dir_path)
                now_para_str = configs['tune']['now_para_str']
                torch.save(
                    model_state_dict, '{}/{}-{}-{}.pth'.format(save_dir_path, model_name, data_name, now_para_str))
                self.logger.log("Save model parameters to {}".format(
                    '{}/{}-{}.pth'.format(save_dir_path, model_name, now_para_str)))

    def load_model(self, model):
        if 'pretrain_path' in configs['train']:
            pretrain_path = configs['train']['pretrain_path']
            model.load_state_dict(torch.load(pretrain_path))
            self.logger.log(
                "Load model parameters from {}".format(pretrain_path))
            return model
        else:
            raise KeyError("No pretrain_path in configs['train']")
