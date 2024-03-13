import os
import logging
import datetime

from config.configurator import configs

def get_local_time():
    return datetime.datetime.now().strftime('%b-%d-%Y_%H-%M-%S')

class Logger(object):
    # Initialization Method:
    # The constructor method initializes the logger.
    # Retrieves the model name from configuration settings and creates a directory for logs based on the model name.
    # Configures a logger named 'train_logger' with the INFO level.
    # Determines the log file path based on whether tuning is enabled or not.
    # Adds a file handler to the logger with a formatter for log messages.
    # Logs configuration settings if specified.

    # This Logger class provides a structured way to handle logging in a machine learning project, allowing for configuration settings, saving to log files, and printing to the console. It includes methods for general logging, logging loss information, and logging evaluation results.
    def __init__(self, log_configs=True):
        # Retrieves the model name from the configuration settings and creates a directory path for logs based on the model name.
        model_name = configs['model']['name']
        log_dir_path = './log/{}'.format(model_name)
        if not os.path.exists(log_dir_path):
            os.makedirs(log_dir_path)
        # Creates a logger named 'train_logger' and sets its logging level to INFO.
        self.logger = logging.getLogger('train_logger')
        self.logger.setLevel(logging.INFO)
        # Determines the log file path based on whether tuning is enabled or not. Creates a file handler for logging.
        dataset_name = configs['data']['name']
        if not configs['tune']['enable']:
            log_file = logging.FileHandler('{}/{}_{}.log'.format(log_dir_path, dataset_name, get_local_time()), 'a', encoding='utf-8')
        else:
            log_file = logging.FileHandler('{}/{}-tune_{}.log'.format(log_dir_path, dataset_name, get_local_time()), 'a', encoding='utf-8')
        # Configures a formatter for log messages, including the timestamp.
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        log_file.setFormatter(formatter)
        # Adds the file handler to the logger.
        self.logger.addHandler(log_file)
        # If log_configs is True, it logs the configuration settings using the log method.
        if log_configs:
            self.log(configs)

    def log(self, message, save_to_log=True, print_to_console=True):
        # Logs a message. Here's an explanation:
        # If save_to_log is True, logs the message using the INFO level.
        if save_to_log:
            self.logger.info(message)
        # If print_to_console is True, prints the message to the console.
        if print_to_console:
            print(message)

    def log_loss(self, epoch_idx, loss_log_dict, save_to_log=True, print_to_console=True):
        # Logs loss information. Here's an explanation:
        epoch = configs['train']['epoch']
        # Construct Message:
        message = '[Epoch {:3d} / {:3d}] '.format(epoch_idx, epoch)
        for loss_name in loss_log_dict:
            message += '{}: {:.4f} '.format(loss_name, loss_log_dict[loss_name])
        # Save to Log File and Print to Console:
        if save_to_log:
            self.logger.info(message)
        if print_to_console:
            print(message)

    def log_eval(self, eval_result, k, data_type, save_to_log=True, print_to_console=True, epoch_idx=None):
        # Logs evaluation results. Here's an explanation:
        # Construct Message:
        if epoch_idx is not None:
            message = 'Epoch {:3d} '.format(epoch_idx)
        else:
            message = ''

        for metric in eval_result:
            message += '{} ['.format(data_type)
            for i in range(len(k)):
                message += '{}@{}: {:.4f} '.format(metric, k[i], eval_result[metric][i])
            message += '] '
        # Save to Log File and Print to Console:
        if save_to_log:
            self.logger.info(message)
        if print_to_console:
            print(message)