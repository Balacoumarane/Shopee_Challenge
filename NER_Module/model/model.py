from typing import Tuple
import os
import time
import pandas as pd
import torch
from torch import optim
from tqdm import tqdm
from transformers import BertConfig, BertTokenizer

from .backbone import BertForWordClassification, forward_word_classification
from ..utils import ner_metrics_fn, get_lr, metrics_to_string, set_seed, get_logger

logger = get_logger(__name__)


def load_pretrained_model(num_labels: int, is_cuda: bool = False) -> Tuple[BertForWordClassification, BertTokenizer]:
    """

    Args:
        num_labels (int):
        is_cuda(Bool):
    Returns:

    """
    logger.info('Loading pretrained indobert model.....')
    tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
    config = BertConfig.from_pretrained('indobenchmark/indobert-base-p1')
    config.num_labels = num_labels

    # Instantiate model
    model = BertForWordClassification.from_pretrained('indobenchmark/indobert-base-p1', config=config)
    if is_cuda:
        model = model.cuda()
    logger.info('Initialised model with indo-bert!!!!')
    return model, tokenizer


class BertNerModel(object):

    def __init__(self, num_classes: int, i2w: dict = None, exp_id: str = None, random_state: int = None,
                 prediction_only: bool = False, device: str = None):
        """

        Args:
            num_classes:
            i2w:
            exp_id:
            random_state:
            device:
            prediction_only:
        """
        self.num_classes = num_classes
        if not prediction_only:
            self.model, _ = load_pretrained_model(num_labels=num_classes, is_cuda=True)
        else:
            self.model, self.tokenizer = load_pretrained_model(num_labels=num_classes, is_cuda=True)
        if device is None:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        else:
            assert device in ['cpu', 'cuda', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']  # Assuming 4 GPUs at maximum
            self.device = torch.device(device)
        logger.info('Device: {}'.format(self.device))
        self.random_state = random_state
        self.i2w = i2w
        self.exp_id = exp_id

    def train(self, train_loader, val_loader, model_dir: str = None, num_epochs: int = 10,
              evaluate_every: int = 2, early_stop: int = 5, valid_criterion: str = None, **kwargs):
        """

        Args:
            train_loader:
            val_loader:
            model_dir:
            num_epochs:
            evaluate_every:
            early_stop:
            valid_criterion:
            **kwargs:

        Returns:

        """
        # TODO: Implement other optimizer
        logger.info('Starting Training.....')
        logger.info('Device is set at {}'.format(self.device))
        # if self.device =='cuda':
        #     logger.info('Setting model to device cuda')
        logger.info('The model device is {}'.format(self.model.device))
        optimizer = optim.Adam(self.model.parameters(), lr=2e-5)
        best_val_metric = -100
        count_stop = 0

        for epoch in range(num_epochs):
            set_seed(self.random_state)
            self.model.train()
            torch.set_grad_enabled(True)

            total_train_loss = 0
            list_hyp, list_label = [], []

            train_pbar = tqdm(train_loader, leave=True, total=len(train_loader))
            for i, batch_data in enumerate(train_pbar):
                # Forward model
                # TODO: change device to cpu if needed
                loss, batch_hyp, batch_label = forward_word_classification(self.model, batch_data[:-1], i2w=self.i2w,
                                                                           device='cuda')

                # Update model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                tr_loss = loss.item()
                total_train_loss = total_train_loss + tr_loss

                # Calculate metrics
                list_hyp += batch_hyp
                list_label += batch_label

                train_pbar.set_description("(Epoch {}) TRAIN LOSS:{:.4f} LR:{:.8f}".format((epoch + 1),
                                                                                           total_train_loss / (i + 1),
                                                                                           get_lr(optimizer)))

            # Calculate train metric
            metrics = ner_metrics_fn(list_hyp, list_label)
            logger.info("(Epoch {}) TRAIN LOSS:{:.4f} {} LR:{:.8f}".format((epoch + 1),
                                                                           total_train_loss / (i + 1),
                                                                           metrics_to_string(metrics),
                                                                           get_lr(optimizer)))

            # save model
            if model_dir is not None:
                logger.info("Saving model....")
                self.save(path=model_dir)
            else:
                logger.info('No path to save the model')
            # Evaluate on validation
            # evaluate
            if ((epoch + 1) % evaluate_every) == 0:
                val_loss, val_metrics = self.evaluate(model=self.model, data_loader=val_loader, is_test=False)
                logger.info('(Epoch {}) VAL_METRIC :{:.4f}'.format((epoch + 1), val_metrics[valid_criterion]))
                # Early stopping
                val_metric = val_metrics[valid_criterion]
                if best_val_metric < val_metric:
                    best_val_metric = val_metric
                    count_stop = 0
                else:
                    logger.info('The best val is :{} and val_metric is {}'.format(best_val_metric, val_metric))
                    count_stop += 1
                    logger.info("count stop: {}".format(count_stop))
                    if count_stop == early_stop:
                        break

    def evaluate(self, model, data_loader, is_test=False):
        """

        Args:
            model:
            data_loader:
            is_test:

        Returns:

        """
        model.eval()
        torch.set_grad_enabled(False)
        total_loss, total_correct, total_labels = 0, 0, 0

        list_hyp, list_label, list_seq = [], [], []

        pbar = tqdm(iter(data_loader), leave=True, total=len(data_loader))
        for i, batch_data in enumerate(pbar):
            batch_seq = batch_data[-1]
            # TODO: change device to cpu if needed
            loss, batch_hyp, batch_label = forward_word_classification(model, batch_data[:-1], i2w=self.i2w,
                                                                       device='cuda')

            # Calculate total loss
            test_loss = loss.item()
            total_loss = total_loss + test_loss

            # Calculate evaluation metrics
            list_hyp += batch_hyp
            list_label += batch_label
            list_seq += batch_seq
            metrics = ner_metrics_fn(list_hyp, list_label)

            if not is_test:
                pbar.set_description(
                    "VALID LOSS:{:.4f} {}".format(total_loss / (i + 1), metrics_to_string(metrics)))
            else:
                pbar.set_description("TEST LOSS:{:.4f} {}".format(total_loss / (i + 1), metrics_to_string(metrics)))

        if is_test:
            return total_loss, metrics, list_hyp, list_label, list_seq
        else:
            return total_loss, metrics

    def predict(self, test_loader, save_path: str = None, filename: str = None) -> pd.DataFrame:
        """

        Returns:

        """
        # Evaluate on test
        logger.info('Prediction mode....')
        logger.info('Total sentence to predict: {}'.format(len(test_loader)))
        start = time.time()
        set_seed(self.random_state)
        self.model.eval()
        torch.set_grad_enabled(False)

        total_loss, total_correct, total_labels = 0, 0, 0
        list_hyp, list_label = [], []

        pbar = tqdm(test_loader, leave=True, total=len(test_loader))
        for i, batch_data in enumerate(pbar):
            # TODO: change device to cpu if needed
            _, batch_hyp, _ = forward_word_classification(self.model, batch_data[:-1], i2w=self.i2w, device='cuda')
            list_hyp += batch_hyp

        # Save prediction
        df = pd.DataFrame({'label': list_hyp}).reset_index()
        if save_path is not None:
            filename = filename + '.csv'
            path = os.path.join(save_path, filename)
            df.to_csv(path, index=False)
            logger.info('Results saved in {}'.format(path))
        logger.info('Prediction completed!!!')
        end = time.time()
        logger.info('Total prediction time is:{}'.format(end - start))
        return df

    def save(self, path: str = None):
        """

        Args:
            path:

        Returns:

        """
        if self.exp_id is not None:
            logger.info('Saving model with exp_id')
            torch.save(self.model.state_dict(), path + "/best_model_" + str(self.exp_id) + ".th")
        else:
            logger.info('Saving model')
            torch.save(self.model.state_dict(), path + "/best_model.th")

    def load_modellocal(self, path: str = None, load_tokenizer: bool = False):
        """

        Args:
            path:
            load_tokenizer:

        Returns:

        """
        if self.device == torch.device('cuda'):
            logger.info('Loading model from local path in cuda device')
            state = torch.load(str(path))
        else:
            logger.info('Loading model from local path in cpu device')
            state = torch.load(str(path), map_location='cpu')
        self.model.load_state_dict(state)
        if load_tokenizer:
            logger.info('Loading Tokenizer from indo-Bert')
            return self, self.tokenizer
        else:
            logger.info('Tokenizer not selected')
            return self
