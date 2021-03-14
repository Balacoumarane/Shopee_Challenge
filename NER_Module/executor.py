import random
import numpy as np
import pandas as pd
import torch
from torch import optim
from tqdm import tqdm

from transformers import BertConfig, BertTokenizer
from nltk.tokenize import word_tokenize

from .model import BertForWordClassification, forward_word_classification
from .utils import ner_metrics_fn, get_lr, metrics_to_string, count_param
from .data import NerGritDataset, NerDataLoader


def execute(train_dataset_path, valid_dataset_path, test_dataset_path, predict=False, n_epochs=10):
    """

    Args:
        train_dataset_path:
        valid_dataset_path:
        test_dataset_path:
        predict:
        n_epochs:

    Returns:

    """
    # Load Tokenizer and Config
    tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
    config = BertConfig.from_pretrained('indobenchmark/indobert-base-p1')
    config.num_labels = NerGritDataset.NUM_LABELS

    # Instantiate model
    model = BertForWordClassification.from_pretrained('indobenchmark/indobert-base-p1', config=config)

    # Load dataset
    if not predict:
        train_dataset = NerGritDataset(train_dataset_path, tokenizer, lowercase=True)
        valid_dataset = NerGritDataset(valid_dataset_path, tokenizer, lowercase=True)
        test_dataset = NerGritDataset(test_dataset_path, tokenizer, lowercase=True)

        train_loader = NerDataLoader(dataset=train_dataset, max_seq_len=512, batch_size=16, num_workers=16,
                                     shuffle=True)
        valid_loader = NerDataLoader(dataset=valid_dataset, max_seq_len=512, batch_size=16, num_workers=16,
                                     shuffle=False)
        test_loader = NerDataLoader(dataset=test_dataset, max_seq_len=512, batch_size=16, num_workers=16,
                                    shuffle=False)
    else:
        test_dataset = NerGritDataset(test_dataset_path, tokenizer, lowercase=True)
        test_loader = NerDataLoader(dataset=test_dataset, max_seq_len=512, batch_size=16, num_workers=16,
                                    shuffle=False)

    w2i, i2w = NerGritDataset.LABEL2INDEX, NerGritDataset.INDEX2LABEL

    optimizer = optim.Adam(model.parameters(), lr=2e-5)
    model = model.cuda()

    # Train
    if not predict:
        n_epochs = n_epochs
        for epoch in range(n_epochs):
            model.train()
            torch.set_grad_enabled(True)

            total_train_loss = 0
            list_hyp, list_label = [], []

            train_pbar = tqdm(train_loader, leave=True, total=len(train_loader))
            for i, batch_data in enumerate(train_pbar):
                # Forward model
                loss, batch_hyp, batch_label = forward_word_classification(model, batch_data[:-1], i2w=i2w,
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
            print("(Epoch {}) TRAIN LOSS:{:.4f} {} LR:{:.8f}".format((epoch + 1),
                                                                     total_train_loss / (i + 1),
                                                                     metrics_to_string(metrics),
                                                                     get_lr(optimizer)))

            # Evaluate on validation
            model.eval()
            torch.set_grad_enabled(False)

            total_loss, total_correct, total_labels = 0, 0, 0
            list_hyp, list_label = [], []

            pbar = tqdm(valid_loader, leave=True, total=len(valid_loader))
            for i, batch_data in enumerate(pbar):
                loss, batch_hyp, batch_label = forward_word_classification(model, batch_data[:-1], i2w=i2w,
                                                                           device='cuda')

                # Calculate total loss
                valid_loss = loss.item()
                total_loss = total_loss + valid_loss

                # Calculate evaluation metrics
                list_hyp += batch_hyp
                list_label += batch_label
                metrics = ner_metrics_fn(list_hyp, list_label)

                pbar.set_description("VALID LOSS:{:.4f} {}".format(total_loss / (i + 1), metrics_to_string(metrics)))

            metrics = ner_metrics_fn(list_hyp, list_label)
            print("(Epoch {}) VALID LOSS:{:.4f} {}".format((epoch + 1),
                                                           total_loss / (i + 1), metrics_to_string(metrics)))
    else:
        # Evaluate on test
        model.eval()
        torch.set_grad_enabled(False)

        total_loss, total_correct, total_labels = 0, 0, 0
        list_hyp, list_label = [], []

        pbar = tqdm(test_loader, leave=True, total=len(test_loader))
        for i, batch_data in enumerate(pbar):
            _, batch_hyp, _ = forward_word_classification(model, batch_data[:-1], i2w=i2w, device='cuda')
            list_hyp += batch_hyp

        # Save prediction
        df = pd.DataFrame({'label': list_hyp}).reset_index()
        df.to_csv('pred.txt', index=False)
        print(df)
