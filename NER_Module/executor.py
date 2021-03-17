import os
import pandas as pd

from .data import NerGritDataset, NerDataLoader
from .model import load_pretrained_model, BertNerModel
from .utils import get_logger

logger = get_logger(__name__)


def execute_main(train_dataset_path: str = None, valid_dataset_path: str = None, test_dataset_path: str = None,
                 model_dir: str = None,  model_filename: str = None, predict_only: bool = False, n_epochs: int = 10,
                 exp_id: str = None, random_state: int = 33, device: str = 'cuda', validate_epoch: int = 1,
                 early_stop: int = 3, criteria: str = 'F1', result_path: str = None, result_filename: str = 'result'
                 ) -> pd.DataFrame:
    """

    Args:
        train_dataset_path:
        valid_dataset_path:
        test_dataset_path:
        model_dir:
        model_filename:
        predict_only:
        n_epochs:
        exp_id:
        random_state:
        device:
        validate_epoch:
        early_stop:
        criteria:
        result_path:
        result_filename:

    Returns:

    """
    w2i, i2w = NerGritDataset.LABEL2INDEX, NerGritDataset.INDEX2LABEL

    if not predict_only:
        logger.info('Training mode....')
        # load pretrained model
        base_model, tokenizer = load_pretrained_model(num_labels=NerGritDataset.NUM_LABELS, is_cuda=True)
        # load data
        train_dataset = NerGritDataset(dataset_path=train_dataset_path, tokenizer=tokenizer, lowercase=True)
        valid_dataset = NerGritDataset(dataset_path=valid_dataset_path, tokenizer=tokenizer, lowercase=True)

        train_loader = NerDataLoader(dataset=train_dataset, max_seq_len=512, batch_size=16, num_workers=16,
                                     shuffle=True)
        valid_loader = NerDataLoader(dataset=valid_dataset, max_seq_len=512, batch_size=16, num_workers=16,
                                     shuffle=False)

        # model initialisation
        bert_model = BertNerModel(num_classes=NerGritDataset.NUM_LABELS, i2w=i2w, exp_id=exp_id,
                                  random_state=random_state, device=device)
        bert_model.train(train_loader=train_loader, val_loader=valid_loader, model_dir=model_dir, num_epochs=n_epochs,
                         evaluate_every=validate_epoch, early_stop=early_stop, valid_criterion=criteria)
        if test_dataset_path is not None:
            test_dataset = NerGritDataset(dataset_path=test_dataset_path, tokenizer=tokenizer, lowercase=True)
            test_loader = NerDataLoader(dataset=test_dataset, max_seq_len=512, batch_size=16, num_workers=16,
                                        shuffle=False)
            result_df = bert_model.predict(test_loader=test_loader, save_path=result_path, filename=result_filename)
            return result_df
        else:
            logger.info('Test dataset is not passed, so using validation dataset')
            result_df = bert_model.predict(test_loader=valid_loader, save_path=result_path, filename=result_filename)
            return result_df
    else:
        # model initialisation
        predict_model = BertNerModel(num_classes=NerGritDataset.NUM_LABELS, i2w=i2w, exp_id=None,
                                     prediction_only=predict_only, random_state=random_state, device=device)
        # model_path
        model_folder = model_dir
        filename = model_filename
        model_path = os.path.join(model_folder, filename)
        # load custom model
        bert_predict_model, bert_predict_tokenizer = predict_model.load_modellocal(path=model_path, load_tokenizer=True)

        # load prediction_data
        assert isinstance(test_dataset_path, str)
        predict_dataset = NerGritDataset(dataset_path=test_dataset_path, tokenizer=bert_predict_tokenizer,
                                         lowercase=True)
        # load data trainer
        predict_loader = NerDataLoader(dataset=predict_dataset, max_seq_len=512, batch_size=16, num_workers=16,
                                       shuffle=False)
        predict_df = bert_predict_model.predict(test_loader=predict_loader, save_path=result_path,
                                                filename=result_filename)
        return predict_df
