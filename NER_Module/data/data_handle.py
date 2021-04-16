import os
import pandas as pd
from sklearn.model_selection import train_test_split

from ..utils import get_logger, text_ner_format

logger = get_logger(__name__)


class ShopeeDataset:

    def __init__(self, input_dir: str = None, training_file: str = None, validation_file: str = None,
                 test_file: str = None, valid_split: float = None):
        """

        Args:
            input_dir:
            training_file:
            validation_file:
            test_file:
            valid_split:
        """
        assert (isinstance(input_dir, str))
        assert (isinstance(training_file, str))
        if validation_file is not None:
            training_file = os.path.join(input_dir, training_file)
            self.training_data = pd.read_csv(training_file)
            validation_file = os.path.join(input_dir, validation_file)
            self.validation_data = pd.read_csv(validation_file)
        else:
            logger.info('Validation data is not passed, Splitting train file into training and validation...')
            if valid_split is not None:
                valid_split = valid_split
            else:
                valid_split = 0.2
            training_file = os.path.join(input_dir, training_file)
            training_data = pd.read_csv(training_file)
            self.training_data, self.validation_data = train_test_split(training_data, test_size=valid_split)
        if test_file is not None:
            logger.info('Loading test file')
            test_file = os.path.join(input_dir, test_file)
            self.test_data = pd.read_csv(test_file)
        else:
            logger.info('Test file not loaded')

    def parse_ner_format(self, save_path: str = None, training_text_file: str = None, validation_text_file: str = None,
                         test_text_file: str = None) -> None:
        """

        Args:
            save_path:
            training_text_file:
            validation_text_file:
            test_text_file:

        Returns:

        """
        logger.info('Writing training as {}.txt in {}'.format(training_text_file, save_path))
        text_ner_format(data=self.training_data, address_col='raw_address', street_col='POI/street', is_test=False)
        logger.info('Writing validation as {}.txt in {}'.format(validation_text_file, save_path))
        text_ner_format(data=self.validation_data, address_col='raw_address', street_col='POI/street', is_test=False)
        if test_text_file is not None:
            logger.info('Writing training as {}.txt in {}'.format(test_text_file, save_path))
            text_ner_format(data=self.test_data, address_col='raw_address', street_col=None, is_test=True)
        else:
            logger.info('No test file')
