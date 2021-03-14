import pandas as pd
import os
from pathlib import Path
from ..utils import get_logger

logger = get_logger(__name__)


class ShopeeDataset:
    """

    """
    def __init__(self, input_dir: Path = None):
        """

        Args:
            input_dir:
        """
        assert (isinstance(input_dir, Path))
        self.data = pd.read_csv(input_dir)

    def parse_ner_format(self, is_test: bool = False) -> pd.DataFrame:
        """

        Args:
            is_test(bool):

        Returns:
            data(pd.DataFrame):
        """
        return None


