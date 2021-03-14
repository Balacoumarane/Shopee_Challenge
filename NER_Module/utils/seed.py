import random
import numpy as np

from .log import get_logger

logger = get_logger(__name__)

_MAX_SEED = 2 ** 8 - 1


def set_seed(seed: int, verbose: bool = False) -> int:
    """
    Set seed for reproducibility.

    Args:
        seed (int): Seed
        verbose (bool): False. If true, enable verbose output

    Returns:
        seed (int): Seed

    """
    if seed is None:
        seed = np.random.randint(_MAX_SEED)
        if verbose:
            msg = ('Random seed is not provided. '
                   'Initializing with generated seed: {}')
            logger.info(msg.format(seed))
    else:
        seed %= _MAX_SEED

    random.seed(seed)
    np.random.seed(seed)

    return seed
