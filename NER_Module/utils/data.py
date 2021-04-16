import os
import re
from ast import literal_eval
import pandas as pd

from .log import get_logger

logger = get_logger(__name__)


def text_file_gen(address, ps):
    """

    Args:
        address:
        ps:

    Returns:

    """

    address = address.lower()
    ps = ps.lower()
    POI = ps.split('/')[0]
    street = ps.split('/')[1]
    add = re.split('[ ]', address)
    POI = re.split('[ ]', POI)
    street = re.split('[ ]', street)

    add = [i for i in add if i != '']
    POI = [i for i in POI if i != '']
    street = [i for i in street if i != '']
    add_fl = ['O' for i in add]

    for p in range(len(POI)):
        try:
            if p == 0:
                add_fl[add.index(POI[0])] = 'B-POI'
            else:
                add_fl[add.index(POI[0]) + p] = 'I-POI'
        except:
            pass
    for s in range(len(street)):
        try:
            if s == 0:
                add_fl[add.index(street[0])] = 'B-STREET'
            else:
                add_fl[add.index(street[0]) + s] = 'I-STREET'
        except:
            pass
    return add, add_fl


def test_text_file_gen(address):
    # print(address)
    # print(ps)
    address = address.lower()

    add = re.split('[ ]', address)

    add = [i for i in add if i != '']

    add_fl = ['O' for i in add]

    return add, add_fl


def text_ner_format(data: pd.DataFrame = None, address_col: str = None, street_col: str = None,
                    is_test: bool = False, filename: str = None, save_path: str = None):
    """

    Args:
        data:
        address_col:
        street_col:
        is_test:
        filename:
        save_path:

    Returns:

    """
    file_path = os.path.join(save_path, filename)
    with open(file_path, "a", encoding='utf-8') as file:
        for i, r in data.iterrows():
            if not is_test:
                add_full, add_op = text_file_gen(r[address_col], r[street_col])
            else:
                add_full, add_op = test_text_file_gen(r[address_col])
            for j in range(len(add_op)):
                string = add_full[j] + " " + add_op[j]
                file.write(string + '\n')
            file.write('\n')


def out_gen(c, input_add, op_pred_list):
    add = re.split('[- :,.]', input_add)
    add = [i for i in add if i != '']
    op_pred_list = literal_eval(op_pred_list)

    street_list = [i for i in op_pred_list if "STREET" in i]
    poi_list = [i for i in op_pred_list if "POI" in i]

    try:
        street_index = op_pred_list.index('B-STREET')
        street_list_mapped = add[street_index:street_index + len(street_list)]
        street_list_str = ' '.join(street_list_mapped)
    except:
        street_list_str = ''
        pass
    try:
        poi_index = op_pred_list.index('B-POI')
        poi_list_mapped = add[poi_index:poi_index + len(poi_list)]
        poi_list_str = ' '.join(poi_list_mapped)
    except:
        poi_list_str = ''
        pass
    final_string = poi_list_str + '/' + street_list_str
    return final_string


def convert_shopee_format(input_df: pd.DataFrame = None, address_col: str = None, output_df: pd.DataFrame = None,
                          filename: str = None, save_path: str = None):
    """

    Args:
        input_df:
        address_col:
        output_df:
        filename:
        save_path:

    Returns:

    """
    final_string_list = []
    try:
        input_df.shape[0] = output_df.shape[0]
        for i in range(input_df.shape[0]):
            final_string_list.append(out_gen(i, input_df[address_col][i], output_df['label'][i]))

        Index_list = [i for i in range(len(final_string_list))]
        out_df = pd.DataFrame(list(zip(Index_list, final_string_list)), columns=['id', 'POI/street'])
        save_file_path = os.path.join(save_path, filename)
        out_df.to_csv(save_file_path, encoding='utf-8', index=False)
        logger.info('Results are saved in shopee format in {}'.format(save_file_path))
    except:
        logger.info('input and output do not have same index')

