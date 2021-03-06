import numpy as np
import torch.utils.data


class NerGritDataset(torch.utils.data.Dataset):
    # Static constant variable
    # # TODO: Change this to accept label and any IOB annotation format
    # LABEL2INDEX = {'I-PERSON': 0, 'B-ORGANISATION': 1, 'I-ORGANISATION': 2, 'B-PLACE': 3, 'I-PLACE': 4, 'O': 5,
    #                'B-PERSON': 6}
    # INDEX2LABEL = {0: 'I-PERSON', 1: 'B-ORGANISATION', 2: 'I-ORGANISATION', 3: 'B-PLACE', 4: 'I-PLACE', 5: 'O',
    #                6: 'B-PERSON'}
    # NUM_LABELS = 7
    # TODO: Uncomment to use shopee dataset
    LABEL2INDEX = {'I-POI': 0, 'B-POI': 1, 'I-STREET': 2, 'B-STREET': 3, 'O': 4}
    INDEX2LABEL = {0: 'I-POI', 1: 'B-POI', 2: 'I-STREET', 3: 'B-STREET', 4: 'O'}
    NUM_LABELS = 5

    def load_dataset(self, path):
        # Read file
        data = open(path, 'r', encoding='utf-8').readlines()

        # Prepare buffer
        dataset = []
        sentence = []
        seq_label = []
        for line in data:
            if len(line.strip()) > 0:
                # token, label = line[:-1].split('\t')
                token, label = line[:-1].split(' ')
                sentence.append(token)
                seq_label.append(self.LABEL2INDEX[label])
            else:
                dataset.append({
                    'sentence': sentence,
                    'seq_label': seq_label
                })
                sentence = []
                seq_label = []
        return dataset

    def __init__(self, dataset_path, tokenizer, *args, **kwargs):
        self.data = self.load_dataset(dataset_path)
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        data = self.data[index]
        sentence, seq_label = data['sentence'], data['seq_label']

        # Add CLS token
        subwords = [self.tokenizer.cls_token_id]
        subword_to_word_indices = [-1]  # For CLS

        # Add subwords
        for word_idx, word in enumerate(sentence):
            subword_list = self.tokenizer.encode(word, add_special_tokens=False)
            subword_to_word_indices += [word_idx for i in range(len(subword_list))]
            subwords += subword_list

        # Add last SEP token
        subwords += [self.tokenizer.sep_token_id]
        subword_to_word_indices += [-1]

        return np.array(subwords), np.array(subword_to_word_indices), np.array(seq_label), data['sentence']

    def __len__(self):
        return len(self.data)


class NerDataLoader(torch.utils.data.DataLoader):
    def __init__(self, max_seq_len=512, *args, **kwargs):
        super(NerDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = self._collate_fn
        self.max_seq_len = max_seq_len

    def _collate_fn(self, batch):
        batch_size = len(batch)
        max_seq_len = max(map(lambda x: len(x[0]), batch))
        max_seq_len = min(self.max_seq_len, max_seq_len)
        max_tgt_len = max(map(lambda x: len(x[2]), batch))

        subword_batch = np.zeros((batch_size, max_seq_len), dtype=np.int64)
        mask_batch = np.zeros((batch_size, max_seq_len), dtype=np.float32)
        subword_to_word_indices_batch = np.full((batch_size, max_seq_len), -1, dtype=np.int64)
        seq_label_batch = np.full((batch_size, max_tgt_len), -100, dtype=np.int64)

        seq_list = []
        for i, (subwords, subword_to_word_indices, seq_label, raw_seq) in enumerate(batch):
            subwords = subwords[:max_seq_len]
            subword_to_word_indices = subword_to_word_indices[:max_seq_len]

            subword_batch[i, :len(subwords)] = subwords
            mask_batch[i, :len(subwords)] = 1
            subword_to_word_indices_batch[i, :len(subwords)] = subword_to_word_indices
            seq_label_batch[i, :len(seq_label)] = seq_label

            seq_list.append(raw_seq)

        return subword_batch, mask_batch, subword_to_word_indices_batch, seq_label_batch, seq_list
