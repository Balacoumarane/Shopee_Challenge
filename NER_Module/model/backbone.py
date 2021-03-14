import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertPreTrainedModel, BertModel
from ..utils import get_logger

logger = get_logger(__name__)


# Forward function for word classification
def forward_word_classification(model, batch_data, i2w, is_test=False, device='cpu', **kwargs):
    # Unpack batch data
    if len(batch_data) == 4:
        (subword_batch, mask_batch, subword_to_word_indices_batch, label_batch) = batch_data
        token_type_batch = None
    elif len(batch_data) == 5:
        (subword_batch, mask_batch, token_type_batch, subword_to_word_indices_batch, label_batch) = batch_data

    # Prepare input & label
    subword_batch = torch.LongTensor(subword_batch)
    mask_batch = torch.FloatTensor(mask_batch)
    token_type_batch = torch.LongTensor(token_type_batch) if token_type_batch is not None else None
    subword_to_word_indices_batch = torch.LongTensor(subword_to_word_indices_batch)
    label_batch = torch.LongTensor(label_batch)

    if device == "cuda":
        logger.info('Device is set to cuda')
        subword_batch = subword_batch.cuda()
        mask_batch = mask_batch.cuda()
        token_type_batch = token_type_batch.cuda() if token_type_batch is not None else None
        subword_to_word_indices_batch = subword_to_word_indices_batch.cuda()
        label_batch = label_batch.cuda()

    # Forward model
    outputs = model(subword_batch, subword_to_word_indices_batch, attention_mask=mask_batch,
                    token_type_ids=token_type_batch, labels=label_batch)
    loss, logits = outputs[:2]

    # generate prediction & label list
    list_hyps = []
    list_labels = []
    hyps_list = torch.topk(logits, k=1, dim=-1)[1].squeeze(dim=-1)
    for i in range(len(hyps_list)):
        hyps, labels = hyps_list[i].tolist(), label_batch[i].tolist()
        list_hyp, list_label = [], []
        for j in range(len(hyps)):
            if labels[j] == -100:
                break
            else:
                list_hyp.append(i2w[hyps[j]])
                list_label.append(i2w[labels[j]])
        list_hyps.append(list_hyp)
        list_labels.append(list_label)

    return loss, list_hyps, list_labels


class BertForWordClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            subword_to_word_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to
        :obj:`None`): Labels for computing the token classification loss. Indices should be in ``[0, ...,
        config.num_labels - 1]``. Returns: :obj:`tuple(torch.FloatTensor)` comprising various elements depending on
        the configuration (:class:`~transformers.BertConfig`) and inputs: loss (:obj:`torch.FloatTensor` of shape
        :obj:`(1,)`, `optional`, returned when ``labels`` is provided) : Classification loss. scores (
        :obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.num_labels)`) Classification
        scores (before SoftMax). hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when
        ``config.output_hidden_states=True``): Tuple of :obj:`torch.FloatTensor` (one for the output of the
        embeddings + one for the output of each layer) of shape :obj:`(batch_size, sequence_length, hidden_size)`.
        Hidden-states of the model at the output of each layer plus the initial embedding outputs. attentions (
        :obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``): Tuple of
        :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads, sequence_length,
        sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
        the self-attention heads.
        """
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]

        # average the token-level outputs to compute word-level representations
        max_seq_len = subword_to_word_ids.max() + 1
        word_latents = []
        for i in range(max_seq_len):
            mask = (subword_to_word_ids == i).unsqueeze(dim=-1)
            word_latents.append((sequence_output * mask).sum(dim=1) / mask.sum())
        word_batch = torch.stack(word_latents, dim=1)

        sequence_output = self.dropout(word_batch)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)
