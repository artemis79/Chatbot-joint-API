from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel
from torchcrf import CRF
import torch
from torch import nn
from models.transformer.ID_classifier import SlotClassifier

""" Using Bert with LSTM and CRF """


class BertSF(BertPreTrainedModel):
    def __init__(self, config, slot_label_lst, n_layers=1):
        super().__init__(config)

        self.num_slot_labels = len(slot_label_lst)
        self.bert = BertModel(config=config)  # Load pretrained bert
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.n_layers = n_layers
        self.bidirectional = False
        if hasattr(config, 'bi'):
            if config.bi:
                self.bidirectional = True
        config.lstm_size = self.config.hidden_size
        if self.bidirectional:
            config.lstm_size = int(config.lstm_size / 2)
        self.final_lstm = nn.LSTM(config.hidden_size, config.lstm_size, self.n_layers, bidirectional=self.bidirectional)
        # self.final_gru = nn.GRU(config.hidden_size, config.lstm_size, self.n_layers, bidirectional=self.bidirectional)
        self.slot_classifier = SlotClassifier(config.hidden_size, self.num_slot_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.use_crf = config.use_crf

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=True,
            lens=None,
            device=None
    ):


        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=True
        )
        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)

        # lstm_out, (hn, cn) = self.final_lstm(sequence_output)
        # gru_out, (hn, cn) = self.final_gru(sequence_output)

        slot_logits = self.slot_classifier(sequence_output)

        loss = 0

        # Slot Softmax
        if labels is not None:
            if self.use_crf:
                loss = self.crf(slot_logits, labels, mask=attention_mask.byte(), reduction='mean')
                loss = -1 * loss  # negative log-likelihood
            else:
                loss_fct = nn.CrossEntropyLoss(ignore_index=0)
                # Only keep active parts of the loss
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = slot_logits.view(-1, self.num_slot_labels)[active_loss]
                    active_labels = labels.view(-1)[active_loss].type(torch.long)
                    loss = loss_fct(active_logits, active_labels)
                else:
                    loss = loss_fct(slot_logits.view(-1, self.num_slot_labels), labels.view(-1))


        outputs = ((slot_logits),) + outputs[2:]  # add hidden states and attention if they are here

        outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)  # Logits is a tuple of intent and slot logits
