from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel
from TorchCRF import CRF
from torch import nn
from models.transformer.ID_classifier import SlotClassifier, IntentClassifier

""" Using Bert with LSTM and CRF """


class BertIDSF(BertPreTrainedModel):
    def __init__(self, config, intent_label_lst, slot_label_lst, n_layers=1):
        super().__init__(config)

        self.num_intent_labels = len(intent_label_lst)
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
        self.intent_classifier = IntentClassifier(config.hidden_size, self.num_intent_labels)
        self.slot_classifier = SlotClassifier(config.hidden_size, self.num_slot_labels)
        self.crf = CRF(config.num_labels, batch_first=True)
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
            intents=None,
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

        intent_logits = self.intent_classifier(sequence_output[:, 0, :])
        slot_logits = self.slot_classifier(sequence_output)

        total_loss = 0
        # Intent Softmax
        if intents is not None:
            intent_loss_fct = nn.CrossEntropyLoss()
            intent_loss = intent_loss_fct(intent_logits.view(-1, self.num_intent_labels), intents.view(-1))
            total_loss += 0.5 * intent_loss

        # Slot Softmax
        if labels is not None:
            if self.use_crf:
                slot_loss = self.crf(slot_logits, labels, mask=attention_mask.byte(), reduction='mean')
                slot_loss = -1 * slot_loss  # negative log-likelihood
            else:
                slot_loss_fct = nn.CrossEntropyLoss(ignore_index=0)
                # Only keep active parts of the loss
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = slot_logits.view(-1, self.num_slot_labels)[active_loss]
                    active_labels = labels.view(-1)[active_loss]
                    slot_loss = slot_loss_fct(active_logits, active_labels)
                else:
                    slot_loss = slot_loss_fct(slot_logits.view(-1, self.num_slot_labels), labels.view(-1))
            total_loss += 0.5 * slot_loss

        outputs = ((intent_logits, slot_logits),) + outputs[2:]  # add hidden states and attention if they are here

        outputs = (total_loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)  # Logits is a tuple of intent and slot logits
