import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
import os
from .seqTagger import tag_and_tokens_to_original_form
from seqeval.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import accuracy_score

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from .seqTagger import get_ouputs, if_final_layer_is_crf
# from .eval import conlleval
from tqdm import tqdm



def run_transformer_trainer(Data, batch_size, FULL_FINETUNING, model, tokenizer,
                            device, validation_func=None,
                            lr=5e-5, epochs=10, save_dir=None):
    '''
      train a transformer tagger

      :param nparray input_ids: ids tokens, given by tokenizer
      :param nparray tags: ids of tags
      :param list starts: identifies if the position is start of a token, starts[i] 1 is it is start o.w. 0
      :param int batch_size: model training batch_size
      :param intlist lens: length of sequences in input_ids
      :param Boolean FULL_FINETUNING: if set to true, update BERT (or any language model) parameters too, if set to false, only upates classifier's parameters
      :param tokenizer tokenizer: needed when saving the model
      :param intlist val_idnxs: indices of input_ids to consider for validation part
      :param intlist test_idnxs: indices of input_ids to consider for test part
      :param function validation_func: function to log proper metric in each epoch, takes four args valid_true_labels, valid_predicted_labels, test_true_labels,
                   test_predicted_labels
      :param float lr: learning rate
      :param int epochs: training iterations
      :param string save_dir: directory to save the model
      '''
    final_layer_is_crf = if_final_layer_is_crf(model.__class__)
    tr_inputs = Data["tr_inputs"]
    tr_masks = Data["tr_masks"]
    tr_tags = Data["tr_tags"]
    tr_intents = Data["tr_intents"]

    val_inputs = Data["val_inputs"]
    val_masks = Data["val_masks"]
    val_tags = Data["val_tags"]
    val_intents = Data["val_intents"]

    test_inputs = Data["test_inputs"]
    test_masks = Data["test_masks"]
    test_tags = Data["test_tags"]
    test_intents = Data["test_intents"]

    tr_lens = Data["tr_lens"]
    tr_starts = Data["tr_starts"]
    val_lens = Data["val_lens"]
    val_starts = Data["val_starts"]
    test_lens = Data["test_lens"]
    test_starts = Data["test_starts"]

    tr_tokens = Data["tr_tokens"]
    val_tokens = Data["val_tokens"]
    test_tokens = Data["test_tokens"]

    # tr_lens = get_at(lens, train_ids)
    # val_inputs_ = get_at(input_ids, val_idnxs)
    # test_inputs_ = get_at(input_ids, test_idnxs)
    # val_tokenized_sentences = get_at(tokenized_sentences, val_ids)
    # test_tokenized_sentences = get_at(tokenized_sentences, test_ids)



    train_data = TensorDataset(tr_inputs, tr_masks, tr_tags, tr_intents, torch.Tensor(tr_lens))
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    valid_data = TensorDataset(val_inputs, val_masks, val_tags, val_intents, torch.Tensor(val_lens))
    valid_sampler = SequentialSampler(valid_data)
    valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=batch_size)

    test_data = TensorDataset(test_inputs, test_masks, test_tags, test_intents, torch.Tensor(test_lens))
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)


    # train_data = TensorDataset(tr_inputs, tr_masks, tr_tags, torch.Tensor(tr_lens))
    # train_sampler = RandomSampler(train_data)
    # train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # valid_data = TensorDataset(val_inputs, val_masks, val_tags, torch.Tensor(val_lens))
    # valid_sampler = SequentialSampler(valid_data)
    # valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=batch_size)

    # test_data = TensorDataset(test_inputs, test_masks, test_tags, torch.Tensor(test_lens))
    # test_sampler = SequentialSampler(test_data)
    # test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)


    # train_data = TensorDataset(tr_inputs, tr_masks, tr_tags, tr_intents)
    # train_sampler = RandomSampler(train_data)
    # train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # valid_data = TensorDataset(val_inputs, val_masks, val_tags, val_intents)
    # valid_sampler = SequentialSampler(valid_data)
    # valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=batch_size)

    # test_data = TensorDataset(test_inputs, test_masks, test_tags, test_intents)
    # test_sampler = SequentialSampler(test_data)
    # test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    if device.type == "cuda":
        # print('Its CUDA')
        model.cuda()

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    # no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}]

    # if FULL_FINETUNING:
    #     param_optimizer = list(model.named_parameters())
    #     no_decay = ['bias', 'gamma', 'beta']
    #     optimizer_grouped_parameters = [
    #         {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
    #          'weight_decay_rate': 0.01},
    #         {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
    #          'weight_decay_rate': 0.0}
    #     ]
    # else:
    #     # TODO: edit this
    #     if final_layer_is_crf:
    #         param_optimizer = list(model.classifier.named_parameters()) + list(model.crf.named_parameters())
    #         if hasattr(model, 'final_lstm'):
    #             param_optimizer = param_optimizer + list(model.final_lstm.named_parameters())
    #         # print(param_optimizer)
    #     else:
    #         param_optimizer = list(model.classifier.named_parameters())
    #     optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=lr,
        eps=1e-8
        # ,weight_decay=1e-6, amsgrad=True
    )

    max_grad_norm = 1.0

    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    # for name, param in model.named_parameters():
    #     if 'classifier' not in name: # classifier layer
    #         param.requires_grad = False

    ## Store the average loss after each epoch so we can plot them.
    loss_values, validation_loss_values = [], []
    for epoch in range(epochs):
        print('Epoch', epoch)
        # ========================================
        #               Training
        # ========================================
        # Perform one full pass over the training set.

        # Put the model into training mode.
        model.train()
        # Reset the total loss for this epoch.
        total_loss = 0
        # Training loop
        train_dataloader = tqdm(train_dataloader, mininterval=60)
        for step, batch in enumerate(train_dataloader):
            # add batch to gpu
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels, b_intents, lens = batch
            # b_input_ids, b_input_mask, b_labels, lens = batch
            # try:
            # Always clear any previously calculated gradients before performing a backward pass.
            model.zero_grad()
            # forward pass
            # This will return the loss (rather than the model output)
            # because we have provided the `labels`.
            b_labels = b_labels.long()
            b_intents = b_intents.long()
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask, labels=b_labels, intents=b_intents, lens=lens, device=device)
            # outputs = model(b_input_ids, token_type_ids=None,
            #                 attention_mask=b_input_mask, labels=b_labels, lens=lens, device=device)

            # get the loss
            loss = outputs[0]
            
            # print(loss)
            # Perform a backward pass to calculate the gradients.
            # print(loss)
            loss.backward()
            # track train loss
            total_loss += loss.item()
            # Clip the norm of the gradient
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
            # update parameters
            optimizer.step()
            # Update the learning rate.
            scheduler.step()
        # make_dot(outputs, params=dict(model.named_parameters()), show_attrs=True, show_saved=True)
        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)
        print("Average train loss: {}".format(avg_train_loss))

        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)

        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.
        tr_true_slots, tr_pred_slots,  tr_true_intents, tr_pred_intents = get_predictions(model, train_dataloader, device,
                                                    tr_lens, model.config.dict2, model.config.inte2, tr_starts)

        val_true_slots, val_pred_slots,  val_true_intents, val_pred_intents = get_predictions(model, valid_dataloader, device,
                                                    val_lens, model.config.dict2, model.config.inte2, val_starts)

        test_true_slots, test_pred_slots, test_true_intents, test_pred_intents = get_predictions(model, test_dataloader, device,
                                                    test_lens, model.config.dict2, model.config.inte2, test_starts)
        if validation_func is not None:
            validation_func(tr_true_slots, tr_pred_slots,  tr_true_intents, tr_pred_intents,
                            val_true_slots, val_pred_slots, val_true_intents, val_pred_intents,
                            test_true_slots, test_pred_slots, test_true_intents, test_pred_intents)

        # tr_true_slots, tr_pred_slots = get_predictions(model, train_dataloader, device,
        #                                             tr_lens, model.config.dict2, tr_starts)

        # val_true_slots, val_pred_slots = get_predictions(model, valid_dataloader, device,
        #                                             val_lens, model.config.dict2, val_starts)

        # test_true_slots, test_pred_slots = get_predictions(model, test_dataloader, device,
        #                                             test_lens, model.config.dict2, test_starts)
        # if validation_func is not None:
        #     validation_func(tr_true_slots, tr_pred_slots,
        #                     val_true_slots, val_pred_slots,
        #                     test_true_slots, test_pred_slots)


    if save_dir != None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model,
                                                'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)

    # Good practice: save your training arguments together with the trained model
    # torch.save(args, os.path.join(save_dir, 'training_args.bin'))


def get_predictions(model, dataloader, device, lens, dict2, inte2, starts,
                    ):
    predictions_slots, true_slots, predictions_intents, true_intents = get_ouputs(model, dataloader, device)
    predicted_labels_slots = []
    true_labels_slots = []
    predicted_labels_intents = []
    true_labels_intents = []
    for i in range(len(predictions_slots)):
        ll, ttt = tag_and_tokens_to_original_form(predictions_slots[i][1:lens[i] + 1],
                                                  dict2,
                                                  starts[i], true_slots[i][1:lens[i] + 1])
        # tokens.append(tt)
        predicted_labels_slots.append(ll)
        true_labels_slots.append(ttt)
        predicted_labels_intents.append(inte2[predictions_intents[i]+1])
        true_labels_intents.append(inte2[true_intents[i]+1])
    return true_labels_slots, predicted_labels_slots, predicted_labels_intents, true_labels_intents


# def get_predictions(model, dataloader, device, lens, dict2, starts,
#                     ):
#     predictions_slots, true_slots = get_ouputs(model, dataloader, device)
#     predicted_labels_slots = []
#     true_labels_slots = []
#     for i in range(len(predictions_slots)):
#         ll, ttt = tag_and_tokens_to_original_form(predictions_slots[i][1:lens[i] + 1],
#                                                   dict2,
#                                                   starts[i], true_slots[i][1:lens[i] + 1])
#         # tokens.append(tt)
#         predicted_labels_slots.append(ll)
#         true_labels_slots.append(ttt)
#     return true_labels_slots, predicted_labels_slots



def joint_IDSF_validation(tr_true_slots, tr_pred_slots,  tr_true_intents, tr_pred_intents,
                          val_true_slots, val_pred_slots, val_true_intents, val_pred_intents,
                          test_true_slots, test_pred_slots, test_true_intents, test_pred_intents):
    
    # Train Evaluation
    trppr = precision_score(tr_true_slots, tr_pred_slots)
    trpre = recall_score(tr_true_slots, tr_pred_slots)
    trpf1 = f1_score(tr_true_slots, tr_pred_slots)
    tracc = accuracy_score(tr_true_intents, tr_pred_intents)
    trEM = 0
    for i in range(len(tr_true_slots)):
        if accuracy_score(tr_true_slots[i], tr_pred_slots[i]) == 1 and tr_true_intents[i] == tr_pred_intents[i]:
            trEM += 1
    print("Metrics on Train data: P={0:.5f}, R={1:.5f}, F1={2:.5f}, Acc={3:.5f}, EM={4:.5f}".format(trppr, trpre, trpf1, tracc, trEM/len(tr_true_slots)))
    
    # Validation Evaluation
    ppr = precision_score(val_true_slots, val_pred_slots)
    pre = recall_score(val_true_slots, val_pred_slots)
    valf1 = f1_score(val_true_slots, val_pred_slots)
    acc =  accuracy_score(val_true_intents, val_pred_intents)
    EM = 0
    for i in range(len(val_true_slots)):
        if accuracy_score(val_true_slots[i], val_pred_slots[i])==1 and val_true_intents[i]==val_pred_intents[i]:
            EM+=1
    print("Metrics on Validation data: P={0:.5f}, R={1:.5f}, F1={2:.5f}, Acc={3:.5f}, EM={4:.5f}".format(ppr, pre, valf1, acc, EM/len(val_true_slots)))

    # Test Evaluation
    tppr = precision_score(test_true_slots, test_pred_slots)
    tpre = recall_score(test_true_slots, test_pred_slots)
    tpf1 = f1_score(test_true_slots, test_pred_slots)
    tacc = accuracy_score(test_true_intents, test_pred_intents)
    tEM = 0
    for i in range(len(test_true_slots)):
        if accuracy_score(test_true_slots[i], test_pred_slots[i]) == 1 and test_true_intents[i] == test_pred_intents[i]:
            tEM += 1
    print(
        '------------------------------------------------------------------------------------------------------------------------ do not look (metrics on test data)-------------------------------',
         tppr, tpre, tpf1, tacc, tEM/len(test_true_slots))




# def joint_IDSF_validation(tr_true_slots, tr_pred_slots,
#                           val_true_slots, val_pred_slots,
#                           test_true_slots, test_pred_slots):
    
#     # Train Evaluation
#     trppr = precision_score(tr_true_slots, tr_pred_slots)
#     trpre = recall_score(tr_true_slots, tr_pred_slots)
#     trpf1 = f1_score(tr_true_slots, tr_pred_slots)
#     print("Metrics on Train data: P={0:.5f}, R={1:.5f}, F1={2:.5f}".format(trppr, trpre, trpf1))
    
#     # Validation Evaluation
#     ppr = precision_score(val_true_slots, val_pred_slots)
#     pre = recall_score(val_true_slots, val_pred_slots)
#     valf1 = f1_score(val_true_slots, val_pred_slots)
#     print("Metrics on Validation data: P={0:.5f}, R={1:.5f}, F1={2:.5f}".format(ppr, pre, valf1))

#     # Test Evaluation
#     tppr = precision_score(test_true_slots, test_pred_slots)
#     tpre = recall_score(test_true_slots, test_pred_slots)
#     tpf1 = f1_score(test_true_slots, test_pred_slots)
#     print(
#         '------------------------------------------------------------------------------------------------------------------------ do not look (metrics on test data)-------------------------------',
#          tppr, tpre, tpf1)