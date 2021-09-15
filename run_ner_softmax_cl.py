import argparse
import glob
import logging
import os
import json
import pdb
import time
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from callback.optimizater.adamw import AdamW
from callback.lr_scheduler import get_linear_schedule_with_warmup
from callback.progressbar import ProgressBar
from callback.adversarial import FGM
from models.projector import Projector
from tools.common import seed_everything
from tools.common import init_logger, logger



from transformers import WEIGHTS_NAME, BertConfig,get_linear_schedule_with_warmup,AdamW, BertTokenizer
from models.bert_for_ner import BertSoftmaxForNer, BertCrfForNer, BertSpanForNer
from processors.utils_ner import get_entities
from processors.ner_seq import convert_examples_to_features
from processors.ner_seq import ner_processors as processors
from processors.ner_seq import collate_fn
from metrics.ner_metrics import SeqEntityScore
from tools.finetuning_argparse import get_argparse

from tensorboardX import SummaryWriter
from utils.general_utils import get_chunks
import numpy as np

MODEL_CLASSES = {
    ## bert ernie bert_wwm bert_wwwm_ext
    # change model here
    'bert': (BertConfig, BertSoftmaxForNer, BertTokenizer),
}

alpha_list = [
    [0.9 * 0.001, 0.6],
    [0.9 * 0.001, 0.6],
    [0.9 * 0.001, 0.6],
    [0.9 * 0.001, 0.6],
    [0.9 * 0.001, 0.6],
    [0.9 * 0.001, 0.6],
    [0.9 * 0.001, 0.6],
    [0.9 * 0.001, 0.6],
    [0.9 * 0.001, 0.6],
    [0.9 * 0.001, 0.6],
    [0.9 * 0.001, 0.6],
    [0.9 * 0.001, 0.6],
    [0.9 * 0.001, 0.6],
]

def train(args, train_dataset, valid_dataset, test_dataset, model, tokenizer, projector):
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                  collate_fn=collate_fn)
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    adapter_params = ["adapter", "classifier"]  # the params that have 'adapter' in the name will be included in the training process.
    adapter_params_exclude = ["bias"]
    # test_ = [n for n, _ in model.named_parameters() if any(nd in n for nd in adapter_params)]
    # pdb.set_trace()
    args.warmup_steps = int(t_total * args.warmup_proportion)
    if args.do_adapter:
        # for n_, p_ in enumerate(model.named_parameters()):
        #     pdb.set_trace()
        optimizer_adapter_parameters = [{"params": [p for n, p in model.named_parameters() if any(nd in n for nd in adapter_params)], "weight_decay": 0.0},]
        if not projector.is_inited:
            shape_list = {p[0]: [p[1].shape, p[1], alpha_list[2]] for n, p in enumerate(model.named_parameters()) if any(nd not in p[0] for nd in adapter_params_exclude) and any(nd in p[0] for nd in adapter_params)}
            projector.initial(shape_list)
            # pdb.set_trace()
        optimizer = AdamW(optimizer_adapter_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=t_total)
        # pdb.set_trace()
    else:
        optimizer_grouped_parameters = [
            {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,},
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=t_total)
        # Check if saved optimizer or scheduler states exist
        if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")):
            # pdb.set_trace()
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
            scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    # pdb.set_trace()

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    global_step = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path) and "checkpoint" in args.model_name_or_path:
        # set global_step to gobal_step of last saved checkpoint from model path
        global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)
        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    if args.do_adv:
        fgm = FGM(model, emb_name=args.adv_name, epsilon=args.adv_epsilon)
    model.zero_grad()
    
    pbar = ProgressBar(n_total=len(train_dataloader), desc='Training', num_epochs=int(args.num_train_epochs))
    if args.save_steps==-1 and args.logging_steps==-1:
        args.logging_steps=len(train_dataloader)
        args.save_steps = len(train_dataloader)
        # pdb.set_trace()

    all_step = int(args.num_train_epochs) * len(train_dataloader)
    for epoch in range(int(args.num_train_epochs)):
        pbar.reset()

        pbar.epoch_start(current_epoch=epoch)
        for step, batch in enumerate(train_dataloader):
            lambda_ = step / all_step
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            # print('batch: ', batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if args.model_type != "distilbert":
                # XLM and RoBERTa don"t use segment_ids
                inputs["token_type_ids"] = (batch[2] if args.model_type in ["bert", "xlnet"] else None)
            # print('Inputs here!!!!!!! \n', inputs)
            # input('pause')

            outputs, projector_data = model(**inputs)
            ######################################################################
            if args.do_adapter:
                projector.update(projector_data, lambda_)
            #####################################################################
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            if global_step % 20 == 0:
                args.writer.add_scalar("softmax_ner"+'/train/loss', loss, global_step)

            model.zero_grad()
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            if args.do_adv:
                fgm.attack()
                loss_adv = model(**inputs)[0]
                if args.n_gpu>1:
                    loss_adv = loss_adv.mean()
                loss_adv.backward()
                fgm.restore()
            pbar(step, {'loss': loss.item()})
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                if args.do_adapter:
                    projector.adjust_gradient()

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    # print(" ")
                    if args.local_rank == -1:
                        # Only evaluate when single GPU otherwise metrics may not average well
                        #evaluate(args, model, tokenizer)
                        pass

        logger.info("\n")
        if 'cuda' in str(args.device):
            torch.cuda.empty_cache()
        
        # Evaluation
        best_score, nepoch_no_imprv, test_acc = -1, 0, 0   
        epoch_no_imprv = 5
        results = {}

        prefix = ""
        # print("prefix here!!!!!!: ", prefix)
        # input()
        result = evaluate(args, model, tokenizer, valid_dataset, prefix=prefix, data_type='dev')
        # if global_step:
        #     result = {"{}_{}".format(global_step, k): v for k, v in result.items()}
        # results.update(result)
        # print('result here!!!!!!!!!!: ', result)
        
        args.writer.add_scalar('eval/accuracy', result['acc'], global_step)
        args.writer.add_scalar('eval/recall', result['recall'], global_step)
        args.writer.add_scalar('eval/f1', result['f1'], global_step)

        average_score = (result['acc'] + result['recall'] + result['f1'])/3
        if average_score  > best_score:
            nepoch_no_imprv = 0
            #self.save_session()
            best_score = average_score
            print("- new best score!")
            test = False
            if test:
                test_result = evaluate(args, model, tokenizer, test_dataset, prefix=prefix, data_type='test')
                # self.print_eval_result(test_result)
                print("test sf acc:{}".format(test_result))
        else:
            nepoch_no_imprv += 1
            if nepoch_no_imprv >= epoch_no_imprv:
                print("- early stopping {} epoches without improvement".format(nepoch_no_imprv))
                break

    # test after all epoches training
    test_result = evaluate(args, model, tokenizer, test_dataset, prefix=prefix, data_type='test')
    print(test_result)


    # if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
    save=False
    if save: 
        # Save model checkpoint
        output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # Take care of distributed/parallel training
        model_to_save = (model.module if hasattr(model, "module") else model)
        model_to_save.save_pretrained(output_dir)
        torch.save(args, os.path.join(output_dir, "training_args.bin"))
        tokenizer.save_vocabulary(output_dir)
        logger.info("Saving model checkpoint to %s", output_dir)
        # torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
        logger.info("Saving optimizer and scheduler states to %s", output_dir)

    return test_result

def evaluate(args, model, tokenizer, eval_dataset, prefix="", data_type='dev'):
    # param data_type: 'dev', 'test'

    metric = SeqEntityScore(args.id2label, markup=args.markup)

    eval_output_dir = args.output_dir
    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    # eval_dataset = load_and_cache_examples(args, args.task_name,tokenizer, data_type='dev')
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                 collate_fn=collate_fn)
    # Eval!
    logger.info("***** Running evaluation on %s *****", data_type)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    pbar = ProgressBar(n_total=len(eval_dataloader), desc="Evaluating")
    accs = []

    for step, batch in enumerate(eval_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if args.model_type != "distilbert":
                # XLM and RoBERTa don"t use segment_ids
                inputs["token_type_ids"] = (batch[2] if args.model_type in ["bert", "xlnet"] else None)
            outputs, _ = model(**inputs)
        tmp_eval_loss, logits = outputs[:2]
        if args.n_gpu > 1:
            tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu parallel evaluating
        eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1

        # preds = np.argmax(logits.cpu().numpy(), axis=2).tolist()
        out_label_ids = inputs['labels'].cpu().numpy().tolist()
        batch_onehot_label = []
        for example in out_label_ids:
            tmp = []
            for ind, tag in enumerate(example):
                
                onehot = np.zeros(len(args.id2label))
                # if tag !=0:
                onehot[tag] = 1
                tmp.append(onehot)
            batch_onehot_label.append(tmp)
        input_lens = batch[4].cpu().numpy().tolist()
        # print('preds: ', preds)
        # print('out_label_ids: ', out_label_ids)
        # input('pause')
        # =====================================
        # labels_onehot = predictions['meta_nested_tag_ids']
        # labels_pred = predictions['labels_pred']
        # sequences_lengths = predictions['self_sequence_lengths']

        # labels_onehot = labels_onehot[:sequences_lengths]
        # labels_pred = labels_pred[:sequences_lengths]
        # lab_chunks = set(get_chunks(labels_onehot, meta_nested_tag_vocab))
        # lab_pred_chunks = set(get_chunks(labels_pred, meta_nested_tag_vocab))
        # pred_correct_chunks = lab_chunks & lab_pred_chunks
        # accs += [len(lab_chunks) == len(pred_correct_chunks)]
        debug = False
        if debug:
            print('logits.cpu().numpy(): ', np.shape(logits.cpu().numpy()))
            input('paause')
        
        for ind, tmp in enumerate(logits.cpu().numpy()):
            pred_tags = []
            for tags in tmp:
                tags = [1.0 if x >= np.max(tags) else 0.0 for x in tags]
                pred_tags.append(np.array(tags))
            
            pred_t = set(get_chunks(pred_tags, args.label2id))
            true_tags_ids = set(get_chunks(batch_onehot_label[ind], args.label2id))
            # print('true_tags_ids: ', true_tags_ids)
            # input("pause")
            # print('pred_t: ', pred_t)
            pred_correct_chunks = true_tags_ids & pred_t
            accs += [len(true_tags_ids) == len(pred_correct_chunks)]
            

            

            metric.update_chunks(label_chunks=true_tags_ids, pred_chunks=pred_t)

        # # print(len(true_tags_ids))
        # # print(len(pred_tags))
        # tag_acc.append(np.array_equal(true_tags_ids, pred_tags))
        
        
        
        
        # tag_acc.append(pred_t==true_tags_ids)
        # =======================================
        # for i, label in enumerate(out_label_ids):
        #     temp_1 = []
        #     temp_2 = []
        #     for j, m in enumerate(label):
        #         if j == 0:
        #             continue
        #         elif j == input_lens[i]-1:
        #             metric.update(pred_paths=[temp_2], label_paths=[temp_1])
        #             break
        #         else:
        #             temp_1.append(args.id2label[out_label_ids[i][j]])
        #             temp_2.append(preds[i][j])
        pbar(step)
    print("acc: ", np.mean(accs))
    
    eval_loss = eval_loss / nb_eval_steps
    eval_info, entity_info = metric.result()
    results = {f'{key}': value for key, value in eval_info.items()}
    results['loss'] = eval_loss
    logger.info("***** Eval results %s *****", prefix)
    info = "-".join([f' {key}: {value:.4f} ' for key, value in results.items()])
    logger.info(info)
    logger.info("***** Entity results %s *****", prefix)
    for key in sorted(entity_info.keys()):
        logger.info("******* %s results ********"%key)
        info = "-".join([f' {key}: {value:.4f} ' for key, value in entity_info[key].items()])
        logger.info(info)
    logger.info("\n")
    return results

def predict(args, model, tokenizer, prefix=""):
    pred_output_dir = args.output_dir
    if not os.path.exists(pred_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(pred_output_dir)

    test_dataset = load_and_cache_examples(args, args.task_name,tokenizer, data_type='test')
    # Note that DistributedSampler samples randomly
    test_sampler = SequentialSampler(test_dataset) if args.local_rank == -1 else DistributedSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=1,collate_fn=collate_fn)
    # Eval!
    logger.info("***** Running prediction %s *****", prefix)
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", 1)

    results = []
    output_submit_file = os.path.join(pred_output_dir, prefix, "test_prediction.json")
    pbar = ProgressBar(n_total=len(test_dataloader), desc="Predicting")
    for step, batch in enumerate(test_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": None}
            if args.model_type != "distilbert":
                # XLM and RoBERTa don"t use segment_ids
                inputs["token_type_ids"] = (batch[2] if args.model_type in ["bert", "xlnet"] else None)
            outputs = model(**inputs)
        logits = outputs[0]
        preds = logits.detach().cpu().numpy()
        preds = np.argmax(preds, axis=2).tolist()
        preds = preds[0][1:-1] # [CLS]XXXX[SEP]
        tags = [args.id2label[x] for x in preds]
        label_entities = get_entities(preds, args.id2label, args.markup)
        json_d = {}
        json_d['id'] = step
        json_d['tag_seq'] = " ".join(tags)
        json_d['entities'] = label_entities
        results.append(json_d)
        pbar(step)
    logger.info("\n")
    with open(output_submit_file, "w") as writer:
        for record in results:
            writer.write(json.dumps(record) + '\n')

def load_and_cache_examples(args, task, tokenizer, data_type='train'):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    processor = processors[task]()
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_soft-{}_{}_{}_{}'.format(
        data_type,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.train_max_seq_length if data_type=='train' else args.eval_max_seq_length),
        str(task)))
    if False and os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if data_type == 'train':
            examples = processor.get_train_examples(args.data_dir)
        elif data_type == 'dev':
            examples = processor.get_dev_examples(args.data_dir)
        else:
            examples = processor.get_test_examples(args.data_dir)
        # print('examples here!!!!!!!: ', examples)
        # input('pause')
        features = convert_examples_to_features(examples=examples,
                                                tokenizer=tokenizer,
                                                label_list=label_list,
                                                max_seq_length=args.train_max_seq_length if data_type=='train' \
                                                               else args.eval_max_seq_length,
                                                cls_token_at_end=bool(args.model_type in ["xlnet"]),
                                                pad_on_left=bool(args.model_type in ['xlnet']),
                                                cls_token = tokenizer.cls_token,
                                                cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
                                                sep_token=tokenizer.sep_token,
                                                # pad on the left for xlnet
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
                                                )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)
    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
   
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
   
    all_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
   
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_lens,all_label_ids)
    return dataset


from data import  OSDataset_CL_once_sep
import yaml
def get_data(data_type='train'):
    cfg_path = './config/data_config.yml'
    cfg = yaml.load(open(cfg_path, encoding='utf-8'), Loader=yaml.FullLoader)

    if data_type =='train':
        file_name = "data.train/"
    elif data_type == 'dev':
        file_name = "data.valid/"
    elif data_type == 'test':
        file_name ="data.test/"

    train_set = OSDataset_CL_once_sep(cfg['data_dir'] + file_name, cfg, is_train=1)
    num_labels = len(train_set.tag_vocab.voc2id)
    id2label = train_set.tag_vocab.id2voc
    label2id = train_set.tag_vocab.voc2id
    
    task_name = train_set.task_setup[3]
    print(task_name)
    print(train_set.task_setup)
    len_train = len(train_set.all_id_set[task_name])
    
    features = train_set.all_id_set[task_name]
    all_input_ids = torch.tensor([f[0] for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f[1] for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f[2] for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f[6] for f in features], dtype=torch.long)
    all_lens = torch.tensor([f[3] for f in features], dtype=torch.long)

    
    train_dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_lens, all_label_ids)

    return train_dataset, num_labels, id2label, label2id

def get_data_all_domain(data_type='train'):
    cfg_path = './config/data_config.yml'
    cfg = yaml.load(open(cfg_path, encoding='utf-8'), Loader=yaml.FullLoader)

    if data_type=='train':
        file_name = "data.train/"
    elif data_type == 'dev':
        file_name = "data.valid/"
    elif data_type == 'test':
        file_name ="data.test/"

    train_set = OSDataset_CL_once_sep(cfg['data_dir'] + file_name, cfg, is_train=1)
    num_labels = len(train_set.tag_vocab.voc2id)
    id2label = train_set.tag_vocab.id2voc
    label2id = train_set.tag_vocab.voc2id
    
    
    return train_set, num_labels, id2label, label2id

def process_data(train_set, task_name):
    # return a domain's data specified by task_name
    
    print("\n\n\n current domain: ", task_name)
    # print(train_set.task_setup)
    len_train = len(train_set.all_id_set[task_name])
    
    features = train_set.all_id_set[task_name]
    all_input_ids = torch.tensor([f[0] for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f[1] for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f[2] for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f[6] for f in features], dtype=torch.long)
    all_lens = torch.tensor([f[3] for f in features], dtype=torch.long)

   
    
    train_dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_lens, all_label_ids)

    return train_dataset

def print_eval_matrix(eval_matrix):
    for i in range(len(eval_matrix)):
        print(eval_matrix[i])
        print("\n")


def main():

    # 1. set args and output files
    args = get_argparse().parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    args.output_dir = args.output_dir + '{}'.format(args.model_type)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    time_ = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    init_logger(log_file=args.output_dir + f'/{args.model_type}-{args.task_name}-{time_}.log')
    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    # 2. Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device
    logger.warning(
                "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                args.local_rank,device,args.n_gpu, bool(args.local_rank != -1),args.fp16,)
    # Set seed
    seed_everything(args.seed)


    # 3. Prepare NER task
  
    train_dataset, num_labels, args.id2label, args.label2id = get_data_all_domain("train")
    
    valid_dataset, _, _, _ = get_data_all_domain("dev")
    test_dataset, _, _, _ = get_data_all_domain("test")
    print("All data readin !!!!!!!")
 

    # 4. Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    args.writer = SummaryWriter("./outputs/tensorboard/" )
    config = config_class.from_pretrained(args.model_name_or_path, num_labels=num_labels)
    config.loss_type = args.loss_type
    config.adapter_layers = args.adapter_layers
    config.do_adapter = args.do_adapter

    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path,do_lower_case=args.do_lower_case,)
    model = model_class.from_pretrained(args.model_name_or_path,config=config)
    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)
    eval_matrix = []
    # Training
    projector = None
    if args.do_adapter:
        args.alpha_array = [0.9 * 0.001, 0.6]
        projector = Projector(args)
    if args.do_train:
        for ind, task_name in enumerate(train_dataset.task_setup):
            train_set = process_data(train_dataset, task_name)
            valid_set = process_data(valid_dataset, task_name)
            test_set = process_data(test_dataset, task_name)

            test_result = train(args, train_set, valid_set, test_set, model, tokenizer, projector)
            test_result["task_name"]=task_name

            logger.info(" task name: %s", task_name)
            logger.info(" test result: %s", test_result)
            logger.info("\n")

            # evaluate on past tasks
            past_results = []
            for past_ind in range(ind):
                past_task_name = train_dataset.task_setup[past_ind]
                test_set = process_data(test_dataset, past_task_name)
                test_tmp = evaluate(args, model, tokenizer, test_set, prefix="", data_type='test')
                test_tmp['task_name'] = past_task_name
                past_results.append(test_tmp)
            past_results.append(test_result)

            eval_matrix.append(past_results)

            if ind % 2 == 0:
                np.savez(args.output_dir+ '/eval_matrix.npz', eval_matrix=eval_matrix)        
                print_eval_matrix(eval_matrix)
           
        np.savez(args.output_dir+ '/eval_matrix.npz', eval_matrix=eval_matrix)        
        print_eval_matrix(eval_matrix)
        logger.info(" eval matrix saved ")
    


if __name__ == "__main__":
    main()
