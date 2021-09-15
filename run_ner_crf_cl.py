import glob
import logging
import os
import json
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from callback.optimizater.adamw import AdamW
from callback.lr_scheduler import get_linear_schedule_with_warmup
from callback.progressbar import ProgressBar

from tools.common import seed_everything,json_to_text
from tools.common import init_logger, logger

from transformers import WEIGHTS_NAME, BertConfig,get_linear_schedule_with_warmup,AdamW, BertTokenizer
from models.bert_for_ner import BertCrfForNer
from processors.utils_ner import get_entities
from processors.ner_seq import convert_examples_to_features
from processors.ner_seq import ner_processors as processors
from processors.ner_seq import collate_fn
from metrics.ner_metrics import SeqEntityScore
from tools.finetuning_argparse import get_argparse

from utils.general_utils import get_chunks
from tensorboardX import SummaryWriter
import numpy as np

MODEL_CLASSES = {
    ## bert ernie bert_wwm bert_wwwm_ext
    'bert': (BertConfig, BertCrfForNer, BertTokenizer),
}


def train(args, train_dataset, valid_dataset, test_dataset, model, tokenizer):
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
    bert_param_optimizer = list(model.bert.named_parameters())
    crf_param_optimizer = list(model.crf.named_parameters())
    linear_param_optimizer = list(model.classifier.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.learning_rate},
        {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': args.learning_rate},

        {'params': [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.crf_learning_rate},
        {'params': [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': args.crf_learning_rate},

        {'params': [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.crf_learning_rate},
        {'params': [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': args.crf_learning_rate}
    ]
    args.warmup_steps = int(t_total * args.warmup_proportion)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)
    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))
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
    
    pbar = ProgressBar(n_total=len(train_dataloader), desc='Training', num_epochs=int(args.num_train_epochs))
    if args.save_steps==-1 and args.logging_steps==-1:
        args.logging_steps=len(train_dataloader)
        args.save_steps = len(train_dataloader)
    for epoch in range(int(args.num_train_epochs)):
        pbar.reset()
        pbar.epoch_start(current_epoch=epoch)
        for step, batch in enumerate(train_dataloader):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            
        
            input("see len and label")
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3], "cates_ids": batch[5], "acts_ids": batch[6]}
            if args.model_type != "distilbert":
                # XLM and RoBERTa don"t use segment_ids
                inputs["token_type_ids"] = (batch[2] if args.model_type in ["bert", "xlnet"] else None)
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
            # print('loss: ', loss)
            
            # if global_step % 20 == 0:
            #     args.writer.add_scalar("crf_ner"+'/train/loss', loss, global_step)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            pbar(step, {'loss': loss.item()})
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    # print(" ")
                    if args.local_rank == -1:
                        # Only evaluate when single GPU otherwise metrics may not average well
                        # evaluate(args, model, tokenizer)
                        pass
                if False and args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)
                    tokenizer.save_vocabulary(output_dir)
                    # torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)
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
        
        args.writer.add_scalar('eval/accuracy', result['acc'], epoch)
        args.writer.add_scalar('eval/recall', result['recall'], epoch)
        args.writer.add_scalar('eval/f1', result['f1'], epoch)

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



    return  test_result


def evaluate(args, model, tokenizer, eval_dataset, prefix="",  data_type='dev'):
    metric = SeqEntityScore(args.id2label, markup=args.markup)
    eval_output_dir = args.output_dir
    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)
    # eval_dataset = load_and_cache_examples(args, args.task_name, tokenizer, data_type='dev')
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
    if isinstance(model, nn.DataParallel):
        model = model.module
    
    accs = []

    for step, batch in enumerate(eval_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if args.model_type != "distilbert":
                # XLM and RoBERTa don"t use segment_ids
                inputs["token_type_ids"] = (batch[2] if args.model_type in ["bert", "xlnet"] else None)
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            tags = model.crf.decode(logits, inputs['attention_mask'])
        if args.n_gpu > 1:
            tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu parallel evaluating
        eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1

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

        tags = tags.squeeze(0).cpu().numpy().tolist()

        

        for ind, tmp in enumerate(tags):
            pred_tags = []
            for tag in tmp:
                onehot = np.zeros(len(args.id2label))
                # if tag!=0:
                onehot[tag] = 1
                
                pred_tags.append(onehot)

            # print('tags: ', np.shape(pred_tags))
            # print('baatch_onehot_label:', np.shape(batch_onehot_label))
            # input('pause')
            
            pred_t = set(get_chunks(pred_tags, args.label2id))
            true_tags_ids = set(get_chunks(batch_onehot_label[ind], args.label2id))
            # print('true_tags_ids: ', true_tags_ids)
            # print('pred_t: ', pred_t)
            pred_correct_chunks = true_tags_ids & pred_t
            accs += [len(true_tags_ids) == len(pred_correct_chunks)]
            metric.update_chunks(label_chunks=true_tags_ids, pred_chunks=pred_t)
        # for i, label in enumerate(out_label_ids):
        #     temp_1 = []
        #     temp_2 = []
        #     for j, m in enumerate(label):
        #         if j == 0:
        #             continue
        #         elif j == input_lens[i] - 1:
        #             metric.update(pred_paths=[temp_2], label_paths=[temp_1])
        #             break
        #         else:
        #             temp_1.append(args.id2label[out_label_ids[i][j]])
        #             temp_2.append(args.id2label[tags[i][j]])
        pbar(step)
    logger.info("\n")
    eval_loss = eval_loss / nb_eval_steps
    eval_info, entity_info = metric.result()
    results = {f'{key}': value for key, value in eval_info.items()}
    results['loss'] = eval_loss
    logger.info("***** Eval results %s *****", prefix)
    info = "-".join([f' {key}: {value:.4f} ' for key, value in results.items()])
    logger.info(info)
    logger.info("***** Entity results %s *****", prefix)
    for key in sorted(entity_info.keys()):
        logger.info("******* %s results ********" % key)
        info = "-".join([f' {key}: {value:.4f} ' for key, value in entity_info[key].items()])
        logger.info(info)
    return results


def predict(args, model, tokenizer, prefix=""):
    pred_output_dir = args.output_dir
    if not os.path.exists(pred_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(pred_output_dir)
    test_dataset = load_and_cache_examples(args, args.task_name, tokenizer, data_type='test')
    # Note that DistributedSampler samples randomly
    test_sampler = SequentialSampler(test_dataset) if args.local_rank == -1 else DistributedSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=1, collate_fn=collate_fn)
    # Eval!
    logger.info("***** Running prediction %s *****", prefix)
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", 1)
    results = []
    output_predict_file = os.path.join(pred_output_dir, prefix, "test_prediction.json")
    pbar = ProgressBar(n_total=len(test_dataloader), desc="Predicting")

    if isinstance(model, nn.DataParallel):
        model = model.module
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
            tags = model.crf.decode(logits, inputs['attention_mask'])
            tags  = tags.squeeze(0).cpu().numpy().tolist()
        preds = tags[0][1:-1]  # [CLS]XXXX[SEP]
        label_entities = get_entities(preds, args.id2label, args.markup)
        json_d = {}
        json_d['id'] = step
        json_d['tag_seq'] = " ".join([args.id2label[x] for x in preds])
        json_d['entities'] = label_entities
        results.append(json_d)
        pbar(step)
    logger.info("\n")
    with open(output_predict_file, "w") as writer:
        for record in results:
            writer.write(json.dumps(record) + '\n')
    if args.task_name == 'cluener':
        output_submit_file = os.path.join(pred_output_dir, prefix, "test_submit.json")
        test_text = []
        with open(os.path.join(args.data_dir,"test.json"), 'r') as fr:
            for line in fr:
                test_text.append(json.loads(line))
        test_submit = []
        for x, y in zip(test_text, results):
            json_d = {}
            json_d['id'] = x['id']
            json_d['label'] = {}
            entities = y['entities']
            words = list(x['text'])
            if len(entities) != 0:
                for subject in entities:
                    tag = subject[0]
                    start = subject[1]
                    end = subject[2]
                    word = "".join(words[start:end + 1])
                    if tag in json_d['label']:
                        if word in json_d['label'][tag]:
                            json_d['label'][tag][word].append([start, end])
                        else:
                            json_d['label'][tag][word] = [[start, end]]
                    else:
                        json_d['label'][tag] = {}
                        json_d['label'][tag][word] = [[start, end]]
            test_submit.append(json_d)
        json_to_text(output_submit_file,test_submit)

def load_and_cache_examples(args, task, tokenizer, data_type='train'):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    processor = processors[task]()
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_crf-{}_{}_{}_{}'.format(
        data_type,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.train_max_seq_length if data_type == 'train' else args.eval_max_seq_length),
        str(task)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
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
        features = convert_examples_to_features(examples=examples,
                                                tokenizer=tokenizer,
                                                label_list=label_list,
                                                max_seq_length=args.train_max_seq_length if data_type == 'train' \
                                                    else args.eval_max_seq_length,
                                                cls_token_at_end=bool(args.model_type in ["xlnet"]),
                                                pad_on_left=bool(args.model_type in ['xlnet']),
                                                cls_token=tokenizer.cls_token,
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
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_lens, all_label_ids)
    return dataset


from data import  OSDataset_CL_once_sep
import yaml
def get_data(data_type='train'):
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
    
    task_name = train_set.task_setup[3]
    print("current domain: ", task_name)
    # print(train_set.task_setup)
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
        args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16, )
    # Set seed
    seed_everything(args.seed)

    # train_dataset, num_labels, args.id2label, args.label2id = get_data("train")
    
    # valid_dataset, _, _, _ = get_data("dev")
    # test_dataset, _, _, _ = get_data("test")
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
    config = config_class.from_pretrained(args.model_name_or_path,num_labels=num_labels,)
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,)
    model = model_class.from_pretrained(args.model_name_or_path, config=config)
    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)

    eval_matrix = []
    # Training
    if args.do_train:
        for ind, task_name in enumerate(train_dataset.task_setup):
            train_set = process_data(train_dataset, task_name)
            
            valid_set = process_data(valid_dataset, task_name)
            test_set = process_data(test_dataset, task_name)

            test_result = train(args, train_set, valid_set, test_set, model, tokenizer)
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
