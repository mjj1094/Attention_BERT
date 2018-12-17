#coding=utf8
import os
import sys
import re
import argparse
import math
import timeit
import numpy
import random
from subprocess import *
import pickle
import codecs
from conf import *
from buildTree import get_info_from_file
import utils
from torch import nn
import torch.autograd as autograd
import logging
from data_generater import *
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert import BertModel
print("PID", os.getpid(), file=sys.stderr)
random.seed(0)
numpy.random.seed(0)
torch.manual_seed(args.random_seed)
torch.cuda.manual_seed(args.random_seed)
torch.cuda.set_device(args.gpu)
import pickle
sys.setrecursionlimit(1000000)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
MAX = 2

def generate_vec_bert(data_path):
    zp_vec_index = 0
    zp_sent_mask_bert = []

    read_f = open(data_path + "zp_info", "rb")
    zp_info_test = pickle.load(read_f)
    read_f.close()

    vectorized_sentences_bert = numpy.load(data_path + "sent_cls_output_bert.npy")
    for zp, candi_info in zp_info_test:
        index_in_file, sentence_index, zp_index, ana = zp
        if ana == 1:
            zp_sent_mask_bert.append(vectorized_sentences_bert[index_in_file])
            zp_vec_index += 1

    zp_sent_mask_bert = numpy.array(zp_sent_mask_bert, dtype='int32')
    numpy.save(data_path + "zp_sent_cls_output_bert.npy", zp_sent_mask_bert)

def get_bert_out(output_path):
    parser = argparse.ArgumentParser()

    parser.add_argument("--layers", default="-1", type=str)
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences longer "
                            "than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for predictions.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help = "local_rank for distributed training on gpus")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")

    args = parser.parse_args()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {} distributed training: {}".format(device, n_gpu, bool(args.local_rank != -1)))

    model = BertModel.from_pretrained('/home/miaojingjing/data/chinese_L-12_H-768_A-12')
    model.to(device)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    model.eval()
    zp_sent_bert = numpy.load(output_path + "sen_bert.npy")
    zp_sent_mask_bert = numpy.load(output_path + "sen_mask_bert.npy")

    num=0
    outs=[]
    all_input_ids = torch.tensor(zp_sent_bert, dtype=torch.int64).to(device)
    all_input_mask = torch.tensor(zp_sent_mask_bert, dtype=torch.int64).to(device)
    all_example_index=torch.tensor(list(range(len(zp_sent_bert))), dtype=torch.int64).to(device)
    eval_data = TensorDataset(all_input_ids, all_input_mask,all_example_index)
    if args.local_rank == -1:
        eval_sampler = SequentialSampler(eval_data)
    else:
        eval_sampler = DistributedSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.batch_size)
    for input_ids, input_mask,example_indices in eval_dataloader:
        all_encoder_layers, _ = model(input_ids, token_type_ids=None, attention_mask=input_mask)
        all_encoder_layers = all_encoder_layers
        num+=len(zp_sent_bert)
        for b, example_index in enumerate(example_indices):
            layer_output = all_encoder_layers[-1].detach().cpu().numpy()  # last layer
            layer_output = layer_output[b]  # sent b
            out = [round(x.item(), 6) for x in layer_output[0]]  # [CLS]
            outs.append(out)
    outs = numpy.array(outs)
    numpy.save(output_path+"sent_cls_output_bert.npy", outs)

def getNumZpNp(data_path):
    read_f = open(data_path + "zp_info", "rb")
    zp_info_test = pickle.load(read_f)
    read_f.close()

    zp_vec_index = 0
    candi_index = 0
    for zp, candi_info in zp_info_test:
        index_in_file, sentence_index, zp_index, ana = zp
        if ana == 1:
            for candi_index_in_file, candi_sentence_index, candi_begin, candi_end, res, target, ifl in candi_info:
                candi_index += 1
            zp_vec_index += 1
    return zp_vec_index,candi_index
if __name__ == "__main__":
    # zp_vec_index,candi_index=getNumZpNp("./data/train/")
    # print(zp_vec_index*4*512/1024/1024/1024)#12090，0.023059844970703125
    # print(candi_index*40*4*256/1024/1024/1024)#1811960，6.912078857421875
    # zp_vec_index1, candi_index1 = getNumZpNp("./data/test/")
    # print(zp_vec_index1*4*512/1024/1024/1024)#1711，0.0032634735107421875
    # print(candi_index1*40*4*256/1024/1024/1024)#22545，0.8600234985351562
    # zp_neicun=(zp_vec_index+zp_vec_index1)*4*512/1024/1024/1024
    # candi_neicun=(candi_index+candi_index1)*40*4*256/1024/1024/1024
    # print(zp_neicun)#0.026323318481445312
    # print(candi_neicun)#7.772102355957031

    m = nn.MaxPool1d(2)
    input = torch.randn(2,2,3)
    output = m(input)
    print("sda")