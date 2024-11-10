import os
import torch
import random
import time
import os
import numpy as np

from tqdm import tqdm

import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from models.a_llmrec_model import *
from pre_train.sasrec.utils import data_partition, SeqDataset, SeqDataset_Inference


def setup_ddp(rank, world_size):
    os.environ ["MASTER_ADDR"] = "localhost"
    os.environ ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
def train_model_phase1(args):
    print('A-LLMRec start train phase-1\n')
    if args.multi_gpu:
        world_size = torch.cuda.device_count()
        mp.spawn(train_model_phase1_, args=(world_size, args), nprocs=world_size)
    else:
        train_model_phase1_(0, 0, args)
        
def train_model_phase2(args):
    print('A-LLMRec strat train phase-2\n')
    if args.multi_gpu:
        world_size = torch.cuda.device_count()
        mp.spawn(train_model_phase2_, args=(world_size, args), nprocs=world_size)
    else:
        train_model_phase2_(0, 0, args)

def inference(args):
    print('A-LLMRec start inference\n')
    if args.multi_gpu:
        world_size = torch.cuda.device_count()
        mp.spawn(inference_, args=(world_size, args), nprocs=world_size)
    else:
        inference_(0,0,args)
  
def train_model_phase1_(rank, world_size, args):
    if args.multi_gpu:
        setup_ddp(rank, world_size)
        args.device = 'cuda:' + str(rank)
        
    model = A_llmrec_model(args).to(args.device)
    
    # preprocess data
    dataset = data_partition(args.rec_pre_trained_data, path=f'./data/amazon/{args.rec_pre_trained_data}.txt')
    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    print('user num:', usernum, 'item num:', itemnum)
    num_batch = len(user_train) // args.batch_size1
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))
    # Init Dataloader, Model, Optimizer
    train_data_set = SeqDataset(user_train, usernum, itemnum, args.maxlen)
    if args.multi_gpu:
        train_data_loader = DataLoader(train_data_set, batch_size = args.batch_size1, sampler=DistributedSampler(train_data_set, shuffle=True), pin_memory=True)
        model = DDP(model, device_ids = [args.device], static_graph=True)
    else:
        train_data_loader = DataLoader(train_data_set, batch_size = args.batch_size1, pin_memory=True)        
        
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.stage1_lr, betas=(0.9, 0.98))
    
    epoch_start_idx = 1
    T = 0.0
    model.train()
    t0 = time.time()
    for epoch in tqdm(range(epoch_start_idx, args.num_epochs + 1)):
        if args.multi_gpu:
            train_data_loader.sampler.set_epoch(epoch)
        for step, data in enumerate(train_data_loader):
            u, seq, pos, neg = data
            u, seq, pos, neg = u.numpy(), seq.numpy(), pos.numpy(), neg.numpy()
            model([u,seq,pos,neg], optimizer=adam_optimizer, batch_iter=[epoch,args.num_epochs + 1,step,num_batch], mode='phase1')
            if step % max(10,num_batch//100) ==0:
                if rank ==0:
                    if args.multi_gpu: model.module.save_model(args, epoch1=epoch)
                    else: model.save_model(args, epoch1=epoch)
        if rank == 0:
            if args.multi_gpu: model.module.save_model(args, epoch1=epoch)
            else: model.save_model(args, epoch1=epoch)

    print('train time :', time.time() - t0)
    if args.multi_gpu:
        destroy_process_group()
    return 

def train_model_phase2_(rank,world_size,args):
    if args.multi_gpu:
        setup_ddp(rank, world_size)
        args.device = 'cuda:'+str(rank)
    random.seed(0)

    model = A_llmrec_model(args).to(args.device)
    phase1_epoch = 10
    model.load_model(args, phase1_epoch=phase1_epoch)

    dataset = data_partition(args.rec_pre_trained_data, path=f'./data/amazon/{args.rec_pre_trained_data}.txt')
    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    print('user num:', usernum, 'item num:', itemnum)
    num_batch = len(user_train) // args.batch_size2
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))
    # Init Dataloader, Model, Optimizer
    train_data_set = SeqDataset(user_train, usernum, itemnum, args.maxlen)
    if args.multi_gpu:
        train_data_loader = DataLoader(train_data_set, batch_size = args.batch_size2, sampler=DistributedSampler(train_data_set, shuffle=True), pin_memory=True)
        model = DDP(model, device_ids = [args.device], static_graph=True)
    else:
        train_data_loader = DataLoader(train_data_set, batch_size = args.batch_size2, pin_memory=True, shuffle=True)
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.stage2_lr, betas=(0.9, 0.98))
    
    epoch_start_idx = 1
    T = 0.0
    model.train()
    t0 = time.time()
    for epoch in tqdm(range(epoch_start_idx, args.num_epochs + 1)):
        if args.multi_gpu:
            train_data_loader.sampler.set_epoch(epoch)
        for step, data in enumerate(train_data_loader):
            u, seq, pos, neg = data
            u, seq, pos, neg = u.numpy(), seq.numpy(), pos.numpy(), neg.numpy()
            model([u,seq,pos,neg], optimizer=adam_optimizer, batch_iter=[epoch,args.num_epochs + 1,step,num_batch], mode='phase2')
            if step % max(10,num_batch//100) ==0:
                if rank ==0:
                    if args.multi_gpu: model.module.save_model(args, epoch1=phase1_epoch, epoch2=epoch)
                    else: model.save_model(args, epoch1=phase1_epoch, epoch2=epoch)
        if rank == 0:
            if args.multi_gpu: model.module.save_model(args, epoch1=phase1_epoch, epoch2=epoch)
            else: model.save_model(args, epoch1=phase1_epoch, epoch2=epoch)
    
    print('phase2 train time :', time.time() - t0)
    if args.multi_gpu:
        destroy_process_group()
    return

from utils import mmr_rerank  # Import MMR function


#####################################################################CHANGED INFERENCE FUNTION #################################################################
import json

def inference_(rank, world_size, args):
    if args.multi_gpu:
        setup_ddp(rank, world_size)
        args.device = 'cuda:' + str(rank)
        
    model = A_llmrec_model(args).to(args.device)
    phase1_epoch = 10
    phase2_epoch = 5
    model.load_model(args, phase1_epoch=phase1_epoch, phase2_epoch=phase2_epoch)

    dataset = data_partition(args.rec_pre_trained_data, path=f'./data/amazon/{args.rec_pre_trained_data}.txt')
    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    print('user num:', usernum, 'item num:', itemnum)
    num_batch = len(user_train) // args.batch_size_infer
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))
    model.eval()
    
    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    
    user_list = []
    for u in users:
        if len(user_train[u]) < 1 or len(user_test[u]) < 1: continue
        user_list.append(u)

    inference_data_set = SeqDataset_Inference(user_train, user_valid, user_test, user_list, itemnum, args.maxlen)
    
    if args.multi_gpu:
        inference_data_loader = DataLoader(inference_data_set, batch_size=args.batch_size_infer, sampler=DistributedSampler(inference_data_set, shuffle=True), pin_memory=True)
        model = DDP(model, device_ids=[args.device], static_graph=True)
    else:
        inference_data_loader = DataLoader(inference_data_set, batch_size=args.batch_size_infer, pin_memory=True)

    all_recommendations = []  # Store recommendations for each user
    all_llm_predictions = []  # Store LLM predictions for each user
    
    for _, data in enumerate(inference_data_loader):
        u, seq, pos, neg = data
        u, seq, pos, neg = u.numpy(), seq.numpy(), pos.numpy(), neg.numpy()
        
        # Modify this section in inference_ function
        llm_predictions, recommendations, relevance_scores, embeddings = model([u, seq, pos, neg, rank], mode='generate')

        # Save llm_predictions for further evaluation
        all_llm_predictions.append(llm_predictions)
        
        # MMR Rerank Step
        top_n = args.top_n  # Customize as needed from args
        lambda_param = args.lambda_param  # Customize as per input args
        
        # MMR reranking with embeddings and relevance_scores
        final_recommendations = mmr_rerank(
            recommendations=recommendations,
            embeddings=embeddings,
            relevance_scores=relevance_scores,
            lambda_param=lambda_param,
            top_n=top_n
        )

        # Append final recommendations to the list
        all_recommendations.append(final_recommendations)
        print(f"Final MMR-based Recommendations for user {u}: {final_recommendations}")

    # Save all recommendations and LLM predictions to a JSON file
    output_data = {
        "all_recommendations": all_recommendations,
        "all_llm_predictions": all_llm_predictions
    }
    with open('./inference_output.json', 'w') as json_file:
        json.dump(output_data, json_file, indent=4)
    print("All recommendations and LLM predictions have been saved to 'inference_output.json'")
