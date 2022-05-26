import imp
from posixpath import commonpath
from doc import RelationBatchSampler
import torch
import json
import torch.backends.cudnn as cudnn
import os
from doc import Dataset, RelationBatchSampler
from config import args
from trainer import Trainer
from logger_config import logger
from doc import Dataset, collate, rel_gen_collate
from torch.utils.data import Sampler, DataLoader
import random


def main():
    ngpus_per_node = torch.cuda.device_count()
    cudnn.benchmark = True

    logger.info("Use {} gpus for training".format(ngpus_per_node))

    trainer = Trainer(args, ngpus_per_node=ngpus_per_node)
    logger.info('Args={}'.format(json.dumps(args.__dict__, ensure_ascii=False, indent=4)))
    trainer.train_loop()


if __name__ == '__main__':
    ngpus_per_node = torch.cuda.device_count()
    # trainer = Trainer(args, ngpus_per_node=ngpus_per_node)
    train_dataset = Dataset(path=args.train_path, task=args.task)
    sampler = RelationBatchSampler(batch_size=args.batch_size, commonsense_path= args.commonsense_path, cake_ratio = args.cake_ratio, ds_info=train_dataset.ds_info())
    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler = sampler,
            collate_fn=collate,
            num_workers=args.workers,
            pin_memory=True)
    next(iter(train_loader))
    # main()
   