"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

import minigpt4.tasks as tasks
from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank, init_distributed_mode
from minigpt4.common.logger import setup_logger
from minigpt4.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from minigpt4.common.registry import registry
from minigpt4.common.utils import now

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *
import wandb
# os.environ['WANDB_MODE'] = 'offline'  


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path",default="train_configs_llama2/224_v2_llama2_video.yaml", required=False, help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument("--job_name",default="test",type=str)
    args = parser.parse_args()

    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def get_runner_class(cfg):
    """
    Get runner class from config. Default to epoch-based runner.
    """
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base"))

    return runner_cls


def setup_environ_flags(rank):
    """Set environment flags for debugging purposes"""
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    if rank == 0:
        print(f"--> Running with torch dist debug set to detail")


def main():
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"

    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    setup_environ_flags(get_rank())
    job_id = now()
    args = parse_args()
    cfg = Config(args)
    ######################################################################################################
    # cfg.run_cfg['distributed']=False
    init_distributed_mode(cfg.run_cfg)
    setup_seeds(cfg)

    # set after in
    # it_distributed_mode() to only log on master.
    setup_logger()
    wandb.login()
    # print(wandb.run)
    cfg.pretty_print()

    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    model = task.build_model(cfg)
    flag=0
    # if not hasattr(cfg.run_cfg, 'rank') or cfg.run_cfg.rank == 0:
    #     if flag>0:
    #         print(1)
    #     else:
    #         print("project name", args.job_name)

    #         wandb.init(project="minigpt4-spatial",name=args.job_name)

    #         wandb.config = {"learning_rate": 0.0001, "epochs": 100, "batch_size": 8}
    #         wandb.watch(model)
    #         flag+=1

    # print('+++++++++++++++++')
    # print(type(model))
    # print('+++++++++++++++++')
    # print(model)
    # print('+++++++++++++++++')
    # print(model.super().device)
    # print('+++++++++++++++++')
    # print(model.device)

    runner = get_runner_class(cfg)(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    )
    runner.train()


if __name__ == "__main__":
    main()
