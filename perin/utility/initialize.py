import random
import torch
import os


def seed_everything(seed_value=42):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def initialize(args, init_wandb: bool):
    seed_everything(args.seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False

    if init_wandb:
        import wandb
        print(args.framework, args.language, flush=True)
        tags = args.framework, args.language
        wandb.init(name=f"{args.framework}_{args.language}_{args.graph_mode}_{args.name}", dir="../../wandb", mode="offline", config=args, project="sentiment_graphs", tags=list(tags))
        print("Connection to Weights & Biases initialized.", flush=True)
