import random
import torch


def initialize(args, init_wandb: bool):
    random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if init_wandb:
        import wandb
        tags = {x for f in args.frameworks for x in f}
        wandb.init(name=args.name, dir="../../wandb", mode="offline", config=args.get_hyperparameters(), project="sentiment_graphs", tags=list(tags))
        # args.get_hyperparameters().save("config.json")
        # wandb.save("config.json")
        print("Connection to Weights & Biases initialized.", flush=True)
