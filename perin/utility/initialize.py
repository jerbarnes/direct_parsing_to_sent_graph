import random
import torch


def initialize(args, init_wandb: bool):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if init_wandb:
        import wandb
        print(args.framework, args.language, flush=True)
        tags = args.framework, args.language
        wandb.init(name=f"{args.framework}_{args.language}_{args.graph_mode}_{args.name}", dir="../../wandb", mode="offline", config=args, project="sentiment_graphs", tags=list(tags))
        # args.get_hyperparameters().save("config.json")
        # wandb.save("config.json")
        print("Connection to Weights & Biases initialized.", flush=True)
