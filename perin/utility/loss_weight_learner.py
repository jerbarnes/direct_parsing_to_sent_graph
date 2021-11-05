import torch
import torch.nn.functional as F

from utility.schedule.linear_lr import LinearLr


class LossWeightLearner:
    def __init__(self, args, model, steps_per_epoch: int):
        self.loss_weights = model.head.loss_weights
        self.loss_keys = list(self.loss_weights.keys())
        self.optimizer = torch.optim.SGD(list(self.loss_weights.values()), lr=args.grad_norm_lr)
        self.scheduler = LinearLr(self.optimizer.param_groups[0], args.grad_norm_lr, args.epochs * steps_per_epoch, True)
        self.loss_0 = model.head.loss_0
        self.last_layer = [p for p in model.decoder.layers[-1].parameters() if p.requires_grad]
        self.preference_weights = model.head.preference_weights
        self.alpha = args.grad_norm_alpha
        self.accumulation_steps = args.accumulation_steps
        self.accumulated_grads = 0.0

    def compute_grad(self, losses, epoch: int):
        self.loss_weights.zero_grad()

        grads = {}
        for name, loss in losses.items():
            grads[name] = torch.cat([
                g.flatten()
                for g in torch.autograd.grad(loss, self.last_layer, retain_graph=True)
            ])

        grads = torch.cat([
            (torch.norm(grads[key], 2) / self.loss_weights[key]).detach() * self.loss_weights[key]
            for key in self.loss_keys
        ])

        losses = torch.stack([losses[key] for key in self.loss_keys])
        losses_0 = torch.stack([self.loss_0[key] for key in self.loss_keys]).to(losses.device)

        with torch.no_grad():
            preferences = torch.tensor([self.preference_weights[key] for key in self.loss_keys], device=losses.device)
            target = losses / losses_0 * preferences
            target.div_(target.mean())
            target.pow_(self.alpha).mul_(grads.mean())

        grad_norm_loss = F.l1_loss(grads, target, reduction="sum")
        with torch.no_grad():
            grads = torch.autograd.grad(grad_norm_loss, [self.loss_weights[key] for key in self.loss_keys])
            grads = torch.cat(grads)

        self.accumulated_grads += grads

    @torch.no_grad()
    def step(self, epoch: int):
        for i, key in enumerate(self.loss_keys):
            self.loss_weights[key].grad = self.accumulated_grads[i].unsqueeze(0).clone()
        self.accumulated_grads.zero_()

        self.scheduler(epoch)
        self.optimizer.step()

        normalize_coeff = 1.0 / torch.stack(list(self.loss_weights.values())).abs().sum()
        for key in self.loss_keys:
            self.loss_weights[key].data = self.loss_weights[key].data.abs() * normalize_coeff
