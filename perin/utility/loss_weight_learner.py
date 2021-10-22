import torch
import torch.nn.functional as F
import torch.distributed as dist

from utility.schedule.inverse_sqrt_lr import InverseSqrtLr


class LossWeightLearner:
    def __init__(self, args, model, n_gpus):
        self.all_loss_weights = [head.loss_weights for head in model.heads]
        self.all_loss_keys = [list(loss_weights.keys()) for loss_weights in self.all_loss_weights]
        self.optimizer = torch.optim.SGD([l for ll in self.all_loss_weights for l in ll.values()], lr=args.grad_norm_lr)
        self.scheduler = InverseSqrtLr(self.optimizer.param_groups[0], args.grad_norm_lr, args.warmup_steps, args.encoder_delay_steps // 4)
        self.losses_0 = [head.loss_0 for head in model.heads]
        self.last_layer = [p for p in model.decoder.layers[-1].parameters() if p.requires_grad]
        self.preference_weights = [head.preference_weights for head in model.heads]
        self.n_gpus = n_gpus
        self.alpha = args.grad_norm_alpha
        self.accumulation_steps = args.accumulation_steps
        self.accumulated_grads = 0.0
        self.distributed = args.distributed

        #print("####")
        #print(self.all_loss_keys)
        #print("####")

    def compute_grad(self, all_losses, epoch: int):
        for loss_weights in self.all_loss_weights:
            loss_weights.zero_grad()

        all_grads = []
        for j, losses in enumerate(all_losses):
            if len(losses) == 0:
                grads = [torch.zeros_like(self.all_loss_weights[j][key]) for key in self.all_loss_keys[j]]
            else:
                grads = {}
                for name, loss in losses.items():
                    grads[name] = torch.cat([
                        g.flatten()
                        for g in torch.autograd.grad(loss, self.last_layer, retain_graph=True)
                    ])

                grads = torch.cat([
                    (torch.norm(grads[key], 2) / self.all_loss_weights[j][key]).detach() * self.all_loss_weights[j][key]
                    for key in self.all_loss_keys[j]
                ])

                losses = torch.stack([losses[key] for key in self.all_loss_keys[j]])
                losses_0 = torch.stack([self.losses_0[j][key] for key in self.all_loss_keys[j]]).to(losses.device)

                with torch.no_grad():
                    preferences = torch.tensor([self.preference_weights[j][key] for key in self.all_loss_keys[j]], device=losses.device)
                    target = losses / losses_0 * preferences
                    target.div_(target.mean())
                    target.pow_(self.alpha).mul_(grads.mean())

                grad_norm_loss = F.l1_loss(grads, target, reduction="sum")
                # grad_norm_loss = F.smooth_l1_loss(grads, target, reduction="sum")
                with torch.no_grad():
                    grads = torch.autograd.grad(grad_norm_loss, [self.all_loss_weights[j][key] for key in self.all_loss_keys[j]])

            all_grads.append(torch.cat(grads))

        self.accumulated_grads += torch.cat(all_grads)

    @torch.no_grad()
    def step(self, epoch: int):
        if self.distributed:
            dist.all_reduce(self.accumulated_grads)
            self.accumulated_grads.div_(self.n_gpus)

        offset = 0
        for j in range(len(self.all_loss_weights)):
            for key in self.all_loss_keys[j]:
                self.all_loss_weights[j][key].grad = self.accumulated_grads[offset].unsqueeze(0).clone()
                offset += 1
        self.accumulated_grads.zero_()

        self.scheduler(epoch)
        self.optimizer.step()

        for j, loss_weights in enumerate(self.all_loss_weights):
            normalize_coeff = 1.0 / torch.stack(list(loss_weights.values())).abs().sum()
            for key in self.all_loss_keys[j]:
                self.all_loss_weights[j][key].data = loss_weights[key].data.abs() * normalize_coeff
