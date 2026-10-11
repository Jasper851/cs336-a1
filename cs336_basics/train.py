import torch
import torch.nn as nn
from collections.abc import Callable, Iterable
from typing import Optional
import math

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    shifted_logits = logits - torch.max(logits,dim=-1,keepdim=True).values
    log_sum_exp = torch.log(torch.sum(torch.exp(shifted_logits),dim=-1))
    targets_ex = targets.unsqueeze(-1)
    target_logits = torch.gather(shifted_logits,dim=-1,index=targets_ex).squeeze(-1)
    loss = log_sum_exp - target_logits
    return torch.mean(loss)



class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr = 1e-3, betas = (0.9,0.999), eps = 1e-8, weight_decay = 0.01):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = {"lr": lr, "beta1": betas[0], "beta2": betas[1], "eps": eps, "weight_decay": weight_decay}
        super().__init__(params,defaults)


    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        with torch.no_grad():
            for group in self.param_groups:
                lr = group["lr"]
                beta1 = group["beta1"]
                beta2 = group["beta2"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    grad = p.grad.data
                    t = state.get("t", 0)
                    t += 1
                    state["t"] = t
                    m = state.get("m", torch.zeros_like(p))
                    v = state.get("v", torch.zeros_like(p))
                    m.mul_(beta1).add_((1-beta1) * grad)
                    v.mul_(beta2).addcmul_(grad, grad, value = 1-beta2)
                    state["m"] = m
                    state["v"] = v
                    alpha_t = lr * (1 - beta2 ** t) ** 0.5 / (1 - beta1 ** t)
                    p.mul_(1 - lr * weight_decay)
                    p.addcdiv_(m, v ** 0.5 + eps, value = -alpha_t)
        return loss



def get_lr_cosine_schedule(t, alpha_max, alpha_min, T_w, T_c):
    if t < T_w:
        alpha_t = t * alpha_max / T_w
    elif t > T_c:
        alpha_t = alpha_min
    else:
        alpha_t = alpha_min + 0.5 * (1 + math.cos((t - T_w) * math.pi / (T_c - T_w))) * (alpha_max - alpha_min)
    return alpha_t



def gradient_clipping(params, M) -> None:
    eps = 1e-6
    gradients = [p.grad for p in params if p.grad is not None]
    if not gradients:
        return
    with torch.no_grad():
        l2_norm = torch.zeros((),device=gradients[0].device,dtype=gradients[0].dtype)
        for gradient in gradients:
            l2_norm += torch.sum(gradient ** 2)
        l2_norm = torch.sqrt(l2_norm)
        if l2_norm > M:
            scale = M / (l2_norm + eps)
            for gradient in gradients:
                gradient.mul_(scale)


                