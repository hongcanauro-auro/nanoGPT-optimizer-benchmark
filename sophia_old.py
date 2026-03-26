import torch
from torch.optim.optimizer import Optimizer

class SophiaG(Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.965, 0.99), rho=0.04,
                 weight_decay=1e-1, *, maximize=False, capturable=False):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid lr: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta[0]: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta[1]: {betas[1]}")
        if not 0.0 <= rho:
            raise ValueError(f"Invalid rho: {rho}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        defaults = dict(lr=lr, betas=betas, rho=rho,
                        weight_decay=weight_decay, maximize=maximize,
                        capturable=capturable)
        super(SophiaG, self).__init__(params, defaults)

    @torch.no_grad()
    def update_hessian(self):
        for group in self.param_groups:
            beta2 = group['betas'][1]
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = torch.zeros((1,), dtype=torch.float,
                                                  device=p.device)
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                if 'hessian' not in state.keys():
                    state['hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state['hessian'].mul_(beta2).addcmul_(p.grad, p.grad, value=(1 - beta2) * bs)

    @torch.no_grad()
    def step(self, closure=None, bs=5120):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            hessians = []
            state_steps = []
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError('SophiaG does not support sparse gradients')
                grads.append(p.grad)
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = torch.zeros((1,), dtype=torch.float,
                                                  device=p.device)
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                if 'hessian' not in state.keys():
                    state['hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                exp_avgs.append(state['exp_avg'])
                hessians.append(state['hessian'])
                state_steps.append(state['step'])
            self._single_tensor_step(params_with_grad, grads, exp_avgs,
                                     hessians, state_steps, bs=bs,
                                     beta1=beta1, beta2=beta2,
                                     rho=group['rho'], lr=group['lr'],
                                     weight_decay=group['weight_decay'],
                                     maximize=group['maximize'],
                                     capturable=group['capturable'])
        return loss

    def _single_tensor_step(self, params, grads, exp_avgs, hessians,
                            state_steps, *, bs, beta1, beta2, rho, lr,
                            weight_decay, maximize, capturable):
        for i, param in enumerate(params):
            grad = grads[i] if not maximize else -grads[i]
            exp_avg = exp_avgs[i]
            hessian = hessians[i]
            step_t = state_steps[i]
            if capturable:
                assert param.is_cuda and step_t.is_cuda
            step_t += 1
            if weight_decay != 0:
                grad = grad.add(param, alpha=weight_decay)
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            update_ratio = exp_avg / (rho * bs * hessian + 1e-12)

            update_ratio_clipped = torch.clamp(update_ratio, -1, 1)

            param.add_(update_ratio_clipped, alpha=-lr)