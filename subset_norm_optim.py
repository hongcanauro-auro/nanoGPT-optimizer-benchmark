"""
Subset-Norm Optimizer Implementation
Based on: "Lean and Mean Adaptive Optimization via Subset-Norm and Subspace-Momentum"
Paper: https://arxiv.org/abs/2411.07120
"""

import torch
from torch.optim.optimizer import Optimizer
import math


class AdamSN(Optimizer):
    """
    Adam with Subset-Norm adaptive step sizes.
    
    Reduces memory footprint from O(d) to O(sqrt(d)) for the second moment term
    by using row-wise or column-wise norms instead of element-wise second moments.
    
    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups
        lr: learning rate (default: 1e-3)
        betas: coefficients used for computing running averages of gradient
               and its square (default: (0.9, 0.999))
        eps: term added to the denominator to improve numerical stability (default: 1e-8)
        weight_decay: weight decay coefficient (default: 0.01)
        subset_size: size of subsets for subset-norm. 
                    - 'heuristics': use dimension-based heuristic (d/2 for 2D, full for 1D)
                    - int: specific subset size
                    - -1: use heuristics (default)
        correct_bias: whether to use bias correction (default: True)
    """
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01, subset_size=-1, correct_bias=True):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
            
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                       subset_size=subset_size, correct_bias=correct_bias)
        super(AdamSN, self).__init__(params, defaults)
    
    def _get_reduce_dim(self, grad, subset_size):
        """Determine which dimension to reduce for subset-norm."""
        if len(grad.shape) < 2:
            return None
        
        # For 2D parameters (like Linear layers)
        if len(grad.shape) == 2:
            m, n = grad.shape
            
            if subset_size == 'heuristics' or subset_size == -1:
                # Use heuristic: reduce along the larger dimension
                # This gives us sqrt(d) memory instead of d
                reduce_dim = 0 if m >= n else 1
            elif isinstance(subset_size, int) and subset_size > 0:
                # Use specified subset size
                # Reduce along dimension that gives closest to subset_size groups
                dim0_groups = (m + subset_size - 1) // subset_size
                dim1_groups = (n + subset_size - 1) // subset_size
                reduce_dim = 0 if dim0_groups <= dim1_groups else 1
            else:
                # Default: reduce along larger dimension
                reduce_dim = 0 if m >= n else 1
                
            return reduce_dim
        
        return None
    
    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('AdamSN does not support sparse gradients')
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponentially moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p)
                    
                    # Determine reduction dimension for subset-norm
                    reduce_dim = self._get_reduce_dim(grad, group['subset_size'])
                    state['reduce_dim'] = reduce_dim
                    
                    # Initialize second moment with reduced size
                    if reduce_dim is not None:
                        # For 2D: reduce to row-wise or column-wise norms
                        second_moment_shape = list(grad.shape)
                        second_moment_shape[1 - reduce_dim] = 1
                        state['exp_avg_sq'] = torch.zeros(
                            second_moment_shape, dtype=p.dtype, device=p.device
                        )
                    else:
                        # For 1D or other shapes: use full second moment
                        state['exp_avg_sq'] = torch.zeros_like(p)
                
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                reduce_dim = state['reduce_dim']
                
                state['step'] += 1
                
                # Decay the first moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Compute second moment update
                if reduce_dim is not None:
                    # Subset-norm: sum of squares along one dimension
                    second_moment_update = torch.sum(
                        grad ** 2, dim=(1 - reduce_dim), keepdim=True
                    )
                else:
                    # Standard coordinate-wise second moment
                    second_moment_update = grad ** 2
                
                # Decay the second moment running average coefficient
                exp_avg_sq.mul_(beta2).add_(second_moment_update, alpha=1 - beta2)
                
                # Compute step size
                step_size = group['lr']
                
                if group['correct_bias']:
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1
                
                # Compute adaptive learning rate
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                
                # Compute normalized gradient (with broadcasting for subset-norm)
                norm_grad = exp_avg / denom
                
                # Apply weight decay (decoupled weight decay as in AdamW)
                if group['weight_decay'] > 0:
                    p.mul_(1 - group['lr'] * group['weight_decay'])
                
                # Update parameters
                p.add_(norm_grad, alpha=-step_size)
        
        return loss


class AdamSNSM(Optimizer):

    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01, subset_size='heuristics', rank=128,
                 proj_type='svd', update_proj_gap=200, correct_bias=True):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                       subset_size=subset_size, rank=rank, proj_type=proj_type,
                       update_proj_gap=update_proj_gap, correct_bias=correct_bias)
        super(AdamSNSM, self).__init__(params, defaults)
    
    def _get_reduce_dim(self, grad, subset_size):
        """Determine which dimension to reduce for subset-norm."""
        if len(grad.shape) < 2:
            return None
        
        if len(grad.shape) == 2:
            m, n = grad.shape
            
            if subset_size == 'heuristics' or subset_size == -1:
                reduce_dim = 0 if m >= n else 1
            elif isinstance(subset_size, int) and subset_size > 0:
                dim0_groups = (m + subset_size - 1) // subset_size
                dim1_groups = (n + subset_size - 1) // subset_size
                reduce_dim = 0 if dim0_groups <= dim1_groups else 1
            else:
                reduce_dim = 0 if m >= n else 1
                
            return reduce_dim
        
        return None
    
    def _init_subspace_projection(self, grad, rank, proj_type='svd'):
        """Initialize projection for subspace momentum."""
        if len(grad.shape) != 2:
            return None, None, None
        
        m, n = grad.shape
        # Choose projection dimension
        proj_dim = 0 if m > n else 1
        dim_size = m if proj_dim == 0 else n
        
        # Limit rank to dimension size
        actual_rank = min(rank, dim_size)
        
        if proj_type == 'svd':
            # Use SVD-based projection (computed on first gradient)
            # For efficiency, we'll compute this lazily on first use
            projector = None
        else:  # random projection
            # Random orthogonal projection
            random_matrix = torch.randn(
                dim_size, actual_rank, dtype=grad.dtype, device=grad.device
            )
            projector, _ = torch.linalg.qr(random_matrix)
        
        return projector, proj_dim, actual_rank
    
    def _update_projection(self, grad, projector, proj_dim, rank):
        """Update projection using SVD of gradient."""
        if len(grad.shape) != 2:
            return projector
        
        # Compute SVD
        try:
            if proj_dim == 0:
                # Project rows
                U, S, Vh = torch.linalg.svd(grad, full_matrices=False)
                projector = U[:, :rank]
            else:
                # Project columns
                U, S, Vh = torch.linalg.svd(grad.T, full_matrices=False)
                projector = U[:, :rank]
        except:
            # If SVD fails, keep old projector
            pass
        
        return projector
    
    def _project_momentum(self, momentum, projector, proj_dim):
        """Project momentum to subspace."""
        if projector is None or len(momentum.shape) != 2:
            return momentum
        
        if proj_dim == 0:
            # Project rows: M_proj = P^T @ M
            momentum_proj = torch.matmul(projector.T, momentum)
        else:
            # Project columns: M_proj = M @ P
            momentum_proj = torch.matmul(momentum, projector)
        
        return momentum_proj
    
    def _reconstruct_momentum(self, momentum_proj, projector, proj_dim, shape):
        """Reconstruct momentum from subspace."""
        if projector is None or len(shape) != 2:
            return momentum_proj
        
        if proj_dim == 0:
            # Reconstruct: M = P @ M_proj
            momentum = torch.matmul(projector, momentum_proj)
        else:
            # Reconstruct: M = M_proj @ P^T
            momentum = torch.matmul(momentum_proj, projector.T)
        
        return momentum
    
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('AdamSNSM does not support sparse gradients')
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    
                    # Subset-norm setup
                    reduce_dim = self._get_reduce_dim(grad, group['subset_size'])
                    state['reduce_dim'] = reduce_dim
                    
                    if reduce_dim is not None:
                        second_moment_shape = list(grad.shape)
                        second_moment_shape[1 - reduce_dim] = 1
                        state['exp_avg_sq'] = torch.zeros(
                            second_moment_shape, dtype=p.dtype, device=p.device
                        )
                    else:
                        state['exp_avg_sq'] = torch.zeros_like(p)
                    
                    # Subspace-momentum setup
                    projector, proj_dim, rank = self._init_subspace_projection(
                        grad, group['rank'], group['proj_type']
                    )
                    state['projector'] = projector
                    state['proj_dim'] = proj_dim
                    state['rank'] = rank
                    
                    # Initialize momentum in subspace
                    if projector is not None and len(grad.shape) == 2:
                        # Create momentum in reduced subspace
                        if proj_dim == 0:
                            momentum_shape = (rank, grad.shape[1])
                        else:
                            momentum_shape = (grad.shape[0], rank)
                        state['exp_avg'] = torch.zeros(
                            momentum_shape, dtype=p.dtype, device=p.device
                        )
                    else:
                        # Full momentum for non-2D parameters
                        state['exp_avg'] = torch.zeros_like(p)
                
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                reduce_dim = state['reduce_dim']
                projector = state.get('projector')
                proj_dim = state.get('proj_dim')
                rank = state.get('rank')
                
                state['step'] += 1
                
                # Update projection periodically
                if (projector is not None and 
                    state['step'] % group['update_proj_gap'] == 0 and
                    group['proj_type'] == 'svd'):
                    projector = self._update_projection(grad, projector, proj_dim, rank)
                    state['projector'] = projector
                
                # Project gradient to subspace for momentum
                if projector is not None and len(grad.shape) == 2:
                    grad_proj = self._project_momentum(grad, projector, proj_dim)
                    # Update momentum in subspace
                    exp_avg.mul_(beta1).add_(grad_proj, alpha=1 - beta1)
                    # Reconstruct full momentum
                    exp_avg_full = self._reconstruct_momentum(
                        exp_avg, projector, proj_dim, grad.shape
                    )
                else:
                    # Standard momentum update
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_full = exp_avg
                
                # Compute second moment update (subset-norm)
                if reduce_dim is not None:
                    second_moment_update = torch.sum(
                        grad ** 2, dim=(1 - reduce_dim), keepdim=True
                    )
                else:
                    second_moment_update = grad ** 2
                
                exp_avg_sq.mul_(beta2).add_(second_moment_update, alpha=1 - beta2)
                
                # Compute step size
                step_size = group['lr']
                
                if group['correct_bias']:
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1
                
                # Compute adaptive learning rate
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                
                # Compute normalized gradient
                norm_grad = exp_avg_full / denom
                
                # Apply weight decay (decoupled)
                if group['weight_decay'] > 0:
                    p.mul_(1 - group['lr'] * group['weight_decay'])
                
                # Update parameters
                p.add_(norm_grad, alpha=-step_size)
        
        return loss
