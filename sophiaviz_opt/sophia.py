"""
BASED ON THE OFFICIAL SOPHIA SECOND-ORDER OPTIMIZER:
https://github.com/Liuhong99/Sophia


THIS: SophiaViz is optimized for optimizing images - NOT weights.
Created for use with CLIP Attention Head Max Visualization.
zer0int
https://github.com/zer0int

SophiaViz was vibecoded by GPT-5.
Not sure if the low-pass makes much sense :-), but the exposed
parameters are otherwise meaningfully influencing the outcome
(activation / attention head max visualization image), alas...

"""
import math
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from typing import List, Optional, Callable, Literal

HVPClosure = Optional[Callable[[], torch.Tensor]]  # should return scalar loss J(z)


class SophiaViz(Optimizer):
    r"""
    Sophia-Viz: curvature-aware, per-coordinate clipped optimizer specialized for feature visualization.

    Key additions vs your SophiaG:
      - Hutchinson diagonal Hessian (optionally) every k steps via hvp_closure
      - PSD surrogate & proximal curvature floor tau for mode-locking
      - gamma controls clip fraction directly (ratio = m / (gamma*h + eps))
      - h_subsample to cheapen HVP and reduce VRAM

    Args:
        params: iterable of parameters to optimize (e.g., Fourier/patch params z)
        lr: learning rate
        betas: (beta1, beta2) for EMA(m) and EMA(h)
        weight_decay: decoupled weight-decay on params
        k: refresh period for Hessian/diag-curvature
        hessian_mode: "hutchinson" or "grad_sqr"
        h_subsample: float in (0,1], Bernoulli mask prob for Hutchinson HVP
        gamma: per-coordinate clip scaling (lower => more clipping)
        tau: proximal curvature floor (adds to diag h) for mode-locking
        abs_hessian: if True, take |u ⊙ H u| to enforce PSD surrogate
        eps: numerical floor in denominator
        maximize: if True, maximize the closure objective (standard PyTorch flag)
        capturable: CUDA graph capture support
    """
    def __init__(
        self,
        params,
        lr: float = 3e-2,
        betas=(0.96, 0.99),
        weight_decay: float = 0.0,
        *,
        k: int = 10,
        hessian_mode: Literal["hutchinson","grad_sqr"] = "hutchinson",
        h_subsample: float = 0.25,
        gamma: float = 0.01,
        tau: float = 0.0,
        abs_hessian: bool = True,
        eps: float = 1e-12,
        maximize: bool = False,
        capturable: bool = False,
        u_lowpass: bool = False,
        u_lowpass_cutoff: float = 0.25,
        num_hutch: int = 1,
        u_dist: Literal["gaussian","rademacher"] = "gaussian"
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if k < 1:
            raise ValueError("k must be >= 1")
        if not (0.0 < h_subsample <= 1.0):
            raise ValueError("h_subsample must be in (0,1]")
        if gamma <= 0.0:
            raise ValueError("gamma must be > 0")

        defaults = dict(
            lr=lr, betas=betas, weight_decay=weight_decay,
            k=k, hessian_mode=hessian_mode, h_subsample=h_subsample,
            gamma=gamma, tau=tau, abs_hessian=abs_hessian, eps=eps,
            maximize=maximize, capturable=capturable,
            u_lowpass=u_lowpass,
            u_lowpass_cutoff=float(u_lowpass_cutoff),
            num_hutch=int(num_hutch),
            u_dist=u_dist,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('maximize', False)
            group.setdefault('capturable', False)
        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(state_values[0].get('step', None))
        if not step_is_tensor:
            for s in state_values:
                if 'step' in s:
                    s['step'] = torch.tensor(float(s['step']))

    @staticmethod
    def _fft_lowpass_2d(x: torch.Tensor, cutoff: float) -> torch.Tensor:
        """
        Low-pass filter x over its last two dims (H, W) via FFT.
        cutoff is a fraction of Nyquist (0<cutoff<=0.5). Returns real tensor with unit std.
        """
        if x.ndim < 3:  # not image-like; do nothing
            return x
        H, W = x.shape[-2], x.shape[-1]
        # build frequency radius mask
        fy = torch.fft.fftfreq(H, d=1.0, device=x.device).view(H, 1)
        fx = torch.fft.fftfreq(W, d=1.0, device=x.device).view(1, W)
        r = torch.sqrt(fy * fy + fx * fx)  # 0..~0.5
        mask = (r <= cutoff).to(x.dtype)

        X = torch.fft.fftn(x, dim=(-2, -1))
        X = X * mask
        x_lp = torch.fft.ifftn(X, dim=(-2, -1)).real

        # normalize variance so Hutchinson scaling stays comparable
        s = x_lp.std()
        if torch.isfinite(s) and s > 0:
            x_lp = x_lp / s
        return x_lp

    @torch.no_grad()
    def _init_state_if_needed(self, p, device):
        state = self.state[p]
        if len(state) == 0:
            state['step'] = torch.zeros((1,), dtype=torch.float, device=device) \
                if self.defaults['capturable'] else torch.tensor(0.)
            state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
            state['hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)
        else:
            if 'exp_avg' not in state:
                state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
            if 'hessian' not in state:
                state['hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)
        return state

    @torch.no_grad()
    def update_hessian_grad_sqr(self):  # EF-style surrogate
        """Diagonal curvature refresh via squared gradients (cheap, PSD)."""
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self._init_state_if_needed(p, p.device)
                # EMA of "diagonal Hessian" surrogate using grad^2
                state['hessian'].mul_(beta2).addcmul_(p.grad, p.grad, value=(1 - beta2))

    @torch.no_grad()
    def _fallback_grad_sqr_for_group(self, group):
        beta1, beta2 = group['betas']
        for p in group['params']:
            if p.grad is None:
                continue
            state = self._init_state_if_needed(p, p.device)
            state['hessian'].mul_(beta2).addcmul_(p.grad, p.grad, value=(1 - beta2))

    @torch.no_grad()
    def update_hessian_hutchinson(self, hvp_closure):
        if hvp_closure is None:
            self.update_hessian_grad_sqr()
            return

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            h_subsample = group['h_subsample']
            abs_hessian = group['abs_hessian']
            u_lowpass = bool(group.get('u_lowpass', False))
            u_cut = float(group.get('u_lowpass_cutoff', 0.25))
            num_hutch = max(1, int(group.get('num_hutch', 1)))
            u_dist = group.get('u_dist', 'gaussian')

            params_req = [p for p in group['params'] if p.requires_grad]
            if not params_req:
                continue

            # 1) Build differentiable loss
            with torch.enable_grad():
                J = hvp_closure()
            if not (isinstance(J, torch.Tensor) and J.requires_grad):
                self._fallback_grad_sqr_for_group(group)
                continue

            # 2) First derivatives w.r.t. all params (reused across all probes)
            with torch.enable_grad():
                grads = torch.autograd.grad(
                    J, params_req, create_graph=True, retain_graph=True, allow_unused=True
                )

            # Prepare accumulators for averaged diagonal estimate
            accum_h = [torch.zeros_like(p) for p in params_req]
            valid_draws = 0

            # 3) Multiple iid probes
            for t in range(num_hutch):
                us, dot = [], None
                for g, p in zip(grads, params_req):
                    if g is None or (not g.requires_grad):
                        us.append(None)
                        continue

                    # Sample Hutchinson vector
                    if u_dist == "rademacher":
                        u = torch.empty_like(p).bernoulli_(0.5).mul_(2).add_(-1)  # ±1
                    else:
                        u = torch.randn_like(p)  # gaussian

                    # Optional Fourier low-pass (only for spatial tensors)
                    if u_lowpass and p.ndim >= 3 and p.shape[-2] >= 8 and p.shape[-1] >= 8:
                        u = self._fft_lowpass_2d(u, cutoff=u_cut)

                    # Optional Bernoulli subsample
                    if h_subsample < 1.0:
                        m = (torch.rand_like(p) < h_subsample).to(p.dtype)
                        u = u * m

                    us.append(u)
                    term = (g * u).sum()
                    dot = term if dot is None else (dot + term)

                if dot is None or (not dot.requires_grad):
                    # Degenerate draw; try next
                    continue

                # 4) HVP for this draw
                with torch.enable_grad():
                    hvps = torch.autograd.grad(
                        dot, params_req,
                        retain_graph=(t + 1 < num_hutch),  # keep graph until the last draw
                        create_graph=False, allow_unused=True
                    )

                # 5) Accumulate diagonal estimate u ⊙ (H u)
                for i, (p, u, hvp) in enumerate(zip(params_req, us, hvps)):
                    if u is None or hvp is None:
                        continue
                    h_hat = u * hvp
                    if abs_hessian:
                        h_hat = h_hat.abs()
                    accum_h[i].add_(h_hat)

                valid_draws += 1

            # 6) Finalize average (fallback if nothing valid)
            if valid_draws == 0:
                self._fallback_grad_sqr_for_group(group)
                continue

            inv_valid = 1.0 / float(valid_draws)
            for p, h_acc in zip(params_req, accum_h):
                state = self._init_state_if_needed(p, p.device)
                state['hessian'].mul_(beta2).add_(h_acc, alpha=(1 - beta2) * inv_valid)


    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None, *, hvp_closure: HVPClosure = None):
        """
        Standard PyTorch Optimizer.step.
        Optionally pass hvp_closure (callable returning scalar J) to enable Hutchinson HVP on refresh steps.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            hessian = []
            state_steps = []

            beta1, beta2 = group['betas']
            gamma = group['gamma']
            tau = group['tau']
            eps = group['eps']
            k = group['k']

            # gather
            for p in group['params']:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError('SophiaViz does not support sparse gradients')
                state = self._init_state_if_needed(p, p.device)
                params_with_grad.append(p)
                grads.append(p.grad if not group['maximize'] else -p.grad)
                exp_avgs.append(state['exp_avg'])
                hessian.append(state['hessian'])
                state_steps.append(state['step'])

            # refresh diagonal curvature every k steps
            if len(params_with_grad) > 0:
                # We check the first state's step counter as representative
                t_next = state_steps[0] + 1
                refresh_now = bool(int(t_next.item()) % k == 1)
                if refresh_now:
                    if group['hessian_mode'] == "hutchinson":
                        self.update_hessian_hutchinson(hvp_closure)
                    else:
                        self.update_hessian_grad_sqr()

            _sophiaviz_single_tensor(
                params_with_grad, grads, exp_avgs, hessian, state_steps,
                lr=group['lr'], beta1=beta1, beta2=beta2, weight_decay=group['weight_decay'],
                gamma=gamma, tau=tau, eps=eps,
                maximize=group['maximize'], capturable=group['capturable']
            )

        return loss


def _sophiaviz_single_tensor(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    hessian: List[Tensor],
    state_steps: List[Tensor],
    *,
    lr: float,
    beta1: float,
    beta2: float,
    weight_decay: float,
    gamma: float,
    tau: float,
    eps: float,
    maximize: bool,
    capturable: bool
):
    for i, p in enumerate(params):
        grad = grads[i]
        exp_avg = exp_avgs[i]
        hess = hessian[i]
        step_t = state_steps[i]

        if capturable:
            assert p.is_cuda and step_t.is_cuda

        if torch.is_complex(p):
            # Maintain behavior for complex, though viz params will be real
            grad = torch.view_as_real(grad)
            exp_avg = torch.view_as_real(exp_avg)
            hess = torch.view_as_real(hess)
            p_real = torch.view_as_real(p)
            p_ref = p_real
        else:
            p_ref = p

        # update step
        step_t += 1

        # decoupled weight decay
        if weight_decay != 0.0:
            p_ref.mul_(1 - lr * weight_decay)

        # EMA of gradient (Sophia numerator)
        exp_avg.mul_(beta1).add_(grad, alpha=(1 - beta1))

        # Effective diagonal curvature with proximal floor
        h_eff = (hess + tau).clamp_min(eps)

        # Preconditioned per-coordinate update, then elementwise clipping
        upd = exp_avg / (gamma * h_eff)
        upd = upd.clamp(min=-1.0, max=1.0)

        # Parameter update (descent on J; pass maximize=True if your closure returns activation)
        p_ref.add_(upd, alpha=-lr)
