
import time
import torch
import math
import numpy as np
from sklearn import datasets as datasets
import ot
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import geomloss


#@title

import numpy as np
import torch
import scipy.optimize


# Objective function value and gradient for L-BFGS-B (SciPy version)
def obj_grad_scipy(alpha, *args):
    # args = (sinkhorn_dual,)
    sinkhorn_dual = args[0]
    obj, grad = sinkhorn_dual.forward_numpy(alpha)
    return obj, grad

# Objective function value and gradient for L-BFGS-B (LBFGS++ version)
def obj_grad_lbfgspp(alpha, args):
    # args = (sinkhorn_dual,)
    sinkhorn_dual = args[0]
    obj, grad = sinkhorn_dual.forward_numpy(alpha)
    return obj, grad

# L-BFGS solver for Sinkhorn dual problem (PyTorch version)
def get_pytorch_solver(sinkhorn_dual, options):
    dtype = sinkhorn_dual.M.dtype
    device = sinkhorn_dual.M.device
    lbfgs = BFGSSolver(sinkhorn_dual, sinkhorn_dual.n,
                       geps=options["gtol"], feps=options["ftol"],
                       maxiter=options["maxiter"], fpast=1,
                       dtype=dtype, device=device, verbose=options["iprint"])
    lbfgs = torch.jit.script(lbfgs)
    return lbfgs

# Solve Sinkhorn dual problem using L-BFGS-B (SciPy version)
# alpha0 is Numpy array, res["x"] is Numpy array
def solve_dual_scipy(sinkhorn_dual, alpha0, lams, options):
    opts = dict(maxiter=options["maxiter"], ftol=options["ftol"], gtol=options["gtol"],
                iprint=options["iprint"])

    with torch.no_grad():
        for lam in lams:
            # Set lambda
            sinkhorn_dual.set_lambda(lam)

            # Optimization
            res = scipy.optimize.minimize(
                obj_grad_scipy, alpha0, args=(sinkhorn_dual,),
                method="L-BFGS-B", jac=True, bounds=None, options=opts)
            alpha0 = res["x"]
    return res["x"], res["nit"], np.linalg.norm(res["jac"])

# Solve Sinkhorn dual problem using L-BFGS-B (LBFGS++ version)
# alpha0 is Numpy array, res["x"] is Numpy array
def solve_dual_lbfgspp(sinkhorn_dual, alpha0, lams, options):
    opts = dict(maxiter=options["maxiter"], ftol=options["ftol"], gtol=options["gtol"],
                verbose=(options["iprint"] > 0))
    lb = np.full_like(alpha0, -np.inf)
    ub = np.full_like(alpha0, np.inf)

    with torch.no_grad():
        for lam in lams:
            # Set lambda
            sinkhorn_dual.set_lambda(lam)

            # Optimization
            res = lbfgspp.minimize(
                obj_grad_lbfgspp, alpha0, (sinkhorn_dual,), lb, ub, opts)
            alpha0 = res["x"]
    return res["x"], res["nit"], res["grad_norm"]

# Solve Sinkhorn dual problem using L-BFGS (PyTorch version)
# alpha0 is torch.Tensor, res["x"] is torch.Tensor
def solve_dual_pytorch(sinkhorn_dual, alpha0, lams, options):
    lbfgs = get_pytorch_solver(sinkhorn_dual, options)
    with torch.no_grad():
        for lam in lams:
            sinkhorn_dual.set_lambda(lam)
            alpha0, niter, gnorm = lbfgs(alpha0)
    return alpha0, niter, gnorm

# A simple wrapper
def solve_dual(*args, solver="scipy"):
    if solver == "scipy":
        return solve_dual_scipy(*args)
    elif solver == "lbfgs++":
        return solve_dual_lbfgspp(*args)
    else:
        return solve_dual_pytorch(*args)
      
#@title
import torch
from typing import Tuple
from typing_extensions import Final

# Get the optimal beta give alpha
@torch.jit.script
def optimal_alpha(beta: torch.Tensor, wa: torch.Tensor, M: torch.Tensor, lam: float):
    D = lam * (beta.reshape(1, -1) - M)
    alpha = (wa.log() - torch.logsumexp(D, dim=1)) / lam
    return alpha

# Get the optimal beta give alpha
@torch.jit.script
def optimal_beta(alpha: torch.Tensor, wb: torch.Tensor, M: torch.Tensor, lam: float):
    D = lam * (alpha.reshape(-1, 1) - M)
    beta = (wb.log() - torch.logsumexp(D, dim=0)) / lam
    return beta

# Get the current transport plan from alpha
# alpha [n]
# M [n x m]
@torch.jit.script
def sinkhorn_plan(alpha: torch.Tensor, wb: torch.Tensor, M: torch.Tensor, lam: float, log: bool = False):
    # Get the optimal beta
    beta = optimal_beta(alpha, wb, M, lam)
    # Compute T
    logT = lam * (alpha.reshape(-1, 1) + beta.reshape(1, -1) - M)
    if log:
        return logT, alpha, beta
    T = torch.exp(logT)
    return T, alpha, beta

# Compute the objective function and gradient
# wa [n], wb [m]
@torch.jit.script
def sinkhorn_dual_obj_grad(alpha: torch.Tensor, M: torch.Tensor,
                           wa: torch.Tensor, wb: torch.Tensor, lam: float):
    # Get the transport plan
    T, _, beta = sinkhorn_plan(alpha, wb, M, lam, log=False)
    Trowsum = T.sum(dim=1)
    obj = -alpha.dot(wa) - beta.dot(wb) + Trowsum.sum() / lam
    grad = Trowsum - wa
    return float(obj), grad

# Defines the dual problem of Sinkhorn loss
class SinkhornDual(torch.nn.Module):
    n:        Final[int]
    lam:      float

    def __init__(self, wa, wb, M):
        super().__init__()
        self.wa = wa
        self.wb = wb
        self.M = M
        self.n = wa.shape[0]
        self.lam = 1.0

    @torch.jit.export
    def set_lambda(self, lam: float):
        self.lam = lam

    @torch.jit.export
    def get_plan(self, alpha: torch.Tensor, log: bool = False):
        return sinkhorn_plan(alpha, self.wb, self.M, self.lam, log=log)

    def forward(self, alpha: torch.Tensor) -> Tuple[float, torch.Tensor]:
        return sinkhorn_dual_obj_grad(alpha, self.M, self.wa, self.wb, self.lam)

    @torch.jit.ignore
    def forward_numpy(self, alpha):
        # alpha [n] is a Numpy array, first convert it to torch.Tensor
        alpha = torch.tensor(alpha, dtype=self.M.dtype, device=self.M.device)
        obj, grad = self.forward(alpha)
        # grad is a torch.Tensor, we need to convert it to a Numpy array
        return obj, grad.detach().cpu().numpy()

#@title
import torch
import torch.nn as nn

# Fully-connected layers that can be compiled by TorchScript
class Dense(nn.Module):
    def __init__(
        self, in_dim, hlayers, out_dim,
        nonlinearity=nn.ReLU(inplace=True), device=torch.device("cpu")
    ):
        super(Dense, self).__init__()
        # Layers
        layers = []
        in_size = in_dim
        for out_size in hlayers:
            linear = nn.Linear(in_size, out_size, device=device)
            # nn.init.normal_(linear.bias, 0.0, 0.01)
            # nn.init.normal_(linear.weight, 0.0, 0.01)
            layers.append(linear)
            layers.append(nonlinearity)
            in_size = out_size
        # Output layer
        linear = nn.Linear(in_size, out_dim, device=device)
        # nn.init.normal_(linear.bias, 0.0, 0.01)
        # nn.init.normal_(linear.weight, 0.0, 0.01)
        layers.append(linear)
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for module in self.layers:
            x = module(x)
        return x

#@title
import torch
import torch.nn as nn
from typing import Tuple, List
from typing_extensions import Final

class BFGSMat(nn.Module):
    # Constants
    m:     Final[int]
    # Attributes
    theta: float
    ncorr: int
    ptr:   int

    # Constructor
    def __init__(self, n, m=30, dtype=torch.float64, device=torch.device("cpu")):
        super().__init__()

        # Common arguments for tensor creation
        targs = dict(dtype=dtype, device=device)

        # Maximum number of correction vectors
        self.m = m
        # theta * I is the initial approximation to the Hessian matrix
        self.theta = 1.0
        # History of the s vectors
        self.s = torch.empty(size=(m, n), **targs)
        # History of the y vectors
        self.y = torch.empty(size=(m, n), **targs)
        # History of the s'y values
        self.ys = torch.empty(size=(m,), **targs)
        # Temporary values used in computing H * v
        self.alpha = torch.empty(size=(m,), **targs)
        # Number of correction vectors in the history, ncorr <= m
        self.ncorr = 0
        # A Pointer to locate the most recent history, 1 <= m_ptr <= m
        self.ptr = m

    @torch.jit.export
    def reset(self):
        self.theta = 1.0
        self.ncorr = 0
        self.ptr = self.m

    # Add correction vectors to the BFGS matrix
    @torch.jit.export
    def add_correction(self, s, y):
        loc = self.ptr % self.m
        self.s[loc, :] = s
        self.y[loc, :] = y
        # ys = y's = 1/rho
        ys = s.dot(y)
        self.ys[loc] = ys
        self.theta = float(y.dot(y) / ys)
        # if self.ncorr < self.m:
        #     self.ncorr += 1
        self.ncorr += int(self.ncorr < self.m)
        self.ptr = loc + 1

    # Recursive formula to compute a * H * v,
    # where a is a scalar, and v is [n x 1]
    # H0 = (1/theta) * I is the initial approximation to H
    # Algorithm 7.4 of Nocedal, J., & Wright, S. (2006). Numerical optimization.
    @torch.jit.export
    def apply_Hv(self, v: torch.Tensor, a: float):
        # L-BFGS two-loop recursion

        # Loop 1
        res = a * v
        j = self.ptr % self.m
        for i in range(self.ncorr):
            j = (j + self.m - 1) % self.m
            self.alpha[j] = self.s[j, :].dot(res) / self.ys[j]
            res -= self.alpha[j] * self.y[j, :]

        # Apply initial H0
        res /= self.theta

        # Loop 2
        for i in range(self.ncorr):
            beta = self.y[j, :].dot(res) / self.ys[j]
            res += (self.alpha[j] - beta) * self.s[j, :]
            j = (j + 1) % self.m

        return res

class BFGSSolver(nn.Module):
    # Constants
    m:          Final[int]
    geps:       Final[float]
    feps:       Final[float]
    maxiter:    Final[int]
    wolfe_decr: Final[float]
    wolfe_curv: Final[float]
    maxls:      Final[int]
    fpast:      Final[int]
    verbose:    Final[int]
    # Attributes
    fxs:        List[float]

    # Constructor
    def __init__(self, fn, dim, m=30, geps=1e-5, feps=1e-9,
                 maxiter=1000, maxls=20, fpast=1,
                 dtype=torch.float64, device=torch.device("cpu"), verbose=0):
        super().__init__()

        self.fn = fn
        self.m = m
        self.geps = geps
        self.feps = feps
        self.maxiter = maxiter
        self.wolfe_decr = 1e-4
        self.wolfe_curv = 0.9
        self.maxls = maxls
        self.fpast = fpast
        self.verbose = verbose

        # Approximation to the Hessian matrix
        self.bmat = BFGSMat(dim, m, dtype=dtype, device=device)
        # History of the objective function values
        self.fxs = [0.0 for _ in range(fpast)]

    # Line search by Nocedal and Wright (2006)
    # fn, args: user-supplied function and arguments
    # x: the current point
    # fx: the current objective function
    # grad: the current gradient
    # drt: the current moving direction
    # step: initial step size
    @torch.jit.export
    def line_search(self, x: torch.Tensor, fx: float, grad: torch.Tensor,
                    drt: torch.Tensor, step: float) -> Tuple[torch.Tensor, float, torch.Tensor]:
        expansion = 2.0
        # Save the current point and function value
        x_init = x
        fx_init = fx
        # Projection of gradient on the search direction
        dg_init = float(grad.dot(drt))
        if dg_init > 0.0:
            raise RuntimeError("the moving direction increases the objective function value")

        # Sufficient decrease condition
        test_decr = self.wolfe_decr * dg_init
        # Curvature condition
        test_curv = -self.wolfe_curv * dg_init

        # Ends of the line search range (step_lo > step_hi is allowed)
        step_lo, fx_lo, dg_lo = 0.0, fx_init, dg_init
        step_hi, fx_hi, dg_hi = 0.0, 0.0, 0.0

        # Step 1: Bracketing phase
        # Find a range guaranteed to contain a step satisfying strong Wolfe
        iter = 0
        for i in range(self.maxls):
            x = x_init + step * drt
            fx, grad = self.fn(x)
            dg = float(grad.dot(drt))
            if fx - fx_init > step * test_decr or (0 < step_lo and fx >= fx_lo):
                step_hi, fx_hi, dg_hi = step, fx, dg
                break

            if abs(dg) <= test_curv:
                return x, fx, grad

            step_hi, fx_hi, dg_hi = step_lo, fx_lo, dg_lo
            step_lo, fx_lo, dg_lo = step, fx, dg
            if dg >= 0:
                break

            step *= expansion
            iter += 1

        # Step 2: Zoom phase
        # Given a range (step_lo, step_hi) that is guaranteed to contain a valid
        # strong Wolfe step value, this step finds such a value
        for i in range(iter, self.maxls):
            # Use {fx_lo, fx_hi, dg_lo} to make a quadratic interpolation of the function,
            # and the fitted quadratic is used to estimate the minimum
            step = (fx_hi - fx_lo) * step_lo - (step_hi ** 2 - step_lo ** 2) * dg_lo / 2
            step /= (fx_hi - fx_lo) - (step_hi - step_lo) * dg_lo

            # If interpolation fails, bisection is used
            if step <= min(step_lo, step_hi) or step >= max(step_lo, step_hi):
                step = (step_lo + step_hi) / 2

            x = x_init + step * drt
            fx, grad = self.fn(x)
            dg = float(grad.dot(drt))
            if fx - fx_init > step * test_decr or fx >= fx_lo:
                if step == step_hi:
                    raise RuntimeError(
                        "the line search routine failed, possibly due to insufficient numeric precision")
                step_hi, fx_hi, dg_hi = step, fx, dg
            else:
                if abs(dg) <= test_curv:
                    break
                if step == step_lo:
                    raise RuntimeError(
                        "the line search routine failed, possibly due to insufficient numeric precision")
                step_lo, fx_lo, dg_lo = step, fx, dg

        return x, fx, grad

    def forward(self, x0: torch.Tensor) -> Tuple[torch.Tensor, int, float]:
        # Reset BFGS matrix
        self.bmat.reset()
        # Initial evaluation of fn
        x = x0.clone()
        fx, grad = self.fn(x)
        gnorm = grad.norm()
        if self.fpast > 0:
            self.fxs[0] = fx
        if self.verbose > 0:
            print(f"==> Iter 0, fx = {fx}, ||grad|| = {float(gnorm)}")

        # Early exit if the initial x is already a minimizer
        if gnorm <= self.geps or gnorm <= self.geps * float(x.norm()):
            return x, 1, gnorm

        # Initial direction
        drt = -grad
        # Initial step size
        step = 1.0 / float(drt.norm())

        # Main iterations
        iter = 1
        for k in range(1, self.maxiter):
            # Save the current x and gradient
            xold = x
            gradold = grad

            # Line search to update x, fx, and gradient
            x, fx, grad = self.line_search(x, fx, grad, drt, step)

            # New gradient norm
            gnorm = grad.norm()
            if self.verbose > 0:
                print(f"==> Iter {k}, fx = {fx}, ||grad|| = {float(gnorm)}")

            # Convergence test - gradient
            if gnorm <= self.geps or gnorm <= self.geps * x0.norm():
                return x, k, float(gnorm)
            # Convergence test - objective function value
            if self.fpast > 0:
                fxd = self.fxs[k % self.fpast]
                ftest = abs(fxd - fx) <= self.feps * max(abs(fx), abs(fxd), 1.0)
                if k >= self.fpast and ftest:
                    return x, k, float(gnorm)
                self.fxs[k % self.fpast] = fx

            # Update s and y
            # s_{k+1} = x_{k+1} - x_k
            # y_{k+1} = g_{k+1} - g_k
            self.bmat.add_correction(x - xold, grad - gradold)

            # Recursive formula to compute d = -H * g
            drt = self.bmat.apply_Hv(grad, -1.0)

            # Reset step=1.0 as initial guess for the next line search
            step = 1.0
            iter += 1

        return x, iter, float(gnorm)

# Line search by Nocedal and Wright (2006)
# fn, args: user-supplied function and arguments
# x: the current point
# fx: the current objective function
# grad: the current gradient
# drt: the current moving direction
# step: initial step size
# lsargs: a dictionary containing line search parameters
def line_search(fn, args, x, fx, grad, drt, step, lsargs):
    expansion = 2.0
    # Save the current point and function value
    x_init = x
    fx_init = fx
    # Projection of gradient on the search direction
    dg_init = grad.dot(drt).item()
    if dg_init > 0:
        raise RuntimeError("the moving direction increases the objective function value")

    # Sufficient decrease condition
    test_decr = lsargs["wolfe_decr"] * dg_init
    # Curvature condition
    test_curv = -lsargs["wolfe_curv"] * dg_init

    # Ends of the line search range (step_lo > step_hi is allowed)
    step_lo, fx_lo, dg_lo = 0.0, fx_init, dg_init

    # Step 1: Bracketing phase
    # Find a range guaranteed to contain a step satisfying strong Wolfe
    for i in range(lsargs["maxls"]):
        x = x_init + step * drt
        fx, grad = fn(x, *args)
        dg = grad.dot(drt).item()
        if fx - fx_init > step * test_decr or (0 < step_lo and fx >= fx_lo):
            step_hi, fx_hi, dg_hi = step, fx, dg
            break

        if abs(dg) <= test_curv:
            return x, fx, grad

        step_hi, fx_hi, dg_hi = step_lo, fx_lo, dg_lo
        step_lo, fx_lo, dg_lo = step, fx, dg
        if dg >= 0:
            break

        step *= expansion

    # Step 2: Zoom phase
    # Given a range (step_lo, step_hi) that is guaranteed to contain a valid
    # strong Wolfe step value, this step finds such a value
    iter = i
    for i in range(iter, lsargs["maxls"]):
        # Use {fx_lo, fx_hi, dg_lo} to make a quadratic interpolation of the function,
        # and the fitted quadratic is used to estimate the minimum
        step = (fx_hi - fx_lo) * step_lo - (step_hi ** 2 - step_lo ** 2) * dg_lo / 2
        step /= (fx_hi - fx_lo) - (step_hi - step_lo) * dg_lo

        # If interpolation fails, bisection is used
        if step <= min(step_lo, step_hi) or step >= max(step_lo, step_hi):
            step = (step_lo + step_hi) / 2

        x = x_init + step * drt
        fx, grad = fn(x, *args)
        dg = grad.dot(drt).item()
        if fx - fx_init > step * test_decr or fx >= fx_lo:
            if step == step_hi:
                raise RuntimeError("the line search routine failed, possibly due to insufficient numeric precision")
            step_hi, fx_hi, dg_hi = step, fx, dg
        else:
            if abs(dg) <= test_curv:
                break
            if step == step_lo:
                raise RuntimeError("the line search routine failed, possibly due to insufficient numeric precision")
            step_lo, fx_lo, dg_lo = step, fx, dg

    return x, fx, grad

def lbfgs(fn, x0, args=(), m=30, geps=1e-5, feps=1e-9, maxiter=1000, maxls=20, fpast=1, verbose=0):
    # Tensor configurations
    dtype = x0.dtype
    device = x0.device
    # Common arguments for tensor creation
    targs = dict(dtype=dtype, device=device)
    # Line search parameters
    lsargs = dict(wolfe_decr=1e-4, wolfe_curv=0.9, maxls=maxls)
    # Dimension of the vector
    n = x0.shape[0]
    # Approximation to the Hessian matrix
    bmat = BFGSMat(n, m, dtype=dtype, device=device)
    # History of the objective function values
    fxs = torch.empty(size=(fpast,), **targs)

    # Initial evaluation of fn
    x = x0.clone()
    fx, grad = fn(x, *args)
    gnorm = grad.norm().item()
    if fpast > 0:
        fxs[0] = fx
    if verbose > 0:
        print(f"==> Iter 0, fx = {fx:.5f}, ||grad|| = {gnorm:.5e}")

    # Early exit if the initial x is already a minimizer
    if gnorm <= geps or gnorm <= geps * x.norm().item():
        return dict(x=x, niter=1, gnorm=gnorm)

    # Initial direction
    drt = -grad
    # Initial step size
    step = 1 / drt.norm().item()

    # Main iterations
    for k in range(1, maxiter):
        # Save the current x and gradient
        xold = x
        gradold = grad

        # Line search to update x, fx, and gradient
        x, fx, grad = line_search(fn, args, x, fx, grad, drt, step, lsargs)

        # New gradient norm
        gnorm = grad.norm().item()
        if verbose > 0:
            print(f"==> Iter {k}, fx = {fx:.5f}, ||grad|| = {gnorm:.5e}")

        # Convergence test - gradient
        if gnorm <= geps or gnorm <= geps * x0.norm().item():
            return dict(x=x, niter=k, gnorm=gnorm)
        # Convergence test - objective function value
        if fpast > 0:
            fxd = fxs[k % fpast].item()
            ftest = abs(fxd - fx) <= feps * max(abs(fx), abs(fxd), 1.0)
            if k >= fpast and ftest:
                return dict(x=x, niter=k, gnorm=gnorm)
            fxs[k % fpast] = fx

        # Update s and y
        # s_{k+1} = x_{k+1} - x_k
        # y_{k+1} = g_{k+1} - g_k
        bmat.add_correction(x - xold, grad - gradold)

        # Recursive formula to compute d = -H * g
        drt = bmat.apply_Hv(grad, -1.0)

        # Reset step=1.0 as initial guess for the next line search
        step = 1.0

    return dict(x=x, niter=k, gnorm=gnorm)

#@title
import math
import numpy as np
import warnings
import torch
from torch.autograd import Function


# Check whether "weights" is a legal probability distribution
def check_weights(weights):
    return torch.all(weights >= 0.0).item() and abs(weights.sum() - 1.0) < 1e-6

def SinkhornLossL(a=None, b=None, reg=0.1, L=10):
    # Check the weights if not None
    if a is not None:
        assert a.ndim == 1
        if not check_weights(a):
            raise RuntimeError("'a' must be a probability distribution")
    if b is not None:
        assert b.ndim == 1
        if not check_weights(b):
            raise RuntimeError("'b' must be a probability distribution")

    def lossfn(M):
        # Check dimensions
        if M.ndim != 2:
            raise RuntimeError("'M' must be a matrix")
        n = M.shape[0]
        m = M.shape[1]

        # Common arguments for tensor creation
        targs = dict(dtype=M.dtype, device=M.device)

        # Generate default a or b if they are set to None
        wa = torch.ones(n, **targs) / n if a is None else a.to(**targs)
        wb = torch.ones(m, **targs) / m if b is None else b.to(**targs)

        # Gibbs kernel
        K = torch.exp(-M / reg)

        # Scaling variable
        v = torch.ones_like(wb)

        # Sinkhorn iterations
        for _ in range(L):
            u = wa / torch.mv(K, v)
            v = wb / torch.mv(K.t(), u)

        loss = torch.sum(u.unsqueeze(dim=-1) * v * K * M)
        return loss

    return lossfn

def SinkhornLoss(a=None, b=None, reg=0.1, max_iter=1000, ftol=1e-9, gtol=1e-6,
                 warmup=5, solver="scipy", verbose=-1):
    # Check the weights if not None
    if a is not None:
        assert a.ndim == 1
        if not check_weights(a):
            raise RuntimeError("'a' must be a probability distribution")
    if b is not None:
        assert b.ndim == 1
        if not check_weights(b):
            raise RuntimeError("'b' must be a probability distribution")

    class SinkhornFn(Function):
        @staticmethod
        def forward(ctx, M, return_dual=False, dual_init=None):
            # Check dimensions
            if M.ndim != 2:
                raise RuntimeError("'M' must be a matrix")
            n = M.shape[0]
            m = M.shape[1]

            # Original data type
            ctx.ori_dt = M.dtype
            # scipy.optimize.minimize requires "double" type
            M64 = M.to(dtype=torch.float64)
            # Common arguments for tensor creation
            targs = dict(dtype=torch.float64, device=M.device)

            # Generate default a or b if they are set to None
            wa = torch.ones(n, **targs) / n if a is None else a.to(**targs)
            wb = torch.ones(m, **targs) / m if b is None else b.to(**targs)

            # Initial value
            if solver == "pytorch":
                alpha0 = torch.zeros(n, **targs) if dual_init is None else dual_init.detach().to(**targs)
            else:
                alpha0 = np.zeros(n) if dual_init is None else dual_init.detach().cpu().numpy()
            lam = 1.0 / reg
            ctx.lam = lam

            # An object to represent the Sinkhorn dual problem
            sinkhorn_dual = SinkhornDual(wa, wb, M64)
            sinkhorn_dual = torch.jit.script(sinkhorn_dual)

            # Warm-up
            # Crude initial optimizations with large regularization and few iterations
            if dual_init is None and warmup > 0:
                lend = np.log10(lam)
                lstart = 0.0
                # The sequence of regularization parameters
                lams = np.logspace(lstart, lend, num=warmup + 1)
                lams = lams[:warmup]
                gtol1 = 0.001 / math.sqrt(n + m)
                options = dict(maxiter=50, ftol=1e-6, gtol=gtol1, iprint=verbose)
                alpha0, _, _ = solve_dual(sinkhorn_dual, alpha0, lams, options, solver=solver)

            # Actual optimization
            gtol2 = max(1e-7, gtol / math.sqrt(n))
            options = dict(maxiter=max_iter, ftol=ftol, gtol=gtol2, iprint=verbose)
            alpha, niter, gnorm = solve_dual(sinkhorn_dual, alpha0, [lam], options, solver=solver)
            if solver != "pytorch":
                alpha = torch.tensor(alpha, **targs)

            if gnorm > 0.001:
                warnings.warn(f"the final gradient is {gnorm}, indicating that the Sinkhorn loss may not converge")

            # Compute loss
            T, _, _ = sinkhorn_dual.get_plan(alpha)
            T = T.to(dtype=ctx.ori_dt)
            MT = M * T
            mur = MT.sum(dim=1)
            loss = mur.sum()

            # Save for backward stage
            ctx.save_for_backward(M, T, MT, mur)
            ctx.set_materialize_grads(False)

            # Whether to return the dual variables
            if return_dual:
                return loss, alpha
            return loss

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx, *grad_output):
            # The length of grad_output is determined by the number
            # of outputs in forward(). For example, if only loss
            # is returned, then grad_output has only one element.
            # If both loss and gamma are returned, then grad_output
            # contains two elements
            #
            # However, we never compute gradient for gamma, so we
            # simply ignore the second element of grad_output,
            # even if it exists
            #
            # The number of outputs of backward() is determined by
            # the number of inputs of forward(). Since return_dual
            # and dual_init are optional arguments, we simply return
            # None for them

            # Early exit if M does not require gradient
            if not ctx.needs_input_grad[0]:
                return None, None, None
            # Early exit if gradient for loss is None
            if grad_output[0] is None:
                return None, None, None

            M, T, MT, mur = ctx.saved_tensors

            wa2 = T.sum(dim=1)
            wb2 = T.sum(dim=0)
            # print((wa2 - wa).abs().max())
            # print((wb2 - wb).abs().max())
            muc = MT.sum(dim=0)
            Tt = T[:, :-1]
            D = torch.diag(wb2[:-1] + 1e-10) - torch.mm(Tt.t() / wa2, Tt)
            # print(torch.linalg.cond(D))
            sv_rhs = muc[:-1] - torch.mv(Tt.t(), mur / wa2)
            # print(sv_rhs.abs().max())
            sv = torch.linalg.solve(D, sv_rhs)
            # l = torch.linalg.cholesky(D)
            # sv = torch.cholesky_solve(sv_rhs.unsqueeze(dim=-1), l).squeeze()
            su = (mur - torch.mv(Tt, sv)) / wa2
            suv = su.reshape(-1, 1) + sv
            suv = torch.hstack((suv, su.reshape(-1, 1)))
            res = T + ctx.lam * (suv - M) * T
            return grad_output[0] * res, None, None

    return SinkhornFn.apply

  


def run_experiment(args):
  data = args.data
  device = args.device
  loss_method = args.loss
  L = args.L
  reg = args.reg

  # data = "grid9"
  # device = "cuda"
  # loss_method = "analytical"
  # L = 10
  # reg = 1.0

  dev = torch.device(device)
  config = f"{data}-{loss_method}-{L}-{reg}"

  def gaussian_grid(num_on_edge, width):
      x = np.linspace(-width / 2.0, width / 2.0, num=num_on_edge)
      X, Y = np.meshgrid(x, x)
      locs = np.hstack((X.reshape(-1, 1), Y.reshape(-1, 1)))
      return locs

  def gaussian_circle(modes, radius):
      locs = [[
          math.cos(2.0 * math.pi * i / modes) * radius,
          math.sin(2.0 * math.pi * i / modes) * radius
      ] for i in range(modes)]
      return locs

  x_dim = 2

  if data == "grid9":
      data_sampler = datasets.make_blobs
      locs = gaussian_grid(num_on_edge=3, width=6)
      data_parameters = dict(n_features=x_dim, centers=locs, cluster_std=0.2)
  elif data == "grid25":
      data_sampler = datasets.make_blobs
      locs = gaussian_grid(num_on_edge=5, width=6)
      data_parameters = dict(n_features=x_dim, centers=locs, cluster_std=0.2)
  elif data == "circle8":
      data_sampler = datasets.make_blobs
      locs = gaussian_circle(modes=8, radius=3)
      data_parameters = dict(n_features=x_dim, centers=locs, cluster_std=0.2)
  elif data == "moon":
      data_sampler = datasets.make_moons
      data_parameters = dict(noise=0.1)



  torch.manual_seed(123)
  np.random.seed(123)

  latent_dim = 5
  bs = 1000
  lr = 0.001

  net = Dense(in_dim=latent_dim, hlayers=[64, 128, 64], out_dim=x_dim,
              nonlinearity=nn.ReLU6(inplace=True), device=dev)
  net = torch.jit.script(net)
  opt = torch.optim.Adam(params=net.parameters(), lr=lr)
  # scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=nepoch, cycle_momentum=False)
  losses = []

  # Generate data outside loop
  dat = data_sampler(n_samples=bs, **data_parameters)[0].astype(np.float32)
  x = torch.tensor(dat, device=dev)
  z = torch.randn(bs, latent_dim, device=dev)

  def train(reg, nepoch, loss_method):
      lossfn = SinkhornLoss(reg=reg, warmup=0, gtol=1e-6, max_iter=1000, solver="scipy", verbose=-1)
      lossfnl = SinkhornLossL(reg=reg, L=L)
      sink_loss = geomloss.SamplesLoss("sinkhorn", p=2, blur=np.sqrt(reg))
      n = bs
      a = b= torch.ones(n, device=device)/n
      t1 = time.time()
      alpha0 = None
      for i in range(nepoch):
          zout = net(z)
          if loss_method == "analytic":
              M = torch.cdist(x, zout)
              M = torch.square(M)
              loss, alpha0 = lossfn(M, True, alpha0)
              alpha0 = alpha0.detach()
          elif loss_method == "geomloss":
            loss = sink_loss(x, zout)
          else:
            M = torch.cdist(x, zout)
            M = torch.square(M)
            loss = lossfnl(M)

          opt.zero_grad()
          loss.backward()
          opt.step()
          # scheduler.step()

          loss = loss.item()
          losses.append(loss)

          if i % 100 == 0:
              print(f"reg = {reg:.3f}, epoch {i}, loss = {loss:.6f}")
      t2 = time.time()
      print(f"Time for training: {t2 - t1} seconds")
      return t2-t1, losses

  dur, losses = train(args.reg, args.nepoch, args.loss)
  zout = net(z)
  # Compute squared 2-Wasserstein distance
  dist = ot.dist(x.detach().cpu().numpy(), zout.detach().cpu().numpy(), metric="sqeuclidean")
  a = np.ones(bs) / bs
  b = a
  w2 = ot.emd2(a, b, dist)
  print(f"W2 = {w2}")

  # Compute Sinkhorn loss
  T = ot.smooth.smooth_ot_dual(a, b, dist, reg=reg, reg_type="kl", stopThr=1e-10)
  loss = np.sum(T * dist)
  print(f"Sinkhorn loss = {loss}")

  fig = plt.figure(figsize=(24, 8))
  sub = fig.add_subplot(131)
  plt.scatter(dat[:, 0], dat[:, 1], s=5)
  plt.title("True data points", fontsize=20)
  sub = fig.add_subplot(132)
  plt.plot(losses)
  # plt.plot(losses[-1000:])
  plt.title("Loss function", fontsize=20)
  sub = fig.add_subplot(133)
  plt.scatter(zout[:, 0].detach().cpu().numpy(), zout[:, 1].detach().cpu().numpy(), s=5)
  plt.title("Generated points", fontsize=20)
  plt.show()
  plt.savefig(f"{config}.png", bbox_inches="tight")

  return dur, losses

if __name__ == "__main__":
  import argparse
  args = argparse.Namespace(device='cuda', loss='geomloss',L=10, reg=0.5, data='grid25', nepoch=2000)
  results = {}
  for data in ['grid9', 'grid25', 'circle8', 'moon']:
    args.data = data
    results[data] = {}
    for loss in ['geomloss', 'analytic']:
      results[data][loss] = {}
      args.loss = loss
      dur, losses = run_experiment(args)
      results[data][loss]['time'] = dur
      results[data][loss]['losses'] = losses
      
  import pandas as pd

  rows = []
  for data in ['grid9', 'grid25', 'circle8', 'moon']:
    for loss in ['geomloss', 'analytic']:
      rows.append( [data, loss, results[data][loss]['time'], results[data][loss]['losses'][-1]])

  df = pd.DataFrame(rows, columns=['dataset', 'loss method', 'run time', 'final loss'])
  print(df)



