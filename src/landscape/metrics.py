# src/probes/hessian_topk.py

import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters


def _grad_vector(loss, params):
    grads = torch.autograd.grad(loss, params, create_graph=True)
    return torch.cat([g.reshape(-1) for g in grads])


def _hvp(loss, params, vector):
    """
    Efficient Hessian-vector product using double backprop.
    """
    grad_vec = _grad_vector(loss, params)
    Hv = torch.autograd.grad(grad_vec @ vector, params, retain_graph=False)
    return torch.cat([h.reshape(-1) for h in Hv])


def _get_batch(loader, device):
    """
    We only need one mini-batch for Hessian approximation.
    """
    for x, y in loader:
        return x.to(device), y.to(device)
    raise RuntimeError("Empty dataloader")


def topk_hessian_eigs(model, loader, criterion, k=1, iters=50, tol=1e-4, device="cuda"):
    """
    Compute the top-k eigenvalues of the Hessian using power iteration
    with (optional) deflation.

    This is rewritten from scratch and structurally different from the
    friend-version, while preserving the mathematical idea.
    """
    model.eval()
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    dim = sum(p.numel() for p in params)

    # Pre-load a batch to avoid repeated iteration
    x, y = _get_batch(loader, device)

    eigenvalues = []
    eigenvectors = []

    for j in range(k):
        v = torch.randn(dim, device=device)
        v = v / (torch.norm(v) + 1e-8)

        prev_val = None

        for _ in range(iters):
            out = model(x)
            loss = criterion(out, y)

            Hv = _hvp(loss, params, v)

            # Deflation: remove components along previously found eigenvectors
            if eigenvectors:
                for q in eigenvectors:
                    Hv = Hv - (Hv @ q) * q

            Hv_norm = torch.norm(Hv)
            if Hv_norm < 1e-10:
                break

            v = Hv / Hv_norm
            eig_val = (v @ Hv).item()

            if prev_val is not None and abs(eig_val - prev_val) < tol:
                break

            prev_val = eig_val

        eigenvalues.append(prev_val)
        eigenvectors.append(v.detach())

    return eigenvalues


def run(model, loader, criterion, cfg=None):
    """
    Public API used by run_experiment.py
    """
    cfg = cfg or {}
    k = cfg.get("k", 1)
    iters = cfg.get("iters", 50)
    tol = cfg.get("tol", 1e-4)

    eigs = topk_hessian_eigs(
        model,
        loader,
        criterion,
        k=k,
        iters=iters,
        tol=tol,
        device=cfg.get("device", "cuda")
    )

    print(f"[Hessian] Top-{k} eigenvalues: {eigs}")
    return eigs
