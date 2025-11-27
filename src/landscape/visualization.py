import torch
import numpy as np
import torch.nn as nn


def extract_weights(model):
    """
    Return a detached list of model parameter tensors.
    Useful for freezing initial weights during landscape probing.
    """
    return [p.detach().clone() for p in model.parameters()]


def assign_weights(model, new_params):
    """
    Overwrite model parameters with supplied tensors.

    This is safer than in-place arithmetic on model.parameters(),
    because it avoids silent autograd graph corruption.
    """
    with torch.no_grad():
        for p, src in zip(model.parameters(), new_params):
            p.copy_(src)



def generate_direction(model, seed=None):
    """
    Create a randomly sampled direction tensor for each model layer.

    This does NOT normalize by default.
    """
    if seed is not None:
        torch.manual_seed(seed)
    return [torch.randn_like(p) for p in model.parameters()]


def normalize_direction(direction, reference, mode="per_filter"):
    """
    Normalize direction tensors so that sweeping along them remains
    comparable to typical weight magnitudes.

    Args:
        direction: list of tensors, same shapes as model parameters
        reference: list of baseline weights (e.g., trained weights)
        mode: one of {"per_filter", "per_layer", "global"}

    Returns:
        list of normalized directions
    """

    if mode == "global":
        flat = torch.cat([d.view(-1) for d in direction])
        norm = flat.norm() + 1e-12
        return [d / norm for d in direction]

    normalized = []
    for d, w in zip(direction, reference):

        if mode == "per_filter" and d.dim() > 1:
            # Normalize filter-wise (out_channels wise)
            d_f = d.view(d.size(0), -1)
            w_f = w.view(w.size(0), -1)

            d_norm = d_f.norm(dim=1, keepdim=True) + 1e-12
            w_norm = w_f.norm(dim=1, keepdim=True) + 1e-12

            scaled = (d_f / d_norm) * w_norm
            normalized.append(scaled.view_as(d))

        elif mode == "per_layer":
            scaled = (d / (d.norm() + 1e-12)) * (w.norm() + 1e-12)
            normalized.append(scaled)

        else:
            normalized.append(d)

    return normalized


def evaluate(model, loader, device):
    """
    Compute (loss, accuracy) on a dataloader.
    """
    criterion = nn.CrossEntropyLoss()
    model.eval()

    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    return total_loss / total, 100.0 * correct / total



def sweep_line(model, loader, w0, w1, steps=25, device="cuda"):
    """
    Sweep along the line: w(alpha) = w0 + alpha * (w1 - w0)
    for alpha ∈ [-1, 2].

    Returns:
        alphas, losses, accuracies
    """
    direction = [b - a for a, b in zip(w0, w1)]
    alphas = np.linspace(-1, 2, steps)

    losses, accs = [], []

    for alpha in alphas:
        w_alpha = [a + alpha * d for a, d in zip(w0, direction)]
        assign_weights(model, w_alpha)
        loss, acc = evaluate(model, loader, device)
        losses.append(loss)
        accs.append(acc)

    assign_weights(model, w0)  # restore original
    return alphas, losses, accs



def sweep_contour(model, loader, center, dir_x, dir_y, steps=25, device="cuda"):
    """
    Compute a 2D landscape around the weight center:
        w(i,j) = center + X[i,j]*dir_x + Y[i,j]*dir_y

    Returns:
        X, Y, Z_loss
    """
    grid = np.linspace(-1, 1, steps)
    X, Y = np.meshgrid(grid, grid)
    Z = np.zeros_like(X)

    for i in range(steps):
        for j in range(steps):

            offset = [X[i, j] * dx + Y[i, j] * dy
                      for dx, dy in zip(dir_x, dir_y)]
            w_ij = [c + off for c, off in zip(center, offset)]

            assign_weights(model, w_ij)
            loss, _ = evaluate(model, loader, device)
            Z[i, j] = loss

    assign_weights(model, center)
    return X, Y, Z
