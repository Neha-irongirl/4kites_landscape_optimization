import torch
import torch.nn as nn
import torch.nn.functional as F


def get_weights(model: nn.Module):
    """
    Returns a list of parameter clones.
    Used in tests to verify cloning works (modifying clone should not modify model).
    """
    return [p.detach().clone() for p in model.parameters()]


def set_weights(model: nn.Module, weights):
    """
    Loads provided weights into the model.
    """
    with torch.no_grad():
        for p, w in zip(model.parameters(), weights):
            p.copy_(w)


def evaluate_model(model, data_loader, device="cpu"):
    """
    Compute average loss across a dataset.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            batch_size = y.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / total_samples


def get_1d_interpolation(
    model,
    data_loader,
    start_weights,
    end_weights,
    steps=10,
    device="cpu",
):
    """
    Evaluates the model along a linear interpolation between
    start_weights and end_weights.

    After completion, restores the model back to start_weights
    (required for unit test).
    """
    model.to(device)

    # Ensure model starts at start_weights
    set_weights(model, start_weights)

    losses = []

    for step in range(steps):
        alpha = step / (steps - 1) if steps > 1 else 0
        interpolated = [
            (1 - alpha) * s + alpha * e
            for s, e in zip(start_weights, end_weights)
        ]

        set_weights(model, interpolated)
        loss = evaluate_model(model, data_loader, device)
        losses.append(loss)

    # Restore original weights for test validation
    set_weights(model, start_weights)

    return losses
