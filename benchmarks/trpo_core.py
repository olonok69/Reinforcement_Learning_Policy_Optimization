from __future__ import annotations

import torch


def flat_params(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat([param.data.view(-1) for param in model.parameters()])


def set_flat_params(model: torch.nn.Module, flat: torch.Tensor) -> None:
    offset = 0
    for param in model.parameters():
        size = param.numel()
        param.data.copy_(flat[offset:offset + size].view_as(param))
        offset += size


def flat_grad(grads: tuple[torch.Tensor | None, ...], params: tuple[torch.nn.Parameter, ...]) -> torch.Tensor:
    pieces: list[torch.Tensor] = []
    for grad, param in zip(grads, params):
        if grad is None:
            pieces.append(torch.zeros_like(param).view(-1))
        else:
            pieces.append(grad.contiguous().view(-1))
    return torch.cat(pieces)


def conjugate_gradients(
    avp_fn,
    b: torch.Tensor,
    nsteps: int,
    residual_tol: float = 1e-10,
) -> torch.Tensor:
    x = torch.zeros_like(b)
    residual = b.clone()
    direction = b.clone()
    residual_dot = torch.dot(residual, residual)

    for _ in range(nsteps):
        avp = avp_fn(direction)
        denom = torch.dot(direction, avp) + 1e-8
        alpha = residual_dot / denom
        x = x + alpha * direction
        residual = residual - alpha * avp
        new_residual_dot = torch.dot(residual, residual)
        if new_residual_dot.item() < residual_tol:
            break
        beta = new_residual_dot / (residual_dot + 1e-12)
        direction = residual + beta * direction
        residual_dot = new_residual_dot

    return x


def backtracking_line_search(
    policy: torch.nn.Module,
    prev_params: torch.Tensor,
    full_step: torch.Tensor,
    evaluate,
    max_backtracks: int = 10,
    step_fraction: float = 0.5,
    accept_ratio: float = 0.1,
) -> tuple[bool, torch.Tensor]:
    base_loss, _ = evaluate()

    for i in range(max_backtracks):
        frac = step_fraction**i
        candidate = prev_params + frac * full_step
        set_flat_params(policy, candidate)
        new_loss, new_kl = evaluate()

        actual_improve = (base_loss - new_loss).item()
        expected_improve = abs(base_loss.item()) * frac
        improve_ratio = actual_improve / (expected_improve + 1e-8)

        if actual_improve > 0.0 and improve_ratio > accept_ratio and torch.isfinite(new_kl):
            return True, candidate

    set_flat_params(policy, prev_params)
    return False, prev_params
