"""PyTorch helpers."""


def clip_gradient(model, clip=5):
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.data.clamp_(-clip, clip)
