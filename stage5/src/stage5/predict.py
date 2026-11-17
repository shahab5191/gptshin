from torch import Tensor

from stage5.model import Model


def predict(model: Model, x: Tensor) -> Tensor:
    return model(x).argmax(dim=-1)
