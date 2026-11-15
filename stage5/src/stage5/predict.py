from torch import Tensor

from stage5.model import Mlp


def predict(model: Mlp, x: Tensor) -> Tensor:
    return model(x).argmax(dim=-1)
