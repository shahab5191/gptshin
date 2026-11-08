from torch import Tensor

from stage4.model import Mlp


def predict(model: Mlp, x: Tensor) -> Tensor:
    return model(x).argmax(dim=1)
