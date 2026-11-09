from stage4.chained import Chained
from stage4.deep import Deep

# every model has block_size, tokens_per_frame, device and forward(x, phase)
Model = Chained | Deep
