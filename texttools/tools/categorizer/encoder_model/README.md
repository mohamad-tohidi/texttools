it will take an object with the `.encode()` method on it

to vectorize the input text

and it would take the Enum like this example


```python

from enum import Enum
import numpy as np

class ProductCategory(Enum):
    CAR     = ("car", [
        [0.12, 0.34, 0.56, …],   # prototype vector 1 for “car”
        [0.22, 0.14, 0.76, …],   # prototype vector 2 (optional)
    ])
    PHONE   = ("phone", [[0.91, 0.05, 0.33, …]])
    LAPTOP  = ("laptop", [[0.47, 0.86, 0.12, …]])

    def __init__(self, value: str, embeddings: list[list[float]]):
        # store the textual value
        self._value_ = value
        # convert to numpy arrays immediately
        self.embeddings = [np.array(v, dtype=float) for v in embeddings]



```