
import numpy as np

class IdGenerator:
    _next_id = 0

    def __init__(self):
        self._next_id = np.int64(0)
        
    def generate_ids(self, vec):
        next_id = self._next_id + vec.size
        vec[:] = np.arange(self._next_id, next_id)
        self._next_id = next_id    

    def generated_so_far(self):
        return self._next_id


if __name__ == "__main__":
    id_gen = IdGenerator()
    arr = np.empty(50, dtype=np.int64)
    id_gen.generate_ids(arr)
    id_gen.generate_ids(arr)
    id_gen.generate_ids(arr)
    # id_gen.generate_ids(arr)
    # id_gen.generate_ids(arr)

    print(arr)
