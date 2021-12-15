class VariableLengthLoader:
    def __init__(self, loader, num_iters):
        self.loader = loader
        self.num_iters = num_iters
        self._cur_iters = 0

    def __iter__(self):
        while True:
            for elem in self.loader:
                yield elem
                self._cur_iters += 1
                if self._cur_iters == self.num_iters:
                    self._cur_iters = 0
                    return

    def __len__(self):
        return self.num_iters
