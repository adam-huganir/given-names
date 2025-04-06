class multidict(dict):
    def __setitem__(self, key, value):
        if key in self:
            self[key].append(value)
        else:
            super().__setitem__(key, [value])

    def get_one(self, key, rank: int = 0):
        try:
            return self[key][rank]
        except IndexError as ie:
            raise IndexError(f"Index {rank} out of range for key {key}") from ie

