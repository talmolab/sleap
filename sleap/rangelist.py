class RangeList():
    """
    Class for manipulating a list of range intervals.
    Each range interval in the list is a [start, end)-tuple.
    """

    def __init__(self, range_list: list=None):
        self.list = range_list if range_list is not None else []

    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, self._list)

    @property
    def list(self):
        """Returns the list of ranges."""
        return self._list

    @list.setter
    def list(self, val):
        """Sets the list of ranges."""
        self._list = val
#         for i, r in enumerate(self._list):
#             if type(r) == tuple:
#                 self._list[i] = range(r[0], r[1])

    @property
    def is_empty(self):
        """Returns True if the list is empty."""
        return len(self.list) == 0

    def add(self, val, tolerance=0):
        """Adds a single value, merges to last range if contiguous."""
        if len(self.list) and self.list[-1][1] + tolerance >= val:
            self.list[-1] = (self.list[-1][0], val+1)
        else:
            self.list.append((val, val+1))

    def insert(self, new_range: tuple):
        """Adds a new range, merging to adjacent/overlapping ranges as appropriate."""
        new_range = self._as_tuple(new_range)

        pre, within, post = self.cut_range(new_range)
        self.list = self.join_([pre, [new_range], post])
        return self.list

    def insert_list(self, range_list: list):
        """Adds each range from a list of ranges."""
        for range_ in range_list:
            self.insert(range_)
        return self.list

    def remove(self, remove: tuple):
        """Removes everything that overlaps with given range."""
        pre, within, post = self.cut_range(remove)
        self.list = pre + post

    def cut(self, cut: int):
        """Returns a pair of lists with everything before/after cut."""
        return self.cut_(self.list, cut)

    def cut_range(self, cut: tuple):
        """Returns three lists, everthing before/within/after cut range."""
        if len(self.list) == 0: return [], [], []
        cut = self._as_tuple(cut)

        a, r = self.cut_(self.list, cut[0])
        b, c = self.cut_(r, cut[1])

        return a, b, c

    @staticmethod
    def _as_tuple(x):
        if type(x) == range: return x.start, x.stop
        return x

    @staticmethod
    def cut_(range_list: list, cut: int):
        pre = []
        post = []

        for range_ in range_list:
            if range_[1] <= cut:
                pre.append(range_)
            elif range_[0] >= cut:
                post.append(range_)
            elif range_[0] < cut and range_[1] > cut:
                # two new ranges, split at cut
                a = (range_[0], cut)
                b = (cut, range_[1])
                pre.append(a)
                post.append(b)
        return pre, post

    @classmethod
    def join_(cls, list_list: list):
        if len(list_list) == 1: return list_list[0]
        if len(list_list) == 2: return cls.join_pair_(list_list[0], list_list[1])
        else: return cls.join_pair_(list_list[0], cls.join_(list_list[1:]))

    @staticmethod
    def join_pair_(list_a: list, list_b: list):
        if len(list_a) == 0 or len(list_b) == 0: return list_a + list_b
        
        last_a = list_a[-1]
        first_b = list_b[0]
        if last_a[1] >= first_b[0]:
            return list_a[:-1] + [(last_a[0], first_b[1])] + list_b[1:]
        else:
            return list_a + list_b
