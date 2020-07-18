"""
Module with RangeList class for manipulating a list of range intervals.

This is used to cache the track occupancy so we can keep cache updating
when user manipulates tracks for a range of instances.
"""

from typing import List, Tuple


class RangeList:
    """
    Class for manipulating a list of range intervals.
    Each range interval in the list is a [start, end)-tuple.
    """

    def __init__(self, range_list: List[Tuple[int]] = None):
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

    @property
    def is_empty(self):
        """Returns True if the list is empty."""
        return len(self.list) == 0

    @property
    def start(self):
        """Return the start value of range (or None if empty)."""
        if self.is_empty:
            return None
        return self.list[0][0]

    @property
    def end(self):
        """Returns the end value of range (or None if empty)."""
        if self.is_empty:
            return None
        return self.list[-1][1]

    def add(self, val, tolerance=0):
        """Add a single value, merges to last range if contiguous."""
        if self.list and self.list[-1][1] + tolerance >= val:
            self.list[-1] = (self.list[-1][0], val + 1)
        else:
            self.list.append((val, val + 1))

    def insert(self, new_range: tuple):
        """Add a new range, merging to adjacent/overlapping ranges as appropriate."""
        new_range = self._as_tuple(new_range)

        pre, _, post = self.cut_range(new_range)
        self.list = self.join_([pre, [new_range], post])
        return self.list

    def insert_list(self, range_list: List[Tuple[int]]):
        """Add each range from a list of ranges."""
        for range_ in range_list:
            self.insert(range_)
        return self.list

    def remove(self, remove: tuple):
        """Remove everything that overlaps with given range."""
        pre, _, post = self.cut_range(remove)
        self.list = pre + post

    def cut(self, cut: int):
        """Return a pair of lists with everything before/after cut."""
        return self.cut_(self.list, cut)

    def cut_range(self, cut: tuple):
        """Return three lists, everthing before/within/after cut range."""
        if not self.list:
            return [], [], []
        cut = self._as_tuple(cut)

        a, r = self.cut_(self.list, cut[0])
        b, c = self.cut_(r, cut[1])

        return a, b, c

    @staticmethod
    def _as_tuple(x):
        """Return tuple (converting from range if necessary)."""
        if isinstance(x, range):
            return x.start, x.stop
        return x

    @staticmethod
    def cut_(range_list: List[Tuple[int]], cut: int):
        """Return a pair of lists with everything before/after cut.
        Args:
            range_list: the list to cut
            cut: the value at which to cut list
        Returns:
            (pre-cut list, post-cut list)-tuple
        """
        pre = []
        post = []

        for range_ in range_list:
            if range_[1] <= cut:
                pre.append(range_)
            elif range_[0] >= cut:
                post.append(range_)
            elif range_[0] < cut < range_[1]:
                # two new ranges, split at cut
                a = (range_[0], cut)
                b = (cut, range_[1])
                pre.append(a)
                post.append(b)
        return pre, post

    @classmethod
    def join_(cls, list_list: List[List[Tuple[int]]]):
        """Return a single list that includes all lists in input list.

        Args:
            list_list: a list of range lists
        Returns:
            range list that joins all of the lists in list_list
        """
        if len(list_list) == 1:
            return list_list[0]
        if len(list_list) == 2:
            return cls.join_pair_(list_list[0], list_list[1])
        return cls.join_pair_(list_list[0], cls.join_(list_list[1:]))

    @staticmethod
    def join_pair_(list_a: List[Tuple[int]], list_b: List[Tuple[int]]):
        """Return a single pair of lists that joins two input lists."""
        if not list_a or not list_b:
            return list_a + list_b

        last_a = list_a[-1]
        first_b = list_b[0]
        if last_a[1] >= first_b[0]:
            return list_a[:-1] + [(last_a[0], first_b[1])] + list_b[1:]

        return list_a + list_b
