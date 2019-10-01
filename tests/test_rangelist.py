from sleap.rangelist import RangeList


def test_rangelist():
    a = RangeList([(1, 2), (3, 5), (7, 13), (50, 100)])

    assert a.list == [(1, 2), (3, 5), (7, 13), (50, 100)]
    assert a.cut(8) == ([(1, 2), (3, 5), (7, 8)], [(8, 13), (50, 100)])
    assert a.cut_range((60, 70)) == (
        [(1, 2), (3, 5), (7, 13), (50, 60)],
        [(60, 70)],
        [(70, 100)],
    )

    # Test inserting range as tuple
    assert a.insert((10, 20)) == [(1, 2), (3, 5), (7, 20), (50, 100)]

    # Test insert range as range
    assert a.insert(range(5, 8)) == [(1, 2), (3, 20), (50, 100)]

    a.remove((5, 8))
    assert a.list == [(1, 2), (3, 5), (8, 20), (50, 100)]

    assert a.start == 1
    a.remove((1, 3))
    assert a.start == 3

    b = RangeList()
    b.add(1)
    b.add(2)
    b.add(4)
    b.add(5)
    b.add(6)
    b.add(9)
    b.add(10)

    assert b.list == [(1, 3), (4, 7), (9, 11)]

    empty = RangeList()
    assert empty.start is None
    assert empty.cut_range((3, 4)) == ([], [], [])

    empty.insert((1, 2))
    assert str(empty) == "RangeList([(1, 2)])"

    empty.insert_list([(1, 2), (3, 5), (7, 13), (50, 100)])
    assert empty.list == [(1, 2), (3, 5), (7, 13), (50, 100)]

    # Test special cases for helper functions
    assert RangeList.join_([(1, 2)]) == (1, 2)
    assert RangeList.join_pair_(list_a=[(1, 2)], list_b=[]) == [(1, 2)]
    assert RangeList.join_pair_(list_a=[], list_b=[(1, 2)]) == [(1, 2)]
    assert RangeList.join_pair_(list_a=[], list_b=[]) == []
