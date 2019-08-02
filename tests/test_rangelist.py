from sleap.rangelist import RangeList

def test_rangelist():
    a = RangeList([(1,2),(3,5),(7,13),(50,100)])

    assert a.list == [(1, 2), (3, 5), (7, 13), (50, 100)]
    assert a.cut(8) == ([(1, 2), (3, 5), (7, 8)], [(8, 13), (50, 100)])
    assert a.cut_range((60,70)) == ([(1, 2), (3, 5), (7, 13), (50, 60)], [(60, 70)], [(70, 100)])
    assert a.insert((10,20)) == [(1, 2), (3, 5), (7, 20), (50, 100)]
    assert a.insert((5,8)) == [(1, 2), (3, 20), (50, 100)]

    a.remove((5,8))
    assert a.list == [(1, 2), (3, 5), (8, 20), (50, 100)]
    
    b = RangeList()
    b.add(1)
    b.add(2)
    b.add(4)
    b.add(5)
    b.add(6)
    b.add(9)
    b.add(10)

    assert b.list == [(1, 3), (4, 7), (9, 11)]