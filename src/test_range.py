from range import Range

def test_range_update():
    r = Range()

    r.update(9)
    assert r.upper == 9

    r.update(-1)
    assert r.lower == -1

    r.update(5)
    assert r.lower == -1
    assert r.upper == 9

def test_range_information_loss():
    r = Range(5, 10)
    I = Range(0, 20)

    assert r / I == 0.25

def test_range_divide_by_zero():
    r = Range(42, 42)
    I = Range(42, 42)

    assert r / I == 0

def test_within_range():
    r = Range(5, 10)
    assert r.within_bounds(7)

def test_out_of_range():
    r = Range(5, 10)
    assert not r.within_bounds(11)
