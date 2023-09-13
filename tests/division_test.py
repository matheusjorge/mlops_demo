import pytest

def test_ok_division():
    a = 3/2
    assert a == 1.5
    
def test_not_ok_division():
    with pytest.raises(ZeroDivisionError):
        a = 3/0