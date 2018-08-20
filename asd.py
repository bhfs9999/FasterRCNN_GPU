def test(a):
    assert a == 1

try:
    test(12)
except:
    print('2')