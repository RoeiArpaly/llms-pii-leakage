from utils.validators import luhn_verify


def test_valid():
    assert luhn_verify("356938035643809")
    assert luhn_verify("213174120250531")
    assert luhn_verify("3590520199312676")
    assert luhn_verify("371080715633510")
    assert luhn_verify("4585115860744434064")


def test_invalid():
    assert not luhn_verify("4222222222222222")
