"""
some safety check function
"""
def check_eq(a, b, msg='not equal'):
    assert a == b, msg

def check_gt(a, b, msg='not greater'):
    assert a > b, msg

def check_lt(a, b, msg='not less'):
    assert a < b, msg

def check_gte(a, b, msg='not bigger or equal'):
    assert a >= b, msg

def check_lte(a, b, msg='not less or equal'):
    assert a <= b, msg




