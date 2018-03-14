from random import random
from math import log


def importance_ran(a, exs):
    return random()


def importance_exp(a, exs):
    pos_ex = [e for e in exs if e.target == 1]
    neg_ex = [e for e in exs if e.target == 2]
    p = len(pos_ex)
    n = len(neg_ex)

    entropy_remainder = 0
    for possible in a.pos_values:
        pk = len([e for e in pos_ex if e.attributes[a.id].val == possible])
        nk = len([e for e in neg_ex if e.attributes[a.id].val == possible])
        if pk == 0 or nk == 0:
            return 1
        q = pk/(pk+nk)
        entropy_remainder += (( pk+nk )/( p+n ))*b(q)

    return b(p/(p+n)) - entropy_remainder


def b(q):
    return -(q*log(q,2) + (1-q) * log(q,2) * (1-q))
