from __future__ import absolute_import, print_function

import itertools

from climin import Adam

from .losses import LogisticRegression
from .common import continuation


def test_adam_lr():
    obj = LogisticRegression()
    args = itertools.repeat(((obj.X, obj.Z), {}))
    opt = Adam(obj.pars, obj.fprime, step_rate=0.01, args=args)
    for i, info in enumerate(opt):
        print(obj.f(opt.wrt, obj.X, obj.Z))
        if i > 3000:
            break
    assert obj.solved(0.15), 'did not find solution'


def test_adam_continue():
    obj = LogisticRegression(n_inpt=2, n_classes=2)
    args = itertools.repeat(((obj.X, obj.Z), {}))
    opt = Adam(
        obj.pars, obj.fprime, step_rate=0.01, args=args)

    continuation(opt)
