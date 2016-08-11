# -*- coding: utf-8 -*-

"""This module provides an implementation of Adam."""

from __future__ import absolute_import

import warnings

from .base import Minimizer


class Adam(Minimizer):
    """Adaptive moment estimation optimizer. (Adam).

    Adam is a method for the optimization of stochastic objective functions.

    The idea is to estimate the first two moments with exponentially decaying
    running averages. Additionally, these estimates are bias corrected which
    improves over the initial learning steps since both estimates are
    initialized with zeros.

    The rest of the documentation follows the original paper [adam2014]_ and is
    only meant as a quick primer. We refer to the original source for more
    details, such as results on convergence and discussion of the various hyper
    parameters.

    Let :math:`f_t'(\\theta_t)` be the derivative of the loss with respect to
    the parameters at time step :math:`t`. In its
    basic form, given a step rate :math:`\\alpha`, decay terms :math:`\\beta_1`
    and :math:`\\beta_2` for the first and second moment estimates respectively
    and an offset :math:`\\epsilon` we initialise the following quantities

    .. math::
       m_0 & \\leftarrow 0 \\\\
       v_0 & \\leftarrow 0 \\\\
       t & \\leftarrow 0 \\\\

    and perform the following updates:

    .. math::
        t     & \\leftarrow t + 1 \\\\
        g_t   & \\leftarrow f_t'(\\theta_{t-1}) \\\\
        m_t   & \\leftarrow \\beta_1 \cdot g_t + (1 - \\beta_1) \cdot m_{t-1} \\\\
        v_t   &\\leftarrow  \\beta_2 \cdot g_t^2 + (1 - \\beta_2) \cdot v_{t-1}

        \\hat{m}_t  &\\leftarrow {m_t \\over (1 - (1 - \\beta_1)^t)} \\\\
        \\hat{v}_t  &\\leftarrow {v_t \\over (1 - (1 - \\beta_2)^t)} \\\\
        \\theta_t   &\\leftarrow \\theta_{t-1} - \\alpha {\\hat{m}_t \\over (\\sqrt{\\hat{v}_t} + \\epsilon)}

    As suggested in the original paper, the last three steps are optimized for
    efficiency by using:

    .. math::
        \\alpha_t  &\\leftarrow \\alpha {\\sqrt{(1 - (1 - \\beta_2)^t)} \\over (1 - (1 - \\beta_1)^t)} \\\\
        \\theta_t   &\\leftarrow \\theta_{t-1} - \\alpha_t {m_t \\over (\\sqrt{v_t} + \\epsilon)}

    The quantities in the algorithm and their corresponding attributes in the
    optimizer object are as follows.

    ======================= =================== ===========================================================
    Symbol                  Attribute           Meaning
    ======================= =================== ===========================================================
    :math:`t`               ``n_iter``          Number of iterations, starting at 0.
    :math:`m_t`             ``est_mom_1_b``     Biased estimate of first moment.
    :math:`v_t`             ``est_mom_2_b``     Biased estimate of second moment.
    :math:`\\hat{m}_t`       ``est_mom_1``       Unbiased estimate of first moment.
    :math:`\\hat{v}_t`       ``est_mom_2``       Unbiased estimate of second moment.
    :math:`\\alpha`          ``step_rate``       Step rate parameter.
    :math:`\\beta_1`         ``decay_mom1``      Exponential decay parameter for first moment estimate.
    :math:`\\beta_2`         ``decay_mom2``      Exponential decay parameter for second moment estimate.
    :math:`\\epsilon`        ``offset``          Safety offset for division by estimate of second moment.
    ======================= =================== ===========================================================

    Additionally, when the momentum argument is set to True, a Nesterov-like
    momentum step is taken for the first-order momentum [nadam2015]_.
    In this case the optimization mentioned above is not used.

    .. note::
       The use of decay parameters :math:`\\beta_1` and :math:`\\beta_2` differs
       from the definition in the original paper [adam2014]_:
       With :math:`\\beta^{\\ast}_i` referring to the parameters as defined in
       the paper, we use :math:`\\beta_i`
       with :math:`\\beta_i = 1 - \\beta^{\\ast}_i`

    .. [adam2014] Kingma, Diederik, and Jimmy Ba.
       "Adam: A Method for Stochastic Optimization."
       arXiv preprint arXiv:1412.6980 (2014).
    .. [nadam2015] Dozat, Timothy
       "Incorporating Nesterov Momentum into Adam."
       Stanford University, Tech. Rep (2015).
    """

    state_fields = 'n_iter step_rate decay_mom1 decay_mom2 step offset est_mom1_b est_mom2_b decay_mom1_pow decay_mom2_pow'.split()

    def __init__(self, wrt, fprime, step_rate=.0002,
                 decay=None,
                 decay_mom1=0.1,
                 decay_mom2=0.001,
                 momentum=False,
                 offset=1e-8, args=None):
        """Create an Adam object.

        Parameters
        ----------

        wrt : array_like
            Array that represents the solution. Will be operated upon in
            place.  ``fprime`` should accept this array as a first argument.

        fprime : callable
            Callable that given a solution vector as first parameter and *args
            and **kwargs drawn from the iterations ``args`` returns a
            search direction, such as a gradient.

        step_rate : scalar or array_like, optional [default: 1]
            Value to multiply steps with before they are applied to the
            parameter vector.

        decay_mom1 : float, optional, [default: 0.1]
            Decay parameter for the exponential moving average estimate of the
            first moment.

        decay_mom2 : float, optional, [default: 0.001]
            Decay parameter for the exponential moving average estimate of the
            second moment.

        momentum : bool, optional [default: False]
            If False, regular Adam is used, otherwise results in Adam with
            Nesterov step.

        offset : float, optional, [default: 1e-8]
            Before taking the square root of the running averages, this offset
            is added.

        args : iterable
            Iterator over arguments which ``fprime`` will be called with.
        """
        if not 0 < decay_mom1 <= 1:
            raise ValueError('decay_mom1 has to lie in (0, 1]')
        if not 0 < decay_mom2 <= 1:
            raise ValueError('decay_mom2 has to lie in (0, 1]')
        if not (1 - decay_mom1 * 2) / (1 - decay_mom2) ** 0.5 < 1:
            warnings.warn("constraint from convergence analysis for adam not "
                          "satisfied; check original paper to see if you "
                          "really want to do this.")
        if decay is not None:
            warnings.warn('decay parameter was used in a previous version of '
                          'Adam and no longer has any effect.')

        super(Adam, self).__init__(wrt, args=args)

        self.fprime = fprime
        self.step_rate = step_rate
        self.decay_mom1 = decay_mom1
        self.decay_mom2 = decay_mom2
        self.offset = offset
        self.momentum = momentum
        self.est_mom1_b = 0
        self.est_mom2_b = 0
        self.decay_mom1_pow = 1
        self.decay_mom2_pow = 1
        self.step = 0

    def _iterate(self):
        for args, kwargs in self.args:
            dm1 = self.decay_mom1
            dm2 = self.decay_mom2
            o = self.offset

            est_mom1_b_m1 = self.est_mom1_b
            est_mom2_b_m1 = self.est_mom2_b

            self.decay_mom1_pow *= (1 - dm1)
            self.decay_mom2_pow *= (1 - dm2)

            gradient = self.fprime(self.wrt, *args, **kwargs)
            self.est_mom1_b = dm1 * gradient + (1 - dm1) * est_mom1_b_m1
            self.est_mom2_b = dm2 * gradient ** 2 + (1 - dm2) * est_mom2_b_m1

            if not self.momentum:
                rate_t = self.step_rate * (1 - self.decay_mom2_pow) ** 0.5 / \
                         (1 - self.decay_mom1_pow)
                step = rate_t * self.est_mom1_b / (self.est_mom2_b ** 0.5 + o)
            else:
                est_mom1 = (1 - dm1) * self.est_mom1_b / (1 - (1 - dm1) *
                                                          self.decay_mom1_pow) \
                           + dm1 * gradient / (1 - self.decay_mom1_pow)
                est_mom2 = self.est_mom2_b / (1 - self.decay_mom2_pow)
                step = self.step_rate * est_mom1 / (est_mom2 ** 0.5 + o)

            self.wrt -= step
            self.step = step

            self.n_iter += 1

            yield {
                'n_iter': self.n_iter,
                'gradient': gradient,
                'args': args,
                'kwargs': kwargs,
            }
