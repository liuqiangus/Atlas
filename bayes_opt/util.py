from turtle import xcor
import warnings, math
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize


def acq_max(ac, gp, y_max, bounds, random_state, n_warmup=10000, n_iter=10, weight_ymax=0, bnn=None):
    """
    A function to find the maximum of the acquisition function

    It uses a combination of random sampling (cheap) and the 'L-BFGS-B'
    optimization method. First by sampling `n_warmup` (1e5) points at random,
    and then running L-BFGS-B from `n_iter` (250) random starting points.

    Parameters
    ----------
    :param ac:
        The acquisition function object that return its point-wise value.

    :param gp:
        A gaussian process fitted to the relevant data.

    :param y_max:
        The current maximum known value of the target function.

    :param bounds:
        The variables bounds to limit the search of the acq max.

    :param random_state:
        instance of np.RandomState random number generator

    :param n_warmup:
        number of times to randomly sample the aquisition function

    :param n_iter:
        number of times to run scipy.minimize

    Returns
    -------
    :return: x_max, The arg max of the acquisition function.
    """

    # Warm up with random points
    x_tries = random_state.uniform(bounds[:, 0], bounds[:, 1],
                                   size=(n_warmup, bounds.shape[0]))

    usage = np.array([np.mean(np.divide(x_try, bounds[:, 1])) for x_try in x_tries]) # all usage of actions
    
    if bnn: # used for online learning part, as BNN learn the overall and the GP learn the sim2real gap
        means = bnn.predict(x_tries)
    else:
        means = None

    ys = ac(x_tries, gp=gp, y_max=y_max, usage=usage, weight_ymax=weight_ymax, base_mean=means)
    x_max = x_tries[ys.argmax()]
    max_acq = ys.max()
    # if max_acq == 0: print('.', end='')

    # Explore the parameter space more throughly
    # x_seeds = random_state.uniform(bounds[:, 0], bounds[:, 1],
    #                                size=(n_iter, bounds.shape[0]))
    # for x_try in x_seeds:
    #     # Find the minimum of minus the acquisition function
    #     res = minimize(lambda x: -ac(x.reshape(1, -1), gp=gp, y_max=y_max),
    #                    x_try.reshape(1, -1),
    #                    bounds=bounds,
    #                    method="L-BFGS-B")

    #     # See if success
    #     if not res.success:
    #         continue

    #     # Store it if better than previous minimum(maximum).
    #     try:
    #         if max_acq is None or -res.fun[0] >= max_acq:
    #             x_max = res.x
    #             max_acq = -res.fun[0]
    #     except:
    #         if max_acq is None or -res.fun >= max_acq:
    #             x_max = res.x
    #             max_acq = -res.fun

    # Clip output to make sure it lies within the bounds. Due to floating
    # point technicalities this is not always the case.
    return np.clip(x_max, bounds[:, 0], bounds[:, 1])

class UtilityFunction(object):
    """
    An object to compute the acquisition functions.
    """

    def __init__(self, kind, kappa, xi, dim, delta=0.1, kappa_decay=1, kappa_decay_delay=0, weight=0, availability= 0.9):

        self.kappa = kappa
        self._kappa_decay = kappa_decay
        self._kappa_decay_delay = kappa_decay_delay

        self.xi = xi

        self.dim = dim # dim of the inputs
        self.delta = delta

        self.weight = weight
        self.availability = availability # this is for QoE requirement

        self._iters_counter = 0

        if kind not in ['ucb', 'ei', 'poi', 'gpucb', 'ts', 'ucb_offline', 'ei_offline', 'poi_offline', 'gpucb_offline', 'ts_offline', 'dcb']:
            err = "The utility function " \
                  "{} has not been implemented, " \
                  "please choose one of ucb, ei, poi, gpucb, ts.".format(kind)
            raise NotImplementedError(err)
        else:
            self.kind = kind

    def update_parameter(self, weight):
        self.weight = weight

    def update_params(self):
        self._iters_counter += 1

        if self._kappa_decay < 1 and self._iters_counter > self._kappa_decay_delay:
            self.kappa *= self._kappa_decay

    def utility(self, x, gp, y_max, usage, weight_ymax, base_mean):
        if self.kind == 'ucb':
            return self._ucb(x, gp, self.kappa)
        if self.kind == 'ei':
            return self._ei(x, gp, y_max, self.xi)
        if self.kind == 'poi':
            return self._poi(x, gp, y_max, self.xi)
        if self.kind == 'gpucb':
            return self._gpucb(x, gp, self.dim, self.delta)
        if self.kind == 'ts':
            return self._ts(x, gp)

        if self.kind == 'ucb_offline':
            return self._ucb_offline(x, gp, self.kappa, self.weight, self.availability, usage)
        if self.kind == 'ei_offline':
            return self._ei_offline(x, gp, y_max, self.xi, self.weight, self.availability, usage, weight_ymax=weight_ymax)
        if self.kind == 'poi_offline':
            return self._poi_offline(x, gp, y_max, self.xi, self.weight, self.availability, usage, weight_ymax=weight_ymax)
        if self.kind == 'gpucb_offline':
            return self._gpucb_offline(x, gp, self.dim, self.delta, self.weight, self.availability, usage)
        if self.kind == 'ts_offline':
            return self._ts_offline(x, gp, self.weight, self.availability, usage)

        if self.kind == 'dcb':
            return self._dcb(x, gp, self.kappa, self.weight, self.availability, usage, base_mean)

    @staticmethod
    def _ucb(x, gp, kappa):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)

        return mean + kappa * std

    @staticmethod
    def _gpucb(x, gp, dim, delta):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)

        t = len(gp.y_train_) # TODO assert this can also work for BNN
        beta = 2*np.log(np.power(t,dim/2)+2*np.square(math.pi)/(3*delta))

        return mean + np.sqrt(beta) * std

    @staticmethod
    def _ei(x, gp, y_max, xi):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)
  
        a = (mean - y_max - xi)
        z = a / std
        return a * norm.cdf(z) + std * norm.pdf(z)

    @staticmethod
    def _poi(x, gp, y_max, xi):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)

        z = (mean - y_max - xi)/std
        return norm.cdf(z)

    @staticmethod
    def _ts(x, gp):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                mean, std = gp.predict(x, return_std=True, avg=1)
            except:
                mean, std = gp.predict(x, return_std=True,)
        # here, use BNN to trail once, std always 0
        return mean
    ########################################################################
    @staticmethod
    def _ucb_offline(x, gp, kappa, weight, availability, usage):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)

        mean_ = weight * (mean - availability) - usage
        std_ = weight * std if weight != 0 else std

        return mean_ + kappa * std_

    @staticmethod
    def _gpucb_offline(x, gp, dim, delta, weight, availability, usage):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)

        mean_ = weight * (mean - availability) - usage
        std_ = weight * std if weight != 0 else std

        t = len(gp.y_train_) # TODO assert this can also work for BNN
        beta = 2*np.log(np.power(t,dim/2)+2*np.square(math.pi)/(3*delta))

        return mean_ + np.sqrt(beta) * std_

    @staticmethod
    def _ei_offline(x, gp, y_max, xi, weight, availability, usage, weight_ymax=0):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)

        mean_ = weight * (mean - availability) - usage
        std_ = weight * std if weight != 0 else std

        # as we learn partial of the real target, we here to get the real y_max
        a = (mean_ - weight_ymax - xi)
        z = a / std_
        return a * norm.cdf(z) + std * norm.pdf(z)

    @staticmethod
    def _poi_offline(x, gp, y_max, xi, weight, availability, usage, weight_ymax=0):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)

        mean_ = weight * (mean - availability) - usage
        std_ = weight * std if weight != 0 else std

        # as we learn partial of the real target, we here to get the real y_max
        z = (mean_ - weight_ymax - xi)/std_
        return norm.cdf(z)

    @staticmethod
    def _ts_offline(x, gp, weight, availability, usage):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                mean, std = gp.predict(x, return_std=True, avg=1)
            except:
                mean, std = gp.predict(x, return_std=True,)

        mean_ = weight * (mean - availability) - usage
        std_ = weight * std if weight != 0 else std

        # here, use BNN to trail once, std always 0
        return mean_

    @staticmethod
    def _dcb(x, gp, kappa, weight, availability, usage, base_mean):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)

        mean_ = weight * (base_mean + mean - availability) - usage
        std_ = weight * std if weight != 0 else std

        return mean_ + kappa * std_

def load_logs(optimizer, logs):
    """Load previous ...

    """
    import json

    if isinstance(logs, str):
        logs = [logs]

    for log in logs:
        with open(log, "r") as j:
            while True:
                try:
                    iteration = next(j)
                except StopIteration:
                    break

                iteration = json.loads(iteration)
                try:
                    optimizer.register(
                        params=iteration["params"],
                        target=iteration["target"],
                    )
                except KeyError:
                    pass

    return optimizer


def ensure_rng(random_state=None):
    """
    Creates a random number generator based on an optional seed.  This can be
    an integer or another random state for a seeded rng, or None for an
    unseeded rng.
    """
    if random_state is None:
        random_state = np.random.RandomState()
    elif isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)
    else:
        assert isinstance(random_state, np.random.RandomState)
    return random_state


class Colours:
    """Print in nice colours."""

    BLUE = '\033[94m'
    BOLD = '\033[1m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    END = '\033[0m'
    GREEN = '\033[92m'
    PURPLE = '\033[95m'
    RED = '\033[91m'
    UNDERLINE = '\033[4m'
    YELLOW = '\033[93m'

    @classmethod
    def _wrap_colour(cls, s, colour):
        return colour + s + cls.END

    @classmethod
    def black(cls, s):
        """Wrap text in black."""
        return cls._wrap_colour(s, cls.END)

    @classmethod
    def blue(cls, s):
        """Wrap text in blue."""
        return cls._wrap_colour(s, cls.BLUE)

    @classmethod
    def bold(cls, s):
        """Wrap text in bold."""
        return cls._wrap_colour(s, cls.BOLD)

    @classmethod
    def cyan(cls, s):
        """Wrap text in cyan."""
        return cls._wrap_colour(s, cls.CYAN)

    @classmethod
    def darkcyan(cls, s):
        """Wrap text in darkcyan."""
        return cls._wrap_colour(s, cls.DARKCYAN)

    @classmethod
    def green(cls, s):
        """Wrap text in green."""
        return cls._wrap_colour(s, cls.GREEN)

    @classmethod
    def purple(cls, s):
        """Wrap text in purple."""
        return cls._wrap_colour(s, cls.PURPLE)

    @classmethod
    def red(cls, s):
        """Wrap text in red."""
        return cls._wrap_colour(s, cls.RED)

    @classmethod
    def underline(cls, s):
        """Wrap text in underline."""
        return cls._wrap_colour(s, cls.UNDERLINE)

    @classmethod
    def yellow(cls, s):
        """Wrap text in yellow."""
        return cls._wrap_colour(s, cls.YELLOW)
