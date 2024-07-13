import numpy as np
import scipy.optimize as opt

from .standard_bandits import DBand


def draw_from_p(p, K):
    try:
        return np.random.choice(K, p=p)
    except ValueError:
        print("Value Error in p :", p, np.sum(p))
        raise ValueError


class DivConstraints:
    r"""
    Models the probability set $\mathcal P$ in the diversity-preserving bandit setting
        - implements a method check(x) which returns a boolean stating whether x
        belongs to $\mathcal P$
        - a argmax_dot which returns an object with two attributes: self.x which
        realizes the maximum, and self.fun with the value of the maximum

    For the moment, we included two types of probability set, polytopes and spheric.
    """

    def __init__(self, K):
        pass

    def check(self, x):
        pass

    def argmax_dot(
        self, U
    ):  # returns an object with two attributes, self.x and self.fun
        pass


class PolytopeConstraints(DivConstraints):
    """
    Probability set used for diversity-preserving bandits

    The constraints attribute is a list of triples l(ower), u(pper),
    s where s is a vector of size K with zeros and ones
    """

    def __init__(self, K, constraints):

        self.K = K
        self.constraints = constraints  # iterable version
        self.A_ub = []
        self.b_ub = []
        # LP-friendly version :
        for low, up, s in self.constraints:
            self.A_ub += [-s, s]
            self.b_ub += [-low, up]
        self.A_ub = np.array(self.A_ub)
        self.A_eq = [np.ones(self.K)]
        self.b_eq = 1
        # For standard form
        self.A = []
        self.b = []
        self.__standardize()
        self.feasible = self.__feasible().x

    def __standardize(self):
        nc = len(self.constraints)
        for i, (l, u, s) in enumerate(self.constraints):
            temp = np.zeros(2 * nc + 1)
            temp[i] = 1
            self.A += [np.concatenate([temp, s])]
            self.b += [u]
            temp = np.zeros(2 * nc + 1)
            temp[2 * i] = -1
            self.A += [np.concatenate([temp, s])]
            self.b += [l]
        temp = np.zeros(2 * nc + 1)
        temp[2 * nc] = 1
        self.A += [np.concatenate([temp, np.ones(self.K)])]
        self.b += [1]

    def check(self, x, delta=0.0001):
        r = (np.abs(np.sum(x) - 1) <= delta) and all(x > -delta)
        for l, u, s in self.constraints:
            r = r and (l - delta <= np.dot(x, s) <= u + delta)
        return r

    def __feasible(self):
        return opt.linprog(
            np.zeros(self.K),
            A_ub=self.A_ub,
            b_ub=self.b_ub,
            A_eq=self.A_eq,
            b_eq=self.b_eq,
            bounds=(0, 1),
            method="revised simplex",
        )

    def projection(self, x, delta=0.00001):
        if self.check(x):
            return x
        else:
            y = np.array([max(xi, 0) for xi in x])
            for l, u, s in self.constraints:
                if np.dot(y, s) > u:
                    y = y - (np.dot(y, s) - u) * s / np.linalg.norm(s)
                elif np.dot(y, s) < l:
                    y = y - (np.dot(y, s) - l) * s / np.linalg.norm(s)
            y = y / np.sum(y)
            return self.projection(y, delta=delta)

    def argmax_dot(self, U):
        R = opt.linprog(
            -U,
            A_ub=self.A_ub,
            b_ub=self.b_ub,
            A_eq=self.A_eq,
            b_eq=self.b_eq,
            bounds=(0, 1),
            method="revised simplex",
            x0=self.feasible,
        )
        R.fun = -R.fun
        return R


class SphereConstraints(DivConstraints):
    """
    Works only when the sphere is completely included in the simplex
    """

    def __init__(self, K, r):
        self.K = K
        self.r = r
        self.feasible = np.ones(self.K) / self.K

    def check(self, x, delta=0.0001):
        return (
            (np.abs(np.sum(x) - 1) <= delta)
            and all(x > -delta)
            and (np.linalg.norm(np.ones(self.K) / self.K - x) <= self.r + delta)
        )

    def argmax_dot(self, U):
        center = np.ones(self.K) / self.K
        Uprime = U - np.dot(U, center) * center * self.K
        if np.linalg.norm(Uprime) < 0.0001:
            x = center
        else:
            x = center + self.r * Uprime / np.linalg.norm(Uprime)
        x = np.maximum(x, 0)
        x = x / np.sum(x)

        class opt:
            def __init__(self):
                self.x = x
                self.fun = np.dot(U, x)

        return opt()


#############################################################################


class DivPBand(DBand):
    """
    Main diversity-preserving bandit class.
    Need to specify upon creation:
        - a vector of mean rewards,
        - a probability set given as a DivConstraints object.

    Inherits the play_arm method and keeps track of the played arm.
    Overwrites the cum_regret and point_regret attributes
    """

    def __init__(self, K, mus, setP, noise="gaussian"):
        super().__init__(K, mus, noise=noise)
        self.setP = setP

        optimals = self.setP.argmax_dot(self.mus)
        self.m_p = optimals.fun
        self.p_star = optimals.x

        self.played_ps = []
        self.point_regret = []
        self.cum_regret = []  # overwrites the attribute in the parent class

    def _compute_reward(self, a):
        return np.random.normal(self.mus[a], 1 / 2)

    def play_p(self, p):
        assert self.setP.check(p)
        arm = draw_from_p(p, self.K)
        reward = self._compute_reward(arm)
        self.played_arms.append(arm)
        self.observed_rewards.append(reward)
        self.played_ps += [p]

        self.point_regret += [self.m_p - np.dot(p, self.mus)]
        if self.cum_regret:
            a = self.cum_regret[-1]
            self.cum_regret.append(a + self.m_p - np.dot(p, self.mus))
        else:
            self.cum_regret.append(self.m_p - np.dot(p, self.mus))
        return arm, reward

    def reset(self):
        super().reset()
        self.played_ps = []
        self.point_regret = []
        self.cum_regret = []


class DivPBandFinite(DBand):
    """
    Main diversity-preserving bandit class.
    Need to specify upon creation:
        - a vector of mean rewards,
        - a probability set given as a DivConstraints object.

    Inherits the play_arm method and keeps track of the played arm.
    Overwrites the cum_regret and point_regret attributes
    """

    def __init__(self, K, mus, Plist, noise="gaussian"):
        super().__init__(K, mus, noise=noise)
        self.Plist = Plist

        self.m_p = -1e10
        for p in Plist:
            avg_rew = p @ self.mus
            if avg_rew > self.m_p:
                self.p_star = p
                self.m_p = avg_rew

        self.played_ps = []
        self.point_regret = []
        self.cum_regret = []  # overwrites the attribute in the parent class

    def _compute_reward(self, a):
        return np.random.normal(self.mus[a], 1 / 2)

    def play_p(self, p):
        arm = draw_from_p(p, self.K)
        reward = self._compute_reward(arm)
        self.played_arms.append(arm)
        self.observed_rewards.append(reward)
        self.played_ps += [p]

        self.point_regret += [self.m_p - np.dot(p, self.mus)]
        if self.cum_regret:
            a = self.cum_regret[-1]
            self.cum_regret.append(a + self.m_p - np.dot(p, self.mus))
        else:
            self.cum_regret.append(self.m_p - np.dot(p, self.mus))
        return arm, reward

    def reset(self):
        super().reset()
        self.played_ps = []
        self.point_regret = []
        self.cum_regret = []
