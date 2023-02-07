# -*- coding: utf-8 -*-
"""

.. moduleauthor:: Valentin Emiya
"""

import abc
import numpy as np


class BernoulliMultiArmedBandits:
    """
    Bandit problem with Bernoulli distributions

    Parameters
    ----------
    means : array-like
        True values (expectation of reward) for each arm
    """
    def __init__(self, means):
        self._means = np.array(means)
        assert np.all(0 <= self._means)
        assert np.all(self._means <= 1)

    @property
    def n_arms(self):
        """
        Number of arms

        Returns
        -------
        int
        """
        return self._means.size

    @property
    def _true_values(self):
        return self._means

    def step(self, a):
        """
        Play an arm and return reward

        Parameters
        ----------
        a : int
            Index of arm to be played

        Returns
        -------
        bool
            Reward obtained from playing arm `a` (true if win, false otherwise)
        """
        assert 0 <= a
        assert a < self.n_arms
        return np.random.rand() < self._means[a]

    def __str__(self):
        return '{}-arms bandit problem with Bernoulli distributions'.format(
            self.n_arms)


class NormalMultiArmedBandits:
    """
    Bandit problem with normal distributions with unit variance.

    Parameters
    ----------
    means : array-like
        Mean values for each arm
    stds : array-like
        Standard deviation values for each arm (1 if None)
    """

    def __init__(self, means, stds=None):
        self._means = np.array(means)
        if stds is None:
            stds = np.ones_like(means)
        self._stds = stds

    @property
    def n_arms(self):
        """
        Number of arms

        Returns
        -------
        int
        """
        return self._means.size

    @property
    def _true_values(self):
        return self._means

    def step(self, a):
        """
        Play an arm and return reward

        Parameters
        ----------
        a : int
            Index of arm to be played

        Returns
        -------
        float
            Reward obtained from playing arm `a`
        """
        #assert 0 <= a
        #assert a < self.n_arms
        return np.random.randn() * self._stds[a] + self._means[a]

    def __str__(self):
        return '{}-arms bandit problem with Normal distributions'.format(
            self.n_arms)

    @staticmethod
    def create_random(n_arms):
        """

        Parameters
        ----------
        n_arms : int
            Number of arms or actions

        Returns
        -------

        """
        return NormalMultiArmedBandits(means=np.random.randn(n_arms))


class BanditAlgorithm(abc.ABC):
    """
    A generic abstract class for Bandit Algorithms

    Parameters
    ----------
    n_arms : int
        Number of arms
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, n_arms=10):
        self.n_arms = n_arms

    @abc.abstractmethod
    def get_action(self):
        """
        Choose an action (abstract)

        Returns
        -------
        int
            The chosen action
        """
        raise NotImplementedError

    @abc.abstractmethod
    def fit_step(self, action, reward):
        """
        update current value estimates with an (action, reward) pair (abstract)

        Parameters
        ----------
        action : int
        reward : float

        """
        raise NotImplementedError


class RandomBanditAlgorithm(BanditAlgorithm):
    """
    A generic class for Bandit Algorithms

    Parameters
    ----------
    n_arms : int
        Number of arms
    """
    def __init__(self, n_arms=10):
        BanditAlgorithm.__init__(self, n_arms=n_arms)
        # Estimation of the value of each arm
        self._value_estimates = np.zeros(n_arms)
        # Number of times each arm has been chosen
        self._n_estimates = np.zeros(n_arms)

    def get_action(self):
        """
        Choose an action at random uniformly among the available arms

        Returns
        -------
        int
            The chosen action
        """
        return np.random.randint(self.n_arms)

    def fit_step(self, action, reward):
        """
        Do nothing since actions are chosen at random

        Parameters
        ----------
        action : int
        reward : float

        """
        
        self._n_estimates[action] = self._n_estimates[action] + 1         # update de n(a) 
        n = self._n_estimates[action]                 
    
        value = self._value_estimates[action]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward  
        self._value_estimates[action] = new_value   # update de q(a)
        pass

class GreedyBanditAlgorithm(BanditAlgorithm):
    """
    Greedy Bandit Algorithm

    Parameters
    ----------
    n_arms : int
        Number of arms
    """
    def __init__(self, n_arms=10):
        BanditAlgorithm.__init__(self, n_arms=n_arms)
        # Estimation of the value of each arm
        self._value_estimates = np.zeros(n_arms)
        # Number of times each arm has been chosen
        self._n_estimates = np.zeros(n_arms, dtype=int)

    def get_action(self):
        """
        Choose the action with maximum estimated value

        Returns
        -------
        int
            The chosen action
        """
        for (i,n) in enumerate(self._n_estimates) :
            if n == 0 :
                return i
        action = np.argmax(self._value_estimates)
        return action 
    
    def fit_step(self, action, reward):
        """
        update current value estimates with an (action, reward) pair

        Parameters
        ----------
        action : int
        reward : float

        """
        if self._n_estimates[action] > 1 :
            self._value_estimates[action] = self._value_estimates[action] + 1/self._n_estimates[action] * (reward - self._value_estimates[action])
            self._n_estimates[action] +=1
        else:
            self._value_estimates[action] = reward
            self._n_estimates[action] +=1
        pass        

class EpsilonGreedyBanditAlgorithm(BanditAlgorithm):

#class EpsilonGreedyBanditAlgorithm(GreedyBanditAlgorithm,                                   RandomBanditAlgorithm):
    """
    Epsilon-greedy Bandit Algorithm

    Parameters
    ----------
    n_arms : int
        Number of arms
    epsilon : float
        Probability to choose an action at random
    """
    def __init__(self, n_arms=10, epsilon=0.1):
        GreedyBanditAlgorithm.__init__(self, n_arms=n_arms)
        self.epsilon = epsilon
        
    def get_action(self):
        """
        Get Epsilon-greedy action

        Choose an action at random with probability epsilon and a greedy
        action otherwise.

        Returns
        -------
        int
            The chosen action
        """        
        p = np.random.random()

            # if the probability is less than epsilon then a random arm is chosen from the complete set
        if p < self.epsilon:
            arm_index = np.random.choice(self.n_arms)
            return arm_index

        else:
            # choose the arm with the current highest mean reward or arbitrary select a arm in the case of a tie            
            arm_index = np.argmax(self._value_estimates)               
        
            return arm_index

    #La plupart des algorithmes doivent mettre à jour le nombre de tirages  𝑁(𝑎)  de chaque bras  𝑎  et sa récompense moyenne  𝑄(𝑎) : ces valeurs sont stockées sous forme de tableaux numpy dans les attributs _value_estimates et _n_estimates, respectivement.
        
    def fit_step(self, action, reward):
        """
        update current value estimates with an (action, reward) pair

        Parameters
        ----------
        action : int
        reward : float

        """
        self._n_estimates[action] = self._n_estimates[action] + 1         # update de n(a) 
        n = self._n_estimates[action]                 
    
        value = self._value_estimates[action]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward  
        self._value_estimates[action] = new_value   # update de q(a)
        pass
    
class UcbBanditAlgorithm(GreedyBanditAlgorithm):
    """

    Parameters
    ----------
    n_arms : int
        Number of arms
    c : float
        Positive parameter to adjust exploration/explotation UCB criterion
    """
    def __init__(self, n_arms, c,t):
        GreedyBanditAlgorithm.__init__(self, n_arms=n_arms)
        self.c = c
        self.t = t
        self.n_arms = n_arms

    def get_action(self):
        """
        Get UCB action

        Returns
        -------
        int
            The chosen action
        """
        #initialisation
        for (i,n) in enumerate(self._n_estimates) :
            if n == 0 :
                return i
        #Highest Confidence Bound 
        action = np.argmax(self._value_estimates + self.c * np.sqrt(np.log(self.t)/self._n_estimates))
        
        return action

    def get_upper_confidence_bound(self):

        upper_bound = np.empty(self.n_arms)
        lower_bound = np.empty(self.n_arms)
        
        for i in range(self.n_arms):
            upper_bound[i] = self._value_estimates[i] + self.c * np.sqrt(np.log(self.t)/self._n_estimates[i]) 
            lower_bound[i] = self._value_estimates[i] - self.c * np.sqrt(np.log(self.t)/self._n_estimates[i])
        return upper_bound , lower_bound
        

class ThompsonSamplingAlgorithm(BanditAlgorithm):
    """

    Parameters
    ----------
    n_arms : int
        Number of arms
    """
    def __init__(self, n_arms):
        BanditAlgorithm.__init__(self, n_arms=n_arms)

        self.alpha = np.ones(n_arms)
        self.beta = np.ones(n_arms)
        self.n_arms = n_arms
        self._n_estimates = np.zeros(n_arms,dtype=int)
        self._value_estimates = np.zeros(n_arms)
        pass

    def get_action(self):

        liste = np.empty(self.n_arms)
        for arm in range(self.n_arms):
            liste[arm] = np.random.beta(self.alpha[arm],self.beta[arm])
        action = np.argmax(liste)
        return action

    def fit_step(self, action, reward):
        self.alpha[action]+=reward
        self.beta[action]+=1-reward
        
        if self._n_estimates[action] > 1 :
            self._value_estimates[action] = self._value_estimates[action] + 1/self._n_estimates[action] * (reward - self._value_estimates[action])
            self._n_estimates[action] +=1
        else:
            self._value_estimates[action] = reward
            self._n_estimates[action] +=1
        
        pass
