# PyCV - A Computer Vision Package for Python (Incorporating Fast Training ...)

# Copyright 2007 Nanyang Technological University, Singapore.
# Authors: Minh-Tri Pham, Viet-Dung D. Hoang, and Tat-Jen Cham.

# This file is part of PyCV.

# PyCV is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public 
# License as published by the Free Software Foundation, either version 
# 3 of the License, or (at your option) any later version.

# PyCV is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# ---------------------------------------------------------------------
#!/usr/bin/env python


__all__ = ['DiscreteBoostedClassifier', 'OnlineDiscreteBoostedClassifier']

from math import log, exp, sqrt
from numpy import array, where, zeros, ones, dot, prod
from numpy import exp as NP_exp
from numpy import log as NP_log
from numpy import sqrt as NP_sqrt

from pycv.cs.ml.cla import AdditiveClassifier
from pycv.cs.ml import OnlineLearningInterface

#-------------------------------------------------------------------------------
# Boosted Classifiers
#-------------------------------------------------------------------------------

class DiscreteBoostedClassifier(AdditiveClassifier):
    """Discrete Boosted Classifier of this form:
        F_M(x) = preceeding_classifier(x) + \sum_{m=1}^M c_m f_m(x) + b,
        where
            c_m: coefficient, >= 0
            f_m(x): a binary classifier outputing {-1,1}
            b: shifting amount, default is 0
    """
    
    def __init__(self, sc=None, weaks=[], c=array([],'d'), b=0):
        """Initialize the DiscreteBoostedClassifier with some parameters.

        :Parameters:
            sc : ScoringClassifier
                a ScoringClassifier to preceed this classifier,
                or None if there's no classifier to preceed
            weaks : list
                list of weak classifiers trained
            c : numpy.array of real values
                array of coefficients for weak classifiers
            b : double
                threshold of the classifier
        """
        AdditiveClassifier.__init__(self, sc)
        self.M = len(weaks)
        self.weaks = weaks
        self.c = array(c)
        self.b = b
		
    def current_score(self, input_point, *args, **kwds):
        """Return the score of the classifier before being aggregated."""
        p = array([w.predict(input_point, *args, **kwds) for w in self.weaks])
        #return where(p,self.c,-self.c).sum()
        return (self.c*(p*2-1)).sum()+self.b
        
    def score(self, input_point, *args, **kwds):
        return self.preceeding_score(input_point, *args, **kwds) + \
				self.current_score(input_point, *args, **kwds)
    score.__doc__ = AdditiveClassifier.score.__doc__
    
    def refine(self):
        """Refine the classifier.
        
        Refine the classifier by throwing away weak classifiers with zero 
        coefficients.

        Returns the filtering array so superclass(es) can filter their data.
        """
        filterarray = (self.c >= 1.0e-10)
        self.c = self.c[filterarray]
        self.weaks = [self.weaks[m] for m in xrange(len(self.weaks)) if filterarray[m]]
        self.M = len(self.weaks)
        return filterarray
        
    def normalize_c(self):
        self.c /= self.c.sum()

    
class OnlineDiscreteBoostedClassifier(DiscreteBoostedClassifier, \
    OnlineLearningInterface):
    """DiscreteBoostedClassifier with an ability to learn online."""

    def __init__(self, sc=None, weaks=[], c=array([],'d'), k=1.0, \
        skewness_balancing=0, polarity_balancing=1):
        """Initialize the OnlineDiscreteBoostedClassifier with some parameters

        :Parameters:
            sc : ScoringClassifier
                a ScoringClassifier to preceed this classifier,
                or None if there's no classifier to preceed
            weaks : list
                list of weak classifiers trained
            c : numpy.array of real values
                array of coefficients for weak classifiers
            k : double
                false negatives penalized k times more than false positives
            skewness_balancing : int
                type of skewness balancing among weak classifiers
                    0 = no balancing at all, this is the original AdaBoost's 
                        method
                    1 = asymmetric weight balancing, Viola-Jones (NIPS'02)
                    2 = skewness balancing, Pham-Cham (CVPR'07)
            polarity_balancing : int
                use polarity balancing?
                    0 = no polarity balancing, same as Oza-Rusell (ICSMC'05)
                    1 = polarity balancing, Pham-Cham (CVPR'07)
        """
        DiscreteBoostedClassifier.__init__(self, sc=sc, weaks=weaks, c=c, b=0)
        M = len(weaks)
        self.M = M
        self.k = k
        self.skewness_balancing = skewness_balancing
        self.polarity_balancing = polarity_balancing
        self.tw = 0 # total weights
        if polarity_balancing == 1: # Pham-Cham (CVPR'07 oral)
            self.g = zeros(M) # gamma
            self.v = ones((M,2,2)) # Pham-Cham's v
        elif polarity_balancing == 0: # Oza-Rusell (ICSMC'05)
            self.v = ones((M,2)) # Oza-Rusell's lambda
        else:
            raise IndexError('Out of bound is the value of polarity_balancing')
        
    def learn( self, input_point, j, weight = None, *args, **kwds):
        """Learn incrementally

        Learn incrementally with a new input point, its class, and optionally 
        its weight. Other parameters like k and balancing are derived from 
        the class itself.

        TODO: Re-test this function extensively.

        Input:
            input_point: a new input point
            j: its corresponding class
            w: optionally its weight, or 1 if not specified
            polaritybalancing: use polarity balancing?
                0 = no polarity balancing, same as Oza-Rusell (ICSMC'05)
                1 = polarity balancing, Pham-Cham (CVPR'07)
        """
        M = self.M
        a = zeros(M)
        e = zeros(M)
        kk = zeros(M)
        if self.skewness_balancing == 1:
            k2 = exp(log(self.k)/self.M)
        elif self.skewness_balancing == 2:
            logk = log(self.k)

        v = 1.0 if weight is None else weight
        self.tw += v
            
        for m in xrange(self.M):
            # compute k_m
            if self.skewness_balancing == 0:
                if m == 0:
                    kk[m] = self.k
                else:
                    kk[m] = 1.0
            elif self.skewness_balancing == 1:
                kk[m] = k2
            elif self.skewness_balancing == 2:
                log_k_m = (logk-NP_log(kk[:m]).sum() + (M-m-1)*self.g[m]) / (M-m)
                kk[m] = exp(log_k_m)

            # update weights to deal with k_m
            k3 = sqrt(kk[m])
            if j != 0: # positive
                w = v*k3
            else:
                w = v/k3

            # update weak classifier
            self.weaks[m].learn(input_point,j,w)

            # rerun
            j2 = self.weaks[m].predict(input_point, *args, **kwds)

            # propagate the weights
            if self.polarity_balancing == 1: # Pham-Cham
                self.v[m,j,j2] += v
                a[m] = self.v[m,1].sum() / self.v[m].sum()
                if j2 > 0: # predicted as positive
                    v *= a[m] * 0.5 * self.tw / self.v[m,j,j2]
                else: # predicted as negative
                    v *= (1-a[m]) * 0.5 * self.tw / self.v[m,j,j2]
            else: # Oza-Rusell
                j3 = int(j == j2)
                self.v[m,j3] += v
                v *= 0.5 * self.tw / self.v[m,j3]

            # update other parameters
            if self.polarity_balancing == 1: # Pham-Cham
                self.g[m] = log((1-a[m])/a[m])
                e[m] = (self.v[m,0,1]/k3 + self.v[m,1,0]*k3) / \
                    ((self.v[m,0].sum())/k3 + (self.v[m,1].sum())*k3)
            else: # Oza-Rusell
                e[m] = self.v[m,0] / self.v[m].sum()
            self.c[m] = 0.5 * log((1-e[m])/e[m])

