# PyCV - A Computer Vision Package for Python (Incorporating Fast Training ...)

# Copyright 2007 Nanyang Technological University, Singapore.
# Authors: Minh-Tri Pham, Viet-Dung D. Hoang, and Tat-Jen Cham.

# This file is part of PyCV.

# PyCV is free software: you can redistribute it and/or modify
# it under the following term, and the terms of the GNU General Public 
# License as published by the Free Software Foundation, either version 
# 3 of the License, or (at your option) any later version.

# Any published result, involving training a boosted classifier using
# at least skewness balancing or polarity balancing, based on running
# any method in this file, must cite the following paper:
# @INPROCEEDINGS{Pham2007,
#   author = {Pham, Minh-Tri and Cham, Tat-Jen},
#   title = {Online Learning Asymmetric Boosted Classifiers for Object Detection},
#   booktitle = {2007 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR 2007)},
#   year = {2007},
#   address = {Minneapolis, MN, USA},
#   doi = {10.1109/CVPR.2007.383083}
# }

# PyCV is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# ---------------------------------------------------------------------
#!/usr/bin/env python


__all__ = ['train_DBC', 'train_AdaBoost', 'train_VJ', 'train_PC', \
    'convert_scoring2weighted', 'train_OfflineDBC']

from copy import copy, deepcopy
from math import log, exp, sqrt, fabs
from numpy import array, where, zeros, ones, dot, prod
from numpy import exp as NP_exp
from numpy import log as NP_log
from numpy import sqrt as NP_sqrt

from boosting import DiscreteBoostedClassifier, OnlineDiscreteBoostedClassifier
from pycv.cs.ml.cla import CDataset, ScoringCDataset, WeightedCDataset, \
    BinaryErrorStats, sort_1d
from pycv import tprint, ordinal
from pycv.cs.ml import OnlineLearningInterface


def convert_scoring2weighted(scd):
    """Convert a binary ScoringCDataset into a WeightedCDataset.
    
    The formula is weight(x,y) = exp(-y score(x,y)), where y \in {-1,1}
    
    :Parameters:
        scd : ScoringCDataset
            scd must be a binary dataset
        
    :Returns:
        wcd : WeightedCDataset
    """
    cd = WeightedCDataset(scd.input_data, \
        [NP_exp((1-2*j)*scd.scores[j]) for j in xrange(2)])
    cd.normalize_weights()
    return cd


def _check_stop_2(scores, maxFAR, maxFRR):
    cd = CDataset(scores)

    # all examples classified as positive
    false_pos = cd.nspc[0]
    false_neg = 0
    if float(false_pos)/cd.nspc[0] <= maxFAR and float(false_neg)/cd.nspc[1] <= maxFRR:
        return True
    
    # sort the scores
    sorted_cd = sort_1d(cd)
    
    # move the threshold to check if the the stopping condition has been reached
    for j, _ in sorted_cd:
        if j == 0: false_pos -= 1
        else: false_neg += 1
        if float(false_pos)/cd.nspc[0] <= maxFAR and float(false_neg)/cd.nspc[1] <= maxFRR:
            return True
    # stopping condition has not been reached
    return False


def _check_stop_3(scores, maxFAR, maxFRR):
    cd = CDataset(scores)

    # all examples classified as positive
    false_pos = cd.nspc[0]
    false_neg = 0
    if (float(false_neg)/cd.nspc[1])*maxFAR/maxFRR + (float(false_pos)/cd.nspc[0]) <= maxFAR:
        return True

    # sort the scores
    sorted_cd = sort_1d(cd)

    # move the threshold to check if the the stopping condition has been reached
    for j, _ in sorted_cd:
        if j == 0: false_pos -= 1
        else: false_neg += 1
        if (float(false_neg)/cd.nspc[1])*maxFAR/maxFRR + (float(false_pos)/cd.nspc[0]) <= maxFAR:
            return True
    # stopping condition has not been reached
    return False
    

def train_OfflineDBC(scd, trainfunc, M, criterion = 0, param1 = 1.0, \
    skewness_balancing=1, preceeding_sc=None, extra_output=False):
    """Train an offline DiscreteBoostedClassifier
    
    Criteria:
        criterion=0: 
            \arg \min_f (\lambda P(1) FRR(f) + P(0) FAR(f)) / \
                (\lambda P(1) + P(0))
        criterion=1: 
            \arg \min_f (\lambda FRR(f) + FAR(f)) / (\lambda + 1)

    :Paramters:
        scd : ScoringCDataset
            a binary ScoringCDataset
        trainfunc: a function that takes a WeightedCDataset as input
            and returns a BinaryClassifier as a weak classifier
        M : int
            the maximum number of weak classifiers
        criterion : int
            which criterion
        param1 : double
            \lambda for the criterion
        skewness_balancing : int
            type of balancing among weak classifiers
                0 = no balancing at all, the original AdaBoost's method
                1 = asymmetric weight balancing, Viola-Jones (NIPS'02)
                2 = skewness balancing, Pham-Cham (CVPR'07) 
                    (N/A if criterion=1)
        preceeding_sc : ScoringClassifier
            a classifier to preceed this newly trained one, 
            default is None
        extra_output : boolean
            if True then produce extra useful information
            
    :Returns:
        dbc : DiscreteBoostedClassifier
            the newly trained DiscreteBoostedClassifier
        err : double (extra_output)
            training error, or training criterion function value
        scd2 : ScoringCDataset (extra_output)
            a new ScoringCDataset with scores augmented by this dbc,
    """
    cd = convert_scoring2weighted(scd)
    
    if criterion==1 and skewness_balancing==2:
        raise NotImplementedError, 'this case has not been implemented'
        
    if criterion==1:
        cd.weights[0] /= cd.weights[0].sum()
        cd.weights[1] /= cd.weights[1].sum()

    if skewness_balancing == 0:
        cd.weights[1] *= param1
    elif skewness_balancing == 1:
        kk = exp(log(param1)/M)
    elif skewness_balancing == 2:
        logk = log(param1)
    else:
        raise IndexError('balancing value is out of bound.')

    weaks = []
    cs = []
    tprint("Training a boosted classifier with at most "+str(M)+" weak classifiers...")
    for m in xrange(M):
        if skewness_balancing == 1:
            cd.weights[1] *= kk
        elif skewness_balancing == 2:
            prior = cd.get_twpc()
            gamma = cd.get_skewness()

            # compute k_m
            log_k_m = (logk + (M-m-1)*gamma) / (M-m)
            k_m = exp(log_k_m)

            # update weights to deal with k_m
            cd.weights[1] *= k_m

        cd.normalize_weights()

        if skewness_balancing == 2:
            prior = cd.get_twpc()
            gamma = cd.get_skewness()

        # train the new weak classifier
        tprint("Training the "+ordinal(m+1)+" weak classifier...")
        f = trainfunc(cd)

        err = [f.test(cd.input_data[j]) != j for j in xrange(2)]
        werr = sum([where(err[j],cd.weights[j],0).sum() for j in xrange(2)])

        if werr < 1e-10: # no single error? too good to be true
            weaks = [f]
            cs = [1]
            break

        c = 0.5 * log((1-werr)/werr)

        # append to the boost
        weaks.append(f)
        cs.append(c)

        # update the weights
        for j in xrange(2): cd.weights[j] *= where(err[j],exp(c),exp(-c))

        if skewness_balancing == 2:
            logk -= log_k_m

    dbc = DiscreteBoostedClassifier(preceeding_sc, weaks, array(cs))
    if not extra_output:
        return dbc
    
    # produce extra information
    scores = [dbc.scores(x) for x in scd.input_data]
    scd2 = ScoringCDataset(scd.input_data, scores)
    err0 = float((scores[0] >= 0).sum()) / scd.nspc[0] # FAR
    err1 = float((scores[1] < 0).sum()) / scd.nspc[1] # FRR
    
    if criterion == 0:
        err = (err0*scd.nspc[0] + err1*scd.nspc[1]*param1) / \
            (scd.nspc[0]+scd.nspc[1]*param1)
    else: #if criterion == 1:
        err = (err0+err1*param1) / (1+param1)
    
    return (dbc, err, scd2)

        
        
        
def train_DBC(classification_dataset, trainfunc, M, k = 1.0, balancing = 2, \
    can_learn=True, polarity_balancing=1, previous=None):
    """Train a DiscreteBoostedClassifier

    Input:
        classification_dataset: a WeightedCDataset of 2 classes
        trainfunc: a function that takes a WeightedCDataset as input
            and returns a BinaryClassifier as a weak classifier
        M: the maximum number of weak classifier
        k: false negatives penalized k times more than false positives
        balancing: type of balancing among weak classifiers
            0 = no balancing at all, this is the original AdaBoost's method
            1 = asymmetric weight balancing, Viola-Jones (NIPS'02)
            2 = skewness balancing, Pham-Cham (CVPR'07)
        can_learn : boolean
            whether the resulting DiscreteBoostedClassifier can learn 
                incrementally
        polarity_balancing: use polarity balancing for online-learning?
            0 = no polarity balancing, same as Oza-Rusell (ICSMC'05)
            1 = polarity balancing, Pham-Cham (CVPR'07)
        previous: previous additive classifier, default is None
    Output:
        a DiscreteBoostedClassifier
    """
    cd = classification_dataset.new()
    cd.initialize_weights()

    if balancing == 0:
        cd.weights[1] *= k
    elif balancing == 1:
        kk = exp(log(k)/M)
    elif balancing == 2:
        logk = log(k)
    else:
        raise IndexError('balancing value is out of bound.')

    weaks = []
    cs = []
    tprint("Training a boosted classifier with at most "+str(M)+" weak classifiers...")
    for m in xrange(M):
        if balancing == 1:
            cd.weights[1] *= kk
        elif balancing == 2:
            prior = cd.get_twpc()
            gamma = cd.get_skewness()

            # tprint("Before applying k_m:")
            # tprint("m="+str(m))
            # tprint("log k="+str(logk))
            # tprint("prior="+str(prior))
            # tprint("gamma="+str(gamma))

            # compute k_m
            log_k_m = (logk + (M-m-1)*gamma) / (M-m)
            k_m = exp(log_k_m)
            # tprint("log k_m="+str(log_k_m))
            # tprint("k_m="+str(k_m))

            # update weights to deal with k_m
            cd.weights[1] *= k_m

        cd.normalize_weights()

        if balancing == 2:
            # tprint("After applying k_m:")
            prior = cd.get_twpc()
            gamma = cd.get_skewness()
            # tprint("prior="+str(prior))
            # tprint("gamma="+str(gamma))

        # train the new weak classifier
        tprint("Training the "+ordinal(m+1)+" weak classifier...")
        f = trainfunc(cd)

        err = [f.test(cd.input_data[j]) != j for j in xrange(2)]
        werr = sum([where(err[j],cd.weights[j],0).sum() for j in xrange(2)])

        if werr < 1e-20: # no single error? too good to be true
            weaks = [f]
            cs = [1]
            break

        c = 0.5 * log((1-werr)/werr)

        # append to the boost
        weaks.append(f)
        cs.append(c)

        # update the weights
        for j in xrange(2): cd.weights[j] *= where(err[j],exp(c),exp(-c))

        if balancing == 2:
            logk -= log_k_m

    if can_learn:
        return OnlineDiscreteBoostedClassifier(previous, weaks, array(cs), k,
            balancing, polarity_balancing)
    else:
        return DiscreteBoostedClassifier(previous, weaks, array(cs))

def train_AdaBoost(classification_dataset, trainfunc, M,
    can_learn=True, polarity_balancing=1):
    """Train a DiscreteBoostedClassifier using AdaBoost (Friend et al's DiscreteAdaboost)
    
    Warning:
        This function is now obsolete, use train_DBC() instead.

    Input:
        classification_dataset: a WeightedCDataset of 2 classes
        trainfunc: a function that takes a WeightedCDataset as input
            and returns a BinaryClassifier
        M: the maximum number of stages
        can_learn : boolean
            whether the resulting DiscreteBoostedClassifier can learn 
                incrementally
        polarity_balancing: use polarity balancing for online-learning?
            0 = no polarity balancing, same as Oza-Rusell (ICSMC'05)
            1 = polarity balancing, Pham-Cham (CVPR'07)
    Output:
        a DiscreteBoostedClassifier
    """
    cd = classification_dataset.new()
    cd.initialize_weights()

    weaks = []
    cs = []
    for m in xrange(M):
        # train the new weak classifier
        tprint("Training the "+ordinal(m+1)+" weak classifier.")
        f = trainfunc(cd)
        
        err = [f.test(cd.input_data[j]) != j for j in xrange(2)]
        werr = sum([where(err[j],cd.weights[j],0).sum() for j in xrange(2)])

        if werr < 1e-20: # no single error? too good to be true
            weaks = [f]
            cs = [1]
            break

        c = 0.5 * log((1-werr)/werr)

        # append to the boost
        weaks.append(f)
        cs.append(c)

        # update the weights, then normalize
        for j in xrange(2): cd.weights[j] *= where(err[j],exp(c),exp(-c))
        cd.normalize_weights()

    if can_learn:
        return OnlineDiscreteBoostedClassifier(None, weaks, array(cs), 1.0,
            0, polarity_balancing)
    else:
        return DiscreteBoostedClassifier(None, weaks, array(cs))


def train_VJ( classification_dataset, trainfunc, M, k, evenly = True, 
    can_learn=True, polarity_balancing=1 ):
    """Train a DiscreteBoostedClassifier using Viola and Jones'
        asymmetric boost (NIPS'02)

    Warning:
        This function is now obsolete, use train_DBC() instead.

    Input:
        classification_dataset: a WeightedCDataset of 2 classes
        trainfunc: a function that takes a WeightedCDataset as input
            and returns a BinaryClassifier
        M: the maximum number of stages
        k: false negatives penalized k times more than false positives
        evenly: distribute lambda evenly among the weak classifiers
        can_learn : boolean
            whether the resulting DiscreteBoostedClassifier can learn 
                incrementally
        polarity_balancing: use polarity balancing for online-learning?
            0 = no polarity balancing, same as Oza-Rusell (ICSMC'05)
            1 = polarity balancing, Pham-Cham (CVPR'07)
    Output:
        a DiscreteBoostedClassifier
    """
    cd = classification_dataset.new()
    cd.initialize_weights()

    if evenly == True:
        kk = exp(log(k)/M)
    else:
        cd.weights[1] *= k

    weaks = []
    cs = []
    tprint("Training a boosted classifier with at most "+str(M)+" weak classifiers...")
    for m in xrange(M):
        if evenly == True: cd.weights[1] *= kk
        cd.normalize_weights()

        # train the new weak classifier
        tprint("Training the "+ordinal(m+1)+" weak classifier...")
        f = trainfunc(cd)

        err = [f.test(cd.input_data[j]) != j for j in xrange(2)]
        werr = sum([where(err[j],cd.weights[j],0).sum() for j in xrange(2)])
        
        if werr < 1e-20: # no single error? too good to be true
            weaks = [f]
            cs = [1]
            break

        c = 0.5 * log((1-werr)/werr)

        # append to the boost
        weaks.append(f)
        cs.append(c)

        # update the weights
        for j in xrange(2): cd.weights[j] *= where(err[j],exp(c),exp(-c))

    if can_learn:
        return OnlineDiscreteBoostedClassifier(None, weaks, array(cs), k,
            1, polarity_balancing)
    else:
        return DiscreteBoostedClassifier(None, weaks, array(cs))


def train_PC( classification_dataset, trainfunc, M, k,
    can_learn=True, polarity_balancing=1 ):
    """Train a DiscreteBoostedClassifier using our (Pham and Cham's)
        asymmetric boost (CVPR'07)

    Warning:
        This function is now obsolete, use train_DBC() instead.

    Input:
        classification_dataset: a WeightedCDataset of 2 classes
        trainfunc: a function that takes a WeightedCDataset as input
            and returns a BinaryClassifier
        M: the maximum number of stages
        k: false negatives penalized k times more than false positives
        can_learn : boolean
            whether the resulting DiscreteBoostedClassifier can learn 
                incrementally
        polarity_balancing: use polarity balancing for online-learning?
            0 = no polarity balancing, same as Oza-Rusell (ICSMC'05)
            1 = polarity balancing, Pham-Cham (CVPR'07)
    Output:
        a DiscreteBoostedClassifier
    """
    cd = classification_dataset.new()
    cd.initialize_weights()

    logk = log(k)

    weaks = []
    cs = []
    for m in xrange(M):
        # train the new weak classifier
        tprint("Training the "+ordinal(m+1)+" weak classifier.")

        prior = cd.get_twpc()
        gamma = cd.get_skewness()

        # tprint("Before applying k_m:")
        # tprint("m="+str(m))
        # tprint("log k="+str(logk))
        # tprint("prior="+str(prior))
        # tprint("gamma="+str(gamma))

        # compute k_m
        log_k_m = (logk + (M-m-1)*gamma) / (M-m)
        k_m = exp(log_k_m)
        # tprint("log k_m="+str(log_k_m))
        # tprint("k_m="+str(k_m))

        # update weights to deal with k_m
        cd.weights[1] *= k_m

        # tprint("After applying k_m:")
        cd.normalize_weights()
        prior = cd.get_twpc()
        gamma = cd.get_skewness()
        # tprint("prior="+str(prior))
        # tprint("gamma="+str(gamma))

        f = trainfunc(cd)

        err = [f.test(cd.input_data[j]) != j for j in xrange(2)]
        werr = sum([where(err[j],cd.weights[j],0).sum() for j in xrange(2)])

        if werr < 1e-20: # no single error? too good to be true
            weaks = [f]
            cs = [1]
            break

        c = 0.5 * log((1-werr)/werr)

        # append to the boost
        weaks.append(f)
        cs.append(c)

        # update the weights, then normalize
        for j in xrange(2): cd.weights[j] *= where(err[j],exp(c),exp(-c))
        cd.normalize_weights()
        logk -= log_k_m

    if can_learn:
        return OnlineDiscreteBoostedClassifier(None, weaks, array(cs), k,
            2, polarity_balancing)
    else:
        return DiscreteBoostedClassifier(None, weaks, array(cs))



def train_PC_NIPS( classification_dataset, trainfunc, M, k  ):
    """Train a DiscreteBoostedClassifier using our (Pham and Cham's)
        asymmetric boost (NIPS'07 -- never submitted)

    Goal:
        Assume D^+ is the distribution of x given x positive.
        Assume D^- is the distribution of x given x negative.
        We wish to find F_M(x) to minimize:
            k J^-(F_M) + J^+(F_M) (1)
        where
            J^+(F_M) = E_{D^+} [(F_M(x)-1)^2]
            J^-(F_M) = E_{D^-} [(F_M(x)+1)^2]

        Let:
            pi^+_m = 1 - E_{D^+} [F_M(x)]
            pi^-_m = 1 - E_{D^-} [F_M(x)]
            D^+_m a distribution such that p_{D^+_m}(x) = p_{D^+}(x) (1-F_m(x)) / pi^+_m
            D^-_m a distribution such that p_{D^-_m}(x) = p_{D^-}(x) (1+F_m(x)) / pi^-_m
            FRR_m(f) = E_{D^+_m}[f(x) == -1]
            FAR_m(f) = E_{D^-_m}[f(x) == +1]

        I proved that:
            J^+(F_M) = J^+(F_m) + c^2 + 2 c pi^+_m (2FRR_m(f) - 1)
            J^-(F_M) = J^-(F_m) + c^2 + 2 c pi^-_m (2FAR_m(f) - 1)

        Let's say we want to minimize (1) then we need to
            1) choose f minimizing: (weak classifier)
                epsilon(f) = k pi^-_m FRR_m(f) + k pi^+_m FAR_m(f)
            2) choose  minimizing: (voting coefficient)
                (k J^- + J^+)(F_m) + (k+1) c^2 + 4c epsilon(f) - 2c(k pi^-_m + pi^+_m)
                which means:
                    c* = \frac{k pi^-_m + pi^+_m - 2 epsilon(f)} {k+1}

        I also proved that: (not quite)
            |F_M(x)| <= \sum_{m=1}^M |c_m| <= 1 for all M


    Input:
        classification_dataset: a WeightedCDataset of 2 classes
        trainfunc: a function that takes a WeightedCDataset as input
            and returns a BinaryClassifier
        M: the maximum number of stages
        k: false negative rate penalized k times more than false positive rate
    Output:
        a DiscreteBoostedClassifier
    """
    raise NotImplementedError, "the function has not been implemented"



# class RealAdaboostClassifier(BinaryClassifier): # binary classifier

    # def __init__(self,maxM,weak_probabilistic_binary_classifier):
        # classifier.__init__(self,2)
        # self.maxM = maxM
        # self.apclassifier = weak_probabilistic_binary_classifier

    # def train( self, input_data, weights = None, *args ):
        # BinaryClassifier.train( self, input_data, weights, *args )
        # N = map(len,input_data)
        # if weights is None:
            # w = [ones(n)/sum(N) for n in N]
        # else:
            # w = weights.copy()

        # self.weaks = []
        # self.c = []
        # for m in xrange(maxM):
            #train the new weak classifier
            # weak = copy(self.apclassifier)
            # weak.train(input_data,w,*args)

            #append to the boost
            # self.weaks.append(weak)

            #update the weights
            # for j in xrange(2):
                # z = weak.test_pdf(input_data[j])
                # w[j] *= NP_sqrt(z[1-j]/z[j])

    # def predict(self,input_point):
        # val = 0
        # for w in self.weaks:
            # pdf = w.predict_pdf(input_point)
            # val += 0.5 * log(pdf[1]/pdf[0])
        # return int(val >= 0)


# def main():
    #main idea is for testing
    # a = [array([[0,1],[1,1],[1,0]]),array([[0,2],[1,2],[2,2],[2,1],[2,0]])]
    # b = WeightedCDataset(a)
    # c = train_PC(b,train_NBClassifier,10,10000)
    # tprint(c.test(a[0]))
    # tprint(c.test(a[1]))

# if __name__ == '__main__':
    # main()
