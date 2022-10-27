# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 16:59:03 2022

@author: Zahra
"""

import numpy as np
import cvxpy as cvx
from sklearn import metrics


# Supporting functions

class Distances(object):
    
    def __init__(self,P,Q):
        if sum(P)<1e-20 or sum(Q)<1e-20:
            raise "One or both vector are zero (empty)..."
        if len(P)!=len(Q):
            raise "Arrays need to be of equal sizes..."
        #use numpy arrays for efficient coding
        P=np.array(P,dtype=float);Q=np.array(Q,dtype=float)
        #Correct for zero values
        P[np.where(P<1e-20)]=1e-20
        Q[np.where(Q<1e-20)]=1e-20
        self.P=P
        self.Q=Q
        
    def sqEuclidean(self):
        P=self.P; Q=self.Q; 
        return np.sum((P-Q)**2)
    def probsymm(self):
        P=self.P; Q=self.Q; 
        return 2*np.sum((P-Q)**2/(P+Q))
    def topsoe(self):
        P=self.P; Q=self.Q
        return np.sum(P*np.log(2*P/(P+Q))+Q*np.log(2*Q/(P+Q)))
    def hellinger(self):
        P=self.P; Q=self.Q
        return np.sqrt(np.sum((np.sqrt(P) - np.sqrt(Q))**2))


def distance(sc_1, sc_2, measure):
   
    dist = Distances(sc_1, sc_2)

    if measure == 'sqEuclidean':
        return dist.sqEuclidean()
    if measure == 'topsoe':
        return dist.topsoe()
    if measure == 'probsymm':
        return dist.probsymm()
    if measure == 'hellinger':
        return dist.hellinger()
    print("Error, unknown distance specified, returning topsoe")
    return dist.topsoe()


def TernarySearch(left, right, f, eps=1e-4):

    while True:
        if abs(left - right) < eps:
            return (left + right) / 2, f((left + right) / 2)
    
        leftThird  = left + (right - left) / 3
        rightThird = right - (right - left) / 3
    
        if f(leftThird) > f(rightThird):
            left = leftThird
        else:
            right = rightThird 
            
def LinearSearch(left, right, f, eps=1e-4):
    
    p = np.linspace(left, right, num=21, endpoint=True)
    p[0] += 0.01
    p[-1] -= 0.01
    selected_prev = p[0]
    for prev in p:
        if f(prev)<f(selected_prev):
            selected_prev = prev
    return selected_prev, f(selected_prev)

def getHist(scores, nbins):
    breaks = np.linspace(0, 1, int(nbins)+1)
    breaks = np.delete(breaks, -1)
    breaks = np.append(breaks,1.1)
    
    re = np.repeat(1/(len(breaks)-1), (len(breaks)-1))  
    for i in range(1,len(breaks)):
        re[i-1] = (re[i-1] + len(np.where((scores >= breaks[i-1]) & (scores < breaks[i]))[0]) ) / (len(scores)+1)
    return re

def class_dist(Y, nclasses):
    return np.array([np.count_nonzero(Y == i) for i in range(nclasses)]) / Y.shape[0]

def class2index(labels, classes):
    return np.array([classes.index(labels[i]) for i in range(labels.shape[0])])

# Base quantifiers

def DyS(pos_scores, neg_scores, test_scores, measure, binsize):
    
    if binsize == 'DyS':
        bin_size = np.linspace(2,10,10)   #creating bins from 2 to 10 with step size 2
        bin_size = np.append(bin_size, 30)
    else:
        bin_size = np.linspace(10,110,11)
        
    alphas = np.zeros(len(bin_size))
    dists = np.zeros(len(bin_size))
    for i, bins in enumerate(bin_size):
        #....Creating Histograms bins score\counts for validation and test set...............
        
        p_bin_count = getHist(pos_scores, bins)
        n_bin_count = getHist(neg_scores, bins)
        te_bin_count = getHist(test_scores, bins)
        
        def f(x):            
            return(distance(((p_bin_count*x) + (n_bin_count*(1-x))), te_bin_count, measure = measure))
    
        alphas[i], dists[i] = TernarySearch(0, 1, f)
    
    return np.median(alphas)
 

def HDy(train_scores, test_scores, train_labels, nmodels, nclasses):
    
    p_hat = np.zeros((1, nclasses))
    if nclasses == 2:
        pos_scores = [x[1] for indx,x in enumerate(train_scores) if train_labels[indx] == 1]
        neg_scores = [x[1] for indx,x in enumerate(train_scores) if train_labels[indx] == 0]
        te_scores = [x[1] for x in test_scores[0]]
        p_hat[0][1] = DyS(pos_scores, neg_scores, te_scores, measure='hellinger', binsize = 'HDy')
        p_hat[0][0] = 1 - (p_hat[0][1])
    else:
      
        for c in range(nclasses):
            pos_scores = [x[c] for indx,x in enumerate(train_scores) if train_labels[indx] == c]
            neg_scores = [x[c] for indx,x in enumerate(train_scores) if train_labels[indx] != c]
            te_scores = [x[c] for x in test_scores]
            p_hat[0][c] = DyS(pos_scores, neg_scores, te_scores, measure='hellinger', binsize = 'HDy')
        p_hat[0] = p_hat[0] / np.sum(p_hat[0])
        
    p = np.median(p_hat, axis = 0)
    return(p/np.sum(p))

   
def EMQ(test_scores, train_labels, nclasses):
    max_it = 1000        # Max num of iterations
    eps = 1e-6           # Small constant for stopping criterium

    p_tr = class_dist(train_labels, nclasses)
    p_s = np.copy(p_tr)
    p_cond_tr = np.array(test_scores)
    p_cond_s = np.zeros(p_cond_tr.shape)

    for it in range(max_it):
        r = p_s / p_tr
        p_cond_s = p_cond_tr * r
        s = np.sum(p_cond_s, axis = 1)
        for c in range(nclasses):
            p_cond_s[:,c] = p_cond_s[:,c] / s
        p_s_old = np.copy(p_s)
        p_s = np.sum(p_cond_s, axis = 0) / p_cond_s.shape[0]
        if (np.sum(np.abs(p_s - p_s_old)) < eps):
            break

    return(p_s/np.sum(p_s))

def GAC(train_scores, test_scores, train_labels, nmodels, nclasses):
   
    yt_hat = np.argmax(train_scores, axis = 1)
    y_hat = np.argmax(test_scores, axis = 1)
    CM = metrics.confusion_matrix(train_labels, yt_hat, normalize="true").T
    p_y_hat = np.zeros(nclasses)
    values, counts = np.unique(y_hat, return_counts=True)
    p_y_hat[values] = counts 
    p_y_hat = p_y_hat/p_y_hat.sum()
    
    p_hat = cvx.Variable(CM.shape[1])
    constraints = [p_hat >= 0, cvx.sum(p_hat) == 1.0]
    problem = cvx.Problem(cvx.Minimize(cvx.norm(CM @ p_hat - p_y_hat)), constraints)
    problem.solve()
    return p_hat.value

def GPAC(train_scores, test_scores, train_labels, nmodels, nclasses):

    CM = np.zeros((nclasses, nclasses))
    for i in range(nclasses):
        idx = np.where(train_labels == i)[0]
        CM[i] = np.sum(train_scores[idx], axis=0)
        CM[i] /= np.sum(CM[i])
    CM = CM.T
    p_y_hat = np.sum(test_scores, axis = 0)
    p_y_hat = p_y_hat / np.sum(p_y_hat)
    
    p_hat = cvx.Variable(CM.shape[1])
    constraints = [p_hat >= 0, cvx.sum(p_hat) == 1.0]
    problem = cvx.Problem(cvx.Minimize(cvx.norm(CM @ p_hat - p_y_hat)), constraints)
    problem.solve()
    return p_hat.value

def FM(train_scores, test_scores, train_labels, nmodels, nclasses):

    CM = np.zeros((nclasses, nclasses))
    y_cts = np.array([np.count_nonzero(train_labels == i) for i in range(nclasses)])
    p_yt = y_cts / train_labels.shape[0]
    for i in range(nclasses):
        idx = np.where(train_labels == i)[0]
        CM[:, i] += np.sum(train_scores[idx] > p_yt, axis=0) 
    CM = CM / y_cts
    p_y_hat = np.sum(test_scores > p_yt, axis = 0) / test_scores.shape[0]
    
    p_hat = cvx.Variable(CM.shape[1])
    constraints = [p_hat >= 0, cvx.sum(p_hat) == 1.0]
    problem = cvx.Problem(cvx.Minimize(cvx.norm(CM @ p_hat - p_y_hat)), constraints)
    problem.solve()
    return p_hat.value


# Ensemble quantifiers

def EnsembleDyS(train_scores, test_scores, train_labels, nmodels, nclasses):
    p_hat = np.zeros((nmodels, nclasses))
    
    # if nclasses == 2 and nmodels == 1:
    #     pos_scores = [x[1] for indx,x in enumerate(train_scores) if train_labels[indx] == 1]
    #     neg_scores = [x[1] for indx,x in enumerate(train_scores) if train_labels[indx] == 0]
    #     te_scores = [x[1] for x in test_scores]
    #     p_hat[0, 1] = DyS(pos_scores, neg_scores, te_scores, measure='topsoe', binsize = 'DyS')
    #     p_hat[0, 0] = 1 - (p_hat[0, 1])
    
    # if nclasses > 2 and nmodels == 1:
    #     for c in range(nclasses):
    #         pos_scores = [x[c] for indx,x in enumerate(train_scores) if train_labels[indx] == c]
    #         neg_scores = [x[c] for indx,x in enumerate(train_scores) if train_labels[indx] != c]
    #         te_scores = [x[c] for x in test_scores]
    #         p_hat[0][c] = DyS(pos_scores, neg_scores, te_scores, measure='topsoe', binsize = 'DyS')
    #     p_hat = p_hat / np.sum(p_hat)
            
    if nclasses == 2:
        for m in range(nmodels):
            pos_scores = [x[1] for indx,x in enumerate(train_scores[m]) if train_labels[indx] == 1]
            neg_scores = [x[1] for indx,x in enumerate(train_scores[m]) if train_labels[indx] == 0]
            te_scores = [x[1] for x in test_scores[m]]
            p_hat[m][1] = DyS(pos_scores, neg_scores, te_scores, measure='topsoe', binsize = 'DyS')
            p_hat[m][0] = 1 - (p_hat[m][1])
 
    if nclasses > 2:
        for m in range(nmodels):
            for c in range(nclasses):
                pos_scores = [x[c] for indx,x in enumerate(train_scores[m]) if train_labels[indx] == c]
                neg_scores = [x[c] for indx,x in enumerate(train_scores[m]) if train_labels[indx] != c]
                te_scores = [x[c] for x in test_scores[m]]
                p_hat[m][c] = DyS(pos_scores, neg_scores, te_scores, measure='topsoe', binsize = 'DyS')
            p_hat[m] = p_hat[m] / np.sum(p_hat[m])
            
    p = np.median(p_hat, axis = 0)
    return(p/np.sum(p))

def EnsembleEM(train_scores, test_scores, train_labels, nmodels, nclasses):
    p_hat = np.zeros((nmodels, nclasses))
    for m in range(nmodels):
        p_hat[m] = EMQ(test_scores[m], train_labels, nclasses)
    
    p = np.median(p_hat, axis = 0)
    return(p/np.sum(p))

def EnsembleGAC(train_scores, test_scores, train_labels, nmodels, nclasses):
    p_hat = np.zeros((nmodels, nclasses))
    for m in range(nmodels):
        p_hat[m] = GAC(train_scores[m], test_scores[m], train_labels, nmodels, nclasses)
    
    p = np.median(p_hat, axis = 0)
    return(p/np.sum(p))

def EnsembleGPAC(train_scores, test_scores, train_labels, nmodels, nclasses):
    p_hat = np.zeros((nmodels, nclasses))
    for m in range(nmodels):
        p_hat[m] = GPAC(train_scores[m], test_scores[m], train_labels, nmodels, nclasses)
    
    p = np.median(p_hat, axis = 0)
    return(p/np.sum(p))

def EnsembleFM(train_scores, test_scores, train_labels, nmodels, nclasses):
    p_hat = np.zeros((nmodels, nclasses))
    for m in range(nmodels):
        p_hat[m] = FM(train_scores[m], test_scores[m], train_labels, nmodels, nclasses)
    
    p = np.median(p_hat, axis = 0)
    return(p/np.sum(p))


