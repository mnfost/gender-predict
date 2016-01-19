# -*- coding: utf-8 -*-
"""
This code implements a simple classification neural network that fits through backpropagation and 
gradient descent. It currently only supports one hidden layer.
"""
import numpy as np
import matplotlib.pyplot as plt
#from sklearn import preprocessing
#import random


class NeuralNet():
    
    def __init__(self, num_nodes, weights_=[], bias_set=False, classification=True, penalty=0., learn_rate=0.0001):
        self.num_nodes = num_nodes
        self.weights_ = weights_
        self.bias_set = bias_set
        self.is_fit = False
        self.classification = classification
        self.K = 0
        self.penalty = penalty
        self.learn_rate = learn_rate
        np.seterr(all='warn')
        
    def initWeights(self, X, nclass):
        bias = np.ones((X.shape[0],1))
        X = np.hstack((bias,X)) #add constant bias to each observation, X now N by P+1
        sizeX = X.shape[1]
        size_nodes = sizeX*(self.num_nodes+1) #include bias node
        size_output = nclass*(self.num_nodes+1) #include bias node
#        node_weights_ = np.random.uniform(-0.7,0.7,size_nodes).reshape((self.num_nodes+1,sizeX)) #M+1 by P+1
        node_weights_ = np.random.normal(scale=1./X.shape[0],size=size_nodes).reshape((self.num_nodes+1,sizeX))
#        output_weights_ = np.random.uniform(-0.7,0.7,size_output).reshape((nclass,self.num_nodes+1)) #K by M+1
        output_weights_ = np.random.normal(scale=1./X.shape[0],size=size_output).reshape((nclass,self.num_nodes+1))
        return X, [node_weights_, output_weights_] #nrows = n_nodes, ncols = sizeX
        
    def sigmoid(self, alpha_, X_):
        v_ = alpha_.dot(X_.T)
        v_[v_ < -300] = -300
        v_[v_ > 300] = 300
#        print v_
        return 1./(1+np.exp(-v_))
        
    def relu(self, alpha_, X_):
        v_ = alpha_.dot(X_.T)
        v_[v_ < -300] = -300
        v_[v_ > 300] = 300
        return np.maximum(v_,np.zeros(v_.shape))
        
    def drelu(self, alpha_, X_):
        v_ = alpha_.dot(X_.T)
        v_[v_ <= 0.] = 0
        v_[v_ > 0.] = 1.
        return v_
        
    def tanh(self, alpha_, X_):
        v_ = alpha_.dot(X_.T)
        v_[v_ < -300] = -300
        v_[v_ > 300] = 300
        return np.tanh(v_)
        
    def dtanh(self, alpha_, X_):
        return 1 - np.multiply(self.tanh(alpha_, X_), self.tanh(alpha_, X_))
        
    def softmax(self, T):
        T[T < -300] = -300
        T[T > 300] = 300
        return (np.exp(T)/np.sum(np.exp(T), axis=0)).T #(K by N) / elementwise(1 by N)
        
    def initNodes(self, X, Y):
        K = self.K
        if self.weights_ == []:
            X, weights = self.initWeights(X, K)
        else:
            weights = self.weights_
        sig = self.tanh(weights[0], X) #(M+1 by P+1)*(P+1 by N) = M+1 by N
#        sig = self.relu(weights[0], X)
        output = weights[1].dot(sig) #(K by M+1)*(M+1 by N) = K by N
        if self.classification:
            h_ = self.softmax(output) #N by K
        else:
            h_ = output
        return X, weights, sig, h_
        
    def backPropagate(self, weights, X, Y, old_del_alpha, old_del_beta, _dropout):
        #feed forward then back propagate, update weights
        learn_rate = self.learn_rate
        if _dropout:
            n_drops = np.round(weights[0].shape[0]/2.)
            rands = np.random.randint(0,weights[0].shape[0],n_drops)
            sig = self.tanh(weights[0][rands,:], X)
#            sig = self.relu(weights[0][rands,:], X)
            hidden_out = weights[1][:,rands].dot(sig)
    #        print weights[1]
            if self.classification:
                h = self.softmax(hidden_out)
    #            h = hidden_out.T
                forward_error = h - Y
            else:
                h = hidden_out.T
                forward_error = 2*(h - Y) #both N by K
            dRdBeta = sig.dot(forward_error) #(M+1 by N)*(N by K) = M+1 by K gradient-force for each neuron
    #        dsig = np.multiply(sig,1-sig) #M+1 by N
            dsig = self.dtanh(weights[0][rands,:], X)
#            dsig = self.drelu(weights[0][rands,:], X)
#            back_error = np.multiply((forward_error.dot(weights[1][:,rands])),(dsig.T)) #((N by K)*(K by M+1))*ewise(N by M+1) = N by M+1
            back_error = forward_error.dot(weights[1][:,rands])            
            dRdAlpha = (back_error.T).dot(X) #(M+1 by N)*(N by P+1) = M+1 by P+1 back-propagated gradient
            
            #descent with momentum:
            rho = 0.9

            del_beta = rho*old_del_beta - learn_rate*dRdBeta.T
            del_alpha = rho*old_del_alpha - learn_rate*dRdAlpha
            
            weights[1][:,rands] += del_beta #M+1 by K
            weights[0][rands,:] += del_alpha #M+1 by P+1
            
        else:
            sig = self.tanh(weights[0], X)
#            sig = self.relu(weights[0], X)
            hidden_out = weights[1].dot(sig)
            if self.classification:
                h = self.softmax(hidden_out)
    #            h = hidden_out.T
                forward_error = h - Y
            else:
                h = hidden_out.T
                forward_error = h - Y #both N by K
            dRdBeta = sig.dot(forward_error) #(M+1 by N)*(N by K) = M+1 by K gradient-force for each neuron
    #        dsig = np.multiply(sig,1-sig) #M+1 by N
            dsig = self.dtanh(weights[0], X)
#            dsig = self.drelu(weights[0], X)
            back_error = np.multiply((forward_error.dot(weights[1])),(dsig.T)) #((N by K)*(K by M+1))*ewise(N by M+1) = N by M+1
#            back_error = forward_error.dot(weights[1])            
            dRdAlpha = (back_error.T).dot(X) #(M+1 by N)*(N by P+1) = M+1 by P+1 back-propagated gradient
            
            #descent with momentum:
            rho = 0.9

            del_beta = rho*old_del_beta - learn_rate*dRdBeta.T
            del_alpha = rho*old_del_alpha - learn_rate*dRdAlpha
            
            weights[1] = weights[1] + del_beta + 2*self.penalty*weights[1] #M+1 by K
            weights[0] = weights[0] + del_alpha + 2*self.penalty*weights[0] #M+1 by P+1

        return weights, del_alpha, del_beta, dsig, forward_error
        
    def fit(self, X, Y, maxiter=300, tol=0.000001, xtest=[], xall=[], yall=[], dropout=False, batch=10, SGD=False):
        grad_alpha, grad_beta = 0., 0.
        if self.is_fit:
            self.weights_ = []
            w = []
        self.is_fit = True
        if self.classification:
        #one-hot encode Y
            try:
                #if already one-hot encoded, pass Y as Y_new
                if Y.shape[1] > 1:
                    Y_new = Y
                #else one-hot encode Y as Y_new
                else:
                    self.K = len(set(Y.flatten()))
                    Y_new = np.zeros((len(Y),self.K))
                    for i,v in enumerate(Y):
                        Y_new[i,v] = 1.
            #if Y.shape[1] null (1D array), one-hot encode it as Y_new
            except IndexError:
                self.K = len(set(Y.flatten())) #ditto
                Y_new = np.zeros((len(Y),self.K))
                for i,v in enumerate(Y):
                    Y_new[i,v] = 1.
        else:
            Y_new = Y
            self.K = 1
        X, w, init_node_act, h = self.initNodes(X, Y_new)
        if len(xtest) != 0:
            bias = np.ones((xtest.shape[0],1))
            xtest = np.hstack((bias,xtest)) #add constant bias to each observation, X now N by P+1

        for i in xrange(maxiter):
            if not SGD:
                w, grad_alpha, grad_beta, dsig, error = self.backPropagate(w, X, Y_new, grad_alpha, grad_beta, dropout) 
#                if np.mean(dsig) <= 0.000001:
#                    break
            else:
                samples = np.random.choice(range(X.shape[0]),size=batch,replace=False)
                w, grad_alpha, grad_beta, dsig, error = self.backPropagate(w, X[samples,:], Y_new[samples,:], grad_alpha, grad_beta, dropout) 
            
        self.weights_ = w
        
    def predict(self, X, proba=False):
        if self.is_fit:
            self.predictions = []
            bias = np.ones((X.shape[0],1))
            X = np.hstack((bias,X)) #add constant bias to each observation, X now N by P+1
            activation = self.tanh(self.weights_[0], X)
#            print activation
            response = self.weights_[1].dot(activation)
            if self.classification:
                predictions = self.softmax(response)
                if not proba:
                    self.predictions = np.argmax(predictions, axis=1)
                    return self.predictions
                else:
                    self.predictions = predictions
                    return self.predictions
            else:
                self.predictions = response
                return self.predictions
        else:
            return "Cannot predict without fitting data first!!"
            
    def predict_proba(self, X):
        if self.is_fit:
            self.predictions = []
            bias = np.ones((X.shape[0],1))
            X = np.hstack((bias,X)) #add constant bias to each observation, X now N by P+1
            activation = self.tanh(self.weights_[0], X)
#            print activation
            response = self.weights_[1].dot(activation)
            if self.classification:
                predictions = self.softmax(response)
            self.predictions = predictions
            return self.predictions
        else:
            return "Cannot predict without fitting data first!!"
        
    def score(self, X_test, Y_test):
        predictions = self.predict(X_test)
        if self.classification:
            num_correct = predictions == np.array(Y_test).flatten()
            return float(len(Y_test.flatten()[num_correct]))/len(Y_test)
        else:
            n = len(Y_test)
            diff = predictions - Y_test
            MSE = 1. - sum(np.multiply(diff,diff))/n
            return MSE
        