import numpy as np

class Loss(object):
    def forward(self, y, yhat):
        pass

    def backward(self, y, yhat):
        pass
    
    
class MSELoss(Loss):
    def forward(self, y, yhat):
        return np.sum(np.power(y-yhat,2), axis = 1)

    def backward(self, y, yhat):
        return -2*(y-yhat)

class CELoss(Loss):
    def forward(self, y, yhat, eps = 1e-100):
        return 1-np.sum(yhat*y, axis = 1)
    
    def backward(self, y, yhat, eps = 1e-100):
        return -y
 

class BCELoss(Loss):
    def forward(self, y, yhat, eps = 1e-100):
        return - (y*np.log(yhat+eps) + (1-y)*np.log(1-yhat+eps))
    def backward(self, y, yhat, eps = 1e-100):
        return ((1-y)/(1-yhat +eps)) - (y/(yhat +eps))
    
class CESoftMax(Loss):
    def forward(self, y, yhat, eps=10e-100):
        return - np.sum(y * yhat, axis=1) + np.log(np.sum(np.exp(yhat), axis=1))

    def backward(self, y, yhat, eps=10e-100):
        return - y + np.exp(yhat) / np.sum(np.exp(yhat), axis=1)[..., None]