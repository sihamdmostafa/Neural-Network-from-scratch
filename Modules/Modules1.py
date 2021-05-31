import numpy as np

class Module(object):
    def __init__(self):
        self.parameters = None
        self.gradient = None

    def zero_grad(self):
        ## Annule gradient
        pass

    def forward(self, X):
        ## Calcule la passe forward
        pass

    def updateparameters(self, gradient_step=1e-3):
        ## Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step
        print(gradient_step)
    
        self.parameters -= gradient_step*self.gradient

    def backward_updategradient(self, input, delta):
        ## Met a jour la valeur du gradient
        pass

    def backward_delta(self, input, delta):
        ## Calcul la derivee de l'erreur
        pass
   
class Linear(Module):    
    def __init__(self, dim = None,b = True):
        self.grad   = None
        self.parameters = np.random.random(dim)
        if type(self.parameters) != type(None) and b:
            self.b = np.random.random((1,dim[1]))
        else :
            self.b = None            
    
    def forward(self, X):
        if type(self.b) != type(None):
            return np.dot(X, self.parameters) + self.b
        return np.dot(X, self.parameters)
    
    def update_parameters(self, gradient_step=1e-3):
        self.parameters -= gradient_step*self.grad
        if type(self.b) != type(None):
            self.b -=gradient_step*self.bgrad
    
    def backward_updategradient(self, input, delta):
        try:
            self.grad += np.dot(input.T, delta)
        except (AttributeError, TypeError): 
            self.grad = np.dot(input.T, delta)
        
        if type(self.b) != type(None):
            try:
                self.bgrad += np.sum(delta, axis = 0)
            except (AttributeError, TypeError):
                self.bgrad = np.sum(delta, axis = 0)
                
    def backward_delta(self, input, delta):
        return np.dot(delta, self.parameters.T)
    
    def zero_grad(self):
        self.bgrad = None
        self.grad = None


class TanH(Module):
    def forward(self, X):
        return np.tanh(X)
    def backward_updategradient(self, input, delta):
        pass
    def backward_delta(self, input, delta):
        return delta * (1-np.power(np.tanh(input),2))    
    def update_parameters(self, gradient_step=1e-3):
        pass
    
class Sigmoid(Module):
    def forward(self, X):
        return 1/(1 + np.exp(-X))
    def backward_delta(self, input, delta):
        return delta * (np.exp(-input)/np.power(1+np.exp(-input), 2))   
    def update_parameters(self, gradient_step=1e-3):
        pass



class Sequentiel:
    def __init__(self, m = None, fsortie = None):
        self.modules = m
        self.fsortie = fsortie
    def forward(self, x):
        forward_list = [x]
        for m in self.modules:
            forward_list += [m.forward(forward_list[-1])]
        forward_list.reverse()
        return forward_list
        
    def backward(self, l, delta_first):
        delta = [delta_first]
        for i, m in enumerate(np.flip(self.modules)):
            m.backward_updategradient(l[i+1], delta[-1])
            delta += [m.backward_delta(l[i+1] , delta[-1])]
    
    def update_parameters(self, eps):
        for m in self.modules:
            m.update_parameters(gradient_step=eps)
            m.zero_grad()
                
    def predict(self, x):
        if type(self.fsortie) != type(None):
            return self.fsortie(self.forward(x)[0])
        return self.forward(x)[0]
   
class Optim:
    
    def __init__(self, net, loss, eps = 1e-2):
        self.net  = net
        self.loss = loss
        self.eps  = eps       
    def step(self,batch_x, batch_y):
        list_forward_batch = self.net.forward(batch_x)
        list_delta = self.loss.backward(batch_y, list_forward_batch[0])
        self.net.backward(list_forward_batch, list_delta)
        self.net.update_parameters(self.eps)
        
    def accuracy(self, x, y):
        yhat = self.net.predict(x)
        return np.sum(yhat == y)/len(yhat)
        
    def loss_value(self, x, y):
        l = self.net.forward(x)
        return self.loss.forward(y, l[0])
    
def SGD(data, label, optim, batch_size, iterations):     
    b_data  = data[data.shape[0]%batch_size:].reshape([-1,batch_size] + list(data.shape[1:]))
    b_label = label[data.shape[0]%batch_size:].reshape([-1,batch_size] + list(label.shape[1:]))
    
    mean = []
    std  = []
    for i in range(iterations):
        cpt = []
        for j in range(len(b_label)):
            cpt += [optim.loss_value(b_data[j],b_label[j])[-1]]
            optim.step(b_data[j], b_label[j])
        mean += [np.mean(cpt)]
        std  += [np.std(cpt)]
    return mean, std




class Softmax(Module):
    def forward(self, X):
        return np.exp(X) / np.sum(np.exp(X), axis=1)[..., None]
    def backward_delta(self, input, delta):
        softmax = np.exp(input) / np.sum(np.exp(input), axis=1)[..., None]
        return delta * softmax * (1 - softmax)
    def update_parameters(self, gradient_step=1e-3):
        pass

        
class Flatten(Module):
    def forward(self, X):
        return X.reshape(len(X), -1)

    def backward_delta(self, input, delta):
        return delta.reshape(input.shape)

class ReLU(Module):
    def forward(self, X):
        return np.where(X < 0, 0, X)

    def backward_delta(self, input, delta):
        return delta * np.where(input < 0, 0, 1)

class Conv1D(Module):    
    def __init__(self, k_size, chan_in, chan_out, stride = 0, init = 'xavier'):
        """
        Dimensions : un tuple (dim_in, dim_out), si les dimensions sont passées
        initialise les parameters aléatoirement selon la méthode définit dans init,
        uniforme sinon.
        bias : if true, ajoute un bias au module
        
        """
        self._gradient   = None
        self._parameters = Module.initialise((k_size, chan_in, chan_out), init)
        self.k_size      = k_size
        self.stride      = stride
        self.chan_in     = chan_in
        self.chan_out    = chan_out
    
    def forward(self, X):
        """
        input  : batch * input * chan_in
        output : batch * (input - k_size/stride + 1)* chan_out
        """
        assert X.shape[2] == self.chan_in
        a,_ = np.mgrid[0:X.shape[1]-self.k_size:(self.stride+1), 0:self.k_size] + np.arange(self.k_size)
        a = a.reshape(-1)
        input = X[:,a,:].reshape(X.shape[0], -1, self.k_size* self.chan_in)
        input = np.transpose(input, (1, 0, 2))
        def call(x):
            return np.dot(x,self._parameters.reshape(-1, self.chan_out))
        return np.transpose(np.array(list(map(call, input))), (1,0,2))
    
    def update_parameters(self, gradient_step=1e-3):
        ## Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step
        self._parameters -= gradient_step*self._gradient
    
    def backward_update_gradient(self, input, delta):
        ## Met a jour la valeur du gradient
        if self._gradient is None: 
            self._gradient = np.zeros(self._parameters.shape)
        
        a,_ = np.mgrid[0:input.shape[1]-self.k_size:(self.stride+1), 0:self.k_size] \
        + np.arange(self.k_size)
        a = a.reshape(-1)
        input = input[:,a,:].reshape(input.shape[0], -1, self.k_size* self.chan_in)
        input = np.transpose(input, (1, 0, 2))
        for i in range(input.shape[0]):
            self._gradient += np.dot(input[i,:].T,delta[:, i, :]).reshape(self._gradient.shape)
            

    def backward_delta(self, input, delta):
        z = zip(range(0, input.shape[1], 1+ self.stride), \
        range(self.k_size, input.shape[1], 1+self.stride))
        res = np.zeros(input.shape)
        for i, (begin, end) in enumerate(z):
            d = np.dot(delta[:, i, :], \
                self._parameters.reshape(-1, self.chan_out).T)
            res[:,begin:end] += d.reshape(-1, self.k_size, self.chan_in)           
        return res
    
    def zero_grad(self):
        self._gradient = None

class MaxPool1D(Module):    
    def __init__(self, k_size, stride = 0):
        """
        Dimensions : un tuple (dim_in, dim_out), si les dimensions sont passées
        initialise les parameters aléatoirement selon la méthode définit dans init,
        uniforme sinon.
        bias : if true, ajoute un bias au module
        
        """
        self.k_size = k_size
        self.stride = stride
    
    def forward(self, X):
        """
        input  : batch * input * chan_in
        output : batch * (input - k_size/stride + 1, chan_in)
        """
        z = zip(range(0, X.shape[1], 1+self.stride), \
                range(self.k_size, X.shape[1], 1+self.stride))
        tmp = np.array([np.max(X[:,beg:end], axis = 1) for beg, end in z])
        return np.transpose(tmp, (1,0,2))
    
    def update_parameters(self, gradient_step=1e-3):
        ## Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step
        pass
            
    def backward_delta(self, input, delta):
        #Doit avoir la même dimension que l'inputS
        z = zip(range(0, input.shape[1], 1+self.stride), \
        range(self.k_size, input.shape[1], 1+self.stride))
        dim = input.shape[-1]
        batch = input.shape[0]
        res = np.zeros(input.shape)
        for i, (beg, end) in enumerate(z):
            t = np.argmax(input[:,beg:end], axis = 1)
            res[np.repeat(range(batch),dim),beg + t.reshape(-1),\
                np.tile(range(dim),batch)]\
                += delta[:,i,:].reshape(-1)
        return res