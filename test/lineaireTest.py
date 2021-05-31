from Modules.Loss1 import *
from Modules.Modules1 import *
import numpy as np
import matplotlib.pyplot as plt
from tools.tool import *

a = 15
b = 0
def f(x1):
    return x1*a + b

def f_bruit(x1):
    bruit = np.random.normal(0,110,len(x1)).reshape((-1,1))
    return f(x1) + bruit

nb_data = 100
data = np.random.uniform(-10,10,nb_data).reshape((-1,1))
label = f_bruit(data)

in_size  = 1
out_size = 1

iterations = 300
batch_size = 10

lin1 = Linear((in_size, out_size), b = False)

seq = Sequentiel(m = [lin1], fsortie = None)
optim = Optim(seq, MSELoss(), 1e-3)

moyenne , variance = SGD(data, label, optim,10, 900)

plt.plot(moyenne)
plt.show()

x = np.array([-10, 10])

plt.plot(x,f(x), c= 'r')
plt.plot(x,lin1.forward(x.reshape(-1,1)))
plt.legend()
plt.show()
