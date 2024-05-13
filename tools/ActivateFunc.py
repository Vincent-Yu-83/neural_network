import matplotlib.pyplot as plt
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return 2 / (1 + np.exp(-2*x)) - 1

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    x = np.exp(x) / np.sum(np.exp(x))
    return x

def leaky_relu(x,a=0.01):
    return np.maximum(a*x, x)

def elu(x,alpha=1):
    a = x[x>0]
    b = alpha*(np.exp(x[x<0])-1)
    result=np.concatenate((b,a),axis=0)
    return result

def swish(x):
    return sigmoid(x) * x

def Squareplus(x, b=0.2):
    x = 0.5 * (x + np.sqrt(x**2+b))
    return x


def showImg(title,x,lines):
    
    fig, ax = plt.subplots()
    
    for line in lines: 
        ax.plot(x, line[0],line[1],line[2])
        
    # ax.legend() # 设置图例
    # 画轴
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['left'].set_position(('axes', 0.5))
    plt.grid() # 设置方格
    plt.title(title)
    plt.show()

x = np.linspace(-10, 10, 100)

lines = np.array([[sigmoid(x),'b','Sigmoid']],dtype=object)
showImg("Sigmoid",np.linspace(-10, 10, 100),lines)

lines = np.array([[sigmoid(x),'b','Sigmoid']],dtype=object)
lines = np.vstack((lines_st,[[tanh(x), '-r', 'Tanh']]))
showImg("Tanh and Sigmoid",np.linspace(-10, 10, 100),lines)

lines = np.array([[relu(x),'b','ReLU']],dtype=object)
showImg("ReLU",np.linspace(-10, 10, 100),lines)

lines = np.array([[softmax(x),'b','Softmax']],dtype=object)
showImg("Softmax",np.linspace(-10, 10, 100),lines)

lines = np.array([[leaky_relu(x),'b','Leaky ReLu']],dtype=object)
showImg("Leaky ReLu",np.linspace(-10, 10, 100),lines)

lines = np.array([[elu(x),'b','ELU']],dtype=object)
showImg("ELU",np.linspace(-10, 10, 100),lines)

lines = np.array([[swish(x),'b','Swish']],dtype=object)
showImg("Swish",np.linspace(-10, 10, 100),lines)

lines = np.array([[Squareplus(x),'b','Squareplus']],dtype=object)
showImg("Squareplus",np.linspace(-10, 10, 100),lines)