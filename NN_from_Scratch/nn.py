import numpy as np

def init_weights(prev_units):
    return np.random.rand(prev_units)  
def init_bias():
    return np.random.rand(1)[0]  

def Spike(activation, Act_Pot):
    if activation.lower() == 'relu':
        return max(Act_Pot, 0)
    elif activation.lower() == 'linear':
        return Act_Pot
    elif activation.lower() == 'leaky_relu':
        if Act_Pot > 0:
            return max(Act_Pot, 0)           
        elif Act_Pot <= 0:
            return (0.01)*Act_Pot
    elif activation.lower() == 'noisy_relu':
        return max(Act_Pot + np.random.randn(1)[0], 0)
    elif activation.lower() == 'sigmoid':
        return 1/(1 + np.exp(-Act_Pot))
    
def Sequential():
    def __init__(self, inputs_dim):
        self.layers = []
        layers.append(Layer(inputs_dim, 'linear' , 0))
        self.layerID = 0
        
    
    def Add(self, units ,activation):
        self.layerID += 1
        self.layers.append(Layer(units ,activation, self.layerID))
        
    def access_layer(self, layerID):
        return self.layers[layerID]
    
    def Output(self):
        pass
        
    def Compute()
        pass
    
def Layer():

    def access_layer(self, layerID):
        return super().access_layer(layerID)
    def prev_layer(self):
        return super().access_layer(self.layerID - 1)
    
    def __init__(self,units , activation ,layerID):
        self.layerID = layerID
        self.units = units
        self.activation = activation
        self.neurons = []
        for unit in range(units):
            self.neurons.append(Neuron(super().prev_layer().units ,activation))
        self.outputs = []
        
def Neuron():
    def __init__(self, input_units, activation):
        self.inputs = []
        self.activation = activation
        self.weights = init_weights(self.input_units)
        self.bias = init_bias()
        self.output = 0

    def Compute(self):
        Act_Pot = np.array(self.inputs) * np.array(self.weights).transpose() + self.bias
        return Spike(self.activation, Act_Pot)
    
