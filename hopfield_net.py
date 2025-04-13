import numpy as np

class Hopfield:
    def __init__(self, num_neurons, random_init = True):
        self.num_neurons = num_neurons
        
        if random_init == True:
            self.initialise_neurons()
            self.initialise_weights()
        else:
            self.neurons = None
            self.weights = None
            
            
        
    def update_neurons(self):
        activation =  self.weights @ self.neurons 
        self.neurons = ((activation > 0).astype(int)*2)-1

    def _update_weights(self):
        self.weights = (self.neurons @ self.neurons.T)
        np.fill_diagonal(self.weights, 0)
        
    def update_weights_patterns(self, patterns: np.array):
        no_patterns = patterns.shape[1]
        self.weights = (patterns.T @ patterns) / no_patterns
        np.fill_diagonal(self.weights, 0)
    
    def find_energy(self):
        energy = self.neurons.T @ (self.weights @ self.neurons)
        return -1*(energy/2)[0][0]
    
    
    def initialise_neurons(self):
        self.neurons = np.random.random(self.num_neurons)
        self.neurons = self.neurons*2 - 1
        self.neurons = ((self.neurons > 0).astype(int)*2)-1
        self.neurons = self.neurons[:, None]
        
    def initialise_weights(self):
        self.weights = np.random.randint(-1,2,size=(self.num_neurons, self.num_neurons))
        self.weights = np.tril(self.weights) + np.triu(self.weights.T, 1) # make it symmetrical!
        np.fill_diagonal(self.weights, 0)
    