# Kanchi, Gowtham Kumar
# 1002_044_003
# 2022_09_25
# Assignment_01_01

import numpy as np

class SingleLayerNN(object):
    def __init__(self, input_dimensions=2,number_of_nodes=4):
        
        self.input_dimensions=input_dimensions	
        self.number_of_nodes = number_of_nodes
        self.initialize_weights
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------        
    def initialize_weights(self,seed=None):
        
        if seed != None:
            np.random.seed(seed)
        self.weights = np.random.randn(self.number_of_nodes,self.input_dimensions+1)

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def set_weights(self, W):
        
        x,y = W.shape	
        if x == self.number_of_nodes and y == self.input_dimensions+1 :	
            self.weights = W	
        else:	
            return -1
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def get_weights(self):
        
        return self.weights
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def predict(self, X):
        
        input_bias = np.ones((1,X.shape[1]))	
        X = np.vstack((input_bias,X))	
        net_val = np.dot(self.weights,X)	
        hard_limit = np.multiply(net_val>=0,1)	
        return hard_limit

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def train(self, X, Y, num_epochs=10,  alpha=0.1):
        
        
        for num_epoch in range(num_epochs):	
            for s in range(X.shape[1]):	
                predict = self.predict(X[0:,s:s+1])	
                self.weights[0:,1:] = self.weights[0:,1:] + alpha * np.dot(Y[0:,s:s+1] - predict,X[0:,s:s+1].transpose())	
                self.weights[0:,0:1] = self.weights[0:,0:1] + alpha * (Y[0:,s:s+1] - predict)


----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def calculate_percent_error(self,X, Y):
        
        error = 0	
        for s in range(X.shape[1]):		
            	
            if (not np.array_equal(Y[0:,s:s+1],self.predict(X[0:,s:s+1]))):	
                error+=1	
        percentage_error = 100 * (error/X.shape[1])	
        return percentage_error

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    input_dimensions = 2
    number_of_nodes = 2

    model = SingleLayerNN(input_dimensions=input_dimensions, number_of_nodes=number_of_nodes)
    model.initialize_weights(seed=2)
    X_train = np.array([[-1.43815556, 0.10089809, -1.25432937, 1.48410426],
                        [-1.81784194, 0.42935033, -1.2806198, 0.06527391]])
    print(model.predict(X_train))
    Y_train = np.array([[1, 0, 0, 1], [0, 1, 1, 0]])
    print("****** Model weights ******\n",model.get_weights())
    print("****** Input samples ******\n",X_train)
    print("****** Desired Output ******\n",Y_train)
    percent_error=[]
    for k in range (20):
        model.train(X_train, Y_train, num_epochs=1, alpha=0.1)
        percent_error.append(model.calculate_percent_error(X_train,Y_train))
    print("******  Percent Error ******\n",percent_error)
    print("****** Model weights ******\n",model.get_weights())
