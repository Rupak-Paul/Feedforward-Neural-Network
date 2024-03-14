import numpy as np
from keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split

class NuralNetwork:
    def __init__(self, numOfHiddenLayer, sizeOfHiddenLayer, inputSize, outputSize, activationFun, initilizationMethod):
        self.numOfHiddenLayer = numOfHiddenLayer
        self.sizeOfHiddenLayer = sizeOfHiddenLayer
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.activationFun = activationFun
        self.W, self.B = self.__initilizeWandB(initilizationMethod)

    def __initilizeWandB(self, initilizationMethod="None"):
        W = []
        B = []
        
        if(initilizationMethod == "random"):
            W.append(np.random.rand((self.sizeOfHiddenLayer, self.inputSize)))
            B.append(np.random.rand(self.sizeOfHiddenLayer))
        elif(initilizationMethod == "Xavier"):
            W.append(np.random.normal(0.0, 1.0/np.sqrt(self.inputSize), (self.sizeOfHiddenLayer, self.inputSize)))
            B.append(np.zeros(self.sizeOfHiddenLayer))
        else:
            W.append(np.zeros((self.sizeOfHiddenLayer, self.inputSize)))
            B.append(np.zeros(self.sizeOfHiddenLayer))
        
        for i in range(1, self.numOfHiddenLayer+1):
            if(initilizationMethod == "random"):
                W.append(np.random.rand((self.sizeOfHiddenLayer, self.sizeOfHiddenLayer)))
                B.append(np.random.rand(self.sizeOfHiddenLayer))
            
            elif(initilizationMethod == "Xavier"):
                W.append(np.random.normal(0.0, 1.0/np.sqrt(self.inputSize), (self.sizeOfHiddenLayer, self.sizeOfHiddenLayer)))
                B.append(np.random.normal(0.0, 1.0/np.sqrt(self.inputSize), self.sizeOfHiddenLayer))
            else:
                W.append(np.zeros((self.sizeOfHiddenLayer, self.sizeOfHiddenLayer)))
                B.append(np.zeros(self.sizeOfHiddenLayer))
        
        return W, B
    
    def predict(self, input):
        A, H = self.__forwardPropagation(input)
        output = H[self.numOfHiddenLayer]
        output = output[:self.outputSize]
        return np.argmax(output)

    def trainByMomentumGradientDescent(self, epochs, batchSize, eta, beta, train_X, train_Y, val_X, val_Y):
        prev_uw, prev_ub = self.__initilizeWandB()
        
        for i in range(epochs):
            dw, db = self.__initilizeWandB()
            pointSeen = 0
            loss = 0.0
            
            for X, Y in zip(train_X, train_Y):
                A, H = self.__forwardPropagation(X)
                diffW, diffB = self.__backwardPropagation(A, H, X, Y)
                
                dw = self.__addGradient(dw, diffW)
                db = self.__addGradient(db, diffB)
                
                loss += self.__calculateLoss(H[self.numOfHiddenLayer], Y)
                pointSeen += 1
                if(pointSeen % batchSize == 0):
                    uw = self.__addGradient(self.__multiplyGradient(prev_uw, beta), self.__multiplyGradient(dw, eta))
                    ub = self.__addGradient(self.__multiplyGradient(prev_ub, beta), self.__multiplyGradient(db, eta))
                    
                    self.W = self.__subtractGradient(self.W, uw)
                    self.B = self.__subtractGradient(self.B, ub)
                    
                    prev_uw = uw
                    prev_ub = ub
                    dw, db = self.__initilizeWandB()
                
            print("Average Loss after epoch ", i+1, " : ", loss/len(train_X))

    def trainByNesterovGradientDescent(self, epochs, batchSize, eta, beta, train_X, train_Y, val_X, val_Y):
        prev_vw, prev_vb = self.__initilizeWandB()
        
        for i in range(epochs):
            dw, db = self.__initilizeWandB()
            v_w = self.__multiplyGradient(prev_vw, beta)
            v_b = self.__multiplyGradient(prev_vb, beta)
            pointSeen = 0
            loss = 0.0
            
            for X, Y in zip(train_X, train_Y):
                self.W = self.__subtractGradient(self.W, v_w)
                self.B = self.__subtractGradient(self.B, v_b)
                
                A, H = self.__forwardPropagation(X)
                diffW, diffB = self.__backwardPropagation(A, H, X, Y)
                
                dw = self.__addGradient(dw, diffW)
                db = self.__addGradient(db, diffB)
                
                self.W = self.__addGradient(self.W, v_w)
                self.B = self.__addGradient(self.B, v_b)
                
                loss += self.__calculateLoss(H[self.numOfHiddenLayer], Y)
                pointSeen += 1
                if(pointSeen % batchSize == 0):
                    vw = self.__addGradient(self.__multiplyGradient(prev_vw, beta), self.__multiplyGradient(dw, eta))
                    vb = self.__addGradient(self.__multiplyGradient(prev_vb, beta), self.__multiplyGradient(db, eta))
                    
                    self.W = self.__subtractGradient(self.W, vw)
                    self.B = self.__subtractGradient(self.B, vb)
                    
                    prev_vw = vw
                    prev_vb = vb
                    dw, db = self.__initilizeWandB()
            
            print("Average Loss after epoch ", i+1, " : ", loss/len(train_X))

    def trainByStochasticGradientDescent(self, epochs, batchSize, eta, train_X, train_Y, val_X, val_Y): 
        for i in range(epochs):
            dw, db = self.__initilizeWandB()
            pointSeen = 0
            loss = 0.0
            
            for X, Y in zip(train_X, train_Y):
                A, H = self.__forwardPropagation(X)
                diffW, diffB = self.__backwardPropagation(A, H, X, Y)
                
                dw = self.__addGradient(dw, diffW)
                db = self.__addGradient(db, diffB)
                
                loss += self.__calculateLoss(H[self.numOfHiddenLayer], Y)
                pointSeen += 1
                if(pointSeen % batchSize == 0):
                    self.W = self.__subtractGradient(self.W, self.__multiplyGradient(dw, eta))
                    self.B = self.__subtractGradient(self.B, self.__multiplyGradient(db, eta))
                    dw, db = self.__initilizeWandB()
            
            print("Average Loss after epoch ", i+1, " : ", loss/len(train_X))
    
    def __forwardPropagation(self, input):
        A = []
        H = []

        A.append(np.matmul(self.W[0], input) + self.B[0])
        H.append(self.__computeH(A[0]))

        for i in range(1, self.numOfHiddenLayer+1):
            A.append(np.matmul(self.W[i], H[i-1]) + self.B[i])
            H.append(self.__computeH(A[i]))

        sum = np.sum((np.exp(A[self.numOfHiddenLayer]))[:self.outputSize])
        H[self.numOfHiddenLayer] = np.exp(A[self.numOfHiddenLayer]) / sum

        return A, H

    def __backwardPropagation(self, A, H, input, output):
        Y = np.zeros(self.sizeOfHiddenLayer)
        Y[output] = 1.0
        
        YHat = H[self.numOfHiddenLayer]
                 
        diffW, diffB = self.__initilizeWandB()
        
        delA = -1.0 * (Y - YHat)
        
        for i in range(self.numOfHiddenLayer, 0, -1):
            delW = np.matmul(delA.reshape(delA.size, 1), H[i-1].reshape(1, H[i-1].size))
            delB = delA
            
            diffW[i] = delW
            diffB[i] = delB
            
            delH = np.matmul(self.W[i].T, delA.reshape(delA.size, 1))
            delH = np.ravel(delH)
            
            if(self.activationFun == "sigmoid"):
                delA = delH * np.vectorize(self.__diffSigmoid)(A[i-1])
            elif(self.activationFun == "tanh"):
                delA = delH * np.vectorize(self.__diffTanh)(A[i-1])
            elif(self.activationFun == "ReLU"):
                delA = delH * np.vectorize(self.__diffRelu)(A[i-1])

        diffW[0] = np.matmul(delA.reshape(delA.size, 1), input.reshape(1, input.size))
        diffB[0] = delA
        
        return diffW, diffB
        
    def __computeH(self, A):
        H = np.zeros(self.sizeOfHiddenLayer)
        
        for i in range(len(A)):
            if(self.activationFun == "sigmoid"):
                H[i] = self.__sigmoid(A[i])
            elif(self.activationFun == "tanh"):
                H[i] = self.__tanh(A[i])
            elif(self.activationFun == "ReLU"):
                H[i] = self.__relu(A[i])
        
        return H

    def __addGradient(self, grad, x):
        ans = []
        for i in range(len(grad)):
            ans.append(grad[i] + x[i])
        
        return ans
    
    def __subtractGradient(self, grad, x):  
        ans = []
        for i in range(len(grad)):
            ans.append(grad[i] - x[i])
        
        return ans
    
    def __multiplyGradient(self, grad, x):
        ans = []
        for i in range(len(grad)):
            ans.append(x * grad[i])
        
        return ans

    def __calculateLoss(self, y_hat, y):
        loss = -1.0 * np.log(y_hat[y] + 1e-8)
        return loss

    def __sigmoid(self, x):
        if(x > 100.0):
            return 1.0
        elif(x < -100.0):
            return 1e-8
        else:
            return 1.0 / (1.0 + np.exp(-x))
    
    def __diffSigmoid(self, x):
        return self.__sigmoid(x) * (1.0 - self.__sigmoid(x))
    
    def __tanh(self, x):
        return np.tanh(x)
    
    def __diffTanh(self, x):
        return 1.0 - (self.__tanh(x) * self.__tanh(x))
    
    def __relu(self, x):
        return np.maximum(1e-8, x)
    
    def __diffRelu(self, x):
        if(x <= 0.0):
            return 1e-8
        else:
            return 1.0


def main():
    # Load MNIST data using Keras
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # Train validation division
    train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.1, shuffle=True, random_state=42)

    # Preprocess the data (Normalize pixel values to be between 0 and 1)
    train_images = train_images / 255.0
    val_images = val_images / 255.0
    test_images = test_images / 255.0

    # Flatten the images
    train_images = train_images.reshape((-1, 28 * 28))
    val_images = val_images.reshape((-1, 28 * 28))
    test_images = test_images.reshape((-1, 28 * 28))

    # epochs:               5, 10
    # hiddenLayers:         3, 4, 5
    # hiddenLayerSize:      32, 64, 128
    # weightDecay:          0, 0.0005, 0.5
    # learningRate:         1e-3, 1 e-4
    # optimizer:            sgd, momentum, nesterov, rmsprop, adam, nadam
    # batchSize:            16, 32, 64
    # weightInitialisation: random, Xavier
    # activationFunctions:  sigmoid, tanh, ReLU

    epochs = 10
    hiddenLayers = 3
    hiddenLayerSize = 32
    weightDecay = 0.5
    learningRate = 1e-3
    beta = 0.9
    optimizer = "sgd"
    batchSize = 16
    weightInitialisation = "Xavier"
    activationFunction = "sigmoid"
    inputSize = 28 * 28
    outputSize = 10
    
    NL = NuralNetwork(hiddenLayers, hiddenLayerSize, inputSize, outputSize, activationFunction, weightInitialisation)
    NL.trainByMomentumGradientDescent(epochs, batchSize, learningRate, beta, train_images, train_labels, val_images, val_labels)
    
    correctPrediction = 0
    for X, Y in zip(test_images, test_labels):
        ans = NL.predict(X)
        if(ans == Y):
            correctPrediction += 1
    
    print("Test accuracy: ", correctPrediction/len(test_images))

main()