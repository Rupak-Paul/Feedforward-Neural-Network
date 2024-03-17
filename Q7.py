import numpy as np
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

class NuralNetwork:
    def __init__(self, numOfHiddenLayer, sizeOfHiddenLayer, inputSize, outputSize, activationFun, initilizationMethod):
        self.numOfHiddenLayer = numOfHiddenLayer
        self.sizeOfHiddenLayer = sizeOfHiddenLayer
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.activationFun = activationFun
        self.W, self.B = self.__initilizeWandB(initilizationMethod)
        self.eps = 1e-4

    def __initilizeWandB(self, initilizationMethod="None"):
        W = []
        B = []
        
        if(initilizationMethod == "random"):
            W.append(np.random.rand(self.sizeOfHiddenLayer, self.inputSize))
            B.append(np.random.rand(self.sizeOfHiddenLayer))
        elif(initilizationMethod == "Xavier"):
            W.append(np.random.normal(0.0, 1.0/np.sqrt(self.inputSize), (self.sizeOfHiddenLayer, self.inputSize)))
            B.append(np.zeros(self.sizeOfHiddenLayer))
        else:
            W.append(np.zeros((self.sizeOfHiddenLayer, self.inputSize)))
            B.append(np.zeros(self.sizeOfHiddenLayer))
        
        for i in range(1, self.numOfHiddenLayer+1):
            if(initilizationMethod == "random"):
                W.append(np.random.rand(self.sizeOfHiddenLayer, self.sizeOfHiddenLayer))
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

    def trainByMomentumGradientDescent(self, epochs, batchSize, eta, beta, train_X, train_Y, val_X, val_Y, lossFunction, weightDecay):
        prev_uw, prev_ub = self.__initilizeWandB()
        
        for i in range(epochs):
            dw, db = self.__initilizeWandB()
            pointSeen = 0
            
            for X, Y in zip(train_X, train_Y):
                A, H = self.__forwardPropagation(X)
                diffW, diffB = self.__backwardPropagation(A, H, X, Y, weightDecay)
                
                dw = self.__addTwoGradient(dw, diffW)
                db = self.__addTwoGradient(db, diffB)
                
                pointSeen += 1
                if(pointSeen % batchSize == 0):
                    uw = self.__addTwoGradient(self.__multiplyGradientByConstant(prev_uw, beta), self.__multiplyGradientByConstant(dw, eta))
                    ub = self.__addTwoGradient(self.__multiplyGradientByConstant(prev_ub, beta), self.__multiplyGradientByConstant(db, eta))
                    
                    self.W = self.__subtractTwoGradient(self.W, uw)
                    self.B = self.__subtractTwoGradient(self.B, ub)
                    
                    prev_uw = uw
                    prev_ub = ub
                    dw, db = self.__initilizeWandB()
             
            print("Loss after epoch ", i+1, ": ")
            trainLoss, trainAccuracy, validationLoss, validationAccuracy = self.__calculateAndPrintLossAndAccuracy(train_X, train_Y, val_X, val_Y, lossFunction)
            
    def trainByNesterovGradientDescent(self, epochs, batchSize, eta, beta, train_X, train_Y, val_X, val_Y, lossFunction, weightDecay):
        prev_vw, prev_vb = self.__initilizeWandB()
        
        for i in range(epochs):
            dw, db = self.__initilizeWandB()
            v_w = self.__multiplyGradientByConstant(prev_vw, beta)
            v_b = self.__multiplyGradientByConstant(prev_vb, beta)
            pointSeen = 0
            
            for X, Y in zip(train_X, train_Y):
                self.W = self.__subtractTwoGradient(self.W, v_w)
                self.B = self.__subtractTwoGradient(self.B, v_b)
                
                A, H = self.__forwardPropagation(X)
                diffW, diffB = self.__backwardPropagation(A, H, X, Y, weightDecay)
                
                dw = self.__addTwoGradient(dw, diffW)
                db = self.__addTwoGradient(db, diffB)
                
                self.W = self.__addTwoGradient(self.W, v_w)
                self.B = self.__addTwoGradient(self.B, v_b)
                
                pointSeen += 1
                if(pointSeen % batchSize == 0):
                    vw = self.__addTwoGradient(self.__multiplyGradientByConstant(prev_vw, beta), self.__multiplyGradientByConstant(dw, eta))
                    vb = self.__addTwoGradient(self.__multiplyGradientByConstant(prev_vb, beta), self.__multiplyGradientByConstant(db, eta))
                    
                    self.W = self.__subtractTwoGradient(self.W, vw)
                    self.B = self.__subtractTwoGradient(self.B, vb)
                    
                    prev_vw = vw
                    prev_vb = vb
                    dw, db = self.__initilizeWandB()
               
            print("Loss after epoch ", i+1, ": ")
            trainLoss, trainAccuracy, validationLoss, validationAccuracy = self.__calculateAndPrintLossAndAccuracy(train_X, train_Y, val_X, val_Y, lossFunction)

    def trainByStochasticGradientDescent(self, epochs, batchSize, eta, train_X, train_Y, val_X, val_Y, lossFunction, weightDecay):  
        for i in range(epochs):
            dw, db = self.__initilizeWandB()
            pointSeen = 0
            
            for X, Y in zip(train_X, train_Y):
                A, H = self.__forwardPropagation(X)
                diffW, diffB = self.__backwardPropagation(A, H, X, Y, weightDecay)
                
                dw = self.__addTwoGradient(dw, diffW)
                db = self.__addTwoGradient(db, diffB)
                
                pointSeen += 1
                if(pointSeen % batchSize == 0):
                    self.W = self.__subtractTwoGradient(self.W, self.__multiplyGradientByConstant(dw, eta))
                    self.B = self.__subtractTwoGradient(self.B, self.__multiplyGradientByConstant(db, eta))
                    dw, db = self.__initilizeWandB()
               
            print("Loss after epoch ", i+1, ": ")
            trainLoss, trainAccuracy, validationLoss, validationAccuracy = self.__calculateAndPrintLossAndAccuracy(train_X, train_Y, val_X, val_Y, lossFunction)
    
    def trainByRmsprop(self, epochs, batchSize, eta, beta, train_X, train_Y, val_X, val_Y, lossFunction, weightDecay):
        v_w, v_b = self.__initilizeWandB()
        
        for i in range(epochs):
            dw, db = self.__initilizeWandB()
            pointSeen = 0
            
            for X, Y in zip(train_X, train_Y):
                A, H = self.__forwardPropagation(X)
                diffW, diffB = self.__backwardPropagation(A, H, X, Y, weightDecay)
                
                dw = self.__addTwoGradient(dw, diffW)
                db = self.__addTwoGradient(db, diffB)
                
                pointSeen += 1
                if(pointSeen % batchSize == 0):
                    v_w = self.__addTwoGradient(self.__multiplyGradientByConstant(v_w, beta), self.__multiplyGradientByConstant(self.__squareOfElementsOfGradient(dw), (1.0-beta)))
                    v_b = self.__addTwoGradient(self.__multiplyGradientByConstant(v_b, beta), self.__multiplyGradientByConstant(self.__squareOfElementsOfGradient(db), (1.0-beta)))
                    
                    self.W = self.__subtractTwoGradient(self.W, self.__divideTwoGradient(self.__multiplyGradientByConstant(dw, eta), self.__addGradientbyConstant(self.__rootOfElementsOfGradient(v_w), self.eps)))
                    self.B = self.__subtractTwoGradient(self.B, self.__divideTwoGradient(self.__multiplyGradientByConstant(db, eta), self.__addGradientbyConstant(self.__rootOfElementsOfGradient(v_b), self.eps)))
                    
                    dw, db = self.__initilizeWandB()
                   
            print("Loss after epoch ", i+1, ": ")
            trainLoss, trainAccuracy, validationLoss, validationAccuracy = self.__calculateAndPrintLossAndAccuracy(train_X, train_Y, val_X, val_Y, lossFunction)
    
    def trainByAdam(self, epochs, batchSize, eta, beta1, beta2, train_X, train_Y, val_X, val_Y, lossFunction, weightDecay):
        m_w, m_b = self.__initilizeWandB()
        v_w, v_b = self.__initilizeWandB()
        
        for i in range(epochs):
            dw, db = self.__initilizeWandB()
            pointSeen = 0
            
            for X, Y in zip(train_X, train_Y):
                A, H = self.__forwardPropagation(X)
                diffW, diffB = self.__backwardPropagation(A, H, X, Y, weightDecay)
                
                dw = self.__addTwoGradient(dw, diffW)
                db = self.__addTwoGradient(db, diffB)
                
                pointSeen += 1
                if(pointSeen % batchSize == 0):
                    m_w = self.__addTwoGradient(self.__multiplyGradientByConstant(m_w, beta1), self.__multiplyGradientByConstant(dw, (1.0-beta1)))
                    m_b = self.__addTwoGradient(self.__multiplyGradientByConstant(m_b, beta1), self.__multiplyGradientByConstant(db, (1.0-beta1)))
                    v_w = self.__addTwoGradient(self.__multiplyGradientByConstant(v_w, beta2), self.__multiplyGradientByConstant(self.__squareOfElementsOfGradient(dw), (1.0-beta2)))
                    v_b = self.__addTwoGradient(self.__multiplyGradientByConstant(v_b, beta2), self.__multiplyGradientByConstant(self.__squareOfElementsOfGradient(db), (1.0-beta2)))
                    
                    c1 = 1.0 / (1.0 - np.power(beta1, i+1))
                    c2 = 1.0 / (1.0 - np.power(beta2, i+1))
                    m_w_hat = self.__multiplyGradientByConstant(m_w, c1)
                    m_b_hat = self.__multiplyGradientByConstant(m_b, c1)
                    v_w_hat = self.__multiplyGradientByConstant(v_w, c2)
                    v_b_hat = self.__multiplyGradientByConstant(v_b, c2)
                    
                    self.W = self.__subtractTwoGradient(self.W, self.__divideTwoGradient(self.__multiplyGradientByConstant(m_w_hat, eta), self.__addGradientbyConstant(self.__rootOfElementsOfGradient(v_w_hat), self.eps)))
                    self.B = self.__subtractTwoGradient(self.B, self.__divideTwoGradient(self.__multiplyGradientByConstant(m_b_hat, eta), self.__addGradientbyConstant(self.__rootOfElementsOfGradient(v_b_hat), self.eps)))
                    
                    dw, db = self.__initilizeWandB()
                   
            print("Loss after epoch ", i+1, ": ")
            trainLoss, trainAccuracy, validationLoss, validationAccuracy = self.__calculateAndPrintLossAndAccuracy(train_X, train_Y, val_X, val_Y, lossFunction)
    
    def trainByNadam(self, epochs, batchSize, eta, beta1, beta2, train_X, train_Y, val_X, val_Y, lossFunction, weightDecay):
        m_w, m_b = self.__initilizeWandB()
        v_w, v_b = self.__initilizeWandB()
        
        for i in range(epochs):
            dw, db = self.__initilizeWandB()
            pointSeen = 0
            
            for X, Y in zip(train_X, train_Y):
                A, H = self.__forwardPropagation(X)
                diffW, diffB = self.__backwardPropagation(A, H, X, Y, weightDecay)
                
                dw = self.__addTwoGradient(dw, diffW)
                db = self.__addTwoGradient(db, diffB)
               
                pointSeen += 1
                if(pointSeen % batchSize == 0):
                    m_w = self.__addTwoGradient(self.__multiplyGradientByConstant(m_w, beta1), self.__multiplyGradientByConstant(dw, (1.0-beta1)))
                    m_b = self.__addTwoGradient(self.__multiplyGradientByConstant(m_b, beta1), self.__multiplyGradientByConstant(db, (1.0-beta1)))
                    v_w = self.__addTwoGradient(self.__multiplyGradientByConstant(v_w, beta2), self.__multiplyGradientByConstant(self.__squareOfElementsOfGradient(dw), (1.0-beta2)))
                    v_b = self.__addTwoGradient(self.__multiplyGradientByConstant(v_b, beta2), self.__multiplyGradientByConstant(self.__squareOfElementsOfGradient(db), (1.0-beta2)))
                    
                    c1 = 1.0 / (1.0 - np.power(beta1, i+1))
                    c2 = 1.0 / (1.0 - np.power(beta2, i+1))
                    m_w_hat = self.__multiplyGradientByConstant(m_w, c1)
                    m_b_hat = self.__multiplyGradientByConstant(m_b, c1)
                    v_w_hat = self.__multiplyGradientByConstant(v_w, c2)
                    v_b_hat = self.__multiplyGradientByConstant(v_b, c2)
                   
                    c3 = (1.0-beta1) / (1.0-beta1**(i+1))
                    self.W = self.__subtractTwoGradient(self.W, self.__multiplyTwoGradient(self.__divideGrdientByConstant(eta, self.__addGradientbyConstant(self.__rootOfElementsOfGradient(v_w_hat), self.eps)), self.__addTwoGradient(self.__multiplyGradientByConstant(m_w_hat, beta1), self.__multiplyGradientByConstant(dw, c3))))
                    self.B = self.__subtractTwoGradient(self.B, self.__multiplyTwoGradient(self.__divideGrdientByConstant(eta, self.__addGradientbyConstant(self.__rootOfElementsOfGradient(v_b_hat), self.eps)), self.__addTwoGradient(self.__multiplyGradientByConstant(m_b_hat, beta1), self.__multiplyGradientByConstant(db, c3))))
                
                    dw, db = self.__initilizeWandB()
                    
            print("Loss after epoch ", i+1, ": ")
            trainLoss, trainAccuracy, validationLoss, validationAccuracy = self.__calculateAndPrintLossAndAccuracy(train_X, train_Y, val_X, val_Y, lossFunction)
    
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

    def __backwardPropagation(self, A, H, input, output, weightDecay):
        Y = np.zeros(self.sizeOfHiddenLayer)
        Y[output] = 1.0
        
        YHat = H[self.numOfHiddenLayer]
                 
        diffW, diffB = self.__initilizeWandB()
        
        delA = -1.0 * (Y - YHat)
        
        for i in range(self.numOfHiddenLayer, 0, -1):
            delW = np.matmul(delA.reshape(delA.size, 1), H[i-1].reshape(1, H[i-1].size)) + (weightDecay * self.W[i])
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

        diffW[0] = np.matmul(delA.reshape(delA.size, 1), input.reshape(1, input.size)) + (weightDecay * self.W[0])
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

    def __addTwoGradient(self, grad, x):
        ans = []
        for i in range(len(grad)):
            ans.append(grad[i] + x[i])
        return ans
    
    def __subtractTwoGradient(self, grad, x):  
        ans = []
        for i in range(len(grad)):
            ans.append(grad[i] - x[i])
        return ans
    
    def __divideTwoGradient(self, grad1, grad2):
        ans = []
        for i in range(len(grad1)):
            ans.append(grad1[i] / grad2[i])
        return ans
    
    def __multiplyTwoGradient(self, grad1, grad2):
        ans = []
        for i in range(len(grad1)):
            ans.append(grad1[i] * grad2[i])
        return ans
    
    def __squareOfElementsOfGradient(self, grad):
        ans = []
        for i in range(len(grad)):
            ans.append(np.square(grad[i]))
        return ans
    
    def __rootOfElementsOfGradient(self, grad):
        ans = []
        for i in range(len(grad)):
            ans.append(np.sqrt(grad[i]))
        return ans       
    
    def __divideGrdientByConstant(self, const, grad):
        ans = []
        for i in range(len(grad)):
            ans.append(const / grad[i])      
        return ans
    
    def __multiplyGradientByConstant(self, grad, x):
        ans = []
        for i in range(len(grad)):
            ans.append(x * grad[i])
        return ans

    def __addGradientbyConstant(self, grad, x):
        ans = []
        for i in range(len(grad)):
            ans.append(grad[i] + x)
        return ans

    def __calculateCrossEntropyLoss(self, y_hat, y):
        loss = -1.0 * np.log(y_hat[y] + 1e-8)
        return loss

    def __calculateSquareLoss(self, y_hat, y):
        y_hat = y_hat[:self.outputSize]
        y = (np.zeros(self.outputSize))[y] = 1.0
        return np.sum(np.square(y_hat - y)) / self.outputSize

    def __calculateAvgLoss(self, X, Y, lossFunction):
        loss = 0.0
        for input, output in zip(X, Y):
            A, H = self.__forwardPropagation(input)
            if(lossFunction == "cross entropy"):
                loss += self.__calculateCrossEntropyLoss(H[self.numOfHiddenLayer], output)
            elif(lossFunction == "squared error"):
                loss += self.__calculateSquareLoss(H[self.numOfHiddenLayer], output)
        return loss / len(X)
            
    def __calculateAvgAccuracy(self, X, Y):
        correctPrediction = 0
        for input, output in zip(X, Y):
            predicetedOutput = self.predict(input)
            if(predicetedOutput == output):
                correctPrediction += 1
        return correctPrediction / len(X)

    def __calculateAndPrintLossAndAccuracy(self, train_X, train_Y, val_X, val_Y, lossFunction):
        trainLoss = self.__calculateAvgLoss(train_X, train_Y, lossFunction)
        trainAccuracy = self.__calculateAvgAccuracy(train_X, train_Y)
        validationLoss = self.__calculateAvgLoss(val_X, val_Y, lossFunction)
        validationAccuracy = self.__calculateAvgAccuracy(val_X, val_Y)
     
        print("Train Loss: ", trainLoss)
        print("Train Accuracy: ", trainAccuracy)
        print("Validation Loss: ", validationLoss)
        print("Validation Accuracy: ", validationAccuracy)
        
        return trainLoss, trainAccuracy, validationLoss, validationAccuracy
    
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
    wandb.login(key='64b2775be5c91a3a2ab0bac3d540a1d9f6ea7579')
    wandb.init(project='CS23M056_DL_Assignment_1_Q7', name='confusion-matrix')
    
    epochs = 10
    hiddenLayers = 1
    hiddenLayerSize = 64
    weightDecay = 0
    learningRate = 0.01
    optimizer = 'nadam'
    batchSize = 32
    weightInitialisation = 'Xavier'
    activationFunction = 'ReLU'
    lossFunction = 'cross entropy'
    beta = 0.9
    beta1 = 0.9
    beta2 = 0.999
    inputSize = 28*28
    outputSize = 10
    
    NL = NuralNetwork(hiddenLayers, hiddenLayerSize, inputSize, outputSize, activationFunction, weightInitialisation)
    NL.trainByNadam(epochs, batchSize, learningRate, beta1, beta2, train_images, train_labels, val_images, val_labels, lossFunction, weightDecay)
    
    correctPrediction = 0
    for X, Y in zip(test_images, test_labels):
        ans = NL.predict(X)
        if(ans == Y):
            correctPrediction += 1
    print("Test accuracy: ", correctPrediction/len(test_images))
    
    class_names = {
        0: 'T-shirt/top',
        1: 'Trouser',
        2: 'Pullover',
        3: 'Dress',
        4: 'Coat',
        5: 'Sandal',
        6: 'Shirt',
        7: 'Sneaker',
        8: 'Bag',
        9: 'Ankle boot'
    }
    
    y_true = test_labels
    y_pred = np.zeros(len(test_images))
    for i in range(len(test_images)):
        y_pred[i] = NL.predict(test_images[i])
    
    conf_matrix = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', cbar=False)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.xticks(ticks=np.arange(10) + 0.5, labels=[class_names[i] for i in range(10)], rotation=45)
    plt.yticks(ticks=np.arange(10) + 0.5, labels=[class_names[i] for i in range(10)], rotation=0)
    plt.show()
    
    # Log the confusion matrix
    wandb.log({
        "confusion_matrix": wandb.plot.confusion_matrix(probs=None, y_true=y_true, preds=y_pred, class_names=class_names)
    })
    
    wandb.finish()


##########################################################################################################
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.1, shuffle=True, random_state=42)

train_images = train_images / 255.0
val_images = val_images / 255.0
test_images = test_images / 255.0

train_images = train_images.reshape((-1, 28 * 28))
val_images = val_images.reshape((-1, 28 * 28))
test_images = test_images.reshape((-1, 28 * 28))

main()