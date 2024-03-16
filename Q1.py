import numpy as np
import matplotlib.pyplot as plt
import wandb
from keras.datasets import fashion_mnist

wandb.login(key='64b2775be5c91a3a2ab0bac3d540a1d9f6ea7579')
wandb.init(project='CS23M056_DL_Assignment_1_Q1')

# Load Fashion MNIST dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Mapping of class labels to class names
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

# Initialize a figure to plot sample images
plt.figure(figsize=(10, 10))

# Plot one sample image for each class
for i in range(10):
    # Find index of first sample with class label i
    idx = np.where(y_train == i)[0][0]
    
    # Plot the sample image
    plt.subplot(2, 5, i+1)
    plt.imshow(x_test[idx], cmap='gray')
    plt.title(class_names[i])
    plt.axis('off')
    
    wandb.log({
        "From Fashion_MNIST dataset 1 image per class is displayed": [wandb.Image(x_train[idx], caption=class_names[i])]
    })

plt.tight_layout()
plt.show()
wandb.finish()