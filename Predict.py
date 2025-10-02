## The CNN model that was persisted in the first part of this project was retrieved.
# Import Various Libraries
from keras.models import load_model
import numpy as np
import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Import Pre-Trained Model saved on C Drive
my_model = load_model('C:/Users/jaish/cifar10_model_64_1.keras')

## Variables
# Class Names
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Image Names stored in C Drive to be used for Testing.
 
image_names = ["Picture1.jpg" , "Picture2.jpg" , "Picture3.jpg" , "Picture4.jpg" , "Picture5.jpg" , 
               "Picture6.jpg" , "Picture7.jpg" , "Picture8.jpg" , "Picture9.jpg" , "Picture10.jpg" ]
test_images = []

# Upload images from C drive
i = 0
for i in image_names:
    image_path = 'C:/Users/jaish/' + i
    imgp = mpimg.imread(image_path)
    test_images.append(imgp)

# Predict the image
model_predictions = []  #Store Model predictions
i=0
for i in range(10):
    image = np.array(test_images[i])
    image = np.expand_dims(image, axis=0)  
    y_predictions = my_model.predict(image)
    y_predictions = np.argmax(y_predictions, axis=1)
    y_predictions = y_predictions[0]

    model_predictions.append(labels[y_predictions])
    
## Print Actual Image along with the Predicted Class
fig, axes = plt.subplots(nrows=5, ncols=2) # Creates a 2x2 grid of subplots

j = 0
for i in range(0,5):
    axes[j,0].imshow(test_images[i])
    axes[j,0].set_title(model_predictions[i])
    
    axes[j,1].imshow(test_images[i+1])
    axes[j,1].set_title(model_predictions[i+1])
    j+=1

plt.subplots_adjust(wspace=0.4 , hspace = 1.0)
