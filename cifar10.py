### FINAL VERSION ###
## Import Necessary libraries
import tensorflow as tf

from tensorflow import keras
from keras import layers, models, datasets
from keras.layers import SpatialDropout2D

import numpy as np

## Libraries for visualization
import matplotlib.pyplot as plt
import seaborn as sns

## The Class Names are a given. These have been mapped to integers 0 to 9. So airplane =0, aotomobile = 1 etc44
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
class_ids = [0,1,2,3,4,5,6,7,8,9]

## Import the CIFAR10 dataset
from keras.datasets import cifar10

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

## Variables
# Define the batch size while training
batch_size = 32
epochs = 20
image_size = 64
new_shape = (64,64,3)

## Build the CNN layered model
# Build a Sequential Model

model = models.Sequential([
# Input shape
    layers.Input(shape= new_shape),
    layers.Resizing(image_size, image_size),
    layers.Rescaling(1./255),

# First layer with 32 filters size 3 X 3
    layers.Conv2D(32, (3, 3), activation='relu', padding = 'same' ),
    layers.BatchNormalization(),
    layers.SpatialDropout2D(0.2),
    layers.MaxPooling2D((2, 2), padding='same'),
    
# Second layer with 64 filters size 3 X 3
    layers.Conv2D(64, (3, 3), activation='relu', padding = 'same'),
    layers.BatchNormalization(),
    layers.SpatialDropout2D(0.2),
    layers.MaxPooling2D((2, 2), padding='same'),

# Third layer with 128 filters size 3 X 3
    layers.Conv2D(128, (3, 3), activation='relu' , padding = 'same'),
    layers.BatchNormalization(),
    layers.SpatialDropout2D(0.2),
    layers.MaxPooling2D((2, 2), padding='same'),

# Fourth layer with 256 filters size 3 X 3
    layers.Conv2D(256, (3, 3), activation='relu' , padding = 'same'),
    layers.BatchNormalization(),
    layers.SpatialDropout2D(0.2),
    layers.MaxPooling2D((2, 2), padding='same'),

# Two fully connected layers for classification one with 512 neurons and the other with 256 neurons
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),
# Output layer with 10 neurons representing the 10 possible classes
    layers.Dense(10)
])


# Create an Adam optimizer with a specific learning rate hyperparameter
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.003,
    decay_steps=7500,
    decay_rate=0.9
)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

## Compile the model
model.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Split Test Dataset into validation (val_images) and final testing set (new_test_images)
from sklearn.model_selection import train_test_split
# test_size determines the proportion of the original test set that becomes the *new* test set
val_images, new_test_images, val_labels, new_test_labels = train_test_split(
        test_images, test_labels, test_size=0.5, random_state=42
    )

## Train the model
history = model.fit(train_images, train_labels, batch_size = batch_size, epochs=epochs, validation_data=(val_images, val_labels))


## Save the Model
model.save('myfinal_cifar10_model.keras')

## Print a Summary of the Model
print('--- Model Summary ----')
print(model.summary())

## Save Visualization
!pip install visualkeras
import visualkeras
visualkeras.layered_view(model, to_file='final_cnn_architecture.png').show()


## Predictions on Test Data
y_predictions = model.predict(new_test_images)
y_predictions = np.argmax(y_predictions, axis=1)

## Examine the predictions
#Create confusion matrix and normalizes it over predicted (columns)
from sklearn.metrics import confusion_matrix
result = confusion_matrix(new_test_labels, y_predictions , normalize='pred')
test_loss, test_acc = model.evaluate(new_test_images,  new_test_labels, verbose=2)
print(f'Accuracy: {test_acc}')
print(f'Loss: {test_loss}')


# Create and display the heatmap
sns.heatmap(result, annot=True, fmt='.1%', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()


## PLot Validation Accuracy
plt.plot(history.history['val_accuracy'])
plt.plot(history.history['accuracy'])

# Add title and labels
plt.title('Model Validation Accuracy')
plt.ylabel('Validation Accuracy')
plt.xlabel('Epoch')
# Add a legend if you are also plotting training accuracy
plt.legend(['Validation'], loc='upper left')

# Display the plot
plt.show()
