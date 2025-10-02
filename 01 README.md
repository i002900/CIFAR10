# CIFAR10
Multi-Class Classification.

The CIFAR10 dataset contains 60000 32X32 colour images in 10 mutually exclusive classes: Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship and Truck.
This project illustrates thde following:
a) Build and deploy a multi-layer convolution neural network comprising of CONV2D filters, Batch normalization, Spatialdropout and Maxpoolng.
b) Ability to accept images of any size
c) Persist the model and use it for predicting a different of images

After 20 epochs, the model gave an accuracy of a little over 80%
<img width="1257" height="160" alt="image" src="https://github.com/user-attachments/assets/41b6ea30-01e4-4cd9-93b8-a2703987d829" />

The confusion matrix shows that Airplanes/Birds were most mis-classified followed by Cats/Dogs.
<img width="908" height="681" alt="image" src="https://github.com/user-attachments/assets/8e013abd-dfd4-47f3-b198-ce4f7b7b5f55" />


In the second half of this exercise, the model persisted above, was used to classify various images of varying sizes downloaded from the internet. 
The preduction accuracy was a 100%. 

The screen shot below, shows the predicted label for each image.
<img width="632" height="553" alt="image" src="https://github.com/user-attachments/assets/c5776a92-8672-4876-89d2-3819891c90ae" />
