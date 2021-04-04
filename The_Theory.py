# Reading in the Data
import pandas as pd
# Pandas has a read_csv method that expects a csv file, and returns a DataFrame:
train_df = pd.read_csv("data/asl_data/sign_mnist_train.csv")
valid_df = pd.read_csv("data/asl_data/sign_mnist_valid.csv")
# Exploring the Data
train_df.head()
# Extracting the Labels
y_train = train_df['label']
y_valid = valid_df['label']
del train_df['label']
del valid_df['label']
# Extracting the Images
x_train = train_df.values
x_valid = valid_df.values
# Summarizing the Training and Validation Data
# We now have 27,455 images with 784 pixels each for training...
x_train.shape
# ...as well as their corresponding labels:
y_train.shape
# For validation, we have 7,172 images...
x_valid.shape
# Visualizing the Data
# To visualize the images, we will again use the matplotlib library. We don't need to worry about the details of this visualization, but if interested, you can learn more about matplotlib at a later time.
import matplotlib.pyplot as plt
plt.figure(figsize=(40,40))

num_images = 20
for i in range(num_images):
    row = x_train[i]
    label = y_train[i]
    
    image = row.reshape(28,28)
    plt.subplot(1, num_images, i+1)
    plt.title(label, fontdict={'fontsize': 30})
    plt.axis('off')
    plt.imshow(image, cmap='gray')

# Exercise: Normalize the Image Data
# As we did with the MNIST dataset, we are going to normalize the image data, meaning that their pixel values, instead of being between 0 and 255 as they are currently:
x_train.min()
x_train.max()
# ...should be floating point values between 0 and 1. Use the following cell to work. If you get stuck, look at the solution below.
# TODO: Normalize x_train and x_valid.
x_train = x_train / 255
x_valid = x_valid / 255

# Exercise: Categorize the Labels
# As we did with the MNIST dataset, we are going to categorically encode the labels. Recall that we can use the keras.utils.to_categorical method to accomplish this by passing it the values to encode, and, the number of categories to encode it into. Do your work in the cell below. We have imported keras and set the number of categories (24) for you.
import tensorflow.keras as keras
num_classes = 24
# TODO: Categorically encode y_train and y_valid.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_valid = keras.utils.to_categorical(y_valid, num_classes)

# Exercise: Build the Model
# The data is all prepared, we have normalized images for training and validation, as well as categorically encoded labels for training and validation.

# For this exercise we are going to build a sequential model. Just like last time, build a model that:

# Has a dense input layer. This layer should contain 512 neurons, use the relu activation function, and expect input images with a shape of (784,)
# Has a second dense layer with 512 neurons which uses the relu activation function
# Has a dense output layer with neurons equal to the number of classes, using the softmax activation function

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# TODO: build a model following the guidelines above.

model = Sequential()
model.add(Dense(units = 512, activation='relu', input_shape=(784,)))
model.add(Dense(units = 512, activation='relu'))
model.add(Dense(units = num_classes, activation='softmax'))

# Summarizing the Model
# Run the cell below to summarize the model you just created:
model.summary()

# Compiling the Model
# We'll compile our model with the same options as before, using categorical crossentropy to reflect the fact that we want to fit into one of many categories, and measuring the accuracy of our model:

model.compile(loss='categorical_crossentropy', metrics=['accuracy'])

# Exercise: Train the Model
# Use the model's fit method to train it for 20 epochs using the training and validation images and labels created above:
# TODO: Train the model for 20 epochs.

model.fit(x_train, y_train, epochs=20, verbose=1, validation_data=(x_valid, y_valid))

