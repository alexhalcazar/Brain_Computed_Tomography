import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelBinarizer
# "Sequential" models let us define a stack of neural network layers
from keras.models import Sequential
# import the core layers:
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Input

training_dir_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'Training')
testing_dir_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'Testing')

def create_df(directory_path):
    filepaths = []
    labels = []

    directory = os.listdir(directory_path)
    for folder in directory:
        folder_path = os.path.join(directory_path, folder)
        if os.path.isdir(folder_path):
            files = os.listdir(folder_path)
            for file in files:
                file_path = os.path.join(folder_path, file)
                filepaths.append(file_path)
                labels.append(folder)


    # Concatenate data paths with labels into one dataframe
    Fseries = pd.Series(filepaths, name= 'filepaths')
    Lseries = pd.Series(labels, name='labels')
    df = pd.concat([Fseries, Lseries], axis= 1)

    return df


def transform_data(dataframe):
    X = []
    y = []
    for _, row in dataframe.iterrows():
        image_path = row['filepaths']
        image = load_img(image_path, target_size=(256, 256))  # Load image and resize to target size
        image_array = img_to_array(image)  # Convert image to numpy array
        X.append(image_array)
        y.append(row['labels'])
    X = np.array(X)
    y = np.array(y)
    return X, y

training_df = create_df(training_dir_path)
testing_df = create_df(testing_dir_path)


X_train, y_train = transform_data(training_df)
X_test, y_test = transform_data(testing_df)

print(X_train.shape)
print(X_test.shape)

# Display the first image from X_train list for testing purposes
# plt.imshow(X_train[0].astype('uint8'))  # Convert image array back to uint8 format
# plt.axis('off')  # Turn off axis
# plt.show()

# Reshape each image pixels into a row of feature table with 256*256*3=196608 features (each pixel is a feature):
X_train = X_train.reshape(X_train.shape[0], 256 * 256 * 3)
X_test = X_test.reshape(X_test.shape[0], 256 * 256 * 3)

print(X_train.shape)
print(X_test.shape)

# Normalize pixel values
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

print (X_train.shape)
print (X_train[:10])

# Initialize LabelBinarizer
label_binarizer = LabelBinarizer()

# Encode class labels
y_train_encoded = label_binarizer.fit_transform(y_train)
y_test_encoded = label_binarizer.transform(y_test)

# Declare Sequential model to build our network:
model = Sequential()

input_shape = (196608,) 
hidden_neurons = 100
out_size = 4

## Designing the ANN Structure (with 150,528 inputs, 4 outputs and 100 neuron in a hidden layer):

# -----------------------------------------
# first layer: input layer
# Input layer does not do any processing, so no need to define the input layer in this problem.

# -----------------------------------------
# second layer: hidden layer:
model.add(Dense(hidden_neurons, input_shape= input_shape))  # Nuerons
model.add(Activation('sigmoid')) # Activation
# model.add(Dropout(0.1))
# -----------------------------------------
# third layer: output layer:
model.add(Dense(out_size))  # Nuerons
model.add(Activation('softmax')) # Activation

# Compile the model:
model.compile(loss='categorical_crossentropy',
              metrics=['accuracy'],
              optimizer='adam')

# Training:
fitted_model = model.fit(X_train, y_train_encoded, validation_split=0.33, batch_size=32, epochs=15, verbose=1)

# summarize history for accuracy
plt.plot(fitted_model.history['accuracy'])
plt.plot(fitted_model.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(fitted_model.history['loss'])
plt.plot(fitted_model.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# Testing, Prediciton, Evaluation:

y_predict = model.predict(X_test, verbose=1)
print (y_predict.shape)

score = model.evaluate(X_test, y_test_encoded, verbose=1)
print('The accuracy is: ', score[1])