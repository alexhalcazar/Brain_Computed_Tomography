import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
# "Sequential" models let us define a stack of neural network layers
from keras.models import Sequential
# import the core layers:
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.layers import Input
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc

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

# Reshape each image pixels into a row of feature table with 256*256*3=196608 features (each pixel is a feature):
X_train_reshape = X_train.reshape(X_train.shape[0], 256 * 256 * 3)
X_test_reshape = X_test.reshape(X_test.shape[0], 256 * 256 * 3)

# Normalize pixel values
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# Initialize LabelBinarizer
label_binarizer = LabelBinarizer()

# Encode class labels
y_train_encoded = label_binarizer.fit_transform(y_train)
y_test_encoded = label_binarizer.transform(y_test)

# Declare Sequential model to build our network:
# model = Sequential()

# input_shape = (196608,) 
# hidden_neurons = 100
# out_size = 4

# # Designing the ANN Structure (with 196608 inputs, 4 outputs and 100 neuron in a hidden layer):

# # second layer: hidden layer:
# model.add(Dense(hidden_neurons, input_shape=input_shape))  # Nuerons
# model.add(Activation('sigmoid')) # Activation
# model.add(Dropout(0.1))

# # third layer: output layer:
# model.add(Dense(out_size))  # Nuerons
# model.add(Activation('softmax')) # Activation

# # Compile the model:
# model.compile(loss='categorical_crossentropy',
#               metrics=['accuracy'],
#               optimizer='adam')

# # Training:
# fitted_model = model.fit(X_train_reshape, y_train_encoded, validation_split=0.33, batch_size=32, epochs=15, verbose=1)

# # summarize history for accuracy
# plt.plot(fitted_model.history['accuracy'])
# plt.plot(fitted_model.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper left')
# plt.show()

# # summarize history for loss
# plt.plot(fitted_model.history['loss'])
# plt.plot(fitted_model.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper left')
# plt.show()

# # Testing, Prediciton, Evaluation:

# y_predict = model.predict(X_test_reshape, verbose=1)
# print (y_predict.shape)

# score = model.evaluate(X_test_reshape, y_test_encoded, verbose=1)
# print('The accuracy is: ', score[1])

# # Lets try splitting our training data
# # Split the training data into training and testing sets
# X_train_all, X_test, y_train_all, y_test = train_test_split(training_df['filepaths'], training_df['labels'], test_size=0.2, random_state=42)
# # Split the training data into training and validation sets
# X_train, X_val, y_train, y_val = train_test_split(X_train_all, y_train_all, test_size=0.25, random_state=42)

# # Transform data for training, validation, and testing sets
# X_train, y_train = transform_data(pd.DataFrame({'filepaths': X_train, 'labels': y_train}))
# X_val, y_val = transform_data(pd.DataFrame({'filepaths': X_val, 'labels': y_val}))
# X_test, y_test = transform_data(pd.DataFrame({'filepaths': X_test, 'labels': y_test}))

# # Print shapes of training, validation, and testing sets to confirm splitting
# print("X_train shape:", X_train.shape)
# print("y_train shape:", y_train.shape)
# print("X_val shape:", X_val.shape)
# print("y_val shape:", y_val.shape)
# print("X_test shape:", X_test.shape)
# print("y_test shape:", y_test.shape)


# # Lets try a nueral network with 64 neurons in the first layer and 32 neurons in the second layer
# my_ANN = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', random_state=42, early_stopping=True, validation_fraction=0.1)

# # Train your model
# # Original data is a 4D array (num_samples, height, width, channels)
# # reshape into 2D array 256*256*3
# my_ANN.fit(X_train.reshape(X_train.shape[0], -1), y_train)

# # Evaluate your model
# test_accuracy = my_ANN.score(X_test.reshape(X_test.shape[0], -1), y_test)
# print("Test accuracy:", test_accuracy)
# # Accuracy was 80%

# # Get the training history
# training_loss = my_ANN.loss_curve_
# validation_scores = my_ANN.validation_scores_
# best_validation_score = my_ANN.best_validation_score_

# # Plot the loss curve
# plt.plot(training_loss, label='Training Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.title('Training Loss Curve')
# plt.legend()
# plt.show()

# # Plot validation scores
# plt.plot(validation_scores, label='Validation Score')
# plt.xlabel('Iterations')
# plt.ylabel('Accuracy')
# plt.title('Validation Score Curve')
# plt.legend()
# plt.show()

# print("Best Validation Score:", best_validation_score)
# # Best Validation Score: 0.8483965014577259

# # Lets try using PCA to reduce the number of features
# # Initialize PCA with the desired number of components
# n_components = 100
# pca = PCA(n_components=n_components)

# # Fit PCA to the training data and transform the training and testing data
# X_train_pca = pca.fit_transform(X_train.reshape(X_train.shape[0], -1))
# X_test_pca = pca.transform(X_test.reshape(X_test.shape[0], -1))

# # Define your MLPClassifier model
# my_ANN_pca = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', random_state=42, early_stopping=True, validation_fraction=0.1)

# # Train your model on the PCA-transformed training data
# my_ANN_pca.fit(X_train_pca, y_train)

# # Evaluate PCA-transformed testing data
# test_accuracy_pca = my_ANN_pca.score(X_test_pca, y_test)
# print("Test accuracy with PCA:", test_accuracy_pca)
# # Accuracy was 87% and much faster

# # Get the training history
# training_loss_pca = my_ANN_pca.loss_curve_
# validation_scores_pca = my_ANN_pca.validation_scores_
# best_validation_score_pca = my_ANN_pca.best_validation_score_

# # Plot the loss curve
# plt.plot(training_loss_pca, label='Training Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.title('Training Loss Curve')
# plt.legend()
# plt.show()

# # Plot validation scores
# plt.plot(validation_scores_pca, label='Validation Score')
# plt.xlabel('Iterations')
# plt.ylabel('Accuracy')
# plt.title('Validation Score Curve')
# plt.legend()
# plt.show()

# # Using Bagging
# b_base_model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', random_state=42, verbose=True)

# # Define the number of base models
# b_estimators = 10

# # Initialize the BaggingClassifier
# bagging_model = BaggingClassifier(b_base_model, n_estimators=b_estimators, random_state=42, verbose=True)

# # Train the model
# bagging_model.fit(X_train_pca, y_train)

# # Evaluate the bagging model on the PCA-transformed testing data
# test_accuracy_bagging = bagging_model.score(X_test_pca, y_test)
# print("Test accuracy with Bagging and PCA:", test_accuracy_bagging)
# # Accuracy was 91%

# # Loss curve for each individual base model
# for base_model in bagging_model.estimators_: 
#     plt.plot(base_model.loss_curve_, label='Base Model Loss Curve')
# plt.xlabel('Number of Iterations')
# plt.ylabel('Loss')
# plt.title('Loss Curve for Individual Base Models')
# plt.legend()
# plt.show()


# # Lets try using a Decision Tree
# my_decisiontree = DecisionTreeClassifier(random_state=1)
# my_decisiontree.fit(X_train_pca, y_train)

# y_predict = my_decisiontree.predict(X_test_pca)

# score = accuracy_score(y_test, y_predict)

# print("Accuracy:", score)
# # Plot the decision tree
# plt.figure(figsize=(15, 10))
# plot_tree(my_decisiontree)
# plt.show()

# # Accuracy was 86%

# # Using Boosting which is an iterative procedure that adaptively changes the distribution of the data
# # Focuses more on the previously misclassified data samples
# # Define the number of base learners for AdaBoost
# dt_estimators = 50

# # Initialize the AdaBoostClassifier
# adaboost_model = AdaBoostClassifier(n_estimators=dt_estimators, algorithm='SAMME', random_state=42)

# # Train the model
# adaboost_model.fit(X_train_pca, y_train)

# # Evaluate the AdaBoost model on the PCA-transformed testing data
# test_accuracy_adaboost = adaboost_model.score(X_test_pca, y_test)
# print("Test accuracy with AdaBoost and PCA:", test_accuracy_adaboost)
# # Accuracy was 62%

# # Using Random Forest
# # Initialize the Random Forest classifier
# random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)

# # Train the model
# random_forest_model.fit(X_train_pca, y_train)

# # Evaluate the Random Forest model on the PCA-transformed testing data
# test_accuracy_random_forest = random_forest_model.score(X_test_pca, y_test)
# print("Test accuracy with Random Forest and PCA:", test_accuracy_random_forest)
# # Accuracy was 92%

print("Shape of X_train before flattening:", X_train.shape)
print("Shape of X_test before flattening:", X_test.shape)

# Lets try building our own convolutional nueral network
model2 = Sequential()

# CNN first layer (with 32 3x3 filter)
model2.add(Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(256,256,3)))
print('First shape', model2.output_shape)
# apply pooling
model2.add(MaxPooling2D(2,2))
print('After Maxpooling', model2.output_shape)
# Apply a second convolutional and pooling layer
model2.add(Conv2D(32, (3,3), activation='relu', padding='same'))
print('Second shape', model2.output_shape)
model2.add(MaxPooling2D(2,2))
print('After Maxpooling 2', model2.output_shape)

# Dropout layer to avoud overfitting
model2.add(Dropout(0.25))
# Flatten to 1D
model2.add(Flatten())
print('after flatten', model2.output_shape)

# if we are having an overfitting problem increase this value
model2.add(Dropout(0.5))
model2.add(Dense(64, activation='relu'))
# Number of classes (use softmax for more than 2 labels)
model2.add(Dense(4, activation='softmax'))

# compile the model
model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model2.summary())

# train the model
batch_size = 32
epochs = 5

history = model2.fit(X_train, y_train_encoded, epochs=epochs, batch_size=batch_size, verbose=1)
history2 = model2.fit(X_train, y_train_encoded, validation_split=0.25, batch_size=batch_size, epochs=epochs, verbose=1)

# evaluate our model
test_loss, test_accuracy = model2.evaluate(X_test, y_test_encoded)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

accuracy = history.history['accuracy']
# val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
#val_loss = history.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'b-', label='Training accuracy')
#plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'r-', label='Training loss')
#plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training loss')
# plt.title('Training and validation loss')
plt.legend()
plt.show()


# Lets use 25% of our training data as validation
accuracy = history2.history['accuracy']
val_accuracy = history2.history['val_accuracy']
loss = history2.history['loss']
val_loss = history2.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'r-', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'r-', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

predicted_classes = model2.predict(X_test)
# Testing:
score = model2.evaluate(X_test, y_test_encoded, verbose=1)
print('The accuracy is: ', score[1])

# with no padding
# Test Loss: 5.188281059265137
# Test Accuracy: 0.267734557390213

# with normalization
# Test Loss: 8.74139404296875
# Test Accuracy: 0.30892449617385864

# with padding
# Test Loss: 3.6883513927459717
# Test Accuracy: 0.408085435628891

# with dropout added
# Test Loss: 0.21483555436134338
# Test Accuracy: 0.9328756928443909

# batch size 64
# Test Loss: 0.1453573852777481
# Test Accuracy: 0.950419545173645

# With validation split
# Test Loss: 8.018810272216797
# Test Accuracy: 0.7604881525039673

# Lets change batch size from 64 to 32 
# Test Loss: 0.1591634452342987
# Test Accuracy: 0.9519450664520264

