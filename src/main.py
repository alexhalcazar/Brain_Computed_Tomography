import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder



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
        image = load_img(image_path, target_size=(224, 224))  # Load image and resize to target size
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

# Reshape each image pixels into a row of feature table with 224*224*3=150,528 features (each pixel is a feature):
X_train = X_train.reshape(X_train.shape[0], 224 * 224 * 3)
X_test = X_test.reshape(X_test.shape[0], 224 * 224 * 3)

print(X_train.shape)
print(X_test.shape)

# Normalize pixel values
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

print (X_train.shape)
print (X_train[:10])

# Initialize label encoder
label_encoder = LabelEncoder()

# Encode class labels
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)
