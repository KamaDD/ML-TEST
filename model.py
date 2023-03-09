import os
import cv2
from jax import checkpoint_policies
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.callbacks import ModelCheckpoint

#def function; images and labels
def load_dataset(dataset_path):
    images = []
    labels = []
    for folder_name in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder_name)
        if not os.path.isdir(folder_path):
            continue
        label = folder_name
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            image = cv2.imread(image_path)
            image = cv2.resize(image, (64, 64))
            images.append(image)
            labels.append(label)
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

#Call the load_dataset() function to load the dataset into memory.
# Replace "dataset_path" with the path to the directory where you extracted the ASL dataset.
dataset_path = "./asl_dataset"
images, labels = load_dataset(dataset_path)

#convert to one hot encoding
unique_labels = np.unique(labels)
label_to_index = {label: index for index, label in enumerate(unique_labels)}
index_to_label = {index: label for label, index in label_to_index.items()}
labels_one_hot = np.array([label_to_index[label] for label in labels])
labels_one_hot = np.eye(len(unique_labels))[labels_one_hot]

#split dataset
X_train, X_test, y_train, y_test = train_test_split(images, labels_one_hot, test_size=0.2, random_state=42)

#define the model
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(len(unique_labels), activation="softmax")
])

checkpoint = ModelCheckpoint('model-{epoch:03d}.h5', verbose=1, monitor='val_loss',save_best_only=True, mode='auto')
# fit the model on the training data
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, callbacks=[checkpoint_policies])

# evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)


#compile model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])


#save the model
checkpoint = ModelCheckpoint("best_model.h5")
