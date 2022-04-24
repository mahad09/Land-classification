import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from keras.layers import Activation, Dense, Dropout, Flatten
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.layers import Dense, BatchNormalization, Flatten, Activation, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
import seaborn as sns
import datetime
import pdb
from skimage.io import imread, imsave
from tqdm import tqdm
import cv2



TRAINING_DATA_DIR = 'dataset_1/train/'
VALIDATION_DATA_DIR = 'dataset_1/validate/'

IMG_HEIGHT = 256
IMG_WIDTH = 256
INPUT_SHAPE = (256, 256, 3)
BATCH_SIZE = 32
EPOCHS = 200
NUM_OF_CLASSES = 21


# for class_name in os.listdir(DATA_DIR):
#   for image_name in os.listdir(os.path.join(DATA_DIR, class_name)):
#     image = imread(os.path.join(DATA_DIR, class_name, image_name))
#     os.remove(os.path.join(DATA_DIR, class_name, image_name))
#     basename, _ = os.path.splitext(image_name)
#     target_image_name = os.path.join(DATA_DIR, class_name, basename+'.png')
#     imsave(target_image_name, image)


def data_generators():
  print('defining generators of training and validation sets...')
  train_datagen = ImageDataGenerator(
    rotation_range = 40,                  
    width_shift_range = 0.2,                  
    height_shift_range = 0.2,                  
    rescale = 1./255,                  
    shear_range = 0.2,                  
    zoom_range = 0.2,                     
    horizontal_flip = True
    )

  validation_datagen = ImageDataGenerator(rescale = 1./255)

  train_data = train_datagen.flow_from_directory(
    TRAINING_DATA_DIR, target_size=(IMG_WIDTH, IMG_HEIGHT), batch_size=BATCH_SIZE, class_mode='categorical')

  validation_data = validation_datagen.flow_from_directory(
    VALIDATION_DATA_DIR, target_size=(IMG_WIDTH, IMG_HEIGHT), batch_size=BATCH_SIZE, class_mode='categorical')

  return train_data, validation_data


def custom_cnn():
  model = Sequential()
  model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(IMG_WIDTH, IMG_HEIGHT, 3),
                 data_format='channels_last'))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Conv2D(32, (3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Dropout(0.25))
  model.add(Flatten())
  model.add(Dense(128, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(21, activation='softmax'))
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

  return model


def model_training(model, train_generator, validation_generator):
  history = model.fit(
      train_generator,
      steps_per_epoch = train_generator.samples // BATCH_SIZE,
      validation_data = validation_generator, 
      validation_steps = validation_generator.samples // BATCH_SIZE,
      epochs = EPOCHS,
      workers=4)

  model.save('land_classifier.h5')

  return history, model


def model_evaluation(test_data):
  trained_model = keras.models.load_model('land_classifier.h5')
  overall_result = trained_model.evaluate(test_data)
  print(dict(zip(trained_model.metrics_names, overall_result)))

  y_pred = []  # store predicted labels
  y_true = []  # store true labels

  testing = []
  counter = 0
  actual = []
  for folder in tqdm(os.listdir(VALIDATION_DATA_DIR)):
    for image in os.listdir(VALIDATION_DATA_DIR + folder + '/'):
      img = cv2.imread(VALIDATION_DATA_DIR + folder + '/' + image)
      img = cv2.resize(img, (256, 256))
      img = np.array(img)
      img = img.reshape(1, 256, 256, 3)
      # img = img.flatten()
      img *= 255
      predict = trained_model.predict(img)
      # print(counter, 'actual: ', validation_generator.class_indices[folder],'  ', 'predicted: ', np.argmax(predict, axis=1))
      testing.append(np.argmax(predict))
      actual.append(test_data.class_indices[folder])
      counter += 1
    print(counter)

  cf_matrix(actual, testing)
  cls_report(actual, testing)


def cf_matrix(predicted_labels, correct_labels):
  cf_matrix = confusion_matrix(predicted_labels, correct_labels)
  fig, ax = plt.subplots(figsize=(8, 6))
  sns.heatmap(cf_matrix, cmap="YlGnBu", annot=True, linewidths=.5, ax=ax, fmt=".0f")
  plt.show()


def cls_report(predicted_labels, correct_labels):
  print(classification_report(predicted_labels, correct_labels))


def plot_accuracy(history):
    plt.title("Accuracy Graph")
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train_accuracy', 'validation_accuracy'], loc='best')
    plt.show()


def plot_loss(history):
    plt.title("Loss Graph")
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train_loss', 'validation_loss'], loc='best')
    plt.show()



train_data, validation_data = data_generators()
# compiled_model = custom_cnn()
# history, trained_model = model_training(compiled_model, train_data, validation_data)
# plot_accuracy(history)
# plot_loss(history)
model_evaluation(validation_data)





  


