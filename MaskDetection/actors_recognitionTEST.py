from itertools import groupby
import random
from matplotlib import pyplot as plt
import cv2
import numpy as np
from PIL import Image
import random
from WebCamMaskDetection.MaskCropper import cropEyeLineFromMasked
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input
from keras.applications import VGG16
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.applications.vgg16 import preprocess_input
from keras.callbacks import ModelCheckpoint
import math
import os
import shutil
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import classification_report

IMAGE_SIZE = (224, 224)

def splitDataset(trainPath,valPath):
  listOfactorsTrain = os.listdir(trainPath)
  listOfactorsVal = os.listdir(valPath)
  print(" listOfactorsTrain :" ,listOfactorsTrain)
  print(" listOfactorsVal :" ,listOfactorsVal)
  valPercent = 30
  for actor in listOfactorsTrain:
    actorPath = trainPath + '/' + actor
    actorsImages = os.listdir(actorPath)
    # print(actor," : ",actorsImages)
    numOfImages = len(actorsImages)
    print(actor," LEN : ",numOfImages)
    # numOfImages : 100 = x : valPercent
    valSize = int((numOfImages * valPercent)/100)
    print(actor," valSize : ",valSize)
    valImages = actorsImages[:valSize]
    for image in valImages:
      shutil.move(trainPath + '/' + actor + '/' + image, valPath + '/' + actor)
      print("moved: ",image)


def __training():
    random.seed(3)
    # path = '/gdrive/My Drive/SysAgdatasetCropped/train' + '/' + 'Angelina Jolie' + '/' + '166.jpg'
    # im = pyplot.imread(path)

    # def printImage(image):
    #   pyplot.imshow(image)
    #   pyplot.show()

    # printImage(im)

    # image_resized = cv2.resize(im, (224, 224))
    # printImage(image_resized)
    vgg16 = VGG16(weights=None, include_top=False, input_shape=(224, 224, 3))
    for layer in vgg16.layers[:-4]:
        layer.trainable = False
    model = Sequential()

    model.add(vgg16)

    model.add(Flatten())
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation="softmax"))

    vgg16.summary()
    model.summary()
    train_folder = "./SysAgdatasetCropped/train"
    val_folder = "./SysAgdatasetCropped/val"
    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    #splitDataset(train_folder, val_folder)
    train_batchsize = 20  # Testare gli IperParametri
    val_batchsize = 10

    train_generator = train_datagen.flow_from_directory(
        train_folder,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=train_batchsize,
        class_mode="categorical"
    )

    val_generator = val_datagen.flow_from_directory(
        val_folder,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=val_batchsize,
        class_mode="categorical",
        shuffle=False
    )
    filepath = "./Models/vgg16_v1.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    num_epochs = 30
    learning_rate = 1.25e-4
    adam = Adam(lr=learning_rate)
    model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=['acc'])
    history = model.fit_generator(
        train_generator,
        epochs=num_epochs,
        validation_data=val_generator,
        verbose=1,
        callbacks=callbacks_list
    )
    ###############################
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

def show_confusion_matrix(validations, predictions, labels):

    matrix = metrics.confusion_matrix(validations, predictions)
    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix,
                cmap='coolwarm',
                linecolor='white',
                linewidths=1,
                xticklabels=labels,
                yticklabels=labels,
                annot=True,
                fmt='d')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def __testing():
    test_folder = "./Images4TestCropped/"
    #IMAGE_SIZE = 224
    random.seed(3)
    test_batchsize = 1
    model = load_model('./models/vgg16_v1.hdf5')
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_generator = test_datagen.flow_from_directory(
        test_folder,
        target_size=(224, 224),
        batch_size=test_batchsize,
        class_mode='categorical',
        shuffle=False)

    ground_truth = test_generator.classes
    label2index = test_generator.class_indices
    idx2label = dict((v, k) for k, v in label2index.items())
    predictions = model.predict_generator(test_generator,
                                          steps=test_generator.samples / test_generator.batch_size, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    print(predicted_classes)
    print(ground_truth)

    errors = np.where(predicted_classes != ground_truth)[0]
    print("No of errors = {}/{}".format(len(errors), test_generator.samples))
    labels = ['Andrew Garfield', 'Angelina Jolie', 'Anthony Hopkins', 'Ben Affleck', 'Beyonce Knowles']
    show_confusion_matrix(predicted_classes, ground_truth, labels)
    print(classification_report(predicted_classes, ground_truth))

def actor_recognition(img, original_img):
    model = load_model('./models/vgg16_v1.hdf5')
    # image = cv2.imread('./Images4Test/')
    # imageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(img, IMAGE_SIZE)
    # image_np = image_resized / 255.0  # Normalized to 0 ~ 1
    image_exp = np.expand_dims(image_resized, axis=0)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # result = vgg16.predict_classes(image_exp)
    result = model.predict(image_exp)
    print(result)
    key = np.argmax(result, axis=1)
    actor = actors_dict.get(key[0])
    cv2.putText(original_img, actor, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    Image.fromarray(original_img).show()
    # print(result)


actors_dict = {0: 'Andrew Garfield',
               1: 'Angelina Jolie',
               2: 'Anthony Hopkins',
               3: 'Ben Affleck',
               4: 'Beyonce Knowles'}


if __name__ == '__main__':
    '''
    for actor in range (0, 5):
        print(actors_dict.get(actor))
        for index in range(1, 31):
            PATH_IMG = './SysAgWmask/' +actors_dict.get(actor)+ '/' + '('+str(index) + ').jpg'
            img = cv2.imread(PATH_IMG)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_crop = cropEyeLineFromMasked(img)#100dpi-->96
            #image_resized = cv2.resize(img_crop, (224,112))#media train set
            #image_resized = cv2.resize(img_crop, IMAGE_SIZE)
            #image_exp = np.expand_dims(image_resized, axis=0)#,cv2.cvtColor(img_crop, cv2.COLOR_RGB2BGR)
            cv2.imwrite('./Images4TestCropped/' +actors_dict.get(actor)+ '/'+ str(index) + '.jpg', cv2.cvtColor(img_crop, cv2.COLOR_RGB2BGR))
    '''
    __testing()
    #actor_recognition(img_crop, img)

