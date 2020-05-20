from itertools import groupby
import random
from matplotlib import pyplot as plt
import cv2
import numpy as np
from PIL import Image
import random
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
"""
La classe actor recognitionTEST serve per testare il modello appreso su un insieme di immagini classificate all'interno delle cartelle denominate con il nome ell'attore.

"""
def splitDataset(trainPath,valPath):
  """
  tale metodo ripartisce l'insieme in validatoin e train assegnado a il 30% dell'insieme al validation set e il resto al training
  :param trainPath: percorso train
  :param valPath: percorso validation
  :return:
  """
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
    """
    tale metodo effettua il training del modello.
    crea un modello vgg16 definendo lo shape delle immagini e la mappa dei colori.
    Definisce un modello sequenziale con una rectified linear unit e una funzione di attivazione softmax
    Definisce i folder di train e validation con i batchsize
    Crea generatori di train e valid. in modo categorical cioè definendo un problema di classificazione categorica.
    Infine stabilisce numero epoche, learning rate e compila il modello.
    Come ultima cosa mostra a video l'accuratezza raggiunta nelle varie epoche.
    :return:
    """
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
    """
    La matrice di mostra a video la differenza tra predizione e effettiva classificazione dell'immagine.
    :param validations: validazione (ground)
    :param predictions: predizioni effettuate
    :param labels: corrispondenze intero attore
    :return:
    """
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
    """
    Tale metodo effettua il testing del modello appreso. A tal fine legge il folder contenente le immagini di test croppate ed effettua la
    predizione per ognuna di esse.
    Il risultato è una lista dove ad ogni cella corrisponde la classificaizone per l'immagine.
    Infine si confronta la predizione con il ground truth e si calcola il n di errori
    Viene mostrata a video la matrice di confusione.
    :return:
    """
    test_folder = "H:\SysAg\SysAg\ActorsWMFinal"
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
    show_confusion_matrix(ground_truth, predicted_classes, labels)
    print(classification_report(ground_truth, predicted_classes))


def createTestTree():
    testPath = "H:/SysAg/SysAg/ActorsSingleTest/"
    for x in range(6):
        try:
            os.makedirs(testPath + "/" + str(x), exist_ok=True)
            print("Directory '%s' created successfully" % directory)
        except OSError as error:
            print("Directory '%s' can not be created")

def Single_test(original_img, imageTest):
    """
    Tale metodo effettua il testing del modello appreso. A tal fine legge il folder contenente l'immagine di test croppata ed effettua la
    predizione per ognuna di esse.
    Il risultato è la predizione dell'immagine in input
    :return:
    """

    # Insert Test image in Test-Tree
    testPath = "H:/SysAg/SysAg/ActorsSingleTest/0/"
    imageName = "Test.jpg"

    model = ""

    if imageTest == "X":
        cv2.imwrite(testPath + imageName, original_img)
        model = "actor_recognition.hdf5"
    else:
        cv2.imwrite(testPath + imageName, imageTest)
        model = "vgg16_v1.hdf5"

    # Make Prediction on Single Image
    test_folder = "H:\SysAg\SysAg\ActorsSingleTest"
    random.seed(3)
    test_batchsize = 1
    model = load_model('H:\SysAg\SysAg\MaskDetection/models/' + model)
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

    # Print Prediction
    labels = ['Andrew Garfield', 'Angelina Jolie', 'Anthony Hopkins', 'Ben Affleck', 'Beyonce Knowles']
    print("I think it is: ", labels[predicted_classes[0]])

    # Insert name of Actor on image
    cv2.putText(original_img, labels[predicted_classes[0]], (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    Image.fromarray(original_img).show()

    # Clean Test-Tree
    os.remove(testPath+imageName)


actors_dict = {0: 'Andrew Garfield',
               1: 'Angelina Jolie',
               2: 'Anthony Hopkins',
               3: 'Ben Affleck',
               4: 'Beyonce Knowles'}

"""
Il main effettua il la crop delle immagini con mascherina e la resize e scrive tali eimmaginin nella cartella Images4TestCropped
Dopodiche avvia il testing su tali immagini
"""
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
    # actor_recognition(img_crop, img)

    # Single_test(original_img, imageTest)