import cv2
import numpy as np
from matplotlib import pyplot
import os
from PIL import Image
"""
Script dedicato al cropping dell'immagine
Necessario all'ottenimento dell'immagine/immagini delle sole "Linee degli occhi" dei volti mascherati
"""


def printImage(image):
    """
    Stampa a video l'immagine (Matrice di pixel) data in input
    :param image:  Matrice di pixel
    :return: None
    """
    pyplot.imshow(image)
    pyplot.show()


def cropEyeLine(pixels, x, y):
    """
    Taglia l'immagine alla linea degli occhi
    :param pixels: matrice di pixel rappresentante l'immagine
    :param x: Coordinata di taglio orizzontale
    :param y: Coordinata di taglio verticale
    :return: immagine tagliata
    """
    crop = pixels[0:y][0:x]
    # cv2_imshow(crop)
    return crop


def serchNotBlackPixel(image):
    """
    Cerca il pixel non nero nell'immagine di coordinata y inferiore (dove la mascherina inizia)
    :param image: matrice di pixel
    :return: coordinate (i,j) del pixel individuato
    """
    imgShape = image.shape
    print("Size img : ", imgShape)
    for i in range(imgShape[0]):
        for j in range(imgShape[1]):
            # if image[i, j] != [0, 0, 0]:
            if (image[i, j] != [0, 0, 0]).all():
                # print("Trovato = ", i, " - ", j)
                # print("image[i, j] = ", image[i, j])
                return (i, j)


def get_average_color(x, y, n, image):  # Non usata
    """
    Restituisce una tripla contenente il valore RGB della media colore nel riquadro nell'area di lunghezza n
    nel quale l'origine è nell'angolo in alto a sinistra
    :param x: coordinata x dell'area
    :param y: coordinata y dell'area
    :param n: larghezza del riquadro
    :param image: immagine di input (matrice di pixel)
    :return: tripla RGB
    """
    r, g, b = 0, 0, 0
    count = 0
    for s in range(x, x + n):
        for t in range(y, y + 10):
            pixlr, pixlg, pixlb = image[s, t]
            r += pixlr
            g += pixlg
            b += pixlb
            count += 1
    return ((r / count), (g / count), (b / count))


def hsv2rgb(hsvColor):
    """
    Conversione del formato colore HSV in RGB
    :param hsvColor: Stringa del colore HSV
    :return: Tripla RGB
    """
    # hsvColor = '#1C8A88'  #LOW
    # hsvColor = '#BDEFEF'   #HIGH
    h = hsvColor.lstrip('#')
    rgb = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    return rgb[0], rgb[1], rgb[2]


def cropEyeLineFromMasked(frame):
    """
    Funzione principale che restituisce la linea degli occhi di un immagine di un volto avente la mascherina
    :param image: immagine come matrice di pixel
    :return: immagine croppata presentante solo la linea degli occhi
    """
    # frame = pyplot.imread(image)
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    low_blue_hsv = '#147B7A'  # DA FARE IL TUNING
    R0, G0, B0 = hsv2rgb(low_blue_hsv)

    high_blue_hsv = '#36D2CD'  # '#4AD9D6'  # DA FARE IL TUNING
    R, G, B = hsv2rgb(high_blue_hsv)

    low_blue = np.array([R0, G0, B0])
    high_blue = np.array([R, G, B])
    blue_mask = cv2.inRange(hsv_frame, low_blue, high_blue)
    blue = cv2.bitwise_and(frame, frame, mask=blue_mask)

    #### printImage(blue)
    # img = Image.fromarray(blue, 'RGB')
    # img.show()

    i, j = serchNotBlackPixel(blue)
    imgShape = frame.shape

    Pixelcropped = cropEyeLine(frame, i, imgShape[1])

    return Pixelcropped


def cropMaskedActors(Path, croppedPath):
    """
    Funzione batch per il preprocessing utile per croppare le immagini di volti nel dataset di Masked Face
    :param Path: Percorso nel quale è collocato il dataset di input
    :param croppedPath:  Percorso nel quale è collocato il dataset di output
    :return:
    """
    listOfActors = os.listdir(Path)
    print(" listOfActors :", listOfActors)

    # for actor in listOfActors:
    #     os.makedirs(croppedPath + '/' + actor)

    # listOfActorsUnMasked = os.listdir(Path)
    # print(" listOfActorsUnMasked :" ,listOfActorsUnMasked)

    for actor in listOfActors:

        actorPath = Path + '/' + actor

        actorsImages = os.listdir(actorPath)
        print(actor, " : ", actorsImages)

        numOfImages = len(actorsImages)
        print(actor, " LEN : ", numOfImages)

        for image in actorsImages:
            try:
                pixelsCropped = cropEyeLineFromMasked(actorPath + '/' + image)

                ### printImage(pixelsCropped)
                # img = Image.fromarray(pixelsCropped, 'RGB')
                # img.show()

                outputpath = croppedPath + '/' + actor + '/' + image
                pyplot.imsave(outputpath, pixelsCropped)
                print("Saved: ", image)
            except:
                print("not Saved: ", image)


# TESTING ZONE

# Path = "H:\SysAg\ChinaMask\wear_mask_to_face\ActorsWMask"
# croppedPath = "H:\SysAg\ChinaMask\wear_mask_to_face\ActorsWMaskCroppedNoAlign"
#
# cropMaskedActors(Path, croppedPath)