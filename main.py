# Esecuzione dell'intera rete di FMD&R

# Ottenere in input la foto
# Verificare se indossa una maschera
# Se si
#   taglia l'Eyeline e esegue il riconoscimento
#    Output foto non croppata con nome previsione
# Se no
#   esegue il riconoscimento su tutto il volto
#   Output foto non croppata con nome previsione

from MaskDetection.actors_recognitionTEST import Single_test
from MaskDetection.MaskCropper import cropEyeLineFromMasked
from MaskDetection.keras_infer import inference
from PIL import Image
import cv2

actors = ['Andrew Garfield', 'Angelina Jolie', 'Anthony Hopkins', 'Ben Affleck',
          'Beyonce Knowles']

actorsWMak = ['Andrew Garfield-Mask', 'Angelina Jolie-Mask', 'Anthony Hopkins-Mask', 'Ben Affleck-Mask',
          'Beyonce Knowles-Mask']

imageName = '3.jpg'
original_img_path = 'H:\OLDSysAg\Datasets\ActorsWMask' + '/' + actorsWMak[1] + '/' + imageName
original_img_path = 'H:\OLDSysAg\Datasets\SysAgDatasetFaceOnly\Test' + '/' + actors[1] + '/' + imageName

original_img = cv2.imread(original_img_path)
original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
Image.fromarray(original_img).show()
checkMask = original_img.copy()

output = inference(checkMask, show_result=True, target_shape=(260, 260))
# print(output[0][0])

if output[0][0] == 0:   # Maschera identificata
    print("Attore con maschera")
    cropped_img = cropEyeLineFromMasked(original_img)
    Image.fromarray(cropped_img).show()
    Single_test(original_img, cropped_img)
else:   # Maschera non identificata
    print("Attore senza maschera")
    Single_test(original_img, "X")
