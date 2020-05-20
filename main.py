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
import cv2
from PIL import Image

actors = ['Andrew Garfield-Mask', 'Angelina Jolie-Mask', 'Anthony Hopkins-Mask', 'Ben Affleck-Mask',
          'Beyonce Knowles-Mask']
original_img_path = 'H:\OLDSysAg\Datasets\Datasets\ActorsWMask' + '/' + actors[4] + '/' + '1.jpg'
original_img = cv2.imread(original_img_path)
original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
Image.fromarray(original_img).show()
checkMask = original_img.copy()

output = inference(checkMask, show_result=True, target_shape=(260, 260))
print(output[0][0])

if output[0][0] == 0:
    cropped_img = cropEyeLineFromMasked(original_img)
    Image.fromarray(cropped_img).show()

    Single_test(original_img, cropped_img)
else:
    ##### DA FARE
    print("Fare ramo senza maschera")
    # Single_test_noMask(original_img)
    ##### DA FARE
