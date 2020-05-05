from keras.models import load_model
import cv2
import numpy as np
from PIL import Image
from WebCamMaskDetection.MaskCropper import cropEyeLineFromMasked


IMAGE_SIZE = (224, 224)
actors_dict = {0: 'Andrew Garfield',
               1: 'Angelina Jolie',
               2: 'Anthony Hopkins',
               3: 'Ben Affleck',
               4: 'Beyonce Knowles'}


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
    key = np.argmax(result, axis=1)
    actor = actors_dict.get(key[0])
    cv2.putText(original_img, actor, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    Image.fromarray(original_img).show()
    # print(result)


PATH_IMG = './Images4Test/3.beyonce-knowles-social-media-10-10-2019-1.jpg'

if __name__ == '__main__':
    img = cv2.imread(PATH_IMG)
    # Image.fromarray(img).show()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Image.fromarray(img).show()
    img_crop = cropEyeLineFromMasked(img)
    Image.fromarray(img_crop).show()
    actor_recognition(img_crop, img)

