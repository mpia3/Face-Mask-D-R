import cv2
import dlib
import numpy as np


def rect_to_bb(rect):
        x = rect.left()
        y = rect.top()
        w = rect.right() - x
        h = rect.bottom() - y
        return x, y, w, h


def shape_to_np(shape, dtype="int"):
        coords = np.zeros((68, 2), dtype=dtype)
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords


PATH_MODEL = './shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
# detector = dlib.cnn_face_detection_model_v1("/gdrive/My Drive/OpenCVFiles/Dlib_Models/mmod_human_face_detector.dat")
predictor = dlib.shape_predictor(PATH_MODEL)

cap = cv2.VideoCapture(0)
frame_number = 1
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    rects = detector(frame, 1)
    print("Frame #{}, number of faces detected: {}".format(frame_number, len(rects)) + '')

    for (i, rect) in enumerate(rects):
        shape = predictor(frame, rect)
        shape = shape_to_np(shape)
        (x, y, w, h) = rect_to_bb(rect)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Face #{}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    frame_number += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
