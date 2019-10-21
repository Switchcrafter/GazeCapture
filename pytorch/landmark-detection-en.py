from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
 
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

cap = cv2.VideoCapture(0)
 
while True:
    _, image = cap.read()

    faceImage = None
    rightEyeImage = None
    leftEyeImage = None
    faceGrid = None
    isValid = False

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faceRects = detector(gray, 0)

    # if faceRects length != 1, then not valid

    for (i, rect) in enumerate(faceRects):
        shape = predictor(gray, rect)
        npshape = face_utils.shape_to_np(shape)

        (leftEyeLandmarksStart, leftEyeLandmarksEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rightEyeLandmarksStart, rightEyeLandmarksEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        (x, y, w, h) = cv2.boundingRect(npshape)

        if x >= 0 and y >= 0 and x + w <= image.shape[1] and y + h <= image.shape[0]:
            print('Face (%4d, %4d, %4d, %4d)' % (x, y, w, h))
            isValid = True
            faceImage = image.copy()
            faceImage = faceImage[y:y + h, x:x + w]
            faceImage = imutils.resize(faceImage, width=225, inter=cv2.INTER_CUBIC)

            (x, y, w, h) = cv2.boundingRect(npshape[leftEyeLandmarksStart:leftEyeLandmarksEnd])
            leftEyeImage = image.copy()
            leftEyeImage = leftEyeImage[y-15:y + h + 15, x-5:x + w+5]
            leftEyeImage = imutils.resize(leftEyeImage, width=61, inter=cv2.INTER_CUBIC)

            (x, y, w, h) = cv2.boundingRect(npshape[rightEyeLandmarksStart:rightEyeLandmarksEnd])
            rightEyeImage = image.copy()
            rightEyeImage = rightEyeImage[y-15:y + h + 15, x-5:x + w+5]
            rightEyeImage = imutils.resize(rightEyeImage, width=61, inter=cv2.INTER_CUBIC)

            if rect.tl_corner().x < 0 or rect.tl_corner().y < 0:
                isValid = False

            cv2.rectangle(image, (rect.tl_corner().x, rect.tl_corner().y), (rect.br_corner().x, rect.br_corner().y),
                          (255, 255, 00), 2)

            imageWidth = image.shape[1]
            imageHeight = image.shape[0]

            faceGridX = int((rect.tl_corner().x / imageWidth) * 25)
            faceGridY = int((rect.tl_corner().y / imageHeight) * 25)
            faceGridW = int((rect.br_corner().x / imageWidth) * 25) - faceGridX
            faceGridH = int((rect.br_corner().y / imageHeight) * 25) - faceGridY

            faceGrid = np.zeros((25, 25, 1), dtype=np.uint8)
            faceGrid.fill(255)
            for m in range(faceGridW):
                for n in range(faceGridH):
                    faceGrid[faceGridY + n, faceGridX + m] = 0

            # Draw on our image, all the found coordinate points (x,y)
            for (x, y) in npshape:
                cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

        else:
            print('Face (%4d, %4d, %4d, %4d) Not Valid' % (x, y, w, h))

    cv2.imshow("WebCam", image)
    if isValid:
        cv2.imshow("face", faceImage)
        cv2.imshow("rightEye", rightEyeImage)
        cv2.imshow("leftEye", leftEyeImage)
        cv2.imshow("faceGrid", faceGrid)
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
