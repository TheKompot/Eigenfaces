import cv2 as cv  # opencv library
import numpy as np
from math import atan, degrees
from urllib.request import urlopen
from urllib.error import HTTPError


class FaceCouldNotBeAligned(Exception):
    """
    Raised when the face could not be aligned,
    most probably at some point some features could not be found
    """
    pass


face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')


def detectFace(image, returnCutOut=True):
    # locate faces
    faces = list(face_cascade.detectMultiScale(image))

    # if no face found, raise an error
    if len(faces) == 0:
        raise FaceCouldNotBeAligned("No face was found")

    # find the largest face found (to ignore "phantom faces")
    faces.sort(key=lambda a: a[2], reverse=True)

    # cut out the face and return it
    x, y, w, h = faces[0]
    if returnCutOut:
        return image[y:y+h, x:x+w]
    return x, y, w, h


def alignFace(image):
    x, y, w, h = detectFace(image, False)
    eyeL, eyeR = findEyes(image, x, x+w, y, y+h)
    return detectFace(rotate(image,
                             degrees(atan((eyeR[1] - eyeL[1]) / (eyeR[0] - eyeL[0]))),
                             ((eyeR[0] + eyeL[0]) / 2, (eyeR[1] + eyeL[1]) / 2)))


def findEyes(image, minX, maxX, minY, maxY):
    # locate eyes
    eyes = list(eye_cascade.detectMultiScale(image))
    # sort by size (as there might be some other "ghost eyes" found, we want the largest)
    eyes.sort(key=lambda a: a[2], reverse=True)
    # compute eye centers
    eyes = list(map(lambda eye: (eye[0] + eye[2] // 2, eye[1] + eye[3] // 2), eyes))
    # filter out eyes outside the boundaries
    eyes = list(filter(lambda eye: (minX <= eye[0] <= maxX) and (minY <= eye[1] <= maxY), eyes))
    # if less than two eyes remain, raise an error
    if len(eyes) < 2:
        raise FaceCouldNotBeAligned("Couldn't find two eyes")
    # discard all but two (biggest) eyes
    eyes = eyes[:2]
    # check that the left eye is first in the list, if not then remedy
    if eyes[0][0] > eyes[1][0]:
        eyes = eyes[::-1]
    return eyes


def rotate(image, angle, point):
    w, h = image.shape
    return cv.warpAffine(image, cv.getRotationMatrix2D(point, angle, 1), (h, w))


def fetchImage(url, mode=cv.IMREAD_GRAYSCALE):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    resp = urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv.imdecode(image, mode)

    # return the image
    return image


def resize(image, width, height):
    return cv.resize(image, (width, height), interpolation=cv.INTER_AREA)


def createVectorAndMatrixFromListOfURLs(baseURL, listOfExtensions, widthOfImage=100, heightOfImage=100):
    """
    :param baseURL:
    :param listOfExtensions: iterable of extensions
    :param widthOfImage: desired unified image width
    :param heightOfImage: desired unified image height
    :return: tuple:
                vector of extensions
                matrix where i-th row is an image of the aligned face corresponding to the i-th extension
    """
    mode = cv.IMREAD_GRAYSCALE
    workingExtensions = []
    images = []
    for extension in listOfExtensions:
        try:
            image = fetchImage(baseURL + extension, mode)
            images.append(image)
            workingExtensions.append(extension)
        except:
            pass
    assert len(workingExtensions) == len(images)

    alignedExtensions = []
    alignedImages = []
    for i, image in enumerate(images):
        try:
            alignedImages.append(resize(alignFace(image), widthOfImage, heightOfImage).flatten())
            alignedExtensions.append(workingExtensions[i])
        except FaceCouldNotBeAligned:
            pass
    assert len(alignedExtensions) == len(alignedImages)

    return np.array(alignedExtensions), np.array(alignedImages)
