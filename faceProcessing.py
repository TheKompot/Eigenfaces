import cv2 as cv  # opencv library
import numpy as np
from math import atan, degrees
from urllib.request import urlopen
from urllib.error import URLError


class FaceCouldNotBeAligned(Exception):
    """
    Raised when the face could not be aligned,
    most probably at some point some features could not be found
    """
    pass


# face and eye detectors
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')


def detectFace(image, returnCutOut=True):
    """
     locate (the largest) frontal face in the image
    :param image: image in the numpy matrix format (most preferably openCV formats, either a grayscale or BGR)
    :param returnCutOut: bool, specifies if an image should be returned or just the location and size of the face
    :return: based on the returnCutOut parameter, returns either the image of the face or the face's location and size
    """
    # locate faces
    faces = list(face_cascade.detectMultiScale(image))

    # if no face found, raise an error
    if len(faces) == 0:
        raise FaceCouldNotBeAligned("No face was found")

    # sort the faces by width (height should be the same as width)
    # and take the largest face found
    # (to ignore "phantom faces")
    faces.sort(key=lambda a: a[2], reverse=True)
    x, y, w, h = faces[0]

    # if we wish to return the image of the face
    if returnCutOut:
        # cut out the face and return it
        return image[y:y+h, x:x+w]

    # if we wish to find the location of the face, just return those
    return x, y, w, h


def alignFace(image):
    """
    extract an aligned face from the image
    :param image: source image
    :return: image of the largest aligned face from the image
    """
    # get the location and size of the largest face
    x, y, w, h = detectFace(image, False)

    # find two largest eyes inside the faces boundaries
    eyeL, eyeR = findEyes(image, x, x+w, y, y+h)

    # return the cut out of the face from the image rotated so that the eye's centres are on a horizontal line
    return detectFace(rotate(image,
                             degrees(atan((eyeR[1] - eyeL[1]) / (eyeR[0] - eyeL[0]))),
                             ((eyeR[0] + eyeL[0]) / 2, (eyeR[1] + eyeL[1]) / 2)))


def findEyes(image, minX, maxX, minY, maxY):
    """
    find two largest eyes in the area bound by the (min|max)[XY] parameters
    :param image: source image
    :param minX: left x boundary
    :param maxX: right x boundary
    :param minY: top Y boundary
    :param maxY: bottom Y boundary
    :return: list with the centres of the eyes, where the first on is the left eye and the second is the right eye
    """
    # locate eyes in the image
    eyes = list(eye_cascade.detectMultiScale(image))

    # sort the eyes by size (as there might be some other "ghost eyes" found, we want the largest)
    eyes.sort(key=lambda a: a[2], reverse=True)

    # compute eye centers
    eyes = list(map(lambda eye: (eye[0] + eye[2] // 2, eye[1] + eye[3] // 2), eyes))

    # filter out eyes with their centres outside the boundaries
    eyes = list(filter(lambda eye: (minX <= eye[0] <= maxX) and (minY <= eye[1] <= maxY), eyes))

    # if less than two eyes remain, raise an error
    if len(eyes) < 2:
        raise FaceCouldNotBeAligned("Couldn't find two eyes")

    # discard all but two (biggest, thanks to the sorting) eyes
    eyes = eyes[:2]

    # check that the left eye is first in the list, if not then remedy
    if eyes[0][0] > eyes[1][0]:
        eyes = eyes[::-1]
    return eyes


def rotate(image, angle, point):
    """
    rotate the image by the specified angle around the specified point
    :param image: image to rotate
    :param angle: angle in degrees to rotate by
    :param point: tuple (x, y), the centre point of the rotation
    :return: rotated image
    """
    w, h = image.shape
    return cv.warpAffine(image, cv.getRotationMatrix2D(point, angle, 1), (h, w))


def fetchImage(url, mode=cv.IMREAD_GRAYSCALE):
    """
    loads image from the url
    :param url: source of the image
    :param mode: format in which we wish to return the image
    :return: loaded image in the specified format(mode) (openCV format: numpy matrix)
    """
    # download the image, convert it to a numpy array, and then read
    # it into openCV format
    resp = urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv.imdecode(image, mode)

    # return the image
    return image


def resize(image, width, height):
    """
    resize the image
    :param image: the image to resize
    :param width: final width
    :param height: final height
    :return: resized image
    """
    return cv.resize(image, (width, height), interpolation=cv.INTER_AREA)


def createVectorAndMatrixFromListOfURLs(baseURL, listOfExtensions, widthOfImage=100, heightOfImage=100):
    """
    :param baseURL: baseURL string
    :param listOfExtensions: iterable of extensions
    :param widthOfImage: desired unified image width
    :param heightOfImage: desired unified image height
    :return: tuple:
                vector of extensions
                matrix where i-th row is an image of the aligned face corresponding to the i-th extension
    """
    # specify format to load the images in
    mode = cv.IMREAD_GRAYSCALE

    # create a list for working extensions and loaded images
    workingExtensions = []
    images = []

    # loop through all the extensions and gather the images
    for extension in listOfExtensions:
        try:
            # load the image from the URL
            image = fetchImage(baseURL + extension, mode)
            images.append(image)

            # add the extension to the list of working extensions
            workingExtensions.append(extension)
        except URLError:
            # if the image could not be loaded, we ignore the extension
            # the reasons for not being able to lead the image are:
            #   no image to load, connection timeout, ...
            pass

    # check that the number of gathered images is the same as the number of extensions deemed working
    assert len(workingExtensions) == len(images)

    # initiate lists for images that a face could be found in
    alignedExtensions = []
    alignedImages = []

    # loop through the images we have gathered and extract the aligned faces
    for i, image in enumerate(images):
        try:
            # extract the face
            alignedImages.append(resize(alignFace(image), widthOfImage, heightOfImage).flatten())

            # save the extension associated with the image
            alignedExtensions.append(workingExtensions[i])
        except FaceCouldNotBeAligned:
            # if the face could not be aligned, we ignore the image
            pass

    # check that the number of remaining images is the same as the number of extensions associated with them
    assert len(alignedExtensions) == len(alignedImages)

    return np.array(alignedExtensions), np.array(alignedImages)
