import cv2 as cv
import numpy as np
class Road_Transform:

    def __init__(self, fliplr, rotate, norm, image_h, image_w, label_h, label_w):
        self.fliplr = fliplr
        self.rotate = rotate
        self.norm = norm
        self.image_h = image_h
        self.image_w = image_w
        self.label_h = label_h
        self.label_w = label_w

    def transform(self, in_data):

        img, label = in_data
        angle = np.random.randint(0, 360)
        sat_crop = patch_transform(img, self.fliplr, self.rotate, self.norm, angle, self.image_h, self.image_w)
        map_crop = patch_transform(label, self.fliplr, self.rotate, self.norm, angle,self.label_h, self.label_w)
        sat_crop = np.transpose(sat_crop.astype(np.float32), (2, 0, 1))
        sat_crop = sat_crop.astype(np.float32)
        map_crop = map_crop.astype(np.int32)
        return sat_crop, map_crop

def patch_transform(img, fliplr, rotate, norm, angle, image_h, image_w):

    img = np.array(img, dtype=np.float64)
    # Flipping L-R
    if(fliplr == True):
        img = cv.flip(img, 1)

    # Rotation with random arngle
    if (rotate == True):
        RotateMatrix = cv.getRotationMatrix2D(center=(img.shape[1]/2, img.shape[0]/2), angle = angle, scale = 1)
        img = cv.warpAffine(img, RotateMatrix, (img.shape[1], img.shape[0]), flags=cv.INTER_NEAREST)

    # Cropping
    crop = np.array(img[img.shape[0] // 2 - image_h // 2: img.shape[0] // 2 + image_h // 2, img.shape[1] // 2 - image_w // 2: img.shape[1] // 2 + image_w // 2], dtype=np.float64)
    
    # patch-wise mean subtraction
    if(norm == True and len(img.shape) == 3 and img.shape[2] == 3):
        mean, stddev = cv.meanStdDev(crop)
        slice = cv.split(crop)
        for i in range(len(slice)):
            slice[i] = cv.subtract(slice[i], mean[i])
            slice[i] /= stddev[i] + 1E-5
        crop = cv.merge(slice)
        del slice

    return crop
