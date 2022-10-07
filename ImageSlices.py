import cv2
import numpy as np

## ImageMaskSlices make slices of shape (Imsize, Imsize) of the Im and returns the upper-left coordinates of
## the slices which has labels on it and have less than a 30% of white background

def ImageSlices(Im, Imsize):
    (ImXsize, ImYsize, _) = Im.shape
    Xslices = np.arange(0, ImXsize, Imsize)
    Xslices[-1] = max(ImXsize - Imsize, 0)
    Yslices = np.arange(0, ImYsize, Imsize)
    Yslices[-1] = max(ImYsize - Imsize, 0)
    return [[x, y] for x in Xslices for y in Yslices]

def ImageMaskSlices(Im, Imsize,row, Imresize = 0):
    #Black and white conversion
    ImG = cv2.cvtColor(Im, cv2.COLOR_BGR2GRAY)
    (thresh, Imbw) = cv2.threshold(ImG, 150, 1, cv2.THRESH_BINARY_INV)
    kernel = np.ones((15, 15), np.uint8)
    Imbw = cv2.dilate(Imbw, kernel, iterations=3)
    (OriImsize, _) = Imbw.shape
    Xslices = np.arange(0, OriImsize, Imsize)
    Xslices[-1] = OriImsize - Imsize
    Yslices = np.arange(0, OriImsize, Imsize)
    Yslices[-1] = OriImsize - Imsize

    Slices = []

    #Mask generation from rle
    w = row['img_width']
    h = row['img_height']
    rle = row['rle']
    s = rle.split()
    starts, lengths = [np.asarray(t, dtype='int') for t in (s[0:][::2], s[1:][::2])]
    starts = starts - 1
    mask = np.zeros(h * w, dtype=np.uint8)
    for s, l in zip(starts, lengths):
        mask[s:s + l] = 1
    mask = mask.reshape((h, w)).T
    if (Imresize > 0):
        mask = cv2.resize(mask, (Imresize, Imresize))


    for x in Xslices:
        for y in Yslices:
            MaskAny = mask[x:x+Imsize, y:y+Imsize].sum() != 0
            BwProp = Imbw[x:x+Imsize, y:y+Imsize].sum()/(Imsize**2)
            #Append slices with mask and low proportion of white background
            if((MaskAny == True) and (BwProp > 0.7)):
                Slices.append([x,y])
    return Slices








