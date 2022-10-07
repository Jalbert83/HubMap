import pandas as pd
from DataSequence import ImSequence, SliceSequence, SliceTTASequence
import segmentation_models as sm
from segmentation_models import Unet
import tensorflow as tf
from ImageSlices import ImageSlices
import cv2
import matplotlib.pyplot as plt
import numpy as np
import albumentations as A


def rle2mask(mask_rle, shape=(3000, 3000)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (width,height) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img.reshape(shape).T

def mask2rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)

def DfSlices(ds, ImageSize, impath):
    df = pd.DataFrame()
    df['Xslice'] = int(0)
    df['Yslice'] = int(0)
    df['img_width'] = int(0)
    df['img_height'] = int(0)
    df['id'] = int(0)
    df['rle'] = ""
    s = 0

    for i, row in ds.iterrows():
        path = impath + str(int(row['id'])) + ".tiff"
        im = cv2.imread(path)
        slices = ImageSlices(im, ImageSize)
        for slice in slices:
            df.loc[s, 'Xslice'] = int(slice[0])
            df.loc[s, 'Yslice'] = int(slice[1])
            df.loc[s, 'img_width'] = int(ds.loc[i, 'img_width'])
            df.loc[s, 'img_height'] = int(ds.loc[i, 'img_height'])
            df.loc[s, 'id'] = int(ds.loc[i, 'id'])
            s += 1
    return df


def ComposeImage(ImChunks, row, ImgSize) :
    h = int(row['img_height'])
    w = int(row['img_width'])
    OutIm = np.zeros((h,w,3))
    j = 0
    im_array = ImChunks.__getitem__(0)
    Xslices = np.arange(0, w, ImgSize)
    Xslices[-1] = max(w - ImgSize, 0)
    Yslices = np.arange(0, h, ImgSize)
    Yslices[-1] = max(h - ImgSize, 0)
    xy = [[y, x] for y in Yslices for x in Xslices]

    for (j,im) in enumerate(im_array):
        x = xy[j][1]
        y = xy[j][0]
        if ((h < ImgSize) | (w < ImgSize)):
            OutIm = cv2.resize(im,(h, w))
        else:
            OutIm[x:x+ImgSize, y:y+ImgSize, :] = im

    return OutIm



def ComposeMask(MaskChunks, Df, ImgSize):
    h = int(row['img_height'])
    w = int(row['img_width'])
    OutIm = np.zeros((h, w), dtype = bool)
    Xslices = np.arange(0, w, ImgSize)
    Xslices[-1] = max(w - ImgSize, 0)
    Yslices = np.arange(0, h, ImgSize)
    Yslices[-1] = max(h - ImgSize, 0)
    xy = [[y, x] for y in Yslices for x in Xslices]

    for j in range(len(xy)):
        x = xy[j][1]
        y = xy[j][0]

        mask = MaskChunks[j].round()
        if ((h < ImgSize) | (w < ImgSize)):
            OutIm = cv2.resize(mask,(h,w))
        else:
            OutIm[x:x + ImgSize, y:y + ImgSize] |= np.array(mask[:,:,0], dtype= bool)

    return np.array(OutIm, dtype='uint8')


IMG_SIZE = 512

# tta functions
tta_transforms = [

    A.HorizontalFlip(p=1),
    A.VerticalFlip(p=1),
    A.Transpose(p=1),
    A.Compose([
        A.Transpose(p=1),
        A.VerticalFlip(p=1),
    ])#,



]

tta_inv_transforms = [
    A.HorizontalFlip(p=1),
    A.VerticalFlip(p=1),
    A.Transpose(p=1),
    A.Compose([
        A.VerticalFlip(p=1),
        A.Transpose(p=1),
    ])
]


INTERNET = True

TEST_CSV_PATH = 'test2.csv'

# Prediction and file submission
TEST_BATCH_SIZE = 32





df_submission = pd.DataFrame(columns=["id", "rle"]).set_index("id")

df_test = pd.read_csv(TEST_CSV_PATH)

remaining_test = len(df_test)

test_i = 0

ModelSliced = Unet(backbone_name='efficientnetb5', encoder_weights='imagenet', encoder_freeze=True)

ModelSliced.load_weights("best_model_Sliced_30Epoch.h5")

ModelFull = Unet(backbone_name='efficientnetb5', encoder_weights='imagenet', encoder_freeze=True)

ModelFull.load_weights("best_model_Fullimage_80Epoch.h5")


DEBUG = True
IMG_SIZE2 = 640


for (iterId, row) in df_test.iterrows():
    h = int(row['img_height'])
    w = int(row['img_width'])
    FullMask = np.zeros((h, w))
    MaskArray = np.zeros((2 * (len(tta_transforms) + 1), h, w))
    FullIm = np.zeros((h, w, 3))
    for i, (transform, transform_inv) in enumerate(zip(([None] + tta_transforms), ([None] + tta_inv_transforms))):
        imgs = SliceTTASequence(IMG_SIZE, row, ttatransform=transform)
        masks = ModelSliced.predict(imgs).round()

        FullMask = ComposeMask(masks, row, IMG_SIZE)
        FullIm = ComposeImage(imgs, row, IMG_SIZE)
        if (transform_inv != None):
            FullMaskInverted = transform_inv(mask=FullMask, image=FullIm)
            FullMask = FullMaskInverted['mask']
            FullIm = FullMaskInverted['image']
        MaskArray[i] = FullMask
        if (DEBUG):
            plt.figure(i)
            plt.subplot(1, 2, 1)
            plt.imshow(FullIm)
            # plt.imshow(FullMask, alpha=0.5, cmap='gray')
            plt.subplot(1, 2, 2)
            plt.imshow(FullMask, alpha=0.5, cmap='gray')
            plt.axis("off")

    for i, (transform, transform_inv) in enumerate(zip(([None] + tta_transforms), ([None] + tta_inv_transforms)),
                                                   start=(len(tta_transforms) + 1)):
        imgs = ImSequence(1, IMG_SIZE2, df_test[df_test['id'] == int(row['id'])], transform=transform, set_type="test")
        masks = ModelFull.predict(imgs).round()

        if (transform_inv != None):
            FullMaskInverted = transform_inv(mask=masks[0], image=imgs.__getitem__(0)[0])
            FullMask = FullMaskInverted['mask']
            FullIm = FullMaskInverted['image']
        else:
            FullMask = masks[0]
            FullIm = imgs.__getitem__(0)[0]

        MaskArray[i] = cv2.resize(FullMask, (h, w))
        if (DEBUG):
            plt.figure(i)
            plt.subplot(1, 2, 1)
            plt.imshow(FullIm)

            plt.subplot(1, 2, 2)
            plt.imshow(FullMask, alpha=0.5, cmap='gray')
            plt.axis("off")

    FullMaskMean = MaskArray.mean(axis=0).round()
    if (DEBUG):
        plt.figure(2 * (len(tta_transforms) + 1) + 1)
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.resize(FullIm, (h, w)))
        plt.subplot(1, 2, 2)
        plt.imshow(FullMaskMean, alpha=0.5, cmap='gray')
        plt.axis("off")
        plt.show()

    rle = mask2rle(FullMaskMean)
    df_submission.loc[int(row['id']), :] = [rle]



df_submission.to_csv("submission.csv")










