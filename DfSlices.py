import pandas as pd
import numpy as np
import cv2
from ImageSlices import ImageSlices
from ImageSlices import ImageMaskSlices

ImageResize = 0

ImageSize = 512
ds = pd.read_csv('train.csv')
PATH = r"./train_images/"



def DfSlices(ds, test = False):
    df = pd.DataFrame(columns = ['Xslice', 'Yslice', 'img_width', 'img_width', 'img_height', 'id', 'rle'])
    s = 0
    for i, row in ds.iterrows():
        path = PATH + str(int(row['id'])) + ".tiff"
        im = cv2.imread(path)
        if(ImageResize > 0):
            im = cv2.resize(im, (ImageResize, ImageResize))
        slices = ImageMaskSlices(im, ImageSize, row, ImageResize)#, Test=test)
        for slice in slices:
            df.loc[s, 'Xslice'] = slice[0]
            df.loc[s, 'Yslice'] = slice[1]
            df.loc[s, 'img_width'] = ds.loc[i, 'img_width']
            df.loc[s, 'img_height'] = ds.loc[i, 'img_height']
            df.loc[s, 'id'] = ds.loc[i, 'id']
            if(test == False):
                df.loc[s, 'rle'] = ds.loc[i, 'rle']
            s += 1
    return df


df = DfSlices(ds)
df.to_csv('TrainSliced.csv')
print(len(df))

