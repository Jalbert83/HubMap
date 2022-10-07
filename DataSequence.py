import segmentation_models as sm
import numpy as np
import tensorflow as tf
import cv2
import albumentations as A

sm.set_framework('tf.keras')
sm.framework()

TEST_IMAGES_PATH = r"./test_images/"
TRAIN_IMAGES_PATH = r"./train_images/"


# SliceSequence resizes each image to img_Resize and takes slices as stored by the script Dfslices in the Dataframe
class SliceSequence(tf.keras.utils.Sequence):
    def __init__(self, batch_size, img_size, df, train = True, ttatransforms = None):
        super().__init__()
        self.batch_size = batch_size
        self.img_size = img_size
        self.df = df
        self.indexes = np.arange(0, int(np.floor(len(df)/batch_size))*batch_size)
        self.train = train
        self.Basepath = TRAIN_IMAGES_PATH if train == True else TEST_IMAGES_PATH
        self.transform = A.OneOf([
            A.HorizontalFlip(p=1),
            A.VerticalFlip(p=1),
            A.Transpose(p=1),
            A.Compose([
                A.Transpose(p=1),
                A.VerticalFlip(p=1),
            ])
        ], p=0.8)
        self.ttatransforms = ttatransforms
    


    def __len__(self):
        return int(np.floor(len(self.df)/self.batch_size))

    def __getitem__(self, idx):
        ind = self.indexes[idx:idx+self.batch_size]

        x = np.zeros((self.batch_size, self.img_size, self.img_size, 3), dtype='uint8')
        y = np.zeros((self.batch_size, self.img_size, self.img_size, 1), dtype='uint8')

        IdPrev = -1
        for j, (indexm, row) in enumerate(self.df.iloc[ind].iterrows()):
            w = int(row['img_width'])
            h = int(row['img_height'])
            if(IdPrev != int(row['id'])):
                img_path = self.Basepath / (str(int(row["id"])) + ".tiff")
                if((h < self.img_size) | (w < self.img_size)):
                    img = tf.keras.utils.load_img(img_path, target_size=(self.img_size, self.img_size))
                else:
                    img = tf.keras.utils.load_img(img_path)
                img_array = tf.keras.utils.img_to_array(img, dtype='uint8')    
            Xslice = int(row['Xslice'])
            Yslice = int(row['Yslice'])
            ImSlice = img_array[Xslice:Xslice+self.img_size, Yslice:Yslice+self.img_size]
             

            # calculate mask
            if ((IdPrev != int(row['id'])) & (self.train == True)):
                rle = row['rle']
                s = rle.split()
                starts, lengths = [np.asarray(t, dtype='int') for t in (s[0:][::2], s[1:][::2])]
                starts = starts - 1
                original_mask = np.zeros(h * w, dtype=np.uint8)
                for s, l in zip(starts, lengths):
                    original_mask[s:s + l] = 1
                original_mask = original_mask.reshape((h, w)).T
                original_mask = tf.keras.utils.array_to_img(original_mask[:, :, tf.newaxis], scale=False)
                mask_array = tf.keras.utils.img_to_array(original_mask, dtype='uint8')
                
                
            if(self.train == True):
                MaskSlice = mask_array[Xslice:Xslice+self.img_size, Yslice:Yslice+self.img_size]
                #transformed = self.transform(image = ImSlice, mask = MaskSlice)
                x[j] = ImSlice#transformed['image']
                y[j] = MaskSlice#transformed['mask']
            else:
                x[j] = ImSlice
                
            IdPrev = int(row['id'])
        if(self.train == True):
            return x.astype('float32') / 255, y
        else:
            return x.astype('float32') / 255
        
        
        
class SliceTTASequence(tf.keras.utils.Sequence):
    def __init__(self, img_size, row, ttatransform=None):
        super().__init__()
        w = int(row['img_width'])
        h = int(row['img_height'])
        self.batch_size = int(np.ceil(h/img_size)*np.ceil(w/img_size))
        self.img_size = img_size
        self.row = row
        self.Basepath = TEST_IMAGES_PATH
        self.ttatransform = ttatransform

    def __len__(self):
        return 1

    def __getitem__(self, idx):

        x = np.zeros((self.batch_size, self.img_size, self.img_size, 3), dtype='uint8')

        img_path = self.Basepath / (str(int(self.row["id"])) + ".tiff")
        img = tf.keras.utils.load_img(img_path)
        img = tf.keras.utils.img_to_array(img, dtype='uint8')
        i = 0
        if (self.ttatransform != None):
            transform = self.ttatransform(image=img)
            img_array = transform['image']
        else:
            img_array = img
        w = img_array.shape[0]
        h = img_array.shape[1]
        if ((h < self.img_size) | (w < self.img_size)):
            img_array = cv2.resize(img_array, (self.img_size, self.img_size))
            h = self.img_size
            w = self.img_size

        Xslices = np.arange(0, w, self.img_size)
        Xslices[-1] = max(w - self.img_size, 0)
        Yslices = np.arange(0, h, self.img_size)
        Yslices[-1] = max(h - self.img_size, 0)
        for Y in Yslices:
            for X in Xslices:

                ImSlice = img_array[X:X + self.img_size, Y:Y + self.img_size]
                x[i] = ImSlice

                i += 1

        return x.astype('float32') / 255       
    
    
    
## ImSequence with a resize of each image to img_size
class ImSequence(tf.keras.utils.Sequence):

    def __init__(self, batch_size, img_size, df, transform=None, set_type="train"):
        super().__init__()
        self.batch_size = batch_size
        self.img_size = img_size
        self.df = df
        self.set_type = set_type
        self.img_path = TRAIN_IMAGES_PATH if self.set_type == "train" else TEST_IMAGES_PATH
        self.transform = transform

        
    def __len__(self):
        return int(np.floor(len(self.df) / self.batch_size))

    
    def __getitem__(self, idx):
        ind = np.arange(idx * self.batch_size, (idx + 1) * self.batch_size) % len(self.df)

        x = np.zeros((self.batch_size, self.img_size, self.img_size, 3), dtype="uint8")
        y = np.zeros((self.batch_size, self.img_size, self.img_size, 1), dtype="uint8")
        for j, (indexm, row) in enumerate(self.df.iloc[ind].iterrows()):
            img_path = self.img_path / (str(row["id"]) + ".tiff")
            #img = tf.keras.utils.load_img(img_path, target_size=(self.img_size, self.img_size))

            # calculate mask
            if self.set_type == "train":
                
                img = tf.keras.utils.load_img(img_path, target_size=(3000, 3000))
                img_array = tf.keras.utils.img_to_array(img, dtype="uint8")
                
                w = row["img_width"]
                h = row["img_height"]
                rle = row["rle"]
                original_mask = rle2mask(rle)
                # we need at least three channels to save an image so here expand mask to (3000, 3000, 1)
                original_mask = tf.keras.utils.array_to_img(original_mask[:, :, tf.newaxis], scale=False)

                # resize mask
                #mask = original_mask.resize((self.img_size, self.img_size))
                #mask_array = tf.keras.utils.img_to_array(mask, dtype="uint8").reshape((self.img_size,self.img_size, 1))
                mask = original_mask.resize((3000, 3000))
                mask_array = tf.keras.utils.img_to_array(mask, dtype="uint8").reshape((3000, 3000, 1))
                
                # data augmentation                                    
                transformed = self.transform(image=img_array, mask=mask_array)
                transformed_image = transformed["image"]
                transformed_mask = transformed["mask"]
                
                x[j] = transformed_image
                y[j] = transformed_mask
                
            else:
                img = tf.keras.utils.load_img(img_path, target_size=(self.img_size, self.img_size))
                img_array = tf.keras.utils.img_to_array(img, dtype="uint8")
                
                if self.transform:
                    transformed = self.transform(image=img_array)
                    transformed_image = transformed["image"]
                    x[j] = transformed_image
                    
                else:
                    x[j] = img_array 

        if self.set_type == "train":
            return x.astype("float32") / 255, y
            
        else:
            return x.astype("float32") / 255