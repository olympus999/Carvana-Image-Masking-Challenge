import numpy as np
import pandas as pd
from threading import Thread
import time
from os.path import join
from scipy import misc
import gc
from keras.preprocessing import image

# Augmentations functions

def random_flip(img, mask, u=0.5):
    if np.random.random() < u:
        img = image.flip_axis(img, 1)
        mask = image.flip_axis(mask, 1)
    return img, mask

def random_flip(img, mask, u=0.5):
    if np.random.random() < u:
        img = image.flip_axis(img, 1)
        mask = image.flip_axis(mask, 1)
    return img, mask

def shift(x, wshift, hshift, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.):
    h, w = x.shape[row_axis], x.shape[col_axis]
    tx = hshift * h
    ty = wshift * w
    translation_matrix = np.array([[1, 0, tx],
                                   [0, 1, ty],
                                   [0, 0, 1]])
    transform_matrix = translation_matrix  # no need to do offset
    x = image.apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x

def random_shift(img, mask, w_limit=(-0.1, 0.1), h_limit=(-0.1, 0.1), u=0.5):
    if np.random.random() < u:
        wshift = np.random.uniform(w_limit[0], w_limit[1])
        hshift = np.random.uniform(h_limit[0], h_limit[1])
        img = shift(img, wshift, hshift)
        mask = shift(mask, wshift, hshift)
    return img, mask

def zoom(x, zx, zy, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.):
    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])
    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = image.transform_matrix_offset_center(zoom_matrix, h, w)
    x = image.apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x

def random_zoom(img, mask, zoom_range=(0.8, 1), u=0.5):
    if np.random.random() < u:
        zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
        img = zoom(img, zx, zy)
        mask = zoom(mask, zx, zy)
    return img, mask

def random_brightness(img, limit=(-0.3, 0.3), u=0.5):
    if np.random.random() < u:
        alpha = 1.0 + np.random.uniform(limit[0], limit[1])
        img = alpha * img
        img = np.clip(img, 0., 1.)
    return img

def random_channel_shift(x, limit, channel_axis=2):
    x = np.rollaxis(x, channel_axis, 0)
    min_x, max_x = np.min(x), np.max(x)
    channel_images = [np.clip(x_ch + np.random.uniform(-limit, limit), min_x, max_x) for x_ch in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x

def random_saturation(img, limit=(-0.3, 0.3), u=0.5):
    if np.random.random() < u:
        alpha = 1.0 + np.random.uniform(limit[0], limit[1])
        coef = np.array([[[0.114, 0.587, 0.299]]])
        gray = img * coef
        gray = np.sum(gray, axis=2, keepdims=True)
        img = alpha * img + (1. - alpha) * gray
        img = np.clip(img, 0., 1.)
    return img



# Color augmentation

def random_contrast(img, limit=(-0.3, 0.3), u=0.5):
    if np.random.random() < u:
        alpha = 1.0 + np.random.uniform(limit[0], limit[1])
        coef = np.array([[[0.114, 0.587, 0.299]]])  # rgb to gray (YCbCr)
        gray = img * coef
        gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
        img = alpha * img + gray
        img = np.clip(img, 0., 1.)
    return img




# Class for getting images and masks

class ImagesAndMasks:

    '''
    getData() returns images and masks
    '''
    
    
    def __init__(self, x_dir, y_dir, df_list, size=10, workers=1):
        self.x_dir = x_dir
        self.y_dir = y_dir
        self.df_list = df_list
        self.size = size
        self.workers = workers

    def getData(self):
        indx = self.getIndexes()
        masks = self.getMasks(indx)
        imgs = self.getImgs(indx)

        return imgs, masks
        
    def getIndexes(self):
        return np.random.randint(len(self.df_list), size=self.size)
        
    def getMasks(self, indx):
        temp_list = []
        for i in indx:
            temp_list.append(join(self.y_dir, self.df_list['img'][i] + '_mask.gif'))
            
        return self.getFiles(temp_list)
    
    def getImgs(self, indx):
        temp_list = []
        for i in indx:
            temp_list.append(join(self.x_dir, self.df_list['img'][i] + '.jpg'))
            
        return self.getFiles(temp_list)
    
    def getFiles(self, image_list_dir):
        threads = []
        results = [None] * len(image_list_dir)
        for i, img in enumerate(image_list_dir):
            thread = Thread(target=self.readImage, args=(i, img, results));
            thread.start()
            threads.append(thread)
        for t in threads:
            t.join()
            
        return np.array(results)

    def readImage(self, loc, file, results):
        if file.rsplit('.', 1)[1] == 'gif':
            results[loc] = misc.imread(file, mode='L')
            results[loc][results[loc] <= 127] = 0
            results[loc][results[loc] > 127] = 1
        else:
            results[loc] = misc.imread(file)
        
# Class to feed images and maks + augment
        
class returnTransferedImages():
    '''
    get() yields the data
    example:
    data = returnTransferedImages(self, x_dir, y_dir, df_list, datagen, batch_size=10, workers=5, max_queue_size=3)
    for x, y in data.get()
        // now u have x and y
    '''
    
    def __init__(self, x_dir, y_dir, df_list, datagen, batch_size=10, workers=5, max_queue_size=3):
        self.batch_size = batch_size
        self.max_queue_size = max_queue_size
        self.datagen = datagen
        self.queue = []
        self.obj = ImagesAndMasks(x_dir, y_dir, df_list, size=batch_size, workers=workers)
        self.kill = False
        
        thread = Thread(target=self.run)
        thread.start()
        
    def run(self):
        while(True or self.kill):
            if len(self.queue) < self.max_queue_size:
                self.addToQueue()
            else:
                time.sleep(1)
        
    def addToQueue(self):
        img, mask = self.obj.getData()
        img, mask = self.augment_thread(img, mask)
        img = np.array(img)
        mask = np.array(mask)
        for x, y in self.datagen.flow(img, mask, batch_size=self.batch_size):
            x = np.array(x)
            y = np.array(y)
            self.queue.append((x, y))
            break
            
    def get(self):
        while(True):
            gc.collect()
            while(len(self.queue) == 0):
                time.sleep(1)
            x, y = self.queue.pop()
            yield x, y

    def augment_thread(self, x, y):
        res_x = [None] * len(x)
        res_y = [None] * len(y)
        threads = []
        y = np.expand_dims(y, axis=4)
        for i, xi in enumerate(x):
            thread = Thread(target=self.augment, args=(x[i], y[i], res_x, res_y, i))
            thread.start()
            threads.append(thread)
            
        for t in threads:
            t.join()
        
        return res_x, res_y
            
    def augment(self, x, y, res_x, res_y, i):

        # x = random_contrast(x, u=0.5)
        # x = random_brightness(x, u=0.5)
        # x = random_saturation(x, u=0.5)

        x, y = random_zoom(x, y, zoom_range=(0.90, 1), u=0.5)
        x, y = random_shift(x, y, w_limit=(-0.1, 0.1), h_limit=(-0.1, 0.1), u=0.5)
        x, y = random_flip(x, y, u=0.5)
        
        res_x[i] = x
        res_y[i] = y
