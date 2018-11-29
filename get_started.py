import numpy as np
import os
from matplotlib import pyplot as plt

import pandas as pd
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.segmentation import find_boundaries

#%%
#________________________________
# TOOLS
#________________________________

"""
Create Overlay
==============

Verifies if the mask and image have the same size. If so it will create a green
contour (0,1,0) around the nodule

Arguments:
    * im: image
    * mask: mask
    * color: color of the contour (green by default)
    * contour: bool, it true it will draw the contour on the image (true by default)
    
Return:
    * im: image
"""

def createOverlay(im,mask,color=(0,1,0),contour=True):
    if len(im.shape)==2:
        im = np.expand_dims(im,axis=-1)
        im = np.repeat(im,3,axis=-1)
    elif len(im.shape)==3:
        if im.shape[-1] != 3:
            ValueError('Unexpected image format. I was expecting either (X,X) or (X,X,3), instead found', im.shape)

    else:
        ValueError('Unexpected image format. I was expecting either (X,X) or (X,X,3), instead found', im.shape)
   
    if contour:
        bw = find_boundaries(mask,mode='thick') #inner
    else:
        bw = mask
    for i in range(0,3):
        im_temp = im[:,:,i]
        im_temp = np.multiply(im_temp,np.logical_not(bw)*1)
        im_temp += bw*color[i]
        im[:,:,i] = im_temp
    return im

"""
Find Extensions
===============
Finds the files on the directory with the extension provided.

Arguments:
    * directory: path in which we want to find a file
    * extension: type of file. (.npy by default)
    
Returns:
    * files: list of all the files found
    * full path: full path to each of the files
"""
    
def findExtension(directory,extension='.npy'):
    files = []
    full_path = []
    for file in os.listdir(directory):
        if file.endswith(extension):
            files += [file]
            full_path += [os.path.join(directory,file)]
            
    files.sort()
    full_path.sort()
    return files, full_path

"""
Get Middle Slice
================
Returns the middle slice of an volume (in this case a cube)

Arguments:
    * volume
    
Returns:
    * image: middle slice
"""

def getMiddleSlice(volume):
    sh = volume.shape
    
    return volume[...,np.int(sh[-1]/2)]

#%%
#________________________________
# LOAD DATA
#________________________________

"""
Load Data
================
Returns the middle slice of an volume (in this case a cube)

Returns:
    * nodules: numpy array with the the nodules names and paths
    * masks: numpy array the the masks names and paths
    * metadata: ground truth
    
"""

def loadData():

    #find the current working directory
    curr_path = os.getcwd()
    
    #find the files
    nodule_names, nodules = findExtension(os.path.join(curr_path,'images'))
    #remove the extension from the nodule names
    nodule_names = [os.path.splitext(x)[0] for x in nodule_names]
    nodules = np.asarray([nodule_names, nodules])
    
    mask_names, masks = findExtension(os.path.join(curr_path,'masks'))
    #remove the extension from the mask names
    mask_names = [os.path.splitext(x)[0] for x in mask_names]
    masks = np.asarray([mask_names, masks])
    
    #read the metadata
    metadata = pd.read_excel('ground_truth.xls')
    
    return nodules, masks, metadata

#%%
#________________________________
# SHOW IMAGES
#________________________________
    
"""
Show Images
================
Arguments:
    * nodules: numpy array with the the nodules names and paths
    * masks: numpy array  the the masks names and paths
    * nodules_indexes: indexes of the nodules we want to show
    * nodules_and_mask: show nodule and mask side by side (true by default)
    * overlay: show nodule and mask overlay (true by default)
     
"""

def showImages(nodules, masks, nodules_indexes, nodules_and_mask = True, overlay = True):
    plot_args={}
    plot_args['vmin']=0
    plot_args['vmax']=1
    plot_args['cmap']='gray'
    
    if nodules_and_mask:
        for n in nodules_indexes:
            nodule = np.load(nodules[1,n])
            mask = np.load(masks[1,n])
            #since we have volume we must show only a slice
            fig,ax = plt.subplots(1,2)
            ax[0].imshow(getMiddleSlice(nodule),**plot_args) #plots the image
            ax[1].imshow(getMiddleSlice(mask),**plot_args) #plots the mask
            plt.show()
        
    #if instead you want to overlay
    if overlay:
        for n in nodules_indexes:
            nodule = np.load(nodules[1,n])
            mask = np.load(masks[1,n])
            over = createOverlay(getMiddleSlice(nodule),getMiddleSlice(mask))
            #since we have volume we must show only a slice
            fig,ax = plt.subplots(1,1)
            ax.imshow(over,**plot_args)
            plt.show()

"""
Apply mask to the image
=======================     
""" 
def applyMask():  
    for n in nb:
        nodule = np.load(nodules[n])
        mask = np.load(masks[n])
        masked = nodule*mask
        #since we have volume we must show only a slice
        fig,ax = plt.subplots(1,1)
        ax.imshow(getMiddleSlice(masked),**plot_args)
        plt.show()

#%%
#________________________________
# ALGORITHMS TO HELP YOU GER STARTED
#________________________________
        
"""
Mean Intensity
=======================
Arguments:
    * nodule: numpy array with the nodule name and path
    * mask: numpy array with the mask name and path

Returns:
    * intensity: mean intensity of the nodule
"""
        
def meanIntensity(nodule,mask):
    nodule = np.load(nodule[1])
    mask = np.load(mask[1])
    
    return np.mean(nodule[mask!=0])
    #print('The intensity of nodule',str(index),'is',intensity)


"""
PARA EXTRAÇÃO DA TEXTURA - AINDA NÃO COMENTADOS
===============================================
"""   
def sample_nodule():    
    #sample points from a nodule mask
    np.random.seed(0)
    for n in nb:
        nodule = np.load(nodules[n])
        mask = np.load(masks[n])
        
        sampled = np.zeros(mask.shape)
        
        loc = np.nonzero(mask)
        
        indexes = [x for x in range(loc[0].shape[0])]
        np.random.shuffle(indexes)
        
        #get 10% of the points
        indexes = indexes[:int(len(indexes)*0.1)]
        
        sampled[loc[0][indexes],loc[1][indexes],loc[2][indexes]]=True
        fig,ax = plt.subplots(1,2)
        ax[0].imshow(getMiddleSlice(nodule),**plot_args)
        ax[1].imshow(getMiddleSlice(sampled),**plot_args)
        plt.show()    


def feature_vector_segmentation():
    #create a simple 2 feature vector for 2D segmentation
    np.random.seed(0)
    features = []
    labels = []
    for n in nb:
        nodule = np.load(nodules[n])
        mask = np.load(masks[n])
        
        nodule = getMiddleSlice(nodule)
        mask = getMiddleSlice(mask)
    
        
        #collect itensity and local entropy
        
        entrop = np.ravel(entropy(nodule,disk(5)))
        inten = np.ravel(nodule)
        
        
        labels.append([1 for x in range(int(np.sum(mask)))])
        
        
        features.append([entrop,inten])
    
        entrop = np.ravel(entropy(nodule==0,disk(5)))
        inten = np.ravel(nodule==0)
        features.append([entrop,inten])
        labels.append([0 for x in range(int(np.sum(mask==0)))])
    
        
    features = np.hstack(features).T
    labels = np.hstack(labels)
        
def feature_vector_texture():       
    #create a simple 2 feature vector for 2D texture analysis
    np.random.seed(0)
    features = []
    labels = []
    for n in nb:
        nodule = np.load(nodules[n])
        mask = np.load(masks[n])
        
        nodule = getMiddleSlice(nodule)
        mask = getMiddleSlice(mask)
        
        texture = int(metadata[metadata['Filename']==nodule_names[n]]['texture'])
    
        
        #collect itensity and local entropy
        
        entrop = np.mean(entropy(nodule,disk(5)))
        inten = np.mean(nodule)
        
        
        labels.append(texture)
        
        
        features.append([entrop,inten])
    
    features_tex = np.vstack(features)
    labels_tex = np.hstack(labels) 
        




