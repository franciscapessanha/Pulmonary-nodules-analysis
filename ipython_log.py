# IPython log file

runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/get_started.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/get_started.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
"""
Show Images
================
Arguments:
    * nodules: list with the the nodules names and paths
    * masks: list the the masks names and paths
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
            print(n)
            nodule = np.load(nodules[n,1])
            mask = np.load(masks[n,1])
            #since we have volume we must show only a slice
            fig,ax = plt.subplots(1,2)
            ax[0].imshow(getMiddleSlice(nodule),**plot_args) #plots the image
            ax[1].imshow(getMiddleSlice(mask),**plot_args) #plots the mask
            plt.show()
    
    #if instead you want to overlay
    if overlay:
        for n in nb:
            nodule = np.load(nodules[n,1])
            mask = np.load(masks[n,1])
            over = createOverlay(getMiddleSlice(nodule),getMiddleSlice(mask))
            #since we have volume we must show only a slice
            fig,ax = plt.subplots(1,1)
            ax.imshow(over,**plot_args)
            plt.show()
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/get_started.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/get_started.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/get_started.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/get_started.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/get_started.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/get_started.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/get_started.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/get_started.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/get_started.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/get_started.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/get_started.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
sigma = [0.5, 1, 1.5, 2, 2.5, 3, 3.5]
smooth_nodules = []
for i in range(x_train.shape[0]):
    nodule = np.load(x_train[i,1])
    smooth_nod = []
    for s in range(len(sigma)):
        smooth_nod.append(gaussian_filter(nodule, sigma[s])
    smooth_nodules.append(smooth_nod)
sigma = [0.5, 1, 1.5, 2, 2.5, 3, 3.5]
smooth_nodules = []
for i in range(x_train.shape[0]):
    nodule = np.load(x_train[i,1])
    smooth_nod = []
    for s in range(len(sigma)):
        smooth_nod.append(gaussian_filter(nodule, sigma[s]))
    
    smooth_nodules.append(smooth_nod)
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
from scipy.ndimage import gaussian_filter

#1. Multiscale Gaussian smoothing using sigm in the range 0.5 to 3.5
sigma = [0.5, 1, 1.5, 2, 2.5, 3, 3.5]
smooth_nodules = []
for i in range(x_train.shape[0]):
    nodule = np.load(x_train[i,1])
    smooth_nod = []
    for s in range(len(sigma)):
        smooth_nod.append(gaussian_filter(nodule, sigma[s]))
    
    smooth_nodules.append(smooth_nod)
runfile('/Users/mariafranciscapessanha/Desktop/DACO/Project/main.py', wdir='/Users/mariafranciscapessanha/Desktop/DACO/Project')
