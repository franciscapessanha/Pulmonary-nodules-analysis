import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from get_data import getData



# Multiscale Gaussian smoothing using sigm in the range 0.5 to 3.5
# ====================================================================

def gaussianSmooth(nodules):
    sigma = [0.5, 1, 1.5, 2, 2.5, 3, 3.5]
    smooth_nodules = []
    for nodule in nodules:
        smooth_node = []
        for s in range(len(sigma)):
             smooth_node.append(gaussian_filter(nodule, sigma[s]))
  
        smooth_nodules.append(smooth_node)
        #plotImage(gaussian_filter(nodule, sigma[s]))
    
    return smooth_nodules

# 2. Compute the Hessian matrix and eig values

def getEig(image):
    [gx, gy, gz] = np.gradient(image)
    [gxx, gxy, gxz] = np.gradient(gx)
    [gxy, gyy, gyz] = np.gradient(gy)
    [gxz, gyz, gzz] = np.gradient(gy)
    
    eig_vals = np.asarray([[[np.array([0.0,0.0,0.0]) for col in range(51)] for col in range(51)] for row in range(51)])

    for i in range(len(image)):
        for j in range(len(image)):
            for k in range(len(image)):
                row_1 = [gxx[i,j,k], gxy[i,j,k], gxz[i,j,k]]
                row_2 = [gxy[i,j,k], gyy[i,j,k], gyz[i,j,k]]
                row_3 = [gxz[i,j,k], gyz[i,j,k], gzz[i,j,k]]
                eig_vals[i,j,k] = np.asarray(np.linalg.eigvals([row_1, row_2, row_3]))
            
    
    return eig_vals

def plotEig(nodule_eig):
    lower_eig = nodule_eig[0][1]
    higher_eig = nodule_eig[0][0]
    plot_args={}
    plot_args['vmin']= np.min(lower_eig)
    plot_args['vmax']= np.max(higher_eig)
    plot_args['cmap']='gray'
    plt.imshow(lower_eig, **plot_args)
    plt.show()
    plt.imshow(higher_eig, **plot_args)
    plt.show()
  

def getEigNodules(smooth_nodules):
    eig_nodules = []   
    for nodule in range(len(smooth_nodules)):
        all_nodule_sigmas = smooth_nodules[nodule]
        nodule_eig = []
        
        for s in range(len(all_nodule_sigmas)):
            eig_vals = getEig(all_nodule_sigmas[s])
            nodule_eig.append(eig_vals)
            #plotEig(eig_vals)
        eig_nodules.append(nodule_eig)
    
    return eig_nodules

# 3.1 Shape index 
# ===============
    
"""
The shape index, as defined by Koenderink & van Doorn [1], is a single valued
measure of local curvature, assuming the image as a 3D plane with intensities 
representing heights.
It is derived from the eigen values of the Hessian, and its value ranges from 
-1 to 1 (and is undefined (=NaN) in flat regions), with following ranges 
representing following shapes
"""
def plotImage(image):
    plot_args={}
    plot_args['vmin']= np.min(image)
    plot_args['vmax']= np.max(image)
    plot_args['cmap']='gray'
    plt.imshow(image, **plot_args)
    plt.show()

def getSI(eig_nodules):
    SI_nodules = [] 
    
    for all_sigmas_nodule in eig_nodules:
        print("entrou")
        all_SI = []
        for s in range(len(all_sigmas_nodule)):
            nodule = all_sigmas_nodule[s]
            lower_eig = nodule[:][:][:][2]
            print(np.shape(lower_eig))
            higher_eig = nodule[:][:][:][0]
            shape_indexes = (2/np.pi) * np.arctan((lower_eig + higher_eig)/(lower_eig - higher_eig))    
            all_SI.append(shape_indexes)
        
        SI_nodule = np.zeros((51,51,51))
        for i in range(51):
            for j in range(51):
                 for l in range(51):
                     values = []
                     for k in range(len(all_sigmas_nodule)):
                         si = all_SI[k]
                         print(np.shape(si))
                         px = si[i,j,l]
                         values.append(px)
                     SI_nodule[i][j][l] = np.max(values)
    
        SI_nodules.append(SI_nodule)
    
    return SI_nodules

# 3.2 Curvedness approach 
# ======================== 
#we will compute the curvedness manually

def getCV(eig_nodules):
    CV_nodules = []
    for all_sigmas_nodule in eig_nodules:
        all_CV = []
        for s in range(len(all_sigmas_nodule)):
            nodule = all_sigmas_nodule[s]
            lower_eig = nodule[:][:][:][2]
            higher_eig = nodule[:][:][:][0]
            curvedness = np.sqrt(lower_eig**2 + higher_eig**2)    
            all_CV.append(curvedness)
        
        CV_nodule = np.zeros((51,51,51))
        for i in range(51):
            for j in range(51):
                for l in range(51):
                    values = []
                    for k in range(len(all_sigmas_nodule)):
                        cv_ = all_CV[k]
                        px = cv_[i,j,l]
                        values.append(px)
                    CV_nodule[i][j][l] = np.max(values)
        CV_nodules.append(CV_nodule)
    return CV_nodules
    
# 3.3 Central adaptive miedialness approach 
# ==========================================

def getVmed(eig_nodules):     
    Vmed_nodules = []
    for all_sigmas_nodule in eig_nodules:
        all_Vmed = []
        for s in range(len(all_sigmas_nodule)):
            nodule = all_sigmas_nodule[s]
            lower_eig = nodule[:][:][:][2]
            int_eig = nodule[:][:][:][1]
            higher_eig = nodule[:][:][:][0]
            Vmed = np.zeros((51,51,51))
            for i in range(len(Vmed)):
                for j in range(len(Vmed)):
                    for l in range(len(Vmed)):
                        if lower_eig[i][j][l] + higher_eig[i][j][l] < 0:
                            Vmed[i][j][l] = -(int_eig[i][j][l]/lower_eig[i][j][l]) * (int_eig[i][j][l] + lower_eig[i][j][l])
            all_Vmed.append(Vmed)
        Vmed_nodule = np.zeros((51,51,51))
        for i in range(51):
            for j in range(51):
                 for j in range(51):
                     values = []
                     for k in range(len(all_sigmas_nodule)):
                         vmed = all_Vmed[k]
                         px = vmed[i,j,l]
                         values.append(px)
                     Vmed_nodule[i][j][l] = np.max(values)
        Vmed_nodules.append(Vmed_nodule)
    return Vmed_nodules

#train_volumes, _, val_volumes, val_masks, test_volumes, _ = getData(mode = "default", type = "volume")
#smooth_nodules = gaussianSmooth(train_volumes)

eigen_nodules = getEigNodules(smooth_nodules)
SI_train = getSI (eigen_nodules)

#%%%
CV_train = getCV(eigen_nodules)
Vmed_train = getVmed(eigen_nodules)

smooth_nodules = gaussianSmooth(val_volumes)
eigen_nodules = getEigNodules(smooth_nodules)
SI_val = getSI (eigen_nodules)
CV_val = getCV(eigen_nodules)
Vmed_val = getVmed(eigen_nodules)


smooth_nodules = gaussianSmooth(test_volumes)
eigen_nodules = getEigNodules(smooth_nodules)
SI_test = getSI (eigen_nodules)
CV_test = getCV(eigen_nodules)
Vmed_test = getVmed(eigen_nodules)