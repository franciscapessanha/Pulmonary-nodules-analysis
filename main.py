#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 12:51:18 2018
 @author: mariafranciscapessanha
"""
import numpy as np
from get_data import loadData
from get_started import showImages

nodules, masks, metadata = loadData()

nodules_indexes = [i for i in range(nodules.shape[1])]

non_solid = []
sub_solid = []
solid = []

for index in range(nodules.shape[1]):
    texture = int(metadata[metadata['Filename']==nodules[0, index]]['texture'])
    if texture <=2:
        non_solid.append(index)
    elif texture >= 3 and texture <= 4:
        sub_solid.append(index)
    elif texture == 5:
        solid.append(index)
non_solid = np.asarray(non_solid)
sub_solid = np.asarray(sub_solid)
solid = np.asarray(solid)

#show non_solid images
#print("NON SOLID \n==========")
#showImages(nodules, masks, non_solid, nodules_and_mask = False)


#show sub_solid images
#print("SUB SOLID \n==========")
#showImages(nodules, masks, sub_solid, nodules_and_mask = False)

#show sub_solid images
print("SOLID \n==========")
showImages(nodules, masks, solid, nodules_and_mask = False)
