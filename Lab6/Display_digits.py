#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Display module

@author: SlimBenAmor
"""
import numpy as np
import matplotlib.pyplot as plt
import sys

def Display_digits(X_display,Y_display):
    """
    display digit image contained in X_display and print their labels contained in Y_display
    """
    
    # calculate number and size of patches fo each digit image
    nbr_select=X_display.shape[0] # number of digits to display
    origin_img_size=int(np.sqrt(X_display.shape[1]))
    nbr_patch_horiz=int(np.floor(np.sqrt(nbr_select)))
    nbr_patch_vertic=int(np.ceil(nbr_select/nbr_patch_horiz))
    img_width=20
    img_height=20
    
    # create Display_matrix that contains all image patches
    Display_matrix=-np.ones((nbr_patch_vertic*(img_height+1)-1,nbr_patch_horiz*(img_width+1)-1))
    for i in range(nbr_patch_vertic):
        for j in range(nbr_patch_horiz):
            if (i*nbr_patch_horiz+j>=nbr_select):
                break
            Display_matrix[i*(img_height+1):(i+1)*img_height+i,j*(img_width+1):(j+1)*img_width+j]=np.reshape(X_display[i*nbr_patch_horiz+j,np.arange(origin_img_size**2)%origin_img_size<img_height][:img_width*img_height],(img_height,img_width),order='F')
    plt.figure('dataset',figsize=(max(3,nbr_patch_horiz),max(3,nbr_patch_vertic)))
    plt.imshow(Display_matrix,cmap="gray",vmax=1,vmin=-1)
    plt.axis('off')
    plt.show()
    
    # print digit labels
    print(np.reshape(np.concatenate((Y_display[:,0].astype(int),-np.ones((nbr_patch_horiz*nbr_patch_vertic-nbr_select),dtype=int))),(nbr_patch_vertic,nbr_patch_horiz)))



# test Display_digits function if module called as main
if __name__ == "__main__":
    if (len(sys.argv)>1):
        nbr=int(sys.argv[1])
    else:
        nbr=4
    Display_digits(np.random.rand(nbr,400),np.random.randint(0,10,(nbr,1)))