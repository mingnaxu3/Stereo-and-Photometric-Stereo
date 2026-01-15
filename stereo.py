import numpy as np
#==============No additional imports allowed ================================#

def get_ncc_descriptors(img, patchsize):
    '''
    Prepare normalized patch vectors for normalized cross
    correlation.

    Input:
        img -- height x width x channels image of type float32
        patchsize -- integer width and height of NCC patch region.
    Output:
        normalized -- height* width *(channels * patchsize**2) array

    For every pixel (i,j) in the image, your code should:
    (1) take a patchsize x patchsize window around the pixel,
    (2) compute and subtract the mean for every channel
    (3) flatten it into a single vector
    (4) normalize the vector by dividing by its L2 norm
    (5) store it in the (i,j)th location in the output

    If the window extends past the image boundary, zero out the descriptor
    
    If the norm of the vector is <1e-6 before normalizing, zero out the vector.

    '''
    height, width, channel = img.shape
    # padded_img = np.pad(img, ((patchsize//2, patchsize//2), (patchsize//2, patchsize//2), (0, 0)), mode='constant', constant_values=0)
    
    normalized = np.zeros((height, width, (img.shape[2] * patchsize*patchsize)))
    pad = patchsize//2

    for y in range(pad,height):
        for x in range(pad,width):
            patch = img[y-pad : y + pad+1, x-pad : x+pad+1, : ]
            channel_arr = []
            for c in range(channel):
                patch_channel = patch[:, :, c]
               
                patch_mean = np.mean(patch_channel)
                new_patch = patch_channel - patch_mean
                
                flat_patch = np.ndarray.flatten(new_patch)
                print(flat_patch.shape)
                print(type(flat_patch))
                channel_arr.append(np.ndarray.flatten(new_patch))
                
            flattened = np.ndarray.flatten(channel_arr)
            norm = np.linalg.norm(flattened)
            if norm >= 1e-6:
                normalized[y,x]= flattened / norm
            else:
                normalized[y,x] = 0.0
    return normalized
    
   



def compute_ncc_vol(img_right, img_left, patchsize, dmax):
    '''
    Compute the NCC-based cost volume
    Input:
        img_right: the right image, H x W x C
        img_left: the left image, H x W x C
        patchsize: the patchsize for NCC, integer
        dmax: maximum disparity
    Output:
        ncc_vol: A dmax x H x W tensor of scores.

    ncc_vol(d,i,j) should give a score for the (i,j)th pixel for disparity d. 
    This score should be obtained by computing the similarity (dot product)
    between the patch centered at (i,j) in the right image and the patch centered
    at (i, j+d) in the left image.

    Your code should call get_ncc_descriptors to compute the descriptors once.
    '''
    height, width = img_left.shape[:2]
    right_ncc_d = get_ncc_descriptors(img_right, patchsize)
    
    left_ncc_d = get_ncc_descriptors(img_left, patchsize)
    ncc_vol = np.zeros((dmax, height, width))
    for d in range(dmax):
        for x in range(height):
            for y in range(width-d):
                ncc_vol[d][x][y] = np.dot(right_ncc_d[x][y], left_ncc_d[x][y+d]) 
    return ncc_vol


    


def get_disparity(ncc_vol):
    '''
    Get disparity from the NCC-based cost volume
    Input: 
        ncc_vol: A dmax X H X W tensor of scores
    Output:
        disparity: A H x W array that gives the disparity for each pixel. 

    the chosen disparity for each pixel should be the one with the largest score for that pixel
    '''
    height, width = ncc_vol.shape[1:]
    disparity = np.zeros((height, width))

    for x in range(ncc_vol.shape[1]):
        for y in range(ncc_vol.shape[2]):
            for d in range (ncc_vol.shape[0]):
                if ncc_vol[d][x][y] > disparity[x][y]:
                    disparity[x][y] = ncc_vol[d][x][y]
    return disparity





    
