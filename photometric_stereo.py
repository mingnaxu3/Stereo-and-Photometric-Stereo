import numpy as np
##======================== No additional imports allowed ====================================##






def photometric_stereo_singlechannel(I, L):
    #L is 3 x k
    #I is k x n
    G = np.linalg.inv(L @ L.T) @ L @ I
    # G is  3 x n 
    albedo = np.sqrt(np.sum(G*G, axis=0))

    normals = G/(albedo.reshape((1,-1)) + (albedo==0).astype(float).reshape((1,-1)))
    return albedo, normals


def photometric_stereo(images, lights):
    '''
        Use photometric stereo to compute albedos and normals
        Input:
            images: A list of N images, each a numpy float array of size H x W x 3
            lights: 3 x N array of lighting directions. 
        Output:
            albedo, normals
            albedo: H x W x 3 array of albedo for each pixel
            normals: H x W x 3 array of normal vectors for each pixel

        Assume light intensity is 1.
        Compute the albedo and normals for red, green and blue channels separately.
        The normals should be approximately the same for all channels, so average the three sets
        and renormalize so that they are unit norm

    '''
    height, width,channel = images[0].shape 
    albedo = np.zeros([images[0].shape[0], images[0].shape[1], 3])
    normals = np.zeros([images[0].shape[0], images[0].shape[1], 3])
    red_vals = np.zeros((len(images), (height*width)))
    green_vals = np.zeros((len(images), (height*width)))
    blue_vals = np.zeros((len(images), (height*width)))
    

    #red channel 
    #get the red values 
    for i in range(len(images)): 
        red_pixel = images[i][:,:,0]
        red_vals[i] = np.ndarray.flatten(red_pixel)
    red_albedo, red_norm = photometric_stereo_singlechannel(red_vals, lights)
    red_albedo = red_albedo.reshape((height, width))

    #green channel
    for i in range(len(images)): 
        green_pixel = images[i][:,:,1]
        green_vals[i] = np.ndarray.flatten(green_pixel)
    green_albedo, green_norm = photometric_stereo_singlechannel(green_vals, lights)
    green_albedo = green_albedo.reshape((height, width))
    
    #blue channel 
    for i in range(len(images)): 
        blue_pixel = images[i][:,:,2]
        blue_vals[i] = np.ndarray.flatten(blue_pixel)
    blue_albedo, blue_norm = photometric_stereo_singlechannel(blue_vals, lights)
    blue_albedo = blue_albedo.reshape((height, width))

    norm_avg = (red_norm + green_norm + blue_norm )/3.0
    
    
    renormed = np.linalg.norm(norm_avg, axis=0, keepdims=True )
    norm_flat = norm_avg/(renormed + 1e-6)

    normals = norm_flat.T.reshape((height,width,3))

    albedo = np.stack([red_albedo, green_albedo, blue_albedo], axis=2)


    return albedo, normals



