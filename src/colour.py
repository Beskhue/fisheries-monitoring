import numpy as np
import preprocessing
import skimage.transform

def acc(x,dim):
    if isinstance(x, np.ndarray):
        return x[:,:,dim]
    else:
        return x[dim]

def intensity(vc):
    return np.sqrt(np.power(acc(vc,0),2) + np.power(acc(vc,1),2) + np.power(acc(vc,2),2))
    
def colourdist(frm, to):
    """
    Computes the distance from colour frm to another colour to.
    Exact formula is sqrt(norm(from)^2 - dot(from,to)/norm(to)^2)
    which is equivalent to norm(from)*sqrt(1 - cos angle(from,to))
    """
    If = acc(frm, 0)**2 + acc(frm, 1)**2 + acc(frm, 2)**2 
    It = acc(to, 0)**2 + acc(to, 1)**2 + acc(to, 2)**2 
    dot = acc(frm, 0)*acc(to, 0) + acc(frm, 1)*acc(to, 1) + acc(frm, 2)*acc(to, 2)
    return np.sqrt(preprocessing.clamp(If - dot**2 / (It+1e-10)))

def brightnessmatch(I, clmeta, alpha=0.5, beta=1.25):
    return I >= alpha*clmeta[2] and I <= min(beta * clmeta[2], clmeta[1]/alpha)

def colourchange(img):
    """
    Compute the change in colour around each pixel, by
    computing the colour distance between the pixels to the 
    left and right, as well as the up and down.
    """
    imgl = skimage.transform.warp(img, skimage.transform.AffineTransform(translation=(1, 0)).inverse, mode='edge')
    imgr = skimage.transform.warp(img, skimage.transform.AffineTransform(translation=(-1,0)).inverse, mode='edge')
    imgu = skimage.transform.warp(img, skimage.transform.AffineTransform(translation=(0, 1)).inverse, mode='edge')
    imgd = skimage.transform.warp(img, skimage.transform.AffineTransform(translation=(0,-1)).inverse, mode='edge')
    difflr = colourdist(imgl, imgr)
    diffud = colourdist(imgu, imgd)
    diffimg = np.sqrt(difflr**2 + diffud**2)
    return (255*preprocessing.clamp(diffimg / np.percentile(diffimg, 99))).astype(np.uint8)

NIGHT_VISION_THRESHOLD = 1.3 # threshold on mean(G)/mean(R+G+B)
def is_night_vision(img):
    """
    Returns true when an image is taken with a night vision camera,
    false otherwise; night vision images can be distinguished
    by the fact that they are much greener than other (boat) images.
    :param img: The image to test.
    """
    return np.mean(img[:,:,1]) / np.mean(img) > NIGHT_VISION_THRESHOLD
