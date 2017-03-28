import collections
import numpy as np
import scipy.linalg
import scipy.misc
import skimage.color
import skimage.util

DEFAULT_HIST_MATCH_TEMPLATES = ['img_01678', 'img_06382', 'img_04391', 'img_04347', 'img_05883'] # picked from different boats and perspectives

# Based on http://stackoverflow.com/questions/32655686/histogram-matching-of-two-images-in-python-2-x
def build_template(train_imgs, train_meta, template_files=DEFAULT_HIST_MATCH_TEMPLATES):
    """
    Build histogram matching template, to be used as argument for hist_match.
    :param train_imgs: pipeline.DataLoader().get_train_images_and_classes()['x']
    :param template_idxs: images to be used as templates, as indices into train_imgs
    """
    template_idxs = [i for i in range(len(train_imgs)) if train_meta[i]['filename'] in template_files]
    
    templates = []
    for d in range(3):
        # build template histogram
        t_hist = collections.defaultdict(int)
        for idx in template_idxs:
            template = skimage.color.rgb2lab(train_imgs[idx]())
            template = template[:,:,d].ravel()
            t_values, t_counts = np.unique(template, return_counts=True)
            for i in range(len(t_values)):
                t_hist[t_values[i]] += t_counts[i]
        t_values, t_counts = zip(*sorted(t_hist.items()))
        
        # maps pixel value --> quantile
        t_quantiles = np.cumsum(t_counts).astype(np.float64)
        t_quantiles /= t_quantiles[-1]
        
        templates.append((t_quantiles, t_values))
    return templates

def hist_match(source, template):
    """
    Perform histogram matching, applying the template colouring (illumination, colour scheme, etc.)
    to the source image. This can be used to colour in night-vision images
    :param source: The image to be transformed / coloured.
    :param template: Result of build_template (see above), whose colours are to be applied to source.
    """
    source = skimage.color.rgb2lab(source)
    
    for d in range(3):
        # get source histogram
        sourced = source[:,:,d]
        oldshape = sourced.shape
        sourced = sourced.ravel()
        s_values, bin_idx, s_counts = np.unique(sourced, return_inverse=True, return_counts=True)

        # maps pixel value --> quantile
        s_quantiles = np.cumsum(s_counts).astype(np.float64)
        s_quantiles /= s_quantiles[-1]
        
        # unpack template
        t_quantiles, t_values = template[d]

        # interpolate linearly to find the pixel values in the template image
        # that correspond most closely to the quantiles in the source image
        interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
        source[:,:,d] = interp_t_values[bin_idx].reshape(oldshape)
    
    return (255*skimage.color.lab2rgb(source)).astype(np.uint8)

# From scikit image github
yuv_from_rgb = np.array([[ 0.299     ,  0.587     ,  0.114      ],
                         [-0.14714119, -0.28886916,  0.43601035 ],
                         [ 0.61497538, -0.51496512, -0.10001026 ]])
rgb_from_yuv = scipy.linalg.inv(yuv_from_rgb)

def clamp(arr):
    arr[arr < 0] = 0
    arr[arr > 1] = 1
    return arr

def rgb2yuv(arr):
    arr = skimage.util.dtype.img_as_float(arr)
    return clamp(np.dot(arr, yuv_from_rgb.T.copy()))

def yuv2rgb(arr):
    arr = skimage.util.dtype.img_as_float(arr)
    return clamp(np.dot(arr, rgb_from_yuv.T.copy()))

def crop(img, bounding_boxes, out_size = (300, 300), zoom_factor = 0.7):
    """
    Crop an image using the bounding box(es) to one or more cropped image(s).

    :param img: The image to generate the crops for
    :param bounding_boxes: A list of dictionaries of the form x, y, width, height, and optionally class.
    :param out_size: The size of the output crops
    :param zoom_factor: The factor with which to zoom in to the crop (relative to the fish size)
                        == 1: no zoom
                        >  1: zoom in
                        <  1: zoom out
    :return: A list of crops (and potentially a list of classes for each crop, if classes were given in the input)
    """

    (size_y, size_x, channels) = img.shape

    zoom_out_factor = 1 / zoom_factor

    crops = []

    for bounding_box in bounding_boxes:
        box_x = bounding_box['x']
        box_y = bounding_box['y']
        box_w = bounding_box['width']
        box_h = bounding_box['height']

        # Make square by creating seperate zoom-out factors for each axis
        if box_w > box_h:
            zoom_out_factor_x = zoom_out_factor
            zoom_out_factor_y = zoom_out_factor * (box_w / box_h)
        else: 
            zoom_out_factor_x = zoom_out_factor * (box_h / box_w)
            zoom_out_factor_y = zoom_out_factor

        # Zoom out (or in)
        box_x = box_x + (box_w - zoom_out_factor_x * box_w) / 2;
        box_y = box_y + (box_h - zoom_out_factor_y * box_h) / 2;
        box_w *= zoom_out_factor_x;
        box_h *= zoom_out_factor_y;

        # To integer values
        box_x = round(box_x)
        box_y = round(box_y)
        box_w = round(box_w)
        box_h = round(box_h)

        # Ensure the bounding box is not greater than the image itself
        if box_w > size_x:
            box_w = size_x

        if box_h > size_y:
            box_h = size_y

        # Translate (and potentially shrink) box to ensure it falls within the image region
        if box_x < 0:
            box_x = 0
        elif box_x + box_w > size_x:
            box_x = size_x - box_w

        if box_y < 0:
            box_y = 0
        elif box_y + box_h > size_y:
            box_y = size_y - box_h

        # Crop
        crop = img[box_y : box_y + box_h, box_x : box_x + box_w, :]

        # Resize to output size
        crop = scipy.misc.imresize(crop, size = out_size)

        # Add to crops list
        if "class" in bounding_box:
            crops.append((crop, bounding_box["class"]))
        else:
            crops.append(crop)

    return zip(*crops)
