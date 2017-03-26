import collections
import numpy as np
import skimage.color

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

