import colour
import itertools
import json
import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import operator
import os
import pipeline
import preprocessing
import random
import settings
import skimage.color
import skimage.exposure
import skimage.filters
import skimage.future.graph
import skimage.measure
import skimage.segmentation
import skimage.util


# Parameters
bbox_size = 300
overlap_ratio = 0.65

# this library function did not work together with the below functions so we change one line with a monkey-patch..
def _merge_nodes(self, src, dst, weight_func=skimage.future.graph.rag.min_weight, in_place=True,
                extra_arguments=[], extra_keywords={}):
    src_nbrs = set(self.neighbors(src))
    dst_nbrs = set(self.neighbors(dst))
    neighbors = (src_nbrs | dst_nbrs) - set([src, dst])

    if in_place:
        new = dst
    else:
        new = self.next_id()
        self.add_node(new)

    for neighbor in neighbors:
        w = weight_func(self, src, new, neighbor, *extra_arguments,
                        **extra_keywords)
        self.add_edge(neighbor, new, attr_dict=w)

    self.node[new]['labels'] = (self.node[src]['labels'] +
                                self.node[dst]['labels'])
    self.remove_node(src)

    if not in_place:
        self.remove_node(dst)

    return new
skimage.future.graph.RAG.merge_nodes = _merge_nodes

# two functions below adapted from scikit-image example code
def _merge_boundary(graph, src, dst):
    pass

def _weight_boundary(graph, src, dst, n):
    default = {'weight': 0.0, 'count': 0}
    count_src = graph[src].get(n, default)['count']
    count_dst = graph[dst].get(n, default)['count']
    count = count_src + count_dst
    weight_src = graph[src].get(n, default)['weight']
    weight_dst = graph[dst].get(n, default)['weight']
    return { 'count': count, 'weight': (count_src * weight_src + count_dst * weight_dst)/count }

def colour_segmentation(img, num_segments=1000, round_schedule = [0.02, 0.04, 0.06, 0.08], colour_median_prop=0, max_clust_size=0.05, min_clust_size=0.002):
    """
    Segment an image into clusters based on the colours in the image.
    Description of the algorithm can be found in comments in the code.
    Default parameters were tuned for optimal recall, so many false positives will be found.
    :param img: Image to segment
    :param num_segments: Number of segments to segment the image into, prior to merging
    :param round_schedule: Distance threshold for merging two clusters, per round
    :param colour_median_prop: Replace all pixels in the clusters whose areas exceed colour_median_prop*image_size by their median colour
    :param max_clust_size: Discard all clusters with areas larger than max_clust_size*image_size
    :param min_clust_size: Discard all clusters with areas smaller than min_clust_size*image_size
    :return: Tuple of an image coloured by cluster (cluster 0 means discarded), and a list of the cluster centroids
    """
    origimg = img
    
    # Initial segmentation
    regions = skimage.segmentation.slic(img, n_segments=num_segments)

    for round_thr in round_schedule:
        # Compute colour change of each pixel
        edges = skimage.util.dtype.img_as_float(colour.colourchange(img))
        
        # Merge clusters hierarchically based on above distance
        rag = skimage.future.graph.rag_boundary(regions, edges)
        regions = skimage.future.graph.merge_hierarchical(regions, rag, thresh=round_thr, rag_copy=False, in_place_merge=True, merge_func=_merge_boundary, weight_func=_weight_boundary)

        # Replace all pixels in (some?) clusters with their median colour
        clust_sizes = skimage.exposure.histogram(regions)[0]
        clust_sizes = clust_sizes / float(sum(clust_sizes))
        medianclusters = np.where(clust_sizes > colour_median_prop)[0]

        img = origimg.copy()
        for mediancluster in medianclusters:
            img[regions == mediancluster] = np.median(img[regions == mediancluster], axis=0)
        
        if len(clust_sizes) == 1:
            break
    
    # Filter out too small and too large clusters
    num_clusters = 0
    for clust in range(len(clust_sizes)):
        if clust_sizes[clust] > max_clust_size or clust_sizes[clust] < min_clust_size: # background or noise resp.
            regions[regions == clust] = 0
        else:
            num_clusters += 1
            regions[regions == clust] = num_clusters
    
    # Extract centroids
    centroids = [clust.centroid for clust in skimage.measure.regionprops(regions)]
    
    return (regions, centroids)


def unique(iterable, key=None):
    #Adapted from http://stackoverflow.com/questions/7973933/removing-duplicate-elements-from-a-python-list-containing-unhashable-elements-wh
    return list(map(next, map(operator.itemgetter(1), itertools.groupby(sorted(iterable, key=key), key))))


def do_segmentation(img_idxs=None, output=True, save_candidates=True, data='train'):
    """
    Master script for segmentation. Can be used both for testing method performance and generating candidates.
    :param img_idxs: If supplied, the function is applied only to these 
    :param output: Output results to console if true
    :param save_candidates: If True, save generated candidates bounding boxes as JSON
    :param data: Which data to use, as either 'train', 'test' or 'final'
    """
    
    # Load images
    dl = pipeline.DataLoader()
    
    if data == 'train':
        data_imgs = dl.get_train_images_and_classes()
    elif data == 'test':
        data_imgs = dl.get_test_images()
    elif data == 'final':
        print('Final stage not started yet')
        exit()
    else:
        print('Unknown data set: ' + data)
        exit()
    
    data_x = data_imgs['x']
    data_meta = data_imgs['meta']
    
    if img_idxs is None:
        random.seed(42)
        img_idxs = list(range(len(data_x)))
        random.shuffle(img_idxs)

    if len(img_idxs) == 0:
        print('Empty index range given.')
        exit()
    if img_idxs[-1] >= len(data_x):
        print('Invalid index range ending in %d for used data set of size %d' % (img_idxs[-1], len(data_x)))
        exit()
    
    # Prepare output file
    if save_candidates:
        if data == 'train':
            classlist = dl.get_classes()
            out_train_json_objs = {}
            for cls in classlist:
                out_train_json_objs[cls] = []
        else:
            out_json_obj = []
    
    # Prepare performance measurements
    tp_boxes = 0
    num_boxes = 0
    tp_fish = 0
    num_fish = 0
    num_impossible = 0
    
    # See how well the centroids match
    lower = lambda centroid, dim: min(max(centroid[dim] - bbox_size/2.0, 0), img.shape[dim] - bbox_size)
    upper = lambda centroid, dim: max(bbox_size, min(centroid[dim] + bbox_size/2.0, img.shape[dim]))
    intersection = lambda bbox, centroid: max(0, min(upper(centroid, 1), bbox['x']+bbox['width']) - max(lower(centroid, 1), bbox['x'])) * max(0, min(upper(centroid, 0), bbox['y']+bbox['height']) - max(lower(centroid, 0), bbox['y']))
    matches_centroid = lambda bbox, centroid: intersection(bbox, centroid) / float(bbox['width']*bbox['height']) >= overlap_ratio
    
    # Prepare histogram matching template
    if data == 'train':
        template = preprocessing.build_template(data_x, data_meta)
    else:
        hist_template_data_imgs = dl.get_train_images_and_classes(file_filter=preprocessing.DEFAULT_HIST_MATCH_TEMPLATES)
        template = preprocessing.build_template(hist_template_data_imgs['x'], hist_template_data_imgs['meta'])
    
    for idx_idx in range(len(img_idxs)):
        idx = img_idxs[idx_idx]
        
        # Load image
        img = data_x[idx]()
        if 'bounding_boxes' in data_meta[idx]:
            imgboxes = data_meta[idx]['bounding_boxes']
        else:
            imgboxes = []
        
        # Use histogram matching for night vision images
        nvg = False
        if colour.is_night_vision(img): # night vision
            nvg = True
            img = preprocessing.hist_match(img, template)
        
        # Perform actual segmentation
        regions, centroids = colour_segmentation(img)
        
        num_matching_boxes = sum(any(matches_centroid(bbox, centroid) for bbox in imgboxes) for centroid in centroids)
        num_found_fish = sum(any(matches_centroid(bbox, centroid) for centroid in centroids) for bbox in imgboxes)
        num_impossible_here = sum(overlap_ratio * max(bbox['width'], bbox['height']) >= bbox_size for bbox in imgboxes)
        
        # Record this information
        tp_boxes += num_matching_boxes
        num_boxes += len(centroids)
        tp_fish += num_found_fish
        num_fish += len(imgboxes) - num_impossible_here
        num_impossible += num_impossible_here
        
        if output:
            # Output performance for this image
            if data == 'train':
                print('Image %d (found %d/%d%s, %d FPs%s)' % (idx, num_found_fish, len(imgboxes)-num_impossible_here, (', %d impossible' % num_impossible_here) if num_impossible_here > 0 else '', len(centroids)-num_matching_boxes, '; NVG' if nvg else ''))
            else:
                print('Image %d (%d candidates)' % (idx, len(centroids)))
            
            # Summarise performance up till now
            if idx_idx%50 == 49:
                if data == 'train':
                    box_precision = 100*tp_boxes / float(num_boxes) if num_boxes > 0 else -1
                    fish_recall = 100*tp_fish / float(num_fish) if num_fish > 0 else -1
                    print('Box precision after %d images: %g%% (%d/%d)\nFish recall after %d images: %g%% (%d/%d%s)\n' % (idx_idx+1, box_precision, tp_boxes, num_boxes, idx_idx+1, fish_recall, tp_fish, num_fish, (', %d impossible' % num_impossible) if num_impossible > 0 else ''))
                else:
                    print('%d images segmented (%d candidates in total)' % (idx, num_boxes))
        
        if save_candidates:
            img_json_obj = {'filename': data_meta[idx]['filename']}
            img_json_obj['candidates'] = unique([{'x': lower(centroid, 1), 'y': lower(centroid, 0), 'width': bbox_size, 'height': bbox_size} for centroid in centroids], key=lambda cand: (cand['x'], cand['y']))
            if data == 'train':
                out_train_json_objs[data_meta[idx]['class']].append(img_json_obj)
            else:
                out_json_obj.append(img_json_obj)
            
    
    if output:
        # Summarise total performance
        if data == 'train':
            box_precision = 100*tp_boxes / float(num_boxes) if num_boxes > 0 else -1
            fish_recall = 100*tp_fish / float(num_fish) if num_fish > 0 else -1
            print('\n%d images completed!\nTotal box precision: %g%% (%d/%d)\nTotal fish recall: %g%% (%d/%d%s)\n' % (len(img_idxs), box_precision, tp_boxes, num_boxes, fish_recall, tp_fish, num_fish, (', %d impossible' % num_impossible) if num_impossible > 0 else ''))
        else:
            print('%d images segmented (%d candidates in total)' % (idx, num_boxes))

    if save_candidates:
        outdir = settings.SEGMENTATION_CANDIDATES_OUTPUT_DIR
        os.makedirs(outdir)
        filename = 'candidates%s.json' % ('' if img_idxs is None else ('_%d-%d' % (min(img_idxs), max(img_idxs))))
        if data == 'train':
            for cls in classlist:
                with open(os.path.join(outdir, cls + '_' + filename), 'w') as outfile:
                    json.dump(out_train_json_objs[cls], outfile)
        else:
            with open(os.path.join(outdir, filename), 'w') as outfile:
                json.dump(out_json_obj, outfile)

