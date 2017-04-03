import json
import math
import numpy as np
import os
import settings
from sklearn import svm


def train_ensemble(classif_type, model_filters=None):
    # Load models' predictions
    if classif_type == 'fish_or_not':
        path_to_json = settings.TRAIN_FISH_OR_NO_FISH_CLASSIFICATION_DIR
    elif classif_type == 'fish_type':
        path_to_json = settings.TRAIN_FISH_TYPE_CLASSIFICATION_DIR
    else:
        print('Unknown classification type: ' + classif_type)
        exit()
    
    print('Loading models...')
    json_files = sorted([pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json') and not pos_json == 'classification.json'])
    if model_filters is not None:
        json_files = sorted([json_file for json_file in json_files if any(filter in json_file for filter in model_filters)])

    models = []
    for js in json_files:
        with open(os.path.join(path_to_json, js)) as json_file:
            models.append(json.load(json_file))
    
    if len(models) == 0:
        print('No eligible models found!')
        exit()
    
    # Aggregate model predictions
    images = []
    results = []
    for img in models[0].keys():
        images.append(img)
        resultforthiskey = []
        for model in models:
            if img not in model:
                print('Image %s not found in some models; all models need to classify all images/crops' % img)
                exit()
            if classif_type == 'fish_or_not':
                resultforthiskey.append(model[img])
            else:
                resultforthiskey.extend(model[img])
        results.append(resultforthiskey)
    
    # Find true labels
    print('Loading predictions...')
    if classif_type == 'fish_or_not':
        positives = set(os.listdir(os.path.join(settings.TRAIN_CANDIDATES_CROPPED_IMAGES_DIR, 'positive')))
        y = [img + '.jpg' in positives for img in images]
        num_preds = 1
    else:
        print('Fish type labels not implemented for ensembles yet')
        exit()
    
    # Fit ensemble learner
    print('Fitting classifier...')
    m = svm.LinearSVC()
    m = m.fit(results,y)
    
    # Evaluate
    print('Evaluating...')
    ensemble_preds = m.predict(results)
    print(ensemble_preds)
    for i in range(len(models)+1):
        if classif_type == 'fish_or_not':
            if i < len(models):
                ypreds = [result[i] for result in results]
            else:
                ypreds = ensemble_preds
            
            prec = binary_precision(y, ypreds)
            rec = binary_recall(y, ypreds)
            loss = binary_loss(y, ypreds)
            modelstr = 'Model ' + get_model_name(json_files[i]) if i<len(models) else 'Ensemble model'
            print('%s: precision %g, recall %g, avg. loss %g' % (modelstr, prec, rec, loss))
        else:
            if i < len(models):
                ypreds = [result[i*num_preds:(i+1)*num_preds] for result in results]
            else:
                ypreds = ensemble_preds
            
            print('Fish type labels not implemented for ensemble evaluation yet')
            exit()
    
    # Save
    
    

def binary_precision(ys, ypreds, thr=0.5):
    if not any(ypred>thr for ypred in ypreds):
        return -1
    return float(sum(y and (ypred>thr) for y,ypred in zip(ys,ypreds))) / float(sum(ypred>thr for ypred in ypreds))

def binary_recall(ys, ypreds, thr=0.5):
    return float(sum(y and (ypred>thr) for y,ypred in zip(ys,ypreds))) / float(sum(ys))

def binary_loss(ys, ypreds):
    return np.mean([-math.log(clamp(ypred)) if y else -math.log(clamp(1-ypred)) for y,ypred in zip(ys,ypreds)])

def clamp(x, tol=1e-40):
    return (x if x>tol else tol) if x<1-tol else 1-tol

def get_model_name(full_path):
        base = os.path.basename(full_path)
        name = os.path.splitext(base)[0]
        if name.startswith('classification-'):
            name = name[15:]
        if '-' in name:
            name = name[name.find('-')+1:]
        return name