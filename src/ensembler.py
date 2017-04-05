import json
import math
import numpy as np
import os
import settings
import shutil
from sklearn import svm
from time import strftime
import pickle

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
    print('Loading predictions...')
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
    if classif_type == 'fish_or_not':
        negatives = set(os.listdir(os.path.join(settings.TRAIN_CANDIDATES_FULLYCONV_CROPPED_IMAGES_DIR, 'NoF')))
        y = [img + '.jpg' not in negatives for img in images]
        num_preds = 1
    else:
        print('Fish type labels not implemented for ensembles yet')
        exit()
    
    # Evaluate models
    print('Evaluating individual models...')
    for i in range(len(models)):
        if classif_type == 'fish_or_not':
            ypreds = [result[i] for result in results]
            
            prec = binary_precision(y, ypreds)
            rec = binary_recall(y, ypreds)
            loss = binary_loss(y, ypreds)
            modelstr = 'Model ' + get_model_name(json_files[i])
            print('%s: precision %g, recall %g, avg. loss %g' % (modelstr, prec, rec, loss))
        else:
            ypreds = result[i*num_preds:(i+1)*num_preds]
            
            print('Fish type labels not implemented for ensemble evaluation yet')
            exit()
    
    # Fit ensemble learner
    print('Fitting ensemble learner...')
    m = svm.SVC(kernel='linear', probability=True, class_weight = 'balanced')
    m = m.fit(results,y)
    
    outpath = os.path.join(path_to_json, "ensemble.pickle")
    outpath2 = os.path.join(path_to_json, "ensemble-%s.pickle" % strftime("%Y%m%dT%H%M%S"))
    with open(outpath, 'wb') as outfile:
        pickle.dump(m, outfile)
    
    shutil.copyfile(outpath, outpath2)
    
    # Evaluate ensemble
    print('Evaluating ensemble...')
    ensemble_preds = m.predict_proba(results)
    outdict = {}
    
    if classif_type == 'fish_or_not':
        ypreds = [result[1] for result in ensemble_preds]
        
        prec = binary_precision(y, ypreds)
        rec = binary_recall(y, ypreds)
        loss = binary_loss(y, ypreds)
        modelstr = 'Ensemble model'
        print('%s: precision %g, recall %g, avg. loss %g' % (modelstr, prec, rec, loss))
        
        # Save predictions
        for img, ypred in zip(images, ypreds):
            outdict[img] = ypred            
    else:
        print('Fish type labels not implemented for ensemble evaluation yet')
        exit()
    
    outpath = os.path.join(path_to_json, "classification.json")
    outpath2 = os.path.join(path_to_json, "classification-%s.json" % strftime("%Y%m%dT%H%M%S"))
    with open(outpath, 'w') as outfile:
        json.dump(outdict, outfile)
    
    shutil.copyfile(outpath, outpath2)
    
    
def ensemble_predict():
    # Load models' predictions
    if classif_type == 'fish_or_not':
        path_to_train = settings.TRAIN_FISH_OR_NO_FISH_CLASSIFICATION_DIR
        path_to_json = settings.TEST_FISH_OR_NO_FISH_CLASSIFICATION_DIR
    elif classif_type == 'fish_type':
        path_to_train = settings.TRAIN_FISH_TYPE_CLASSIFICATION_DIR
        path_to_json = settings.TEST_FISH_TYPE_CLASSIFICATION_DIR
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
    
    # Load ensemble learner
    print('Loading ensemble learner...')
    m = pickle.load(os.path.join(path_to_train, "ensemble.pickle"))
    
    # Evaluate ensemble
    print('Evaluating ensemble...')
    ensemble_preds = m.predict_proba(results)
    outdict = {}
    
    if classif_type == 'fish_or_not':
        ypreds = [result[1] for result in ensemble_preds]
        for img, ypred in zip(images, ypreds):
            outdict[img] = ypred            
    else:
        print('Fish type labels not implemented for ensemble evaluation yet')
        exit()
    
    outpath = os.path.join(path_to_json, "classification.json")
    outpath2 = os.path.join(path_to_json, "classification-%s.json" % strftime("%Y%m%dT%H%M%S"))
    with open(outpath, 'w') as outfile:
        json.dump(outdict, outfile)
    
    shutil.copyfile(outpath, outpath2)
    
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
        return name