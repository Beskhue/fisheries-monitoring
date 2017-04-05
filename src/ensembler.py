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
    json_files = sorted([pos_json for pos_json in os.listdir(path_to_json) if pos_json.startswith('classification') and pos_json.endswith('.json') and not pos_json == 'classification.json'])
    if model_filters is not None:
        json_files = sorted([json_file for json_file in json_files if any(filter in json_file for filter in model_filters)])
    
    models = []
    modelindices = []
    modelindex = 0
    for js in json_files:
        with open(os.path.join(path_to_json, js)) as json_file:
            model = json.load(json_file)
            models.append(model)
            modelindices.append(modelindex)
            
            for key in model:
                if isinstance(model[key], list):
                    modelindex += len(model[key])
                else:
                    modelindex += 1
                break
    modelindices.append(modelindex)
    
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
        for i in range(len(models)):
            model = models[i]
            if img not in model:
                print('Image %s not found in some models; all models need to classify all images/crops' % img)
                exit()
            if classif_type == 'fish_or_not' or modelindices[i+1]-modelindices[i] == 1:
                resultforthiskey.append(model[img])
            else:
                resultforthiskey.extend(model[img])
        results.append(resultforthiskey)
    
    # Find true labels
    if classif_type == 'fish_or_not':
        negatives = set(os.listdir(os.path.join(settings.TRAIN_CANDIDATES_FULLYCONV_CROPPED_IMAGES_DIR, 'NoF')))
        ys = [img + '.jpg' not in negatives for img in images]
        num_preds = 1
    else:
        classes = sorted([dir for dir in os.listdir(settings.TRAIN_CANDIDATES_FULLYCONV_CROPPED_IMAGES_DIR) if os.path.isdir(os.path.join(settings.TRAIN_CANDIDATES_FULLYCONV_CROPPED_IMAGES_DIR, dir))])
        imagesets = [set(os.listdir(os.path.join(settings.TRAIN_CANDIDATES_FULLYCONV_CROPPED_IMAGES_DIR, clss))) for clss in classes]
        find_class_of = lambda img: [i for i in range(len(classes)) if img+'.jpg' in imagesets[i]][0]
        ys = [find_class_of(img) for img in images]
        num_preds = 8
    
    # Evaluate models
    print('Evaluating individual models...')
    for i in range(len(models)):
        modelstr = 'Model ' + get_model_name(json_files[i])
        modellen = modelindices[i+1] - modelindices[i]
        if modellen == num_preds:
            if classif_type == 'fish_or_not':
                ypreds = [result[i] for result in results]
                
                prec = binary_precision(ys, ypreds)
                rec = binary_recall(ys, ypreds)
                loss = binary_loss(ys, ypreds)
                print('%s: avg. loss %g, precision %g%%, recall %g%%' % (modelstr, loss, 100*prec, 100*rec))
            else:
                ypreds = [result[modelindices[i]:modelindices[i+1]] for result in results]
                
                loss = categorical_loss(ys, ypreds)
                acc = categorical_accuracy(ys, ypreds)
                print('%s: avg. loss %g, accuracy %g%%' % (modelstr, loss, 100*acc))
        else:
            print('%s is not an %d-class model (%d outputs)' % (modelstr, num_preds, modellen))
    
    # Fit ensemble learner
    print('Fitting ensemble learner...')
    m = svm.SVC(kernel='rbf', probability=True, class_weight = 'balanced')
    m = m.fit(results,ys)
    
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
        
        prec = binary_precision(ys, ypreds)
        rec = binary_recall(ys, ypreds)
        loss = binary_loss(ys, ypreds)
        print('Ensemble model: avg. loss %g, precision %g%%, recall %g%%' % (loss, 100*prec, 100*rec))
        
        # Save predictions
        for img, ypred in zip(images, ypreds):
            outdict[img] = ypred            
    else:
        loss = categorical_loss(ys, ypreds)
        acc = categorical_accuracy(ys, ypreds)
        print('Ensemble model: avg. loss %g, accuracy %g%%' % (loss, 100*acc))
    
    outpath = os.path.join(path_to_json, "classification.json")
    outpath2 = os.path.join(path_to_json, "classification-%s.json" % strftime("%Y%m%dT%H%M%S"))
    with open(outpath, 'w') as outfile:
        json.dump(outdict, outfile)
    
    shutil.copyfile(outpath, outpath2)
    
    
def ensemble_predict(classif_type, model_filters=None):
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
    json_files = sorted([pos_json for pos_json in os.listdir(path_to_json) if pos_json.startswith('classification') and pos_json.endswith('.json') and not pos_json == 'classification.json'])
    if model_filters is not None:
        json_files = sorted([json_file for json_file in json_files if any(filter in json_file for filter in model_filters)])
    
    models = []
    modelindices = []
    modelindex = 0
    for js in json_files:
        with open(os.path.join(path_to_json, js)) as json_file:
            model = json.load(json_file)
            models.append(model)
            modelindices.append(modelindex)
            
            for key in model:
                modelindex += len(model[key])
                break
    modelindices.append(modelindex)
    
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
        for i in range(len(models)):
            model = models[i]
            if img not in model:
                print('Image %s not found in some models; all models need to classify all images/crops' % img)
                exit()
            if classif_type == 'fish_or_not' or modelindices[i+1]-modelindices[i] == 1:
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

def categorical_loss(ys, ypreds):
    return np.mean([-math.log(clamp(ypred[y])) for y,ypred in zip(ys, ypreds)])

def categorical_accuracy(ys, ypreds):
    return np.mean([all(pr<y for pr in ypred) for y,ypred in zip(ys,ypreds)])

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