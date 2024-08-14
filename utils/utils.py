# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 20:33:23 2019

@author: Ronglai Zuo
"""
# utils
import numpy as np
# import visdom
from scipy.special import logit
from scipy.special import expit
from collections import deque
from queue import Queue

def soften_labels(lom, softness=0.05):
    return np.clip(lom, softness, 1-softness).astype(np.float32)

def crop(data, offset, crop_shape):
    # tensor [batch,channel,x,y,z] 49,49,49  crop_shape 33,33,33
    shape = np.array(data.shape[2:])
    crop_shape = np.array(crop_shape)
    offset = np.array(offset)
    start = shape // 2 - crop_shape // 2 + offset
    end = start + crop_shape
    selector = [slice(s, e) for s, e in zip(start, end)]
    selector = tuple([slice(None)] + [slice(None)] + selector)
    
    cropped = data[selector]
    return cropped

def initial_seed(shape, pad=0.05, seed=0.95):
    seed_array = np.full(([1,1] + list(shape)), pad, dtype=np.float32)
    idx = tuple([slice(None), slice(None)] + list(np.array(shape) // 2))
    seed_array[idx] = seed # center point = 0.95
    return seed_array

def update_seed(to_update, offset, new_value):
    shape = np.array(to_update.shape[2:])
    crop_shape = np.array(new_value.shape[2:])
    offset = np.array(offset)

    start = shape // 2 - crop_shape // 2 + offset
    end = start + crop_shape

    selector = [slice(s, e) for s, e in zip(start, end)]
    selector = tuple([slice(None), slice(None)] + selector)
    
    to_update[selector] = new_value

class Eval_tracker(object):
    def __init__(self, shape, port='8097', env='main'):
        self.reset()
        self.eval_threshold = logit(0.9)
        self._eval_shape = shape
        # self.vis = visdom.Visdom(port=port, env=env)
        # self.win_loss = self.vis.line(X = np.array([0]), 
        #                               Y = np.array([0]), 
        #                               opts = dict(title = 'loss'))
        # self.win_precision = self.vis.line(X = np.array([0]), 
        #                                    Y = np.array([0]), 
        #                                    opts = dict(title = 'precision'))
        # self.win_recall = self.vis.line(X = np.array([0]), 
        #                                 Y = np.array([0]), 
        #                                 opts = dict(title = 'recall'))
        # self.win_accuracy = self.vis.line(X = np.array([0]), 
        #                                   Y = np.array([0]), 
        #                                   opts = dict(title = 'accuracy'))
        # self.win_f1 = self.vis.line(X = np.array([0]), 
        #                             Y = np.array([0]), 
        #                             opts = dict(title = 'f1'))
        # self.win_images_xy = self.vis.images(np.random.randn(10,1,49,98), 
        #                                      opts = dict(title = 'images_xy'))
        
    def reset(self):
        # self.num_patches = 0
        self.tp = 0
        self.tn = 0
        self.fn = 0
        self.fp = 0
        self.total_voxels = 0
        self.masked_voxels = 0
        self.images_xy = Queue(maxsize=10)
        self.images_yz = []
        self.images_xz = []
        
    def eval_one_patch(self, labels, predicted):
        labels = crop(labels, (0,0,0), self._eval_shape)
        predicted = crop(predicted, (0,0,0), self._eval_shape)
        
        pred_mask = predicted >= self.eval_threshold
        true_mask = labels > 0.5
        pred_bg = np.logical_not(pred_mask)
        true_bg = np.logical_not(true_mask)
        
        self.tp += np.sum(pred_mask & true_mask)
        self.fp += np.sum(pred_mask & true_bg)
        self.fn += np.sum(pred_bg & true_mask)
        self.tn += np.sum(pred_bg & true_bg)
        # self.num_patches += 1
        
        selector = [slice(None), slice(None), labels.shape[2]//2, slice(None), slice(None)]
        la = (labels[tuple(selector)] * 255).astype(np.int8)
        pred = expit(predicted)
        pred = (pred[tuple(selector)] * 255).astype(np.int8)
        if not self.images_xy.full():
            self.images_xy.put(np.concatenate((la, pred), axis=3))
        
    def plot(self, step, loss):
        precision = self.tp / max(self.tp + self.fp, 1)
        recall = self.tp / max(self.tp + self.fn, 1)
        accuracy = (self.tp + self.tn) / max(self.tp + self.tn + self.fn + self.fp, 1)
        f1 = (2.0 * precision * recall)/max(precision + recall, 1)
        # self.vis.line(X = np.array([step]), 
        #               Y = np.array([precision]), 
        #               win=self.win_precision, 
        #               update='append')       
        
        # self.vis.line(X = np.array([step]), 
        #               Y = np.array([recall]), 
        #               win=self.win_recall, 
        #               update='append')
        
        # self.vis.line(X = np.array([step]), 
        #               Y = np.array([accuracy]), 
        #               win=self.win_accuracy, 
        #               update='append')
        
        # self.vis.line(X = np.array([step]), 
        #               Y = np.array([f1]), 
        #               win=self.win_f1, 
        #               update='append')
        
        # self.vis.line(X = np.array([step]), 
        #               Y = np.array([loss/100]), 
        #               win=self.win_loss, 
        #               update='append')
        
        pred_and_labels = self.images_xy.get()
        while not self.images_xy.empty():
            pred_and_labels = np.concatenate((pred_and_labels, self.images_xy.get()))
            
        # self.vis.images(pred_and_labels, nrow=5, 
        #                 padding=2, 
        #                 win=self.win_images_xy)
        
    def plot_loss(self, step, loss):
        # self.vis.line(X = np.array([step]), 
        #               Y = np.array([loss/100]), 
        #               win=self.win_loss, 
        #               update='append')
        pass

