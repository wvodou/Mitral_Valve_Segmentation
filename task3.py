#!/usr/bin/env python
# coding: utf-8

# # Task 3: Helper notebook for loading the data and saving the predictions

# In[1]:


import pickle
import gzip
import numpy as np
import os


# ### Helper functions

# In[2]:


def load_zipped_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        loaded_object = pickle.load(f)
        return loaded_object


# In[3]:


def save_zipped_pickle(obj, filename):
    with gzip.open(filename, 'wb') as f:
        pickle.dump(obj, f, 2)

# In[5]:


from tqdm import tqdm
import matplotlib.pyplot as plt

def preprocess_train_data(data):
    video_frames = []
    mask_frames = []
    names = []
    for item in tqdm(data):
        video = item['video']
        name = item['name']
        height, width, n_frames = video.shape
        mask = np.zeros((height, width, n_frames), dtype=bool)
        for frame in item['frames']:
            mask[:, :, frame] = item['label'][:, :, frame]
            video_frame = video[:, :, frame]
            mask_frame = mask[:, :, frame]
            video_frame = np.expand_dims(video_frame, axis=2).astype(np.float32)
            mask_frame = np.expand_dims(mask_frame, axis=2).astype(np.int32)
            video_frames.append(video_frame)
            mask_frames.append(mask_frame)
            names.append(name)
    return names, video_frames, mask_frames

def preprocess_test_data(data):
    video_frames = []
    names = []
    for item in tqdm(data):
        video = item['video']
        video = video.astype(np.float32).transpose((2, 0, 1))
        video = np.expand_dims(video, axis=3)
        video_frames += list(video)
        names += [item['name'] for _ in video]
    return names, video_frames

def get_sequences2(arr):
    first_indices, last_indices, lengths = [], [], []
    for i in range(len(arr)):

        if arr[i] == 1 and (i == 0 or arr[i - 1] == 0):
            first_indices.append(i)
        if arr[i] == 1 and (i == len(arr) - 1 or arr[i + 1] == 0):
            last_indices.append(i)

    if len(first_indices) != len(last_indices):
        print(f"Mismatch detected: first_indices={first_indices}, last_indices={last_indices}")

    lengths = list(np.array(last_indices)-np.array(first_indices))
    return first_indices, lengths


