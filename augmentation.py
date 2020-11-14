import os
from data_aug.data_aug import *
from data_aug.bbox_util import *
from PIL import Image
import cv2
import numpy as np
import argparse
import json
from config import *

train_folder = TRAIN_FOLDER
train_json = TRAIN_JSON_FILE
augment_folder = AUGMENT_FOLDER
augment_json = AUGMENT_JSON_FILE

with open(augment_json, 'r') as f:
    json_dict = json.load(f)

if not os.path.exists(augment_folder):
    os.mkdir(augment_folder)

with open(train_json, "r") as file:
    annotations = json.load(file)

bboxes = annotations['annotations']
images = annotations['images']

image_id_list = [image['id'] for image in images]
new_image_id = max(image_id_list)

old_bboxes = bboxes.copy()
old_images = images.copy()

# Scaling
for image in old_images:
    new_image_id += 1
    image_id = image['id']
    width, height, street_id = image['width'], image['height'], image['street_id']
    file_name = image['file_name']
    img_path = os.path.join(train_folder, file_name)


    img_dict = {'file_name': '{}.png'.format(new_image_id), 'height': height, 'width': width, 'id': new_image_id, 'street_id': street_id}
    images.append(img_dict)

    img = cv2.imread(img_path)[:, :, ::-1] # BGR -> RGB
    for bbox in old_bboxes:
        if bbox['image_id'] == image_id:
            segmentation, area, iscrowd, category_id, id = bbox['segmentation'], bbox['area'], bbox['iscrowd'], bbox['category_id'], bbox['id']
            anno_dict = {'segmentation': segmentation, 'area': area, 'iscrowd': iscrowd, 'category_id': category_id, 'id': id, 'image_id': new_image_id}
            bb = bbox['bbox']
            bb = [bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]]
            bb = np.array(bb)
            bb.reshape(1, -1)
            bb = np.expand_dims(bb, axis=0)
            bb = bb.astype('float64')

            img_, bboxes_ = RandomScale(0.1)(img.copy(), bb.copy())
            bboxes_ = bboxes_.reshape(-1, 4)
            if bboxes_.shape[0] != 1:
                continue
            bboxes_ = list(bboxes_.squeeze(axis=0))
            bb1 = [bboxes_[0], bboxes_[1], bboxes_[2]-bboxes_[0], bboxes_[3]-bboxes_[1]]
            anno_dict['bbox'] = bb1
            bboxes.append(anno_dict)
      
    # save augment image to augment folder
    Image.fromarray(img_).save(os.path.join(augment_folder, '{}.png'.format(new_image_id)))
    
# Traslation
for image in old_images:
    new_image_id += 1
    image_id = image['id']
    width, height, street_id = image['width'], image['height'], image['street_id']
    file_name = image['file_name']
    img_path = os.path.join(train_folder, file_name)


    img_dict = {'file_name': '{}.png'.format(new_image_id), 'height': height, 'width': width, 'id': new_image_id, 'street_id': street_id}
    images.append(img_dict)

    img = cv2.imread(img_path)[:, :, ::-1] # BGR -> RGB
    for bbox in old_bboxes:
        if bbox['image_id'] == image_id:
            segmentation, area, iscrowd, category_id, id = bbox['segmentation'], bbox['area'], bbox['iscrowd'], bbox['category_id'], bbox['id']
            anno_dict = {'segmentation': segmentation, 'area': area, 'iscrowd': iscrowd, 'category_id': category_id, 'id': id, 'image_id': new_image_id}
            bb = bbox['bbox']
            bb = [bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]]
            bb = np.array(bb)
            bb.reshape(1, -1)
            bb = np.expand_dims(bb, axis=0)
            bb = bb.astype('float64')

            img_, bboxes_ = RandomTranslate(0.3, diff = True)(img.copy(), bb.copy())
            bboxes_ = bboxes_.reshape(-1, 4)
            if bboxes_.shape[0] != 1:
                continue
            bboxes_ = list(bboxes_.squeeze(axis=0))
            bb1 = [bboxes_[0], bboxes_[1], bboxes_[2]-bboxes_[0], bboxes_[3]-bboxes_[1]]
            anno_dict['bbox'] = bb1
            bboxes.append(anno_dict)
      
    # save augment image to augment folder
    Image.fromarray(img_).save(os.path.join(augment_folder, '{}.png'.format(new_image_id)))

# Shearing
for image in old_images:
    new_image_id += 1
    image_id = image['id']
    width, height, street_id = image['width'], image['height'], image['street_id']
    file_name = image['file_name']
    img_path = os.path.join(train_folder, file_name)


    img_dict = {'file_name': '{}.png'.format(new_image_id), 'height': height, 'width': width, 'id': new_image_id, 'street_id': street_id}
    images.append(img_dict)

    img = cv2.imread(img_path)[:, :, ::-1] # BGR -> RGB
    for bbox in old_bboxes:
        if bbox['image_id'] == image_id:
            segmentation, area, iscrowd, category_id, id = bbox['segmentation'], bbox['area'], bbox['iscrowd'], bbox['category_id'], bbox['id']
            anno_dict = {'segmentation': segmentation, 'area': area, 'iscrowd': iscrowd, 'category_id': category_id, 'id': id, 'image_id': new_image_id}
            bb = bbox['bbox']
            bb = [bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]]
            bb = np.array(bb)
            bb.reshape(1, -1)
            bb = np.expand_dims(bb, axis=0)
            bb = bb.astype('float64')

            img_, bboxes_ = RandomShear(0.2)(img.copy(), bb.copy())
            bboxes_ = bboxes_.reshape(-1, 4)
            if bboxes_.shape[0] != 1:
                continue
            bboxes_ = list(bboxes_.squeeze(axis=0))
            bb1 = [bboxes_[0], bboxes_[1], bboxes_[2]-bboxes_[0], bboxes_[3]-bboxes_[1]]
            anno_dict['bbox'] = bb1
            bboxes.append(anno_dict)
      
    # save augment image to augment folder
    Image.fromarray(img_).save(os.path.join(augment_folder, '{}.png'.format(new_image_id)))

# Rotating
for image in old_images:
    new_image_id += 1
    image_id = image['id']
    width, height, street_id = image['width'], image['height'], image['street_id']
    file_name = image['file_name']
    img_path = os.path.join(train_folder, file_name)


    img_dict = {'file_name': '{}.png'.format(new_image_id), 'height': height, 'width': width, 'id': new_image_id, 'street_id': street_id}
    images.append(img_dict)

    img = cv2.imread(img_path)[:, :, ::-1] # BGR -> RGB
    for bbox in old_bboxes:
        if bbox['image_id'] == image_id:
            segmentation, area, iscrowd, category_id, id = bbox['segmentation'], bbox['area'], bbox['iscrowd'], bbox['category_id'], bbox['id']
            anno_dict = {'segmentation': segmentation, 'area': area, 'iscrowd': iscrowd, 'category_id': category_id, 'id': id, 'image_id': new_image_id}
            bb = bbox['bbox']
            bb = [bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]]
            bb = np.array(bb)
            bb.reshape(1, -1)
            bb = np.expand_dims(bb, axis=0)
            bb = bb.astype('float64')

            img_, bboxes_ = RandomRotate(20)(img.copy(), bb.copy())
            bboxes_ = bboxes_.reshape(-1, 4)
            if bboxes_.shape[0] != 1:
                continue
            bboxes_ = list(bboxes_.squeeze(axis=0))
            bb1 = [bboxes_[0], bboxes_[1], bboxes_[2]-bboxes_[0], bboxes_[3]-bboxes_[1]]
            anno_dict['bbox'] = bb1
            bboxes.append(anno_dict)
      
    # save augment image to augment folder
    Image.fromarray(img_).save(os.path.join(augment_folder, '{}.png'.format(new_image_id)))

# Resizing
for image in old_images:
    new_image_id += 1
    image_id = image['id']
    width, height, street_id = image['width'], image['height'], image['street_id']
    file_name = image['file_name']
    img_path = os.path.join(train_folder, file_name)


    img_dict = {'file_name': '{}.png'.format(new_image_id), 'height': height, 'width': width, 'id': new_image_id, 'street_id': street_id}
    images.append(img_dict)

    img = cv2.imread(img_path)[:, :, ::-1] # BGR -> RGB
    for bbox in old_bboxes:
        if bbox['image_id'] == image_id:
            segmentation, area, iscrowd, category_id, id = bbox['segmentation'], bbox['area'], bbox['iscrowd'], bbox['category_id'], bbox['id']
            anno_dict = {'segmentation': segmentation, 'area': area, 'iscrowd': iscrowd, 'category_id': category_id, 'id': id, 'image_id': new_image_id}
            bb = bbox['bbox']
            bb = [bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]]
            bb = np.array(bb)
            bb.reshape(1, -1)
            bb = np.expand_dims(bb, axis=0)
            bb = bb.astype('float64')

            img_, bboxes_ = Resize(608)(img.copy(), bb.copy())
            bboxes_ = bboxes_.reshape(-1, 4)
            if bboxes_.shape[0] != 1:
                continue
            bboxes_ = list(bboxes_.squeeze(axis=0))
            bb1 = [bboxes_[0], bboxes_[1], bboxes_[2]-bboxes_[0], bboxes_[3]-bboxes_[1]]
            anno_dict['bbox'] = bb1
            bboxes.append(anno_dict)
      
    # save augment image to augment folder
    Image.fromarray(img_).save(os.path.join(augment_folder, '{}.png'.format(new_image_id)))

with open(augment_json, 'w') as file:
    json_dict['images'] = images
    json_dict['annotations'] = bboxes
    json.dump(json_dict, file)
