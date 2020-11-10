import os 
import augment.data_aug.data_aug 
import augment.data_aug.bbox_util
import numpy as np 
import cv2 
import matplotlib.pyplot as plt
import json
from pylab import rcParams
rcParams['figure.figsize'] = 20,16
from PIL import Image
from config import *

result_folder = config.PUBLIC_TEST_DIR

def visualize(filename):
    l1 = [1, 2, 3, 4, 5, 6, 7]
    l2 = ['Cấm ngược chiều', 'Cấm dừng và đỗ', 'Cấm rẽ', 'Giới hạn tốc độ', 'Cấm còn lại', 'Nguy hiểm', 'Hiệu lệnh']
    nameDict = dict(zip(l1, l2))
    file_path = os.path.join('./datasets/results', filename)
    with open(file_path, 'r') as f:
        results = json.load(f)

    image_id_list = list(set([result['image_id'] for result in results]))
    for id in image_id_list:
      aList = [] 
      for i in range(len(results)):
        if results[i]['image_id'] == id:
          aList.append(i)
      bb = [results[i]['bbox'] for i in aList]
      bb = np.array(bb)

      category_id = results[aList[0]]['category_id']
      print('Category name: ', nameDict[category_id])

      img_path = os.path.join('./datasets', '{}.png'.format(id))
      img = cv2.imread(img_path)[:, :, ::-1]

      # bb = [bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]]
      for i in range(bb.shape[0]):
        bb[i] = [bb[i][0], bb[i][1], bb[i][0] + bb[i][2], bb[i][1]+bb[i][3]]

      # bb = np.array(bb)
      # bb.reshape(1, -1)
      # bb = np.expand_dims(bb, axis=0)
      bb = bb.astype('float64')

      # img_, bboxes_ = RandomHorizontalFlip(1)(img.copy(), bb.copy())

      plotted_img = draw_rect(img, bb)
      # plotted_img = draw_rect(img, bb)

      plt.imshow(plotted_img)
      plt.show()


if __name__ == "__main__":
    print("Hello")