import os 
import augment.data_aug.data_aug as data_aug
import augment.data_aug.bbox_util as bbox_util
import numpy as np 
import cv2 
import matplotlib.pyplot as plt
import json
from pylab import rcParams
rcParams['figure.figsize'] = 20,16
from PIL import Image
from config import *

def visualize(filename):
    l1 = [1, 2, 3, 4, 5, 6, 7]
    l2 = ['Cấm ngược chiều', 'Cấm dừng và đỗ', 'Cấm rẽ', 'Giới hạn tốc độ', 'Cấm còn lại', 'Nguy hiểm', 'Hiệu lệnh']
    nameDict = dict(zip(l1, l2))
    file_path = os.path.join('.', filename)
    with open(file_path, 'r') as f:
        results = json.load(f)

    image_id_list = list(set([result['image_id'] for result in results]))
    cnt = 0
    for id in image_id_list:
      aList = [] 
      for i in range(len(results)):
        if results[i]['image_id'] == id:
          aList.append(i)
      bb = [results[i]['bbox'] for i in aList]
      bb = np.array(bb)

      img_path = os.path.join(PUBLIC_TEST_DIR, '{}.png'.format(id))
      img = cv2.imread(img_path)[:, :, ::-1]

      bb[:, 2] += bb[:, 0]
      bb[:, 3] += bb[:, 1]
      bb = bb.astype('float64')

      plotted_img = data_aug.draw_rect(img, bb)

      plt.imshow(plotted_img)
      plt.show()

      cnt += 1
      if cnt == 3:
        break


if __name__ == "__main__":
    visualize('results.json')