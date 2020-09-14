from keras.datasets import cifar10
from PIL import Image
import numpy as np
import os

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print("\n mylog: x_test.shape----->", x_test.shape, '\n')
print("\n mylog: y_test.shape----->", y_test.shape, '\n')


max_num_datas = 50
num_classes = 3
num_datas_list = np.zeros(num_classes) 
print("\n mylog: num_datas_list.shape----->", num_datas_list.shape, '\n')
print("\n mylog: num_datas_list----->", num_datas_list, '\n')
# my_query: np.zeros
# generate matrix of the corresponding size, which the value is "0"

img_dir = "data"
id = 0

for x, y in zip(x_train, y_train):
# my_query: zip
# my_note: zip join the list

    if np.sum(num_datas_list) > max_num_datas * len(num_datas_list):
        break

    label = y[0]
    if label >= num_classes:
        continue

    if num_datas_list[label] == max_num_datas:
        continue

    num_datas_list[label] += 1

    img_path = os.path.join(img_dir, "{}_{}.jpg".format(label, id))
    id += 1
    img = Image.fromarray(x)
    img.save(img_path)

