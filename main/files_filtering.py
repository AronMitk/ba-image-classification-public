import os
import random
import re
import shutil

import numpy as np


class DataContainer(object):
    area = ''
    files = []
    size = 0

    def __init__(self, area, files):
        self.area = area
        self.files = files
        self.size = len(files)

    def print(self):
        print(self.area)
        print(self.files)
        print(self.size)


def delete_temp_folder(path):
    path = os.path.join(path, 'temp')
    shutil.rmtree(path)


# -------

def filter_files_by_extension(files):
    p = re.compile("\.(gif|jpe?g|tiff?|png|webp|bmp|JPG)$")
    return [s for s in files if re.search(p, s)]


def filter_files_by_file_name(files):
    p = re.compile("(\d+.(gif|jpe?g|tiff?|png|webp|bmp|JPG))")
    list = [s for s in files if re.fullmatch(p, s)]

    if len(list) < 50:
        list = files

    return list


def get_all_files_container(top_path):
    list = []

    for i in os.listdir(top_path):
        full_path = os.path.join(top_path, i)
        if os.path.isdir(full_path):
            f_list = filter_files_by_extension(os.listdir(full_path))
            f_list = filter_files_by_file_name(f_list)
            list.append(DataContainer(i, [os.path.join(full_path, s) for s in f_list]))

    return list


def filter_by_area(cont, avail):
    return [s for s in cont if s.area in avail]


def filter_images_by_size(cont, max):
    list = []
    for i in cont:
        if i.size > max * 1.5:
            list.append(DataContainer(i.area, random.sample(i.files, max)))
        else:
            list.append(DataContainer(i.area, i.files))
    return list


def copy_to_temp(conts, path):
    temp_dir = os.path.join(path, 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    for i in conts:
        area_dir = os.path.join(temp_dir, i.area)
        os.makedirs(area_dir, exist_ok=True)
        [shutil.copyfile(s, os.path.join(area_dir, os.path.basename(s))) for s in i.files]


def remove_temp_dir(path):
    if os.path.exists(os.path.join(path, 'temp')):
        shutil.rmtree(os.path.join(path, 'temp'))
    else:
        print('no temp folder')


def filter_dirs(top_path, avail):
    conts = get_all_files_container(top_path)

    list = filter_by_area(conts, avail)

    min = np.min([s.size for s in list])
    conts = filter_images_by_size(list, min)

    for i in conts:
        print(i.print())
        print('-----')

    copy_to_temp(conts, top_path)

    return os.path.join(top_path, 'temp')

# 1. get path
# 2. find available image dirs
# 3. check the smallest dataset
# 4. get images number (samples) which all dataset should numbers
# 5. dirs with larger number should be filtered by priorities
# 6. find not original author images and delete them
# 7. if smaller number is still needed, delete original images
# 8. create a temp folder
# 9. copy to it the same structure, e.g. if image is took from LT1 folder, in temp folder it should be also LT1
