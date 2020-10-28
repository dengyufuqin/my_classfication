import os
import copy


def write_txt(path, txt_path):
    num = len(os.listdir(path))
    file_path = txt_path
    file = open(file_path, 'w')
    Y = 0
    c = os.listdir(path)

    for category in c:
        C = c.index(category)
        for imgs in os.listdir(os.path.join(path, category)):
            file.write(category+'/'+imgs+'|'+str(Y)+'\n')
        Y = Y+1



if __name__ == '__main__':
    write_txt('data/',
              'train.txt')
