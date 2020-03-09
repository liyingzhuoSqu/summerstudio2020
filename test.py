import os
import random
import shutil
def delete():
    path = 'slideshow/Ad_images/'
    bin = 'slideshow/BIN/'
    folder = os.listdir(path)
    N = int((len(folder)) * 1)
    print(N)
    i = 0
    for i in range(N):
        filename = random.choice(os.listdir(path))
        print(filename)
        shutil.move(path + filename, bin + filename)
        i += 1

