#!/usr/bin/env python3
import shutil
import os

def move_random(train_dir, val_dir):
    import random
    for d in os.listdir(train_dir):
        print("For directory: {0}".format(d))
        files = os.listdir(os.path.join(train_dir, d))
        files_to_move = random.sample(files, int(len(files)*0.1))
        print("Moving files...")
        print(files_to_move)
        for file in files_to_move:
            src = os.path.join(train_dir, d, file)
            dst = os.path.join(val_dir, d, file)
            shutil.move(src, dst)
        print("Done")

if __name__ == "__main__":
    src = sys.argv[1]
    dst = sys.argv[2]
    move_random(src, dst)

