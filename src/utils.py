import os
import numpy as np
import cv2

def find_name(path: str,base_name:str) -> str:
    files = os.listdir(path)
    n=0
    while True:
        if base_name + str(n) in files:
            n+=1
        else:
            return base_name + str(n)



def load_image(path:str) -> np.array:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} not found")
    return cv2.imread(path)
