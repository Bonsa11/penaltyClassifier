import cv2
import os

def process_img(img_path: str, scale_perc: int):

    if not os.path.exists(img_path):
        raise FileNotFoundError(img_path)

    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    width = int(img.shape[1] * scale_perc / 100)
    height = int(img.shape[0] * scale_perc / 100)
    dim = (width, height)    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    grey_img = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    cv2.imwrite(img_path, grey_img)


def process_dir(fldr_path: str, scale_perc: int):
    for img in os.listdir(fldr_path):
        file_path = os.path.join(fldr_path, img)
        process_img(file_path, scale_perc)

if __name__ == '__main__':
    process_dir('./images/pens', 50) 
