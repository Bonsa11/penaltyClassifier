import cv2
import os

def vid_2_imgs(vid_path: str, imgs_path: str, scale_perc: int):

    vid_path = os.path.abspath(vid_path)
    imgs_path = os.path.abspath(imgs_path)

    if not os.path.exists(vid_path):
        raise FileNotFoundError(vid_path)
    else:
        print(f'found file {vid_path} to download')

    vidcap = cv2.VideoCapture(vid_path)
    fldr_name = vid_path.split('/')[-1].split('.')[0]
    print(f' fldr name is {fldr_name}')
    fldr_num = 0

    fldr_path = os.path.join(imgs_path, fldr_name)
    while os.path.exists(fldr_path):
        fldr_path = os.path.join(imgs_path, f'{fldr_name}_{fldr_num}')  
        fldr_num += 1
    else:
        os.mkdir(fldr_path)

    success,image = vidcap.read()
    count = 0

    while success:
        file_name = os.path.join(fldr_path, f"frame_{count}.jpg")
        width = int(image.shape[1] * scale_perc / 100)
        height = int(image.shape[0] * scale_perc / 100)
        dim = (width, height)    # resize image
        resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        cv2.imwrite(file_name, resized)      
        success,image = vidcap.read()
        count += 1

    print(f'created {len(os.listdir(fldr_path))} files')

if __name__ == '__main__':
    vid_2_imgs('/data/penaltyClassifier/videos/jorg.mp4','/data/penaltyClassifier//images', 50)
