import cv2
import os


def Vid2KeyFrm():
    seq = '_-0cfJOmUaNNI_1'  # par example
    seq_path = os.path.join(os.getcwd(), seq + '.mp4')  # please take care of your own path

    save_path = os.path.join(os.getcwd(), seq)
    if not os.path.exists(save_path): os.makedirs(save_path)
    cap = cv2.VideoCapture(seq_path)
    frames_num = int(cap.get(7))
    countF = 0
    for i in range(frames_num):
        ret, frame = cap.read()
        cv2.imwrite(os.path.join(save_path, format(str(countF), '0>5s') + '.png'), frame)
        print(" {} frames are extracted.".format(countF))
        countF += 1

if __name__ == '__main__':
    Vid2KeyFrm()