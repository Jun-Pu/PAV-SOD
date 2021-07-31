import cv2
import os


def Vid2Frm():
    seqs = os.listdir(os.path.join(os.getcwd(), 'Videos'))  # please take care of your own path
    for seq in seqs:
        seq_path = os.path.join(os.getcwd(), 'Videos', seq)
        save_path = os.path.join(os.getcwd(), 'Frames', seq[:-4])
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
    Vid2Frm()