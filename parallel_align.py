import os
import sys
import bz2
from keras.utils import get_file
from ffhq_dataset.face_alignment import image_align
from ffhq_dataset.landmarks_detector import LandmarksDetector
from multiprocessing import Pool

LANDMARKS_MODEL_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'


def unpack_bz2(src_path):
    data = bz2.BZ2File(src_path).read()
    dst_path = src_path[:-4]
    with open(dst_path, 'wb') as fp:
        fp.write(data)
    return dst_path



landmarks_model_path = unpack_bz2(get_file('shape_predictor_68_face_landmarks.dat.bz2',
                                           LANDMARKS_MODEL_URL, cache_subdir='temp'))
landmarks_detector = LandmarksDetector(landmarks_model_path)

def extract(img_name):
    try:
      raw_img_path = os.path.join(RAW_IMAGES_DIR, img_name)
      n = 0
      for i, face_landmarks in enumerate(landmarks_detector.get_landmarks(raw_img_path), start=1):
          face_img_name = '%s_%02d.png' % (os.path.splitext(img_name)[0], i)
          aligned_face_path = os.path.join(ALIGNED_IMAGES_DIR, face_img_name)
          image_align(raw_img_path, aligned_face_path, face_landmarks)
          n+=1
      print("processed "+img_name+ " produced "+str(n)+ " aligned images from it")
    except Exception as e:
      print(img_name + "failed, too bad")
      print(e)


if __name__ == '__main__':
    RAW_IMAGES_DIR = sys.argv[1]
    ALIGNED_IMAGES_DIR = sys.argv[2]
    n_processed = int(sys.argv[3])

    already_processed = set(map(lambda x: x.split("_")[0], os.listdir(ALIGNED_IMAGES_DIR)))

    to_process_images = list(filter(lambda x: x.split(".")[0] not in already_processed, os.listdir(RAW_IMAGES_DIR)))
    
    print("processing "+str(len(to_process_images))+" images")
    
    # specify number of processes
    p = Pool(n_processed)
    p.map(extract, to_process_images)
