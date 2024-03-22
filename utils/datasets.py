import numpy as np
import pandas as pd
from scipy.io import loadmat
from random import shuffle
import os
import cv2

class DataManager(object):
    """Class for loading fer2013 emotion classification dataset or imdb gender classification dataset."""

    def __init__(self, dataset_name='imdb', dataset_path=None, image_size=(48, 48)):
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.image_size = image_size

        if self.dataset_path is not None:
            self.dataset_path = dataset_path

        elif self.dataset_name == 'imdb':
            self.dataset_path = '../datasets/imdb_crop/imdb.mat'

        elif self.dataset_name == 'fer2013':
            self.dataset_path = '../datasets/fer2013/fer2013.csv'

        elif self.dataset_name == 'KDEF':
            self.dataset_path = '../datasets/KDEF/'

        else:
            raise Exception(
                'Incorrect dataset name, please input imdb or fer2013'
            )

    def get_data(self):
        if self.dataset_name == 'imdb':
            ground_truth_data = self._load_imdb()

        elif self.dataset_name == 'fer2013':
            ground_truth_data = self._load_fer2013()

        elif self.dataset_name == 'KDEF':
            ground_truth_data = self._load_KDEF()

        return ground_truth_data

    def _load_imdb(self):

        """1. Thiết lập ngưỡng điểm ảnh face"""

        face_score_threshold = 3   # -> để qđịnh xem bức ảnh có chứa face hợp lệ ko

        """2. Tải dl từ tệp '.mat'"""

        dataset = loadmat(self.dataset_path)

        """3. Trích xuất các tp cần thiết từ dl tải"""

        # biến chứa tên và giới tính của hình ảnh
        image_name_array = dataset['imdb']['full_path'][0, 0][0]
        gender_classes = dataset['imdb']['gender'][0, 0][0]

        # biến biểu thị điểm số về chất lượng của face trong ảnh
        face_score = dataset['imdb']['face_score'][0, 0][0]
        second_face_score = dataset['imdb']['second_face_score'][0, 0][0]
        
        """4. Xây dựng bộ lọc cho DL"""
    
        face_score_mask = face_score > face_score_threshold   # mask giữ lại các ảnh có điểm só face > thresh
        
        second_face_score_mask = np.isnan(second_face_score)   # mask giữ lại ảnh ko chứa face thứ 2 (nếu có)

        unknown_gender_mask = np.logical_not(np.isnan(gender_classes))   # mask chỉ giữ lại ảnh k phải k xác định đc gender (ý là đã xác định á)
        
        """5. Kết hợp các mask"""

        # sử dụng mệnh đề logic để kết hợp các mask -> final mask
        mask = np.logical_and(face_score_mask, second_face_score_mask)
        mask = np.logical_and(mask, unknown_gender_mask)

        """6. Xây dựng và trả về {} kết quả"""

        # mảng chứa ttin ảnh và gender thoả điều kiện
        image_name_array = image_name_array[mask]
        gender_classes = gender_classes[mask].tolist()

        image_names = []
        for image_names_arg in range(image_name_array.shape[0]):
            image_name = image_name_array[image_names_arg][0]
            image_names.append(image_name)

        # key là tên ảnh - value là gender t.ứng
        return dict(zip(image_names, gender_classes))
    
    def _load_fer2013(self):
        data = pd.read_csv(self.dataset_path)

        """2. Trích xuất dl pixels"""

        pixels = data['pixels'].tolist()   # chuyển dl cột pixels trong data thành list

        """3. Junbi dl ảnh"""
        width = height = 48
        faces = []
        
        # mỗi row trong cột pixels chứa 1 chuỗi các gtri pixel đc ptách = ' '.
        # mỗi chuỗi này đc chuyển đổi thành 1 mảng numpy và sau đó đc chuyển thành 1 mảng 2D 48x48
        for pixel_sequence in pixels:
            face = [int(pixel) for pixel in pixel_sequence.split(' ')]
            face = np.asarray(face).reshape(width, height)
            face = cv2.resize(face.astype('uint8'), self.image_size)   # chỉnh size mỗi ảnh về size đc xđ bởi self.image_size
            faces.append(face.astype('float32'))

        """4. Junbi nhãn emotion"""    
        faces = np.asarray(faces)
        faces = np.expand_dims(faces, -1)

        # cột emotion của df chứa label emotion. 
        # pd.get_dummies chuyển các label này thành dạng one-hot encoding và sau đó đc chuyển thành mảng np
        emotions = pd.get_dummies(data['emotion']).as_matrix()
        
        return faces, emotions   
    
    def _load_KDEF(self):
        class_to_arg = get_class_to_arg(self.dataset_name)
        num_classes = len(class_to_arg)

        file_paths = []
        for folder, subfolders, filenames in os.walk(self.dataset_path):
            for filename in filenames:
                if filename.lower().endswith(('.jpg')):
                    file_paths.append(os.path.join(folder, filename))

        num_faces = len(file_paths)
        y_size, x_size = self.image_size
        faces = np.zeros(shape=(num_faces, y_size, x_size))
        emotions = np.zeros(shape=(num_faces, num_classes))

        for file_arg, file_path in enumerate(file_paths):
            image_array = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            image_array = cv2.resize(image_array, (y_size, x_size))
            faces[file_arg] = image_array
            file_basename = os.path.basename(file_path)
            file_emotion = file_basename[4:6]

            # there are 2 file names in the dataset, that don't match the given classes

            try:
                emotion_arg = class_to_arg[file_emotion]
            except:
                continue
            emotions[file_arg, emotion_arg] = 1

        faces = np.expand_dims(faces, -1)

        return faces, emotions


        