import os
import zipfile
from io import BytesIO
import json

from datasets.SLIViTDataset3D import SLIViTDataset3D, ToTensor
from auxiliaries.pretrain import *
from auxiliaries.finetune import default_transform_gray


class OCTDataset3D(SLIViTDataset3D):
    #  example image name of Hadassah: bscan_12.tiff  ->  012
    #  example image name of Houston: 12.tiff  ->  012

    def __init__(self, meta, label_name, path_col_name, **kwargs):
        with open(os.path.join(os.path.dirname(meta), "meta_info.json"), 'r') as f:
            self.meta_info = json.load(f)

        with open(os.path.join(os.path.dirname(meta), "train_val_test.json"), 'r') as f:
            self.train_val_test = json.load(f)
        self.patient_info = self.train_val_test[kwargs.get('train_val_test')]

        self.scan_paths = []
        self.labels = []
        for patient in self.patient_info:
            cur_zip_paths = []
            for filename in self.patient_info[patient]['filenames']:
                cur_zip_paths.append(os.path.join(meta, filename.split('_')[0], patient + "_" + filename + '.zip'))

            self.scan_paths.extend(cur_zip_paths)
            label = [0, 0, 0, 0, 0, 0, 0, 0]
            for disease in self.patient_info[patient]['diseases']:
                label[int(self.train_val_test['class_mapping'][disease])] = 1
            self.labels.extend([label] * len(cur_zip_paths))

        self.t = default_transform_gray
        self.filter = lambda x: x
        self.num_classes = 8

        self.num_slices_to_use = kwargs.get('num_slices_to_use')
        self.sparsing_method = kwargs.get('sparsing_method')
        self.filter = lambda x: x.endswith(kwargs.get('img_suffix'))

    def load_scan(self, vol_path, slc_idxs):

        vol = []
        with zipfile.ZipFile(vol_path, 'r') as zip_ref:
            filenames = sorted(zip_ref.namelist(), key=lambda x: int(x.split('_')[-1].split('.')[0]))
            for i, filename in enumerate(filenames):
                if i in slc_idxs:
                    slice_data = zip_ref.read(filename)
                    f = BytesIO(slice_data)
                    img = PIL.Image.open(f)
                    vol.append(ToTensor()(img))

        return vol
