'''
    PyTorch dataset class for COCO-CT-formatted datasets. Note that you could
    use the official PyTorch MS-COCO wrappers:
    https://pytorch.org/vision/master/generated/torchvision.datasets.CocoDetection.html

    We just hack our way through the COCO JSON files here for demonstration
    purposes.

    See also the MS-COCO format on the official Web page:
    https://cocodataset.org/#format-data

    2022 Benjamin Kellenberger
'''
import os
import json
from torch.utils.data import Dataset, ConcatDataset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
import pandas as pd
import numpy as np
import pdb
from torch.utils.data import DataLoader, random_split


def create_dataloader(cfg,split,mapping=None):
    dataset_list = [cfg['{}_dataset'.format(split)]]
    dataset=CTDataset(cfg['data_root'], dataset_list ,mapping=mapping)
    dataloader = DataLoader(
            dataset=dataset,
            batch_size=cfg['batch_size'],
            shuffle=True,
            num_workers=cfg['num_workers']
        )  
    return dataloader

class CTDataset(Dataset):
    def __init__(self, data_root, label_file_list, mapping=None):
        self.data_root = data_root#cfg['data_root']
        if type(label_file_list)==str:
            if label_file.endswith('csv'):
                self.labels = pd.read_csv(os.path.join(self.data_root, label_file))
                self.col_name='labels'
                self.img_name='image_path_rel'  
        else:
            dataframes = []
            print(label_file_list)
            for label_file in label_file_list:
                df = pd.read_csv(label_file)
                dataframes.append(df)
            self.labels = pd.concat(dataframes, ignore_index=True)

        if mapping is not None:
            self.mapping=mapping
            mask = self.labels['question__species'].isin(mapping)
            self.labels = self.labels[mask]
            self.labels['labels']=self.labels['question__species'].map(mapping).astype(int)
        else:
            fact=pd.factorize(self.labels['question__species'])
            self.labels['labels']=fact[0]
            self.mapping=fact[1]
            values,keys=pd.factorize(self.labels['question__species'].unique())
            self.mapping = dict(zip(keys, values))
        self.col_name='labels'
        self.img_name='image_path_rel'
        self.transform = Compose([              
            #Resize((cfg['image_size'])),        
            Resize((224,224)),    
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])                       
        ])
        self.classes = self.labels[self.col_name].unique()
        self.num_classes = len(self.classes)
        

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_path = self.labels.iloc[idx][self.img_name]
        label = self.labels.iloc[idx][self.col_name]
        bounding_box = self.labels.iloc[idx]['bbox']
        image = Image.open(os.path.join(self.data_root,image_path))

        if type(bounding_box)==str:
            image_width, image_height = image.size
            # Convert relative bounding box to absolute pixel coordinates
            x_rel, y_rel, width_rel, height_rel = [float(str_x) for str_x in bounding_box[1:-1].split(',')]
            x = int(x_rel * image_width)
            y = int(y_rel * image_height)
            width = int(width_rel * image_width)
            height = int(height_rel * image_height)
            # Adjust the bounding box if it goes out of bounds
            new_x = max(0, min(x, image_width - width))
            new_y = max(0, min(y, image_height - height))
            # Adjusted bounding box
            adjusted_bbox = [new_x, new_y, width, height]
            # Crop the image using PIL
            image = image.crop((new_x, new_y, new_x + width, new_y + height))
                
        # Load and preprocess the image
        if self.transform:
            image = self.transform(image)
        return image, label

if __name__ == "__main__":
    import yaml
    cfg = yaml.safe_load(open('configs/cfg_kga_kga.yaml', 'r'))
    dl_train, dl_test = create_dataloader(cfg,split="train"), create_dataloader(cfg,split="test")
    dl_val = create_dataloader(cfg,split="val", mapping=dl_train.dataset.mapping)
