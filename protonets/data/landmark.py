import os
import sys
import glob
from functools import partial
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import ToTensor, Grayscale, Resize, Compose
from torchnet.dataset import ListDataset, TransformDataset
from torchnet.transform import compose
import protonets
from protonets.data.base import convert_dict, CudaTransform, EpisodicBatchSampler, SequentialBatchSampler

LANDMARK_DATA_DIR = '/content/drive/MyDrive/landmark_dataset'
LANDMARK_CACHE = {}

def load_image_path(key, out_field, d):
    d[out_field] = Image.open(d[key]).convert('RGB')  # Force RGB for consistency
    return d

def convert_tensor(key, d):
    # Normalize to [0,1] and permute dimensions
    d[key] = torch.from_numpy(np.array(d[key], np.float32, copy=False)).permute(2, 0, 1).contiguous() / 255.0
    return d

def resize_image(key, size, d):
    transform = Compose([
        Resize(size),
        # Add other transforms here if needed
    ])
    d[key] = transform(d[key])
    return d

def load_class_images(d):
    if d['class'] not in LANDMARK_CACHE:
        from torchvision import transforms

def load_class_images(d):
    if d['class'] not in LANDMARK_CACHE:
        class_dir = os.path.join(LANDMARK_DATA_DIR, 'data', d['class'])
        
        # Define transforms once here
        preprocess = transforms.Compose([
            transforms.Resize(256),                   # First resize to 256
            transforms.CenterCrop(224),               # Then crop to 224x224
            transforms.ToTensor(),                    # Convert to tensor
            transforms.Normalize(                     # ImageNet stats
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        class_images = sorted(glob.glob(os.path.join(class_dir, '*')))
        image_ds = TransformDataset(ListDataset(class_images),
                            compose([
                                partial(convert_dict, 'file_name'),
                                partial(load_image_path, 'file_name', 'data'),
                                
                                # Replace old resize/convert with:
                                lambda x: {'data': preprocess(x['data']), 'class': x['class']},
                                
                                # Remove these old lines:
                                # partial(resize_image, 'data', (224, 224)),
                                # partial(convert_tensor, 'data')
                            ]))
        # ...

        loader = torch.utils.data.DataLoader(image_ds, batch_size=len(image_ds), shuffle=False)

        for sample in loader:
            LANDMARK_CACHE[d['class']] = sample['data']
            break

    return {'class': d['class'], 'data': LANDMARK_CACHE[d['class']]}

# Keep the rest similar to omniglot.py but adjust parameters:
def load(opt, splits):
    split_dir = os.path.join(LANDMARK_DATA_DIR, 'splits', opt['data.split']) 

    ret = {}
    for split in splits:
        # Use similar logic but adjust parameters:
        n_way = opt['data.way'] if split == 'train' else opt['data.test_way']
        n_support = opt['data.shot']
        n_query = opt['data.query']
        n_episodes = opt['data.train_episodes'] if split == 'train' else opt['data.test_episodes']

        transforms = [
            partial(convert_dict, 'class'),
            load_class_images,
            partial(extract_episode, n_support, n_query)
        ]
        
        if opt['data.cuda']:
            transforms.append(CudaTransform())

        transforms = compose(transforms)

        class_names = []
        class_paths = []
        min_samples = opt['data.shot'] + opt['data.query']

        with open(os.path.join(split_dir, f"{split}.txt"), 'r') as f:
          for cls in f.readlines():
            cls = cls.strip()
            cls_path = os.path.join(LANDMARK_DATA_DIR, 'data', cls)
            if len(os.listdir(cls_path)) >= min_samples:
              class_names.append(cls)
              class_paths.append(cls_path)
        


        ds = TransformDataset(ListDataset(class_names), transforms)

        #sampler = EpisodicBatchSampler(len(ds), n_way, n_episodes)
        sampler = BalancedEpisodicSampler(class_paths = class_paths, n_way = n_way, n_episodes = n_episodes)
        ret[split] = torch.utils.data.DataLoader(ds, batch_sampler=sampler, num_workers=2, pin_memory=opt['data.cuda'])

    return ret