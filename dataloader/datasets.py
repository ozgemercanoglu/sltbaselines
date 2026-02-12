"""
This version have skeleton dataloader in addition to RGB. Also, optimized to be faster in loading - compared to datasets_wSkel.py
"""
import torch
import utils as utils
import torch.utils.data.dataset as Dataset
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
from PIL import Image
import cv2
import os
import lmdb
import io
from vidaug import augmentors as va
from dataloader.augmentation import *

# global definition
from definition import PAD_IDX

import pickle
from torch import Tensor
from dataloader.database import ImageDatabase

class S2T_Dataset(Dataset.Dataset):
    def __init__(self,path,tokenizer,config,args,phase, training_refurbish=False):
        self.config = config
        self.args = args
        self.training_refurbish = training_refurbish
        
        self.raw_data = utils.load_dataset_file(path)
        self.tokenizer = tokenizer
        # read from images
        if "img_path" in config['data']:
            self.img_path = config['data']['img_path']
        # read from lmdb files
        if "img_lmdb_path" in config['data']:
            self.img_lmdb_path = config['data']['img_lmdb_path']
        self.phase = phase
        self.max_length = config['data']['max_length']
        
        self.list = [key for key,value in self.raw_data.items()]
        self.dataset_name = config["data"].get("dataset_name", "phoenix")

        sometimes = lambda aug: va.Sometimes(0.5, aug) # Used to apply augmentor with 50% probability
        self.seq = va.Sequential([
            sometimes(va.RandomRotate(30)),
            sometimes(va.RandomTranslate(x=10, y=10)),
        ])
        self.seq_color = va.Sequential([
            sometimes(Brightness(min=0.1, max=1.5)),
            sometimes(Color(min=0.1, max=1.5)),
        ])

        self.img_len = 0 # this will be assigned later in either load_imgs or load_imgs_lmdb

    def __len__(self):
        return len(self.raw_data)
    
    def __getitem__(self, index):
        key = self.list[index]
        sample = self.raw_data[key]
        tgt_sample = sample['text']
        name_sample = sample['name']
        if self.dataset_name == "csl-daily":
            name_sample = self.phase + "/" + name_sample
            tgt_sample = " ".join(tgt_sample)

        if "img_lmdb_path" in self.config["data"]:
            img_sample = self.load_imgs_lmdb(name_sample)
        else:
            if self.dataset_name == "phoenix":
                img_sample = self.load_imgs([self.img_path + x for x in sample['imgs_path']])
            elif self.dataset_name ==  "csl-daily":
                image_names = os.listdir(self.img_path + "/" + sample['name'])
                image_names.sort()
                img_sample = self.load_imgs([self.img_path + "/" + sample['name'] + "/" + x for x in image_names])

        # print(len([self.img_path + x for x in sample['imgs_path']]), i3d_sample.shape, vit_sample.shape)
        return name_sample,img_sample,tgt_sample,self.img_len
    
    def load_imgs(self, paths):
    # Load images from raw files

        data_transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), 
                                    ])
        if len(paths) > self.max_length:
            tmp = sorted(random.sample(range(len(paths)), k=self.max_length))
            new_paths = []
            for i in tmp:
                new_paths.append(paths[i])
            paths = new_paths

        self.img_len = len(paths)
        imgs = torch.zeros(len(paths),3, self.args.input_size,self.args.input_size)
        crop_rect, resize = utils.data_augmentation(resize=(self.args.resize, self.args.resize), crop_size=self.args.input_size, is_train=(self.phase=='train'))

        batch_image = []
        for i,img_path in enumerate(paths):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            batch_image.append(img)

        if self.phase == 'train':
            batch_image = self.seq(batch_image)

        for i, img in enumerate(batch_image):
            img = img.resize(resize)
            img = data_transform(img).unsqueeze(0)
            imgs[i,:,:,:] = img[:,:,crop_rect[1]:crop_rect[3],crop_rect[0]:crop_rect[2]]
        
        return imgs

    def load_imgs_lmdb(self, file_name):
    # Load images from lmdb file (created with create_lmdb)

        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        phase, file_name = file_name.split('/')
        folder = os.path.join(self.img_lmdb_path, phase, file_name)
        images_db = ImageDatabase(path=str(folder))
        ind = list(range(0, len(images_db.keys)))
        if len(ind) > self.max_length:
            ind = sorted(random.sample(ind, k=self.max_length))
        self.img_len = len(ind)
        images = images_db[ind]  # list of PIL images

        crop_rect, resize = utils.data_augmentation(resize=(self.args.resize, self.args.resize),
                                                    crop_size=self.args.input_size, is_train=(self.phase == 'train'))

        if self.phase == 'train':
            images = self.seq(images)
        imgs = torch.zeros(len(images), 3, self.args.input_size, self.args.input_size)
        for i, img in enumerate(images):
            img = img.resize(resize)
            img = data_transform(img).unsqueeze(0)
            imgs[i, :, :, :] = img[:, :, crop_rect[1]:crop_rect[3], crop_rect[0]:crop_rect[2]]

        del images
        return imgs

    def load_imgs_lmdb_preprocessed(self, file_name):
    # For speeding up dataloader, images could be preprocessed (crop, resize), and compressed as jpeg into lmdb (created with create_lmdb_compressed.py)
    # Use this function in that case.
        
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        phase, file_name = file_name.split('/')
        lmdb_folder = os.path.join(self.img_lmdb_path, phase, file_name)

        # read num of frame information
        with lmdb.open(
                path=f"{lmdb_folder}",
                readonly=True,
                readahead=False,
                lock=False,
                meminit=False,
        ).begin(write=False) as txn:
            details = pickle.loads(txn.get(key=f"details".encode("ascii")))
            num_frames = details["num_frames"]

        ind = np.arange(0, num_frames)
        frame_sampling = self.config["data"].get("frame_sampling", "random")
        adaptive_frame_skip = self.config["data"].get("adaptive_frame_skip", 0)
        if len(ind) > self.max_length:
            if frame_sampling == "random":
                # pick random indices
                ind = np.sort(np.random.choice(ind, size=self.max_length, replace=False))
            elif frame_sampling == "sequential":
                # if the video is still longer even with frame skipping
                if len(ind) > self.max_length * (adaptive_frame_skip + 1):
                    ind = ind[::adaptive_frame_skip + 1][:self.max_length]
                else:
                    ind = np.linspace(0, num_frames - 1, num=self.max_length, dtype=int)
        self.img_len = len(ind)

        # load images
        with lmdb.open(
                path=f"{lmdb_folder}",
                readonly=True,
                readahead=False,
                lock=False,
                meminit=False,
        ).begin(write=False) as txn:
            images = [np.array(Image.open(io.BytesIO(txn.get(key=f"{idx}".encode("ascii"))))) for idx in ind]

        if self.phase == 'train':
            try:
                images = self.seq(images)
                images = va.RandomCrop(size=(self.args.input_size, self.args.input_size))(images)
            except Exception as e:
                print(file_name)
                print(f"Error: {str(e)}")
        else:
            images = va.CenterCrop(size=(self.args.input_size, self.args.input_size))(images)
        imgs = torch.stack([data_transform(img) for img in images]).float()

        return imgs
    
    def collate_fn(self,batch):
        tgt_batch,img_tmp,src_length_batch,name_batch, img_lens = [],[],[],[],[]
        src_input = {} # return this

        for name_sample, img_sample, tgt_sample, img_len in batch:
            name_batch.append(name_sample)
            img_tmp.append(img_sample)
            tgt_batch.append(tgt_sample)
            img_lens.append(img_len)

        if self.dataset_name == "csl-daily":
            name_batch = [video_name.split("/")[-1]  for video_name in name_batch]

        max_len = max(img_lens)
        left_pad = 8
        # video_length = torch.LongTensor([np.ceil(len(os.listdir(self.img_path + "/" + video_name)) / 4.0) * 4 + 16 for video_name in name_batch])
        video_length = torch.LongTensor([np.ceil(video_len/ 4.0) * 4 + 16 for video_len in img_lens])
        right_pad = int(np.ceil(max_len / 4.0)) * 4 - max_len + 8
        max_len = max_len + left_pad + right_pad
        padded_video = [torch.cat(
            (
                vid[0][None].expand(left_pad, -1, -1, -1),
                vid,
                vid[-1][None].expand(max_len - len(vid) - left_pad, -1, -1, -1),
            )
            , dim=0)
            for vid in img_tmp]
        img_tmp = [padded_video[i][0:video_length[i],:,:,:] for i in range(len(padded_video))]
        # Explain how these padding works..
        # Let's assume number of images are [70, 59, 94, 130, 101, 41, 142, 103]
        # video_length: [ 88,  76, 112, 148, 120,  60, 160, 120] because of ceil(len/ 4.0) * 4 + 16
        # padded_video: [160, 160, 160, 160, 160, 160, 160, 160]
        # then img_tmp = [88, 76, 112, 148, 120, 60, 160, 120] by copying from padded_video -this includes copied first and last frames now
        # finally, img_tmp = torch.Size([884, 3, 224, 224]) by concat

        for i in range(len(img_tmp)):
            src_length_batch.append(len(img_tmp[i]))
        img_tmp = torch.cat(img_tmp, 0)

        src_length_batch = torch.tensor(src_length_batch)
        new_src_lengths = (((src_length_batch-5+1) / 2)-5+1)/2
        new_src_lengths = new_src_lengths.long()
        mask_gen = []
        for i in new_src_lengths:
            tmp = torch.ones([i]) + 7
            mask_gen.append(tmp)
        mask_gen = pad_sequence(mask_gen, padding_value=PAD_IDX,batch_first=True)
        img_padding_mask = (mask_gen != PAD_IDX).long()
        tgt_input = self.tokenizer(text_target=tgt_batch, return_tensors="pt", padding=True, truncation=True)


        src_input['input_ids'] = img_tmp
        src_input['attention_mask'] = img_padding_mask
        src_input['name_batch'] = name_batch

        src_input['src_length_batch'] = src_length_batch
        src_input['new_src_length_batch'] = new_src_lengths

        if self.training_refurbish:
            masked_tgt = utils.NoiseInjecting(tgt_batch, self.args.noise_rate, noise_type=self.args.noise_type, random_shuffle=self.args.random_shuffle, is_train=(self.phase=='train'))
            masked_tgt_input = self.tokenizer(text_target=masked_tgt, return_tensors="pt", padding=True, truncation=True)
            return src_input, tgt_input, masked_tgt_input
        return src_input, tgt_input

    def __str__(self):
        return f'#total {self.phase} set: {len(self.list)}.'
