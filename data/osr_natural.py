from torch.utils.data import Dataset,DataLoader
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid

voc_name_list=["aeroplane","bicycle","bird","boat","bottle","bus","car","cat", "chair",
               "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

voc_known_name=["aeroplane","bird","bus","cat","dog","person","train"]
voc_known_classes=[0,2,6,7,11,14]
voc_unknown_classes=[1,3,4,5,8,9,12,13,15,16,17,18,19]
#class_overlap_coco_indices=[5,2,15,9,40,6,3,16,57,20,61,17,18,4,1,59,19,58,7,63]

nus_known_label = [0, 1, 2, 4, 7, 9, 13, 14, 16, 17, 21, 22, 40, 50, 59]
nus_known_names = ['clouds', 'sky', 'person', 'window', 'animal', 'buildings', 'water', 'grass', 'road', 'snow',
                   'flowers', 'plants', 'toy', 'sign', 'food']
nus_unknown_label = [3, 5, 6, 8, 10, 12, 15, 18, 19, 20, 23, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                     41, 42, 43, 44, 46, 47, 48, 51, 53, 54, 55, 57, 58, 62, 64, 66, 71, 72, 74, 78]


class OSR_dataset(Dataset):
    def __init__(self, split='train', data_root="D:\\datasets\\VOC\\VOCdevkit\\", exp="voc", transform=None):
        self.root_dir=data_root
        self.transform=transform
        self.split=split
        self.exp=exp
        self.name_list=self.__get_name_list(self.split)
        self.transform=transform

        if exp=="voc":
            self.num_known_classes=len(voc_known_name)
            self.class_label = np.load("D:\\Open_world_recognition_with_object_centric_learning\\ours\\osr_closed_set_all_you_need\\data\\cls_labels.npy",allow_pickle=True).item()
        elif exp=="coco":
            self.num_known_classes=20
            self.class_label = np.load(
                "D:\\Open_world_recognition_with_object_centric_learning\\ours\\osr_closed_set_all_you_need\\data\\cls_labels_coco.npy",
                allow_pickle=True).item()
        else:
            self.num_known_classes=len(nus_known_label)
            self.class_label = np.load(
                "D:\\Open_world_recognition_with_object_centric_learning\\ours\\osr_closed_set_all_you_need\\data\\nus_cls_labels.npy",
                allow_pickle=True).item()

    def __get_name_list(self,split):
        assert split in ['train','val','test_single','test_mixture']
        text_file=os.path.join("D:\\Open_world_recognition_with_object_centric_learning\\ours\\osr_closed_set_all_you_need\\data\\"+self.exp+"_"+split+".txt")
        line_list=open(text_file).readlines()
        #if split=='test_mixture':
            #line_list=line_list
        print("Totally have {} samples in {} set.".format(len(line_list),self.split))
        labels=[]
        for idx,line in enumerate(line_list):
            cls=line.split()[1]
            labels.append(cls)
        self.labels=labels
        return line_list
    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        name=self.name_list[index].split()[0]
        #print(name)
        if self.exp=='voc':
            img_path=self.root_dir+"VOC2012/JPEGImages/"+name+".jpg"
        else:
            img_path=self.root_dir+name+".jpg"
        image=Image.open(img_path).convert("RGB")
        #print(np.asarray(image).shape)
        image=self.transform(image)
        label=int(self.labels[index])
        assert label<self.num_known_classes
        label=torch.as_tensor(label,dtype=torch.long)

        return image,label


def Get_OSR_Datasets(train_transform, test_transform,dataroot="D:\\datasets\\VOC\\",exp="voc"):

    train_dataset = OSR_dataset(split='train',data_root=dataroot,exp=exp,transform=train_transform)
    val_dataset = OSR_dataset(split='val',data_root=dataroot,exp=exp,transform=test_transform)
    single_unknown_dataset = OSR_dataset(split='test_single',data_root=dataroot,exp=exp,transform=test_transform)
    mix_known_unknown_dataset = OSR_dataset(split='test_mixture',data_root=dataroot,exp=exp,transform=test_transform)

    print('Train: ', len(train_dataset), 'Test: ', len(val_dataset), 'Single_Out: ', len(single_unknown_dataset),
          'Multiple_Out',len(mix_known_unknown_dataset))

    all_datasets = {
        'train': train_dataset,
        'val': val_dataset,
        'test_known': val_dataset,
        'test_single': single_unknown_dataset,
        'test_mixture':mix_known_unknown_dataset
    }

    return all_datasets