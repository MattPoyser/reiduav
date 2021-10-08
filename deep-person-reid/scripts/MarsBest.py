
# coding: utf-8

# In[1]:

from __future__ import print_function, absolute_import
import math
import random
import torch
from torch.utils.data import Dataset
import random
import torchvision.transforms as transforms
random.seed(1)
import os
from PIL import Image
import numpy as np
import os
import sys
import errno
import shutil
import json
import os.path as osp
import re
import torch
import os
import glob
import re
import sys
# import urllib
import urllib.request
import tarfile
import zipfile
import os.path as osp
from scipy.io import loadmat
import numpy as np

def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj

class PRID(object):
    """
    PRID
    Reference:
    Hirzer et al. Person Re-Identification by Descriptive and Discriminative Classification. SCIA 2011.
    
    Dataset statistics:
    # identities: 200
    # tracklets: 400
    # cameras: 2
    Args:
        split_id (int): indicates which split to use. There are totally 10 splits.
        min_seq_len (int): tracklet with length shorter than this value will be discarded (default: 0).
    """
    root  = "/home2/zwjx97/best/prid_2011"
    
    # root = './data/prid2011'
    dataset_url = 'https://files.icg.tugraz.at/f/6ab7e8ce8f/?raw=1'
    split_path = osp.join(root, 'splits_prid2011.json')
    cam_a_path = osp.join(root, 'multi_shot', 'cam_a')
    cam_b_path = osp.join(root, 'multi_shot', 'cam_b')

    def __init__(self, split_id=0, min_seq_len=0):
        self._check_before_run()
        splits = read_json(self.split_path)
        if split_id >=  len(splits):
            raise ValueError("split_id exceeds range, received {}, but expected between 0 and {}".format(split_id, len(splits)-1))
        split = splits[split_id]
        train_dirs, test_dirs = split['train'], split['test']
        print("# train identites: {}, # test identites {}".format(len(train_dirs), len(test_dirs)))

        train, num_train_tracklets, num_train_pids, num_imgs_train = \
          self._process_data(train_dirs, cam1=True, cam2=True)
        query, num_query_tracklets, num_query_pids, num_imgs_query = \
          self._process_data(test_dirs, cam1=True, cam2=False)
        gallery, num_gallery_tracklets, num_gallery_pids, num_imgs_gallery = \
          self._process_data(test_dirs, cam1=False, cam2=True)

        num_imgs_per_tracklet = num_imgs_train + num_imgs_query + num_imgs_gallery
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_query_pids
        num_total_tracklets = num_train_tracklets + num_query_tracklets + num_gallery_tracklets

        print("=> PRID-2011 loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # tracklets")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_tracklets))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_tracklets))
        print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
        print("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))

    def _process_data(self, dirnames, cam1=True, cam2=True):
        tracklets = []
        num_imgs_per_tracklet = []
        dirname2pid = {dirname:i for i, dirname in enumerate(dirnames)}
        
        for dirname in dirnames:
            if cam1:
                person_dir = osp.join(self.cam_a_path, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.png'))
                assert len(img_names) > 0
                img_names = tuple(img_names)
                pid = dirname2pid[dirname]
                tracklets.append((img_names, pid, 0))
                num_imgs_per_tracklet.append(len(img_names))

            if cam2:
                person_dir = osp.join(self.cam_b_path, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.png'))
                assert len(img_names) > 0
                img_names = tuple(img_names)
                pid = dirname2pid[dirname]
                tracklets.append((img_names, pid, 1))
                num_imgs_per_tracklet.append(len(img_names))

        num_tracklets = len(tracklets)
        num_pids = len(dirnames)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet
class Mars(object):
    """
    MARS
    Reference:
    Zheng et al. MARS: A Video Benchmark for Large-Scale Person Re-identification. ECCV 2016.
    
    Dataset statistics:
    # identities: 1261
    # tracklets: 8298 (train) + 1980 (query) + 9330 (gallery)
    # cameras: 6
    Args:
        min_seq_len (int): tracklet with length shorter than this value will be discarded (default: 0).
    """
    # root = "/scratch/pp1953/data/MARS"
    # root  = "/projects/datasets/MARSFull/"
    root  = "/data/reid/MARS/"
    # root = '/archive/p/pp1953/data/MARS'
    # root = '/mnt/scratch/1/pathak/data/MARS'
    train_name_path = osp.join(root, 'info/train_name.txt')
    test_name_path = osp.join(root, 'info/test_name.txt')
    track_train_info_path = osp.join(root, 'info/tracks_train_info.mat')
    track_test_info_path = osp.join(root, 'info/tracks_test_info.mat')
    query_IDX_path = osp.join(root, 'info/query_IDX.mat')

    def __init__(self, min_seq_len=0, ):
        self._check_before_run()
        
        # prepare meta data
        train_names = self._get_names(self.train_name_path)
        test_names = self._get_names(self.test_name_path)
        track_train = loadmat(self.track_train_info_path)['track_train_info'] # numpy.ndarray (8298, 4)
        track_test = loadmat(self.track_test_info_path)['track_test_info'] # numpy.ndarray (12180, 4)
        query_IDX = loadmat(self.query_IDX_path)['query_IDX'].squeeze() # numpy.ndarray (1980,)
        query_IDX -= 1 # index from 0
        track_query = track_test[query_IDX,:]
        gallery_IDX = [i for i in range(track_test.shape[0]) if i not in query_IDX]
        track_gallery = track_test[gallery_IDX,:]

        train, num_train_tracklets, num_train_pids, num_train_imgs =           self._process_data(train_names, track_train, home_dir='bbox_train', relabel=True, min_seq_len=min_seq_len)

        video = self._process_train_data(train_names, track_train, home_dir='bbox_train', relabel=True, min_seq_len=min_seq_len)


        query, num_query_tracklets, num_query_pids, num_query_imgs =           self._process_data(test_names, track_query, home_dir='bbox_test', relabel=False, min_seq_len=min_seq_len)

        gallery, num_gallery_tracklets, num_gallery_pids, num_gallery_imgs =           self._process_data(test_names, track_gallery, home_dir='bbox_test', relabel=False, min_seq_len=min_seq_len)

        num_imgs_per_tracklet = num_train_imgs + num_query_imgs + num_gallery_imgs
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_query_pids
        num_total_tracklets = num_train_tracklets + num_query_tracklets + num_gallery_tracklets

        print("=> MARS loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # tracklets")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_tracklets))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_tracklets))
        print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
        print("  ------------------------------")

        # self.train_videos = video
        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))
        if not osp.exists(self.train_name_path):
            raise RuntimeError("'{}' is not available".format(self.train_name_path))
        if not osp.exists(self.test_name_path):
            raise RuntimeError("'{}' is not available".format(self.test_name_path))
        if not osp.exists(self.track_train_info_path):
            raise RuntimeError("'{}' is not available".format(self.track_train_info_path))
        if not osp.exists(self.track_test_info_path):
            raise RuntimeError("'{}' is not available".format(self.track_test_info_path))
        if not osp.exists(self.query_IDX_path):
            raise RuntimeError("'{}' is not available".format(self.query_IDX_path))

    def _get_names(self, fpath):
        names = []
        with open(fpath, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                names.append(new_line)
        return names

    def _process_data(self, names, meta_data, home_dir=None, relabel=False, min_seq_len=0):
        assert home_dir in ['bbox_train', 'bbox_test']
        num_tracklets = meta_data.shape[0]
        pid_list = list(set(meta_data[:,2].tolist()))
        num_pids = len(pid_list)

        if relabel: pid2label = {pid:label for label, pid in enumerate(pid_list)}
        tracklets = []
        num_imgs_per_tracklet = []

        for tracklet_idx in range(num_tracklets):
            data = meta_data[tracklet_idx,...]
            start_index, end_index, pid, camid = data
            if pid == -1: continue # junk images are just ignored
            assert 1 <= camid <= 6
            if relabel: pid = pid2label[pid]
            camid -= 1 # index starts from 0
            img_names = names[start_index-1:end_index]

            # make sure image names correspond to the same person
            pnames = [img_name[:4] for img_name in img_names]
            assert len(set(pnames)) == 1, "Error: a single tracklet contains different person images"

            # make sure all images are captured under the same camera
            camnames = [img_name[5] for img_name in img_names]
            assert len(set(camnames)) == 1, "Error: images are captured under different cameras!"

            # append image names with directory information
            img_paths = [osp.join(self.root, home_dir, img_name[:4], img_name) for img_name in img_names]
            if len(img_paths) >= min_seq_len:
                img_paths = tuple(img_paths)
                tracklets.append((img_paths, pid, camid))
                num_imgs_per_tracklet.append(len(img_paths))
                # if camid in video[pid] :
                #     video[pid][camid].append(img_paths)  
                # else:
                #     video[pid][camid] =  img_paths

        num_tracklets = len(tracklets)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet

    def _process_train_data(self, names, meta_data, home_dir=None, relabel=False, min_seq_len=0):
        video = defaultdict(dict)

        assert home_dir in ['bbox_train', 'bbox_test']
        num_tracklets = meta_data.shape[0]
        pid_list = list(set(meta_data[:,2].tolist()))
        num_pids = len(pid_list)
        if relabel: pid2label = {pid:label for label, pid in enumerate(pid_list)}
        for tracklet_idx in range(num_tracklets):
            data = meta_data[tracklet_idx,...]
            start_index, end_index, pid, camid = data
            if pid == -1: continue # junk images are just ignored
            assert 1 <= camid <= 6
            if relabel: pid = pid2label[pid]
            camid -= 1 # index starts from 0
            img_names = names[start_index-1:end_index]
            # make sure image names correspond to the same person
            pnames = [img_name[:4] for img_name in img_names]
            assert len(set(pnames)) == 1, "Error: a single tracklet contains different person images"
            # make sure all images are captured under the same camera
            camnames = [img_name[5] for img_name in img_names]
            assert len(set(camnames)) == 1, "Error: images are captured under different cameras!"

            # append image names with directory information
            img_paths = [osp.join(self.root, home_dir, img_name[:4], img_name) for img_name in img_names]
            if len(img_paths) >= min_seq_len:
                if camid in video[pid] :
                    video[pid][camid].extend(img_paths)  
                else:
                    video[pid][camid] =  img_paths
        return video 



class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img


# In[2]:


from bisect import bisect_right
import torch


# FIXME ideally this would be achieved with a CombinedLRScheduler,
# separating MultiStepLR with WarmupLR
# but the current LRScheduler design doesn't allow it

class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]
    
def make_optimizer(model):
    params = []
    base_learning_rate = 0.00035
    weight_decay = 0.0005
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = base_learning_rate
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
  
    #if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':optimizer = getattr(torch.optim, "Adam")(params, momentum=0.9)else:
    optimizer = getattr(torch.optim, "Adam")(params)
    return optimizer


# In[3]:




import torch
from torch import nn
class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=751, feat_dim=2048, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) +                   torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        # raise AttributeError(distmat.shape, self.centers.t().shape, x.shape)

        # https://discuss.pytorch.org/t/addmm--in-torch-nn-linear/1735
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss


# In[4]:








def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class VideoDataset(Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset, seq_len=15, sample='evenly', transform=None , max_length=40):
        self.dataset = dataset
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)
        # return 100

    def __getitem__(self, index):
        img_paths, pid, camid = self.dataset[index]
        num = len(img_paths)

        # if self.sample == 'restricted_random':
        #     frame_indices = range(num)
        #     chunks = 
        #     rand_end = max(0, len(frame_indices) - self.seq_len - 1)
        #     begin_index = random.randint(0, rand_end)


        if self.sample == 'random':
            """
            Randomly sample seq_len consecutive frames from num frames,
            if num is smaller than seq_len, then replicate items.
            This sampling strategy is used in training phase.
            """
            frame_indices = range(num)
            rand_end = max(0, len(frame_indices) - self.seq_len - 1)
            begin_index = random.randint(0, rand_end)
            end_index = min(begin_index + self.seq_len, len(frame_indices))

            indices = frame_indices[begin_index:end_index]
            # print(begin_index, end_index, indices)
            if len(indices) < self.seq_len:
                indices=np.array(indices)
                indices = np.append(indices , [indices[-1] for i in range(self.seq_len - len(indices))])
            else:
                indices=np.array(indices)
            imgs = []
            for index in indices:
                index=int(index)
                img_path = img_paths[index]
                img = read_image(img_path)
                if self.transform is not None:
                    img = self.transform(img)
                img = img.unsqueeze(0)
                imgs.append(img)
            imgs = torch.cat(imgs, dim=0)
            #imgs=imgs.permute(1,0,2,3)
            return imgs, pid, camid

        elif self.sample == 'dense':
            """
            Sample all frames in a video into a list of clips, each clip contains seq_len frames, batch_size needs to be set to 1.
            This sampling strategy is used in test phase.
            """
            # import pdb
            # pdb.set_trace()
        
            cur_index=0
            frame_indices = [i for i in range(num)]
            indices_list=[]
            while num-cur_index > self.seq_len:
                indices_list.append(frame_indices[cur_index:cur_index+self.seq_len])
                cur_index+=self.seq_len
            last_seq=frame_indices[cur_index:]
            # print(last_seq)
            for index in last_seq:
                if len(last_seq) >= self.seq_len:
                    break
                last_seq.append(index)
            

            indices_list.append(last_seq)
            imgs_list=[]
            # print(indices_list , num , img_paths  )
            for indices in indices_list:
                if len(imgs_list) > self.max_length:
                    break 
                imgs = []
                for index in indices:
                    index=int(index)
                    img_path = img_paths[index]
                    img = read_image(img_path)
                    if self.transform is not None:
                        img = self.transform(img)
                    img = img.unsqueeze(0)
                    imgs.append(img)
                imgs = torch.cat(imgs, dim=0)
                #imgs=imgs.permute(1,0,2,3)
                imgs_list.append(imgs)
            imgs_array = torch.stack(imgs_list)
            return imgs_array, pid, camid

        elif self.sample == 'dense_subset':
            """
            Sample all frames in a video into a list of clips, each clip contains seq_len frames, batch_size needs to be set to 1.
            This sampling strategy is used in test phase.
            """
            frame_indices = range(num)
            rand_end = max(0, len(frame_indices) - self.max_length - 1)
            begin_index = random.randint(0, rand_end)
            

            cur_index=begin_index
            frame_indices = [i for i in range(num)]
            indices_list=[]
            while num-cur_index > self.seq_len:
                indices_list.append(frame_indices[cur_index:cur_index+self.seq_len])
                cur_index+=self.seq_len
            last_seq=frame_indices[cur_index:]
            # print(last_seq)
            for index in last_seq:
                if len(last_seq) >= self.seq_len:
                    break
                last_seq.append(index)
            

            indices_list.append(last_seq)
            imgs_list=[]
            # print(indices_list , num , img_paths  )
            for indices in indices_list:
                if len(imgs_list) > self.max_length:
                    break 
                imgs = []
                for index in indices:
                    index=int(index)
                    img_path = img_paths[index]
                    img = read_image(img_path)
                    if self.transform is not None:
                        img = self.transform(img)
                    img = img.unsqueeze(0)
                    imgs.append(img)
                imgs = torch.cat(imgs, dim=0)
                #imgs=imgs.permute(1,0,2,3)
                imgs_list.append(imgs)
            imgs_array = torch.stack(imgs_list)
            return imgs_array, pid, camid
        
        elif self.sample == 'intelligent_random':
            # frame_indices = range(num)
            indices = []
            each = max(num//seq_len,1)
            for  i in range(seq_len):
                if i != seq_len -1:
                    indices.append(random.randint(min(i*each , num-1), min( (i+1)*each-1, num-1)) )
                else:
                    indices.append(random.randint(min(i*each , num-1), num-1) )
            print(len(indices))
            imgs = []
            for index in indices:
                index=int(index)
                img_path = img_paths[index]
                img = read_image(img_path)
                if self.transform is not None:
                    img = self.transform(img)
                img = img.unsqueeze(0)
                imgs.append(img)
            imgs = torch.cat(imgs, dim=0)
            #imgs=imgs.permute(1,0,2,3)
            return imgs, pid, camid
        else:
            raise KeyError("Unknown sample method: {}. Expected one of {}".format(self.sample, self.sample_methods))

            
class RandomErasing3(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
    def __call__(self, img):
        if random.uniform(0, 1) >= self.probability:
            return img , 0 
        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img , 1
            return img , 0         
        
        
class VideoDataset_inderase(Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset, seq_len=15, sample='evenly', transform=None , max_length=40):
        self.dataset = dataset
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform
        self.max_length = max_length
        self.erase = RandomErasing3(probability=0.5, mean=[0.485, 0.456, 0.406])

    def __len__(self):
        return len(self.dataset)
        # return 100

    def __getitem__(self, index):
        img_paths, pid, camid = self.dataset[index]
        num = len(img_paths)
        if self.sample != "intelligent":
            frame_indices = range(num)
            rand_end = max(0, len(frame_indices) - self.seq_len - 1)
            begin_index = random.randint(0, rand_end)
            end_index = min(begin_index + self.seq_len, len(frame_indices))

            indices1 = frame_indices[begin_index:end_index]
            indices = []
            for index in indices1:
                if len(indices1) >= self.seq_len:
                    break
                indices.append(index)
            indices=np.array(indices)
        else:
            # frame_indices = range(num)
            indices = []
            each = max(num//self.seq_len,1)
            for  i in range(self.seq_len):
                if i != self.seq_len -1:
                    indices.append(random.randint(min(i*each , num-1), min( (i+1)*each-1, num-1)) )
                else:
                    indices.append(random.randint(min(i*each , num-1), num-1) )
            # print(len(indices), indices, num )
        imgs = []
        labels = []
        for index in indices:
            index=int(index)
            img_path = img_paths[index]
            img = read_image(img_path)
            if self.transform is not None:
                img = self.transform(img)
            img , temp  = self.erase(img)
            labels.append(temp)
            img = img.unsqueeze(0)
            imgs.append(img)
        labels = torch.tensor(labels)
        imgs = torch.cat(imgs, dim=0)
        #imgs=imgs.permute(1,0,2,3)
        return imgs, pid, camid , labels



# In[5]:



from collections import defaultdict
import numpy as np

import torch
from torch.utils.data.sampler import Sampler


class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/data/sampler.py.
    Args:
        data_source (Dataset): dataset to sample from.
        num_instances (int): number of instances per identity.
    """
    def __init__(self, data_source, num_instances=4):
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid, _) in enumerate(data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_identities = len(self.pids)

    def __iter__(self):
        indices = torch.randperm(self.num_identities)
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            replace = False if len(t) >= self.num_instances else True
            t = np.random.choice(t, size=self.num_instances, replace=replace)
            ret.extend(t)
        return iter(ret)

    def __len__(self):
        return self.num_identities * self.num_instances
        # return 100

def evaluate(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=21):
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        orig_cmc = matches[q_idx][keep] # binary vector, positions with value 1 are correct matches
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP

def test(model, queryloader, galleryloader, pool='avg', use_gpu=True, ranks=[1, 5, 10, 20]):
    model.eval()
    qf, q_pids, q_camids = [], [], []
    with torch.no_grad():
      for batch_idx, data in enumerate(queryloader):
      # for batch_idx, (imgs, pids, camids) in enumerate(queryloader):
        imgs = data["img"].unsqueeze(1) # sequence length of 1, therefore introduce new dimension at index 1
        pids = data["pid"]
        camids = data["camid"]
        if use_gpu:
            imgs = imgs.cuda()
        imgs = Variable(imgs, volatile=True).unsqueeze(0)
        b, n, s, c, h, w = imgs.size()
        assert(b==1)
        imgs = imgs.view(b*n, s, c, h, w)
        features = model(imgs, test=True)
        features = features.view(n, -1)
        # features = torch.mean(features, 0)
        features = features.data.cpu()
        qf.append(features)
        q_pids.extend(pids)
        q_camids.extend(camids)
      qf = torch.cat(qf, 0)
      q_pids = np.asarray(q_pids)
      q_camids = np.asarray(q_camids)
      print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))
      gf, g_pids, g_camids = [], [], []
      # for batch_idx, (imgs, pids, camids) in enumerate(galleryloader):
      for batch_idx, data in enumerate(galleryloader):
      # for batch_idx, (imgs, pids, camids) in enumerate(queryloader):
        imgs = data["img"].unsqueeze(1) # sequence length of 1, therefore introduce new dimension at index 1
        pids = data["pid"]
        camids = data["camid"]
        if use_gpu:
            imgs = imgs.cuda()
        imgs = Variable(imgs, volatile=True).unsqueeze(0)
        b, n, s, c, h, w = imgs.size()
        imgs = imgs.view(b*n, s , c, h, w)
        assert(b==1)
        features = model(imgs, test=True)
        features = features.view(n, -1)
        # if pool == 'avg':
        #     features = torch.mean(features, 0)
        # else:
        #     features, _ = torch.max(features, 0)
        features = features.data.cpu()
        gf.append(features)
        g_pids.extend(pids)
        g_camids.extend(camids)
    gf = torch.cat(gf, 0)
    g_pids = np.asarray(g_pids)
    g_camids = np.asarray(g_camids)
    print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))
    print("Computing distance matrix")
    m, n = qf.size(0), gf.size(0)
    # euclidean squared distance
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) +               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.numpy()
    gf = gf.numpy()
    qf = qf.numpy()
    
    print("Original Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
    
    # print("Results ---------- {:.1%} ".format(distmat_rerank))
    print("Results ---------- ")
    
    print("mAP: {:.1%} ".format(mAP))
    print("CMC curve")
    return cmc[0], cmc[1], mAP


# In[6]:



# coding: utf-8

# In[1]:


import argparse
import os
import numpy as np
import scipy.io as sio
# from AddingLossToBestModel import *

use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")   # use CPU or GPU

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

# In[2]:


import os
import sys
import time
import numpy as np
import pandas as pd
import collections
import random
import math
## For torch lib
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as T
import torch.nn.functional as F
## For Image lib
from PIL import Image
import scipy.io
'''
For MARS,Video-based Re-ID
'''
def process_labels(labels):
    unique_id = np.unique(labels)
    id_count = len(unique_id)
    id_dict = {ID:i for i, ID in enumerate(unique_id.tolist())}
    for i in range(len(labels)):
        labels[i] = id_dict[labels[i]]
    assert len(unique_id)-1 == np.max(labels)
    return labels,id_count

class Video_train_Dataset(Dataset):
    def __init__(self,db_txt,info,transform,S=6,track_per_class=4,flip_p=0.5,delete_one_cam=False,cam_type='normal'):
        with open(db_txt,'r') as f:
            self.imgs = np.array(f.read().strip().split('\n'))
        # For info (id,track)
        if delete_one_cam == True:
            info = np.load(info)
            info[:,2],id_count = process_labels(info[:,2])
            for i in range(id_count):
                idx = np.where(info[:,2]==i)[0]
                if len(np.unique(info[idx,3])) ==1:
                    info = np.delete(info,idx,axis=0)
                    id_count -=1
            info[:,2],id_count = process_labels(info[:,2])
            #change from 625 to 619
        else:
            info = np.load(info)
            info[:,2],id_count = process_labels(info[:,2])

        self.info = []
        for i in range(len(info)):
            sample_clip = []
            F = info[i][1]-info[i][0]+1
            if F < S:
                strip = list(range(info[i][0],info[i][1]+1))+[info[i][1]]*(S-F)
                for s in range(S):
                    pool = strip[s*1:(s+1)*1]
                    sample_clip.append(list(pool))
            else:
                interval = math.ceil(F/S)
                strip = list(range(info[i][0],info[i][1]+1))+[info[i][1]]*(interval*S-F)
                for s in range(S):
                    pool = strip[s*interval:(s+1)*interval]
                    sample_clip.append(list(pool))
            self.info.append(np.array([np.array(sample_clip),info[i][2],info[i][3]]))

        self.info = np.array(self.info)
        self.transform = transform
        self.n_id = id_count
        self.n_tracklets = self.info.shape[0]
        self.flip_p = flip_p
        self.track_per_class = track_per_class
        self.cam_type = cam_type
        self.two_cam = False
        self.cross_cam = False

    def __getitem__(self,ID):
        sub_info = self.info[self.info[:,1] == ID] 

        if self.cam_type == 'normal':
            tracks_pool = list(np.random.choice(sub_info[:,0],self.track_per_class))
        elif self.cam_type == 'two_cam':
            unique_cam = np.random.permutation(np.unique(sub_info[:,2]))[:2]
            tracks_pool = list(np.random.choice(sub_info[sub_info[:,2]==unique_cam[0],0],1))+                list(np.random.choice(sub_info[sub_info[:,2]==unique_cam[1],0],1))
        elif self.cam_type == 'cross_cam':
            unique_cam = np.random.permutation(np.unique(sub_info[:,2]))
            while len(unique_cam) < self.track_per_class:
                unique_cam = np.append(unique_cam,unique_cam)
            unique_cam = unique_cam[:self.track_per_class]
            tracks_pool = []
            for i in range(self.track_per_class):
                tracks_pool += list(np.random.choice(sub_info[sub_info[:,2]==unique_cam[i],0],1))

        one_id_tracks = []
        labls_erassing=[]
        for track_pool in tracks_pool:
            idx = np.random.choice(track_pool.shape[1],track_pool.shape[0])
            number = track_pool[np.arange(len(track_pool)),idx]
         
            imgs = [self.transform(Image.open(path)) for path in self.imgs[number]]
            imgs = torch.stack(imgs,dim=0)
            labls=[0 for path in self.imgs[number]]
            
            random_p = random.random()
            if random_p  < self.flip_p:
                imgs = torch.flip(imgs,dims=[3])
            one_id_tracks.append(imgs)
            labls_erassing.append(labls)
        return torch.stack(one_id_tracks,dim=0), ID*torch.ones(self.track_per_class,dtype=torch.int64), torch.tensor(labls_erassing)

    def __len__(self):
        return self.n_id
        # return 100

def Video_train_collate_fn(data):
    if isinstance(data[0],collections.Mapping):
        t_data = [tuple(d.values()) for d in data]
        values = MARS_collate_fn(t_data)
        return {key:value  for key,value in zip(data[0].keys(),values)}
    else:
        imgs,labels,labls_erassing = zip(*data)
        imgs = torch.cat(imgs,dim=0)
        labels = torch.cat(labels,dim=0)
        labls_erassing=torch.cat(labls_erassing,dim=0)
        return imgs,labels,labls_erassing

def Get_Video_train_DataLoader(db_txt,info,transform,shuffle=True,num_workers=8,S=10,track_per_class=4,class_per_batch=4):
    dataset = Video_train_Dataset(db_txt,info,transform,S,track_per_class)
    dataloader = DataLoader(dataset,batch_size=class_per_batch,collate_fn=Video_train_collate_fn,shuffle=shuffle,worker_init_fn=lambda _:np.random.seed(),drop_last=True,num_workers=num_workers)
    return dataloader

class Video_test_Dataset(Dataset):
    def __init__(self,db_txt,info,query,transform,S=6,distractor=True):
        with open(db_txt,'r') as f:
            self.imgs = np.array(f.read().strip().split('\n'))
        # info
        info = np.load(info)
        self.info = []
        for i in range(len(info)):
            if distractor == False and info[i][2]==0:
                continue
            sample_clip = []
            F = info[i][1]-info[i][0]+1
            if F < S:
                strip = list(range(info[i][0],info[i][1]+1))+[info[i][1]]*(S-F)
                for s in range(S):
                    pool = strip[s*1:(s+1)*1]
                    sample_clip.append(list(pool))
            else:
                interval = math.ceil(F/S)
                strip = list(range(info[i][0],info[i][1]+1))+[info[i][1]]*(interval*S-F)
                for s in range(S):
                    pool = strip[s*interval:(s+1)*interval]
                    sample_clip.append(list(pool))
            self.info.append(np.array([np.array(sample_clip),info[i][2],info[i][3]]))

        self.info = np.array(self.info)
        self.transform = transform
        self.n_id = len(np.unique(self.info[:,1]))
        self.n_tracklets = self.info.shape[0]
        self.query_idx = np.load(query).reshape(-1)

        if distractor == False:
            zero = np.where(info[:,2]==0)[0]
            self.new_query = []
            for i in self.query_idx:
                if i < zero[0]:
                    self.new_query.append(i)
                elif i <= zero[-1]:
                    continue
                elif i > zero[-1]:
                    self.new_query.append(i-len(zero))
                else:
                    continue
            self.query_idx = np.array(self.new_query)
                
    def __getitem__(self,idx):
        clips = self.info[idx,0]
        imgs = [self.transform(Image.open(path)) for path in self.imgs[clips[:,0]]]
        imgs = torch.stack(imgs,dim=0)
        label = self.info[idx,1]*torch.ones(1,dtype=torch.int32)
        cam = self.info[idx,2]*torch.ones(1,dtype=torch.int32)
        return imgs,label,cam
    def __len__(self):
        return len(self.info)
        # return 100

def Video_test_collate_fn(data):
    if isinstance(data[0],collections.Mapping):
        t_data = [tuple(d.values()) for d in data]
        values = MARS_collate_fn(t_data)
        return {key:value  for key,value in zip(data[0].keys(),values)}
    else:
        imgs,label,cam= zip(*data)
        imgs = torch.cat(imgs,dim=0)
        labels = torch.cat(label,dim=0)
        cams = torch.cat(cam,dim=0)
        return imgs,labels,cams

def Get_Video_test_DataLoader(db_txt,info,query,transform,batch_size=10,shuffle=False,num_workers=8,S=6,distractor=True):
    dataset = Video_test_Dataset(db_txt,info,query,transform,S,distractor=distractor)
    dataloader = DataLoader(dataset,batch_size=batch_size,collate_fn=Video_test_collate_fn,shuffle=shuffle,worker_init_fn=lambda _:np.random.seed(),num_workers=num_workers)
    return dataloader


# In[3]:




import sys
import random
from tqdm import tqdm
import numpy as np
import math
from scipy.io import loadmat

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import Compose,ToTensor,Normalize,Resize
import torch.backends.cudnn as cudnn
cudnn.benchmark=True
import os


# In[4]:


import numpy as np
import torch
import torch.nn.functional as F
import sys
import pandas as pd
# from progressbar import ProgressBar, AnimatedMarker, Percentage
import math
from tqdm import trange


def Video_Cmc(features, ids, cams, query_idx,rank_size):
    """
    features: numpy array of shape (n, d)
    label`s: numpy array of shape (n)
    """
    # Sample query
    data = {'feature':features, 'id':ids, 'cam':cams}
    q_idx = query_idx
    g_idx = np.arange(len(ids))
    #print("g_idx ",g_idx )
    #print("q_idx:",len(q_idx))
    #print("data:",len(data['id']))
    q_data = {k:v[q_idx] for k, v in data.items()}
    g_data = {k:v[g_idx] for k, v in data.items()}
    if len(g_idx) < rank_size: rank_size = len(g_idx)

    CMC, mAP = Cmc(q_data, g_data, rank_size)

    return CMC, mAP

    
def Cmc(q_data, g_data, rank_size):
    n_query = q_data['feature'].shape[0]
    n_gallery = g_data['feature'].shape[0]

    dist = np_cdist(q_data['feature'], g_data['feature']) # Reture a n_query*n_gallery array

    cmc = np.zeros((n_query, rank_size))
    ap = np.zeros(n_query)
    
    widgets = ["I'm calculating cmc! ", AnimatedMarker(markers='←↖↑↗→↘↓↙'), ' (', Percentage(), ')']
    pbar = ProgressBar(widgets=widgets, max_value=n_query)
    for k in range(n_query):
        good_idx = np.where((q_data['id'][k]==g_data['id']) & (q_data['cam'][k]!=g_data['cam']))[0]
        junk_mask1 = (g_data['id'] == -1)
        junk_mask2 = (q_data['id'][k]==g_data['id']) & (q_data['cam'][k]==g_data['cam'])
        junk_idx = np.where(junk_mask1 | junk_mask2)[0]
        score = dist[k, :]
        sort_idx = np.argsort(score)
        sort_idx = sort_idx[:rank_size]

        ap[k], cmc[k, :] = Compute_AP(good_idx, junk_idx, sort_idx)
        pbar.update(k)
    pbar.finish()
    CMC = np.mean(cmc, axis=0)
    mAP = np.mean(ap)
    return CMC, mAP

def Compute_AP(good_image, junk_image, index):
    cmc = np.zeros((len(index),))
    ngood = len(good_image)

    old_recall = 0
    old_precision = 1.
    ap = 0
    intersect_size = 0
    j = 0
    good_now = 0
    njunk = 0
    for n in range(len(index)):
        flag = 0
        if np.any(good_image == index[n]):
            cmc[n-njunk:] = 1
            flag = 1 # good image
            good_now += 1
        if np.any(junk_image == index[n]):
            njunk += 1
            continue # junk image
        
        if flag == 1:
            intersect_size += 1
        recall = intersect_size/ngood
        precision = intersect_size/(j+1)
        ap += (recall-old_recall) * (old_precision+precision) / 2
        old_recall = recall
        old_precision = precision
        j += 1
       
        if good_now == ngood:
            return ap, cmc
    return ap, cmc


def cdist(feat1, feat2):
    """Cosine distance"""
    feat1 = torch.FloatTensor(feat1)#.cuda()
    feat2 = torch.FloatTensor(feat2)#.cuda()
    feat1 = torch.nn.functional.normalize(feat1, dim=1)
    feat2 = torch.nn.functional.normalize(feat2, dim=1).transpose(0, 1)
    dist = -1 * torch.mm(feat1, feat2)
    return dist.cpu().numpy()

def np_cdist(feat1, feat2):
    """Cosine distance"""
    feat1_u = feat1 / np.linalg.norm(feat1, axis=1, keepdims=True) # n * d -> n
    feat2_u = feat2 / np.linalg.norm(feat2, axis=1, keepdims=True) # n * d -> n
    return -1 * np.dot(feat1_u, feat2_u.T)

def np_norm_eudist(feat1,feat2):
    feat1_u = feat1 / np.linalg.norm(feat1, axis=1, keepdims=True) # n * d -> n
    feat2_u = feat2 / np.linalg.norm(feat2, axis=1, keepdims=True) # n * d -> n
    feat1_sq = np.sum(feat1_M * feat1, axis=1)
    feat2_sq = np.sum(feat2_M * feat2, axis=1)
    return np.sqrt(feat1_sq.reshape(-1,1) + feat2_sq.reshape(1,-1) - 2*np.dot(feat1_M, feat2.T)+ 1e-12)
    

def sqdist(feat1, feat2, M=None):
    """Mahanalobis/Euclidean distance"""
    if M is None: M = np.eye(feat1.shape[1])
    feat1_M = np.dot(feat1, M)
    feat2_M = np.dot(feat2, M)
    feat1_sq = np.sum(feat1_M * feat1, axis=1)
    feat2_sq = np.sum(feat2_M * feat2, axis=1)
    return feat1_sq.reshape(-1,1) + feat2_sq.reshape(1,-1) - 2*np.dot(feat1_M, feat2.T)


# In[5]:


def normalize_rank(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist_rank(x, y,embd):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    #print("x=",x.shape)
    
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    #--------------------------------------------
    """
    ####distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) 
    #+torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        
    batch_size = n
    num_classes=embd.size(0)
    #diff=(x.unsqueeze(1) - y.unsqueeze(0))
    diff=x-y
    
   
    
    #yy = y.sum(1, keepdim=True).expand(num_classes, batch_size).t()
    #print("yy",yy.shape)
    #dist = xx + yy
    #dist.addmm_(1, -2, x, y.t())
    #dist = dist.clamp(min=1e-12).sqrt()
    
    
    dist = diff.sum(dim=1, keepdim=True).expand(batch_size, num_classes) 
    
    #torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
    #print("embd in RLL",embd.shape)  
    embd=embd.sum(dim=1, keepdim=True).expand( num_classes,batch_size).t()
    

    
    
    #print("dist  **",dist.shape)

    #print("embd**",embd.shape)
        
    #diff1=dist.view(-1,2048)
    d1=torch.mm(dist, embd.t())#,jacob.shape,diff1.shape)
    d2=torch.mm(embd, dist.t())
        
    dist2=torch.mm(d1,d2)
    #print("dist2",dist2.shape)
    #dist2=dist2.view(n,n,-1)
    #dist2=((dist2).sum(2)+1e-12)
    #dist2= dist2.clamp(min=1e-12)
    dist=dist2
    """
    #--------------------------------------------    
    return dist

class OSM_CAA_Loss(nn.Module):
    def __init__(self, alpha=1.2, l=0.5, use_gpu=True , osm_sigma=0.8):
        super(OSM_CAA_Loss, self).__init__()
        self.use_gpu = use_gpu
        self.alpha = alpha # margin of weighted contrastive loss, as mentioned in the paper 
        self.l = l #  hyperparameter controlling weights of positive set and the negative set  
        # I haven't been able to figure out the use of \sigma CAA 0.18 
        self.osm_sigma = osm_sigma  #\sigma OSM (0.8) as mentioned in paper
        
    def forward(self, x, labels , embd):
        '''
        x : feature vector : (n x d)
        labels : (n,)
        embd : Fully Connected weights of classification layer (dxC), C is the number of classes: represents the vectors for class
        '''
        
        x = nn.functional.normalize(x, p=2, dim=1) # normalize the features
        n = x.size(0)
        dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(n, n)
        #print("befor destance",dist.shape)
        dist = dist + dist.t()
        #print("after adding with transpos destance",dist.shape)
        dist.addmm_(1, -2, x, x.t())
        #print("after addmm",dist.shape)
        dist = dist.clamp(min=1e-12).sqrt() 
        #print("first distanc",dist.shape)
        #___________________________________________________
        
        diff=(x.unsqueeze(1) - x.unsqueeze(0))
        #print("diff",diff.shape)
        
        
        diff1=diff.view(-1,2048)
        d1=torch.mm(diff1, embd)#,jacob.shape,diff1.shape)
        d2=torch.mm(embd.t(), diff1.t())
        #print("d1=",d1.shape)
        #print("d2=",d2.shape)
        dist2=torch.mm(d1,d2)
        dist2=dist2.view(n,n,-1)
        dist2=((dist2).sum(2)+1e-12)
        dist2= dist2.clamp(min=1e-12)
        #print("dis2t",dist2.shape)
        #print("dist2",dist2)
        
        #print("rieman distanc",dist2.shape)
        dist=dist2
        #___________________________________________________
        S = torch.exp( -1.0 *  torch.pow(dist, 2)  / (self.osm_sigma * self.osm_sigma) )
        S_ = torch.clamp( self.alpha - dist , min=1e-12)  #max (0, self.alpha - dij ) 
        p_mask = labels.expand(n, n).eq(labels.expand(n, n).t())
        p_mask = p_mask.float()
        n_mask = 1- p_mask
        S = S * p_mask.float()
        S = S + S_ * n_mask.float()
        embd = nn.functional.normalize(embd, p=2, dim=0) 
        #print("class embding",embd.size())
        
        denominator = torch.exp(torch.mm(x , embd)) 
        A = [] 
        for i in range(n):
            a_i = denominator[i][labels[i]] / torch.sum(denominator[i])
            A.append(a_i)
        atten_class = torch.stack(A)
        
        #print("class attention size",atten_class.size())
        #print("dist",dist.size())
        A = torch.min(atten_class.expand(n,n) , atten_class.view(-1,1).expand(n,n) ) # pairwise minimum of attention weights 
        W = S * A 
        W_P = W * p_mask.float()
        W_N = W * n_mask.float()
        if self.use_gpu:
           W_P = W_P * (1 - torch.eye(n, n).float().cuda()) #dist between (xi,xi) not necessarily 0, avoiding precision error
           W_N = W_N * (1 - torch.eye(n, n).float().cuda())
        else:
           W_P = W_P * (1 - torch.eye(n, n).float())
           W_N = W_N * (1 - torch.eye(n, n).float())
        L_P = 1.0/2 * torch.sum(W_P * torch.pow(dist, 2)) / torch.sum(W_P)
        L_N = 1.0/2 * torch.sum(W_N * torch.pow(S_ , 2)) / torch.sum(W_N)
        L = (1- self.l) * L_P + self.l * L_N
        return L 



def rank_loss(dist_mat, labels, margin,alpha,tval):
    """
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      
    """
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    total_loss = 0.0
    for ind in range(N):
        is_pos = labels.eq(labels[ind])
        is_pos[ind] = 0
        is_neg = labels.ne(labels[ind])
        
        dist_ap = dist_mat[ind][is_pos]
        dist_an = dist_mat[ind][is_neg]
        
        ap_is_pos = torch.clamp(torch.add(dist_ap,margin-alpha),min=0.0)
        ap_pos_num = ap_is_pos.size(0) +1e-5
        ap_pos_val_sum = torch.sum(ap_is_pos)
        loss_ap = torch.div(ap_pos_val_sum,float(ap_pos_num))

        an_is_pos = torch.lt(dist_an,alpha)
        an_less_alpha = dist_an[an_is_pos]
        an_weight = torch.exp(tval*(-1*an_less_alpha+alpha))
        an_weight_sum = torch.sum(an_weight) +1e-5
        an_dist_lm = alpha - an_less_alpha
        an_ln_sum = torch.sum(torch.mul(an_dist_lm,an_weight))
        loss_an = torch.div(an_ln_sum,an_weight_sum)
        
        total_loss = total_loss+loss_ap+loss_an
    total_loss = total_loss*1.0/N
    return total_loss

class RankedLoss(object):
    "Ranked_List_Loss_for_Deep_Metric_Learning_CVPR_2019_paper"
    #RankedLoss(1.3,2.0,1.)
    def __init__(self, margin=None, alpha=None, tval=None):
        self.margin = margin
        self.alpha = alpha
        self.tval = tval
        
    def __call__(self, global_feat, labels,embd, normalize_feature=True):
        if normalize_feature:
            global_feat = normalize_rank(global_feat, axis=-1)
        dist_mat = euclidean_dist_rank(global_feat, global_feat,embd)
        total_loss = rank_loss(dist_mat,labels,self.margin,self.alpha,self.tval)
        
        return total_loss
    
class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss
    
def normalize_rank(x, axis=-1):
        x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
        return x
"""Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
"""



    
class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss
    

class TripletLoss(nn.Module):

    def __init__(self, margin=0, batch_hard=False,dim=2048):
        super(TripletLoss, self).__init__()
        self.batch_hard = batch_hard
        if isinstance(margin, float) or margin == 'soft':
            self.margin = margin
        else:
            raise NotImplementedError(
                'The margin {} is not recognized in TripletLoss()'.format(margin))

    def forward(self, feat, id=None, pos_mask=None, neg_mask=None, mode='id',dis_func='eu',n_dis=0):

        if dis_func == 'cdist':
            feat = feat / feat.norm(p=2,dim=1,keepdim=True)
            dist = self.cdist(feat, feat)
        elif dis_func == 'eu':
            dist = self.cdist(feat, feat)

        if mode == 'id':
            if id is None:
                 raise RuntimeError('foward is in id mode, please input id!')
            else:
                 identity_mask = torch.eye(feat.size(0)).byte()
                 identity_mask = identity_mask.cuda() if id.is_cuda else identity_mask
                 same_id_mask = torch.eq(id.unsqueeze(1), id.unsqueeze(0))
                 negative_mask = same_id_mask ^ 1
                 positive_mask = same_id_mask ^ identity_mask
        elif mode == 'mask':
            if pos_mask is None or neg_mask is None:
                 raise RuntimeError('foward is in mask mode, please input pos_mask & neg_mask!')
            else:
                 positive_mask = pos_mask
                 same_id_mask = neg_mask ^ 1
                 negative_mask = neg_mask
        else:
            raise ValueError('unrecognized mode')
        
        if self.batch_hard:
            if n_dis != 0:
                img_dist = dist[:-n_dis,:-n_dis]
                max_positive = (img_dist * positive_mask[:-n_dis,:-n_dis].float()).max(1)[0]
                min_negative = (img_dist + 1e5*same_id_mask[:-n_dis,:-n_dis].float()).min(1)[0]
                dis_min_negative = dist[:-n_dis,-n_dis:].min(1)[0]
                z_origin = max_positive - min_negative
                # z_dis = max_positive - dis_min_negative
            else:
                max_positive = (dist * positive_mask.float()).max(1)[0]
                min_negative = (dist + 1e5*same_id_mask.float()).min(1)[0]
                z = max_positive - min_negative
        else:
            pos = positive_mask.topk(k=1, dim=1)[1].view(-1,1)
            positive = torch.gather(dist, dim=1, index=pos)
            pos = negative_mask.topk(k=1, dim=1)[1].view(-1,1)
            negative = torch.gather(dist, dim=1, index=pos)
            z = positive - negative

        if isinstance(self.margin, float):
            b_loss = torch.clamp(z + self.margin, min=0)
        elif self.margin == 'soft':
            if n_dis != 0:
                b_loss = torch.log(1+torch.exp(z_origin))+ -0.5* dis_min_negative# + torch.log(1+torch.exp(z_dis))
            else:
                b_loss = torch.log(1 + torch.exp(z))
        else:
            raise NotImplementedError("How do you even get here!")
      
        return b_loss
            
    def cdist(self, a, b):
        '''
        Returns euclidean distance between a and b
        
        Args:
             a (2D Tensor): A batch of vectors shaped (B1, D)
             b (2D Tensor): A batch of vectors shaped (B2, D)
        Returns:
             A matrix of all pairwise distance between all vectors in a and b,
             will be shape of (B1, B2)
        '''
        diff = a.unsqueeze(1) - b.unsqueeze(0)
        return ((diff**2).sum(2)+1e-12).sqrt()


# In[ ]:



import parser

import sys
import random
from tqdm import tqdm
import numpy as np
import math


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import Compose,ToTensor,Normalize,Resize
import torch.backends.cudnn as cudnn
cudnn.benchmark=True
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'



def validation(network,dataloader):
    network.eval()
    pbar = tqdm(total=len(dataloader),ncols=100,leave=True)
    pbar.set_description('Inference')
    gallery_features = []
    gallery_labels = []
    gallery_cams = []
    with torch.no_grad():
        for c,data in enumerate(dataloader):
            seqs = data[0].cuda()
            label = data[1]
            cams = data[2]
            
            feat = network(seqs)#.cpu().numpy() #[xx,128]
            
            gallery_features.append(feat.cpu())
            gallery_labels.append(label)
            gallery_cams.append(cams)
            pbar.update(1)
    pbar.close()

    gallery_features = torch.cat(gallery_features,dim=0).numpy()
    gallery_labels = torch.cat(gallery_labels,dim=0).numpy()
    gallery_cams = torch.cat(gallery_cams,dim=0).numpy()

    Cmc,mAP = Video_Cmc(gallery_features,gallery_labels,gallery_cams,dataloader.dataset.query_idx,10000)
    network.train()

    return Cmc[0],mAP



from torch.autograd import Variable
from main import build_datamanager, check_cfg, reset_config
from torchreid.utils import set_random_seed
from default_config import get_default_config
from model_icpr import Baseline
import argparse
from torchreid.models import build_model

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    '--config-file', type=str, default='', help='path to config file'
)
parser.add_argument(
    '-s',
    '--sources',
    type=str,
    nargs='+',
    help='source datasets (delimited by space)'
)
parser.add_argument(
    '-t',
    '--targets',
    type=str,
    nargs='+',
    help='target datasets (delimited by space)'
)
parser.add_argument(
    '--transforms', type=str, nargs='+', help='data augmentation'
)
parser.add_argument(
    '--root', type=str, default='', help='path to data root'
)
parser.add_argument(
    '--save_path', type=str, default='best.pth', help='name to save best checkpoint'
)
parser.add_argument(
    '--model_name', type=str, default='resnet50_ibn_a', help='name of model to use'
)
parser.add_argument(
    '--ncc', type=bool, default=False, help='are we on ncc'
)
parser.add_argument(
    '--epochs', type=int, default=120, help='number of epochs'
)
parser.add_argument(
    '--model_path', type=str, default='../resnet50_ibn_a.pth.tar', help='name of model to use'
)
parser.add_argument(
    'opts',
    default=None,
    nargs=argparse.REMAINDER,
    help='Modify config options using the command-line'
)
args = parser.parse_args()
   
if __name__ == '__main__':

                               
    train_transform = T.Compose([
            T.Resize ([384,256]),   
            T.RandomHorizontalFlip(p=0.5),
            T.Pad(10),
            T.RandomCrop([224, 112]),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
            RandomErasing(probability=0.5 ,mean=[0.485, 0.456, 0.406])
        ])
    
    test_transform = Compose([Resize((224, 112)),ToTensor(),Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])
    
    print('Start dataloader...')
    """
    train_dataloader = Get_Video_train_DataLoader('/home2/zwjx97/STE-NVAN-master/MARS/train_path.txt', '/home2/zwjx97/STE-NVAN-master/MARS/train_info.npy', train_transform, shuffle=True,num_workers=4,S=8,track_per_class=4,class_per_batch=8)
    num_class = train_dataloader.dataset.n_id# set transformation (H flip is inside dataset)
    test_dataloader = Get_Video_test_DataLoader('/home2/zwjx97/STE-NVAN-master/MARS/test_path.txt','/home2/zwjx97/STE-NVAN-master/MARS/test_info.npy','/home2/zwjx97/STE-NVAN-master/MARS/query_IDX.npy',test_transform,batch_size=50,                                                 shuffle=False,num_workers=4,S=8,distractor=True)
    """
    print('End dataloader...\n')
    
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    transform_train = transforms.Compose([
            transforms.Resize((224, 112), interpolation=3),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Pad(10),
            T.RandomCrop([224, 112]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            RandomErasing(probability=0.5 ,mean=[0.485, 0.456, 0.406])
        ])

    transform_test = transforms.Compose([
       transforms.Resize((224, 112), interpolation=3),
       transforms.ToTensor(),
       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


    
    
    
#     dataset = Mars()
#
#     pin_memory = True
#     trainloader = DataLoader(
#     VideoDataset_inderase(dataset.train, seq_len=8, sample='intelligent',transform=transform_train),
#     sampler=RandomIdentitySampler(dataset.train, num_instances=4),
#     batch_size=2, num_workers=4,
#     pin_memory=pin_memory, drop_last=True,
# )
#     queryloader = DataLoader(
#     VideoDataset(dataset.query, seq_len=8, sample='dense', transform=transform_test),
#     batch_size=1, shuffle=False, num_workers=0,
#     pin_memory=pin_memory, drop_last=False,
# )
#
#     galleryloader = DataLoader(
#     VideoDataset(dataset.gallery, seq_len=8, sample='dense', transform=transform_test),
#     batch_size=1, shuffle=False, num_workers=0,
#     pin_memory=pin_memory, drop_last=False,
# )
    cfg = get_default_config()
    cfg.use_gpu = not torch.cuda.is_available()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    reset_config(cfg, args)
    cfg.merge_from_list(args.opts)
    set_random_seed(cfg.train.seed)
    check_cfg(cfg)
    datamanager = build_datamanager(cfg)
    trainloader = datamanager.train_loader
    if "temporallynear" in args.config_file:
        queryloader = datamanager.test_loader["temporallynear"]['query']
        galleryloader = datamanager.test_loader["temporallynear"]['gallery']
    else:
        queryloader = datamanager.test_loader["bigtosmall"]['query']
        galleryloader = datamanager.test_loader["bigtosmall"]['gallery']

    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    print('End dataloader...\n')
    
    # 1. Criterion
    criterion_triplet = TripletLoss('soft',True)
    
    #criterion_ID = nn.CrossEntropyLoss().cuda()
    #criterion_RLL=RankedLoss(1.3,2.0,1.0)
    criterion_RLL=RankedLoss(1.3,2.0,1.)
     # 2. Optimizer
    backbone = build_model(
        name=cfg.model.name,
        num_classes=datamanager.num_train_pids,
        loss=cfg.loss.name,
        pretrained=cfg.model.pretrained,
        use_gpu=cfg.use_gpu
    )
    # model = Baseline(model_name = 'resnet50_ibn_a', num_classes=625, last_stride=1, model_path='../resnet50_ibn_a.pth.tar', stn_flag='no', pretrain_choice='imagenet').to(device)
    model = Baseline(model_name = args.model_name, num_classes=625, last_stride=1, model_path=args.model_path, stn_flag='no', pretrain_choice='none', backbone=backbone).to(device)

    #optimizer = optim.Adam(model.parameters(),lr = 0.0001,weight_decay = 1e-5)
    base_lr = 0.00035 #0.0002
    momentum = 0.9
    weight_decay = 5e-4
    gamma = 0.1
    
    
    
    sigma =  0.9047814732165316
    alpha =  2.8436551583293728
    l =  0.5873389293193368
    
    margin =  4.4132437486402204e-05
    beta_ratio =  .3
    gamma =  0.3282654557691594
    weight_decay = 0.0005
    
   
    optimizer = make_optimizer(model)
    #scheduler = WarmupMultiStepLR(optimizer, (30, 55))
    scheduler = WarmupMultiStepLR(optimizer, milestones=[40, 70], gamma=gamma, warmup_factor=0.01, warmup_iters=10)
    criterion_osm_caa = OSM_CAA_Loss(alpha=alpha , l=l , osm_sigma=sigma )
    

    
    
    # center_criterion = CenterLoss(use_gpu=True, num_classes=512, feat_dim=512)
    center_criterion = CenterLoss(use_gpu=True)
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=0.5)

    id_loss_list = []
    trip_loss_list = []
    track_id_loss_list = []
    lr_step_size=50
    best_cmc = 0
    best_map = 0
    best_cmc5 = 0
    for e in range(args.epochs):
        print('Epoch',e)
        
        scheduler.step()
        if (e+1)%10 == 0:
            #cmc,map = validation(model,test_dataloader)
            cmc1, cmc5 ,map = test(model, queryloader, galleryloader)
            print('CMC1: %.4f, CMC5: %.4f, mAP : %.4f'%(cmc1,cmc5,map))
            #f = open(os.path.join(args.ckpt,args.log_path),'a')
            #f.write('epoch %d, rank-1 %f , mAP %f\n'%(e,cmc,map))
            #if args.frame_id_loss:
            #    f.write('Frame ID loss : %r\n'%(id_loss_list))
            #if args.track_id_loss:
                #f.write('Track ID loss : %r\n'%(track_id_loss_list))
            #f.write('Trip Loss : %r\n'%(trip_loss_list))

            id_loss_list = []
            trip_RLL_list = []
            track_id_loss_list = []
            if cmc1 >= best_cmc:
                if args.ncc:
                    torch.save(model.state_dict(),os.path.join("/home2/lgfm95/reid/marschkpt", args.save_path))
                else:
                    torch.save(model.state_dict(),os.path.join("/data/reid/marschkpt", args.save_path))
                best_cmc = cmc1
                best_cmc5 = cmc5
                best_map = map
                #f.write('best\n')
            #f.close()
        # Training
        total_id_loss = 0 
        total_RLL_loss = 0 
        total_track_id_loss = 0
        pbar = tqdm(total=len(trainloader),ncols=100,leave=True)
        model.train()
        # for batch_idx, (imgs, pids, camids, labels2) in enumerate(trainloader):
            # labels2 refers to whether the image has been erased or not (0,1)
        for batch_idx, data in enumerate(trainloader):
            imgs = data["img"].unsqueeze(1) # sequence length of 1, therefore introduce new dimension at index 1
            pids = data["pid"]
            camids = data["camid"]
            labels2 = data["erase_label"]
            # labels2 = torch.zeros((len(imgs),1))
            # raise AttributeError(imgs.shape, pids.shape, labels2.shape)
        # print(batch_idx)
            criterion_ID = CrossEntropyLabelSmooth(len(pids)).cuda()
            seqs, labels = imgs.cuda(), pids.cuda()
            # 32,4,3,224,112 , 32
            #print("seqs",seqs.shape)
            #print("labels",labels.shape)
           
            classf,feat,a_vals  = model(seqs)
            #print("classf",classf.shape)
            #print("feat",feat.shape)
            labels2=labels2.cuda()
            attn_noise  = a_vals * labels2
            attn_loss = attn_noise.sum(1).mean()
            
            
            
            cetner_loss_weight = 0.0005
            
            pool_feat = feat
            pool_output = classf
            

            id_loss = criterion_ID(classf,labels)
            #print("id_loss",id_loss)
            center= center_criterion(feat,labels)
            #print("center",center)
            #osm_caa_loss = criterion_osm_caa(feat, labels, center_criterion.centers.t())
            RLL=criterion_RLL(pool_feat,labels, center_criterion.centers.t())
            #print("osm_caa_loss",osm_caa_loss)
            #print("RLL",RLL)
            total_id_loss += id_loss.item()
            total_RLL_loss+=RLL.item()
            coeff = 1
            #loss = ide_loss + (1-beta_ratio )* triplet_loss  + center_loss * cetner_loss_weight + beta_ratio * osm_caa_loss

            total_loss =   coeff*id_loss+center * cetner_loss_weight+ RLL+attn_loss #cetner_loss_weight * center+ RLL  
            
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            total_loss.backward()
            optimizer.step()
            for param in center_criterion.parameters():
                param.grad.data *= (1./cetner_loss_weight)
            optimizer_center.step()
            pbar.update(1)
        pbar.close()
        
        #if lr_step_size !=0:
        #scheduler.step()
        print("total_loss",total_loss)
        avg_id_loss = '%.4f'%(total_id_loss/len(trainloader))
        avg_RLL_loss = '%.4f'%(total_RLL_loss/len(trainloader))
        avg_track_id_loss = '%.4f'%(total_track_id_loss/len(trainloader))
        print('RLL : %s , ID : %s , Track_ID : %s'%(avg_RLL_loss,avg_id_loss,avg_track_id_loss))
        id_loss_list.append(avg_id_loss)
        trip_loss_list.append(avg_RLL_loss)
        track_id_loss_list.append(avg_track_id_loss)
    print('Best CMC1: %.4f, Best CMC5: %.4f, Best mAP : %.4f'%(best_cmc,best_cmc5, best_map))


 

