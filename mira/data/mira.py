import random
from tqdm import tqdm
import pandas as pd
from decord import VideoReader, cpu

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import os


class Mira(Dataset):

    def __init__(self,
                 meta_path,
                 webvid_data_dir=None,
                 subsample=None,
                 video_length=16,
                 resolution=[256, 512],
                 frame_stride=1,
                 spatial_transform=None,
                 crop_resolution=None,
                 fps_max=None,
                 load_raw_resolution=False,
                 fixed_fps=None,
                 fps_cond=False,
                 max_framestride=8,
                 Webvid_meta=None, webvid_prob=0.5,
                 root=None
                 ):
        self.meta_path = meta_path
        self.webvid_data_dir = webvid_data_dir
        self.Webvid_meta = Webvid_meta

        self.subsample = subsample
        self.video_length = video_length
        self.webvid_prob = webvid_prob
        self.resolution = [resolution, resolution] if isinstance(resolution, int) else resolution
        self.fps_max = fps_max
        self.frame_stride = frame_stride
        self.fixed_fps = fixed_fps
        self.load_raw_resolution = load_raw_resolution
        self.fps_cond = fps_cond
        self.root = root
        self._load_metadata()
        self.max_framestride = max_framestride
        if spatial_transform is not None:
            if spatial_transform == "random_crop":
                self.spatial_transform = transforms.RandomCrop(crop_resolution)
            elif spatial_transform == "resize_center_crop":
                # assert(self.resolution[0] == self.resolution[1])
                self.spatial_transform = transforms.Compose([
                    transforms.Resize(min(self.resolution), antialias=True),
                    transforms.CenterCrop(self.resolution),
                ])
            else:
                raise NotImplementedError
        else:
            self.spatial_transform = None

    def _load_metadata(self):

        metadata = pd.read_csv(self.meta_path)
        if self.subsample is not None:
            metadata = metadata.sample(self.subsample, random_state=0)
        self.metadata = metadata
        # self.metadata.dropna(inplace=True)

        ### Load Webvid data
        self.webvid_metadata = None
        if self.Webvid_meta:
            webvid_metadata = pd.read_csv(self.Webvid_meta)
            if self.subsample is not None:
                webvid_metadata = webvid_metadata.sample(self.subsample, random_state=0)
            self.webvid_metadata = webvid_metadata
            self.webvid_metadata.dropna(inplace=True)

    def _get_video_path(self, sample):
        rel_video_fp = sample['index']
        full_video_fp = rel_video_fp
        if self.root:
            full_video_fp = os.path.join(self.root, rel_video_fp)
        return full_video_fp, rel_video_fp

    def _get_webvid_video_path(self, sample):
        rel_video_fp = sample['filename']
        full_video_fp = os.path.join(self.webvid_data_dir, rel_video_fp)
        return full_video_fp, rel_video_fp

    def __getitem__(self, index):
        ## get frames until success
        while True:
            if self.webvid_metadata is None or random.random() > self.webvid_prob:
                if index % len(self) < len(self.metadata):
                    index = index % len(self)
                    sample = self.metadata.iloc[index]
                    video_path, rel_fp = self._get_video_path(sample)
                    caption = "{}. {}".format(sample['tag'] , sample['short_caption']).replace('nan', '')

                elif index % len(self) < 2 * len(self.metadata):
                    index = index % len(self) - len(self.metadata)
                    sample = self.metadata.iloc[index]
                    video_path, rel_fp = self._get_video_path(sample)
                    caption = "{}. {}".format(sample['tag'], sample['dense_caption']).replace('nan', '')
                else:
                    index = index % len(self) - 2 * len(self.metadata)
                    sample = self.metadata.iloc[index]
                    video_path, rel_fp = self._get_video_path(sample)
                    caption = "Tag: {}. Dense caption {}.  Main object: {}. Background: {}.  Style: {}. Camera: {}. ".format(
                        sample['tag'], sample['dense_caption'],
                        sample['main_object_caption'],
                        sample['background_caption'],
                        sample['style_caption'],
                        sample['camera_caption']).replace('nan', '')
            else:
                index = index % len(self.webvid_metadata)
                sample = self.webvid_metadata.iloc[index]
                video_path, rel_fp = self._get_webvid_video_path(sample)
                caption = sample['caption']
            try:
                if self.load_raw_resolution:
                    video_reader = VideoReader(video_path, ctx=cpu(0))
                else:
                    video_reader = VideoReader(video_path, ctx=cpu(0), width=self.resolution[1],
                                               height=self.resolution[0])
                if len(video_reader) < self.video_length:
                    print(f"video length ({len(video_reader)}) is smaller than target length({self.video_length})")
                    index += 1
                    continue
                else:
                    pass
            except:
                index += 1
                print(f"Load video failed! path = {video_path}")
                continue

            fps_ori = video_reader.get_avg_fps()
            if self.fixed_fps is not None:
                frame_stride = int(self.frame_stride * (1.0 * fps_ori / self.fixed_fps))
            elif self.fps_cond:
                frame_stride = random.randint(1, self.max_framestride)
            else:
                frame_stride = self.frame_stride
            ## to avoid extreme cases when fixed_fps is used
            frame_stride = max(frame_stride, 1)

            ## get valid range (adapting case by case)
            required_frame_num = frame_stride * (self.video_length - 1) + 1
            frame_num = len(video_reader)
            if frame_num < required_frame_num:
                ## drop extra samples if fixed fps is required
                if self.fixed_fps is not None and frame_num < required_frame_num * 0.5:
                    index += 1
                    continue
                else:
                    frame_stride = frame_num // self.video_length
                    required_frame_num = frame_stride * (self.video_length - 1) + 1
            else:
                pass

            ## select a random clip
            random_range = frame_num - required_frame_num
            start_idx = random.randint(0, random_range) if random_range > 0 else 0
            frame_indices = [start_idx + frame_stride * i for i in range(self.video_length)]
            try:
                frames = video_reader.get_batch(frame_indices)
                break
            except:
                print(
                    f"Get frames failed! path = {video_path}; [max_ind vs frame_total:{max(frame_indices)} / {frame_num}]")
                index += 1
                continue

        ## process data
        assert (frames.shape[0] == self.video_length), f'{len(frames)}, self.video_length={self.video_length}'
        frames = torch.tensor(frames.asnumpy()).permute(3, 0, 1, 2).float()  # [t,h,w,c] -> [c,t,h,w]
        if self.spatial_transform is not None:
            frames = self.spatial_transform(frames)
        if self.resolution is not None:
            assert (frames.shape[2], frames.shape[3]) == (
                self.resolution[0], self.resolution[1]), f'frames={frames.shape}, self.resolution={self.resolution}'
        frames = (frames / 255 - 0.5) * 2
        fps_clip = fps_ori // frame_stride
        if self.fps_max is not None and fps_clip > self.fps_max:
            fps_clip = self.fps_max
        data = {'video': frames, 'caption': caption, 'path': video_path, 'fps': fps_clip, 'frame_stride': frame_stride}
        return data

    def __len__(self):
        return len(self.metadata) * 3  # short dense full


if __name__ == "__main__":

    meta_path = "/group/40033/public_datasets/miradata/caption/caption_before_03M22D15H.csv"

    dataset = Mira(meta_path,
                   subsample=None,
                   video_length=16,
                   resolution=None,
                   frame_stride=1,
                   spatial_transform=None,
                   crop_resolution=None,
                   fps_max=None,
                   load_raw_resolution=True
                   )

    dataloader = DataLoader(dataset,
                            batch_size=1,
                            num_workers=0,
                            shuffle=True)
    print(f'dataset len={dataset.__len__()}')
    i = 0
    total_fps = set()
    res1 = [336, 596]
    res2 = [316, 600]
    n_videos_res1 = 0
    n_videos_res2 = 0
    n_videos_misc = 0
    other_res = []

    for i, batch in tqdm(enumerate(dataloader), desc="Data Batch"):
        # pass
        print(f"video={batch['video'].shape}, fps={batch['fps']}")
        total_fps.add(batch['fps'].item())
        if batch['video'].shape[-2] == res1[0] and batch['video'].shape[-1] == res1[1]:
            n_videos_res1 += 1
        elif batch['video'].shape[-2] == res2[0] and batch['video'].shape[-1] == res2[1]:
            n_videos_res2 += 1
        else:
            n_videos_misc += 1
            other_res.append(list(batch['video'].shape[3:]))

        if (i + 1) == 1000:
            break

    print(f'total videos = {i}')
    print('======== total_fps ========')
    print(total_fps)
    print('======== resolution ========')
    print(f'res1 {res1}: n_videos = {n_videos_res1}')
    print(f'res2 {res2}: n_videos = {n_videos_res2}')
    print(f'other resolution: n_videos = {n_videos_misc}')
    print(f'other resolutions: {other_res}')
