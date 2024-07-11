
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import omegaconf
import webdataset as wds
import io

import os
import glob
from torchvision import transforms
import numpy as np 
from PIL import Image
from itertools import cycle


def create_webdataset(
    urls,
    image_transform,
    enable_text=True,
    enable_image=True,
    image_key="jpg",
    caption_key="txt",
    enable_metadata=False,
    cache_path=None,
    ddp=False
):
    """Create a WebDataset reader, it can read a webdataset of image, text and json"""
    import clip  # pylint: disable=import-outside-toplevel
    import webdataset as wds  # pylint: disable=import-outside-toplevel

    if ddp:
        dataset = wds.WebDataset(urls, cache_dir=cache_path, 
                                cache_size=10 ** 10, 
                                nodesplitter = wds.shardlists.split_by_node,
                                handler=wds.handlers.warn_and_continue)
    else:
        dataset = wds.WebDataset(urls, cache_dir=cache_path, 
                                cache_size=10 ** 10, 
                                handler=wds.handlers.warn_and_continue)

    tokenizer = lambda text: clip.tokenize([text], truncate=True)[0]

    def filter_dataset(item):
        if enable_text and caption_key not in item:
            return False
        if enable_image and image_key not in item:
            return False
        # if enable_metadata and "json" not in item:
        #     return False
        return True

    # filtered_dataset = dataset.select(filter_dataset)
    filtered_dataset = dataset

    def preprocess_dataset(item):
        output = {}
        if enable_image:
            image_data = item[image_key]
            image = Image.open(io.BytesIO(image_data))
            image_tensor = image_transform(image)
            output["image_filename"] = item["__key__"]
            output["video"] = image_tensor.unsqueeze(1)

        if enable_text:
            text = item[caption_key]
            caption = text.decode("utf-8")
            # tokenized_text = tokenizer(caption)
            # output["text_tokens"] = tokenized_text
            output["caption"] = caption

        # output['fps'] = None

        return output

    transformed_dataset = filtered_dataset.map(preprocess_dataset, handler=wds.handlers.warn_and_continue)
    return transformed_dataset


def dataset_to_dataloader(dataset, batch_size, num_prepro_workers, input_format, ddp=False):
    """Create a pytorch dataloader from a dataset"""

    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        return default_collate(batch)
    
    data = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_prepro_workers,
        pin_memory=False,
        prefetch_factor=2,
        collate_fn=collate_fn if input_format == "files" else None,
    )

    # data = wds.WebLoader(dset, batch_size=None, shuffle=False, num_workers=self.num_workers)
    return data


class WebdatasetReader:
    """WebdatasetReader is a reader that reads samples from a webdataset"""

    def __init__(
        self,
        tar_folder,
        resolution,
        batch_size,
        num_prepro_workers,
        enable_text=True,
        enable_image=True,
        enable_metadata=False,
        wds_image_key="image",
        wds_caption_key="caption",
        cache_path=None,
        ddp=False,
        ):

        self.batch_size = batch_size
        self.resolution = resolution
        input_dataset = glob.glob(os.path.join(tar_folder, "*.tar"))
        input_dataset.sort()
        print(f"loading total {len(input_dataset)} tar files")

        if isinstance(resolution, int):
            preprocess = transforms.Compose([
                transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, 256)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
            ])
        elif isinstance(resolution, list) or isinstance(resolution, omegaconf.listconfig.ListConfig):
            preprocess = transforms.Compose([
                        transforms.Resize(max(self.resolution), antialias=True),
                        transforms.CenterCrop(self.resolution),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
                        ])


        dataset = create_webdataset(
            input_dataset,
            preprocess,
            enable_text=enable_text,
            enable_image=enable_image,
            image_key=wds_image_key,
            caption_key=wds_caption_key,
            enable_metadata=enable_metadata,
            cache_path=cache_path,
            ddp = ddp,
        )
        self.dataset = dataset
        self.dataloader = dataset_to_dataloader(dataset, batch_size,
                                                 num_prepro_workers, "webdataset",
                                                 ddp=ddp)
        self.data_all = cycle(self.dataset)

    def __iter__(self):
        for batch in self.dataset:
            yield batch

    def __getitem__(self, item):
        return next(self.data_all)





def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])



