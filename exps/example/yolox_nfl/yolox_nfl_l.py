# encoding: utf-8
import os

import torch
import torch.distributed as dist

from yolox.data import get_yolox_datadir
from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 1
        self.depth = 1.0
        self.width = 1.0
        self.warmup_epochs = 2
        self.input_size = (704,704)
        self.multiscale_range = 0
        # ---------- transform config ------------ #
        self.max_epoch = 50
        self.no_aug_epochs = 2
        
        self.mosaic_prob = 1.0
        self.mixup_prob = 1.0
        self.hsv_prob = 1.0
        self.flip_prob = 0.5
        self.enable_mixup = True
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.train_ann = "train_detection_info.json"
        self.val_ann = "valid_detection_info.json"
        self.print_interval = 1
        self.eval_interval = 1
        # -----------------  testing config ------------------ #
        self.test_size = (704,704)
        self.test_conf = 0.01
        self.nmsthre = 0.35

    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=False):
        from yolox.data import (
            NFLDataset,
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
            worker_init_reset_seed,
        )
        from yolox.utils import (
            wait_for_the_master,
            get_local_rank,
        )
        local_rank = get_local_rank()

        with wait_for_the_master(local_rank):
            dataset = NFLDataset(
                data_dir=self.data_dir,
                json_file=self.train_ann,
                img_size=self.input_size,
                preproc=TrainTransform(
                    max_labels=50,
                    flip_prob=self.flip_prob,
                    hsv_prob=self.hsv_prob),
                cache=cache_img,
            )

        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=120,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob),
            degrees=self.degrees,
            translate=self.translate,
            mosaic_scale=self.mosaic_scale,
            mixup_scale=self.mixup_scale,
            shear=self.shear,
            perspective=self.perspective,
            enable_mixup=self.enable_mixup,
            mosaic_prob=self.mosaic_prob,
            mixup_prob=self.mixup_prob,
        )

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(
            len(self.dataset), seed=self.seed if self.seed else 0
        )

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug,
        )

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler

        # Make sure each process has different random seed, especially for 'fork' method
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed

        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader

    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.data import NFLDataset, ValTransform

        valdataset = NFLDataset(
            data_dir=self.data_dir,
            json_file=self.val_ann,
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.evaluators import COCOEvaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev, legacy)
        evaluator = COCOEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
        )
        return evaluator
