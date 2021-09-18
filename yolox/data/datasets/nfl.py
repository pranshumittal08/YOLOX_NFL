import os
import loguru
import cv2
import numpy as np
from pycocotools.coco import COCO
from ..dataloading import get_yolox_datadir
from .datasets_wrapper import Dataset

class NFLDataset(Dataset):
    '''
    NFL dataset class.
    '''

    def __init__(
        self,
        data_dir = None,
        json_file = 'train_detection_info.json',
        name = 'train',
        img_size = (720, 1280),
        preproc = None,
        cache = False
    ):
        super().__init__(img_size)
        if data_dir is None:
            data_dir = os.path.join(get_yolox_datadir(), "NFL")
        self.data_dir = data_dir
        self.json_file = json_file

        self.nfl = COCO(os.path.join(self.data_dir, "annotations", self.json_file))
        self.ids = self.nfl.getImgIds() #Image IDs
        self.class_ids = sorted(self.nfl.getCatIds()) #Get class ids 
        cats = self.nfl.loadCats(self.nfl.getCatIds())
        self._classes = tuple([c["name"] for c in cats])#Get class names
        self.imgs = None
        self.name = name
        self.img_size = img_size
        self.preproc = preproc
        self.annotations = self._load_nfl_annotations()
        if cache:
            self._cache_images()

    def __len__(self):
        return len(self.ids)

    def __del__(self):
        del self.imgs

    def _load_nfl_annotations(self):
        return [self.load_anno_from_ids(_ids) for _ids in self.ids]
    
    def load_anno_from_ids(self, id_):
        im_ann = self.nfl.loadImgs(id_)[0] #retrieve img annotation
        width = im_ann["width"]
        height = im_ann["height"]
        anno_ids = self.nfl.getAnnIds(imgIds=[int(id_)], iscrowd=False) #get annotation id
        annotations = self.nfl.loadAnns(anno_ids)
        objs = []
        for obj in annotations:
            x1 = np.max((0, obj["bbox"][0]))
            y1 = np.max((0, obj["bbox"][1]))
            x2 = np.min((width, x1 + np.max((0, obj["bbox"][2]))))
            y2 = np.min((height, y1 + np.max((0, obj["bbox"][3]))))
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)

        num_objs = len(objs)

        res = np.zeros((num_objs, 5))

        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            res[ix, 0:4] = obj["clean_bbox"]
            res[ix, 4] = cls

        r = min(self.img_size[0] / height, self.img_size[1] / width)
        res[:, :4] *= r

        img_info = (height, width)
        resized_info = (int(height * r), int(width * r))

        file_name = (
            im_ann["file_name"]
            if "file_name" in im_ann
            else "{:012}".format(id_) + ".jpg"
        )

        return (res, img_info, resized_info, file_name)

    def load_anno(self, index):
        return self.annotations[index][0]

    def load_resized_img(self, index):
        img = self.load_image(index)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return resized_img

    def load_image(self, index):
        file_name = self.annotations[index][3]

        img_file = os.path.join(self.data_dir, self.name, file_name)

        img = cv2.imread(img_file)
        assert img is not None

        return img

    def pull_item(self, index):
        id_ = self.ids[index]

        res, img_info, resized_info, _ = self.annotations[index]
        if self.imgs is not None:
            pad_img = self.imgs[index]
            img = pad_img[: resized_info[0], : resized_info[1], :].copy()
        else:
            img = self.load_resized_img(index)

        return img, res.copy(), img_info, np.array([id_])

    
    @Dataset.mosaic_getitem
    def __getitem__(self, index):
        """
        One image / label pair for the given index is picked up and pre-processed.

        Args:
            index (int): data index

        Returns:
            img (numpy.ndarray): pre-processed image
            padded_labels (torch.Tensor): pre-processed label data.
                The shape is :math:`[max_labels, 5]`.
                each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
            info_img : tuple of h, w.
                h, w (int): original shape of the image
            img_id (int): same as the input index. Used for evaluation.
        """
        img, target, img_info, img_id = self.pull_item(index)

        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)
        return img, target, img_info, img_id