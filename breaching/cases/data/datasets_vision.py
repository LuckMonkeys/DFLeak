"""Additional torchvision-like datasets."""

import torch
import torchvision

import os
import glob
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from torchvision.datasets.utils import download_and_extract_archive, check_integrity, download_file_from_google_drive, extract_archive, verify_str_arg
from torchvision.datasets.vision import VisionDataset
import hashlib

import concurrent.futures

import csv

from collections import namedtuple
from typing import Any, Callable, List, Optional, Tuple, Union

import PIL
import random
import logging
from .autogument import split_policy, construct_policy
log = logging.getLogger(__name__)


CSV = namedtuple("CSV", ["header", "index", "data"])

def _build_dataset_vision(cfg_data, split, can_download=True):
    _default_t = torchvision.transforms.ToTensor()
    cfg_data.path = os.path.expanduser(cfg_data.path)
    if cfg_data.name == "CIFAR10":
        dataset = torchvision.datasets.CIFAR10(
            root=cfg_data.path, train=split == "training", download=can_download, transform=_default_t,
        )
        dataset.lookup = dict(zip(list(range(len(dataset))), dataset.targets))
    elif cfg_data.name == "CIFAR100":
        dataset = torchvision.datasets.CIFAR100(
            root=cfg_data.path, train=split == "training", download=can_download, transform=_default_t,
        )
        dataset.lookup = dict(zip(list(range(len(dataset))), dataset.targets))
    elif cfg_data.name == "ImageNet":
        # if 'train' in split:
        dataset = torchvision.datasets.ImageFolder(root=cfg_data.path, transform=_default_t)
        # dataset = torchvision.datasets.ImageNet(
        #     root=cfg_data.path, split="train" if "train" in split else "val", transform=_default_t,
        # )
        dataset.lookup = dict(zip(list(range(len(dataset))), [label for (_, label) in dataset.samples]))
        # else:
        #     raise NotImplementedError
    elif cfg_data.name == "ImageNetAnimals":
        if 'train' in split:
            dataset = torchvision.datasets.ImageFolder(root=cfg_data.path, transform=_default_t)
        # dataset = torchvision.datasets.ImageNet(
        #     root=cfg_data.path, split="train" if "train" in split else "val", transform=_default_t,
        # )
        else: # still use training dataset as validation dataset
            dataset = torchvision.datasets.ImageFolder(root=cfg_data.path, transform=_default_t)

        dataset.lookup = dict(zip(list(range(len(dataset))), [label for (_, label) in dataset.samples]))

        indices = [idx for (idx, label) in dataset.lookup.items() if label < 397]
        dataset.classes = dataset.classes[:397]
        dataset.samples = [dataset.samples[i] for i in indices]  # Manually remove samples instead of using a Subset
        dataset.lookup = dict(zip(list(range(len(dataset))), [label for (_, label) in dataset.samples]))
    elif cfg_data.name == "TinyImageNet":
        dataset = TinyImageNet(
            root=cfg_data.path, split=split, download=can_download, transform=_default_t, cached=True,
        )
        dataset.lookup = dict(zip(list(range(len(dataset))), dataset.targets))
    elif cfg_data.name == "ImageNet25":
        root_path = os.path.join(cfg_data.path, "train") if "train" in split else os.path.join(cfg_data.path, "val")
        dataset = torchvision.datasets.ImageFolder(root=root_path, transform=_default_t)
        dataset.lookup = dict(zip(list(range(len(dataset))), [label for (_, label) in dataset.samples]))

    elif cfg_data.name == "Birdsnap":
        dataset = Birdsnap(root=cfg_data.path, split=split, download=can_download, transform=_default_t)
        dataset.lookup = dict(zip(list(range(len(dataset))), dataset.labels))

    elif cfg_data.name == 'CelebaHQ_Gender':
        dataset = CelebaHQ_Gender(root=cfg_data.path, split=split, transform=_default_t)
        dataset.lookup = dict(zip(list(range(len(dataset))), [label for (_, label) in dataset]))
    elif cfg_data.name == "CelebaHQ_Gender_ATS":
        dataset = CelebaHQ_Gender(root=cfg_data.path, split=split, transform=_default_t, subdir="data256_ATS")
        dataset.lookup = dict(zip(list(range(len(dataset))), [label for (_, label) in dataset]))
    elif cfg_data.name == "bFFHQ_Gender":
        dataset = bFFHQ_Gender(root=cfg_data.path, split=split, transform=_default_t)
        dataset.lookup = dict(zip(list(range(len(dataset))), [label for (_, label) in dataset]))
    elif cfg_data.name == "CelebaHQ_Recognition":
        dataset = CelebAHQ_Recognition(root=cfg_data.path, split=split, transform=_default_t)
        dataset.lookup = dict(zip(list(range(len(dataset))), [label for (_, label) in dataset]))
    elif cfg_data.name == "LFWA_Gender":
        dataset = LFWA_Gender(root=cfg_data.path, split=split, transform=_default_t)
        dataset.lookup = dict(zip(list(range(len(dataset))), [label for (_, label) in dataset]))
        dataset.classes = [0,1]
    else:
        raise ValueError(f"Invalid dataset {cfg_data.name} provided.")

    if cfg_data.mean is None and cfg_data.normalize:
        data_mean, data_std = _get_meanstd(dataset)
        cfg_data.mean = data_mean
        cfg_data.std = data_std

    transforms = _parse_data_augmentations(cfg_data, split, ATS=True)

    # Apply transformations
    dataset.transform = transforms if transforms is not None else None

    # Save data mean and data std for easy access:
    if cfg_data.normalize:
        dataset.mean = cfg_data.mean
        dataset.std = cfg_data.std
    else:
        dataset.mean = [0]
        dataset.std = [1]

    # Reduce train dataset according to cfg_data.size:
    if cfg_data.size < len(dataset):
        dataset = Subset(dataset, torch.arange(0, cfg_data.size))

    collate_fn = _torchvision_collate
    return dataset, collate_fn


def _split_dataset_vision(dataset, cfg_data, user_idx=None, return_full_dataset=False):
    if not return_full_dataset:
        if user_idx is None:
            user_idx = torch.randint(0, cfg_data.default_clients, (1,))
        else:
            if user_idx > cfg_data.default_clients:
                raise ValueError("This user index exceeds the maximal number of clients.")

        # Create a synthetic split of the dataset over all possible users if no natural split is given
        if cfg_data.partition == "balanced":
            data_per_class_per_user = len(dataset) // len(dataset.classes) // cfg_data.default_clients
            if data_per_class_per_user < 1:
                raise ValueError("Too many clients for a balanced dataset.")
            data_ids = []
            for class_idx, _ in enumerate(dataset.classes):
                data_with_class = [idx for (idx, label) in dataset.lookup.items() if label == class_idx]
                data_ids += data_with_class[
                    user_idx * data_per_class_per_user : data_per_class_per_user * (user_idx + 1)
                ]
            dataset = Subset(dataset, data_ids)
        elif cfg_data.partition == "unique-class":
            data_ids = [idx for (idx, label) in dataset.lookup.items() if label == user_idx]
            dataset = Subset(dataset, data_ids)
        elif cfg_data.partition == "mixup":
            if "mixup_freq" in cfg_data:
                mixup_freq = cfg_data.mixup_freq
            else:
                # use default mixup_freq=2
                mixup_freq = 2
            data_per_user = len(dataset) // cfg_data.default_clients
            last_id = len(dataset) - 1
            data_ids = []
            for i in range(data_per_user):
                data_ids.append(user_idx * data_per_user + i)
                data_ids.append(last_id - user_idx * data_per_user - i)
            dataset = Subset(dataset, data_ids)
        elif cfg_data.partition == "feat_est":
            if "num_data_points" in cfg_data:
                num_data_points = cfg_data.num_data_points
            else:
                num_data_points = 1

            if "target_label" in cfg_data:
                target_label = cfg_data.target_label
            else:
                target_label = 0

            data_ids = [idx for (idx, label) in dataset.lookup.items() if label == target_label]
            data_ids = data_ids[user_idx * num_data_points : (user_idx + 1) * num_data_points]
            dataset = Subset(dataset, data_ids)
        elif cfg_data.partition == "random-full":  # Data might be repeated across users (e.g. meme images)
            data_per_user = len(dataset) // cfg_data.default_clients
            data_ids = torch.randperm(len(dataset))[:data_per_user]
            dataset = Subset(dataset, data_ids)
        elif cfg_data.partition == "random":  # Data not replicated across users. Split is deterministic over reruns!
            data_per_user = len(dataset) // cfg_data.default_clients
            generator = torch.Generator()
            generator.manual_seed(233)
            data_ids = torch.randperm(len(dataset), generator=generator)

            # torch.set_printoptions(profile="full")
            # log.info(f"Data ids: {data_ids}")
            # torch.set_printoptions(profile="default")

            # torch.save(data_ids, "data_ids2.pt")
            # print(len(dataset), data_ids)
            # exit(0)
            data_ids = data_ids[user_idx * data_per_user : data_per_user * (user_idx + 1)]
            dataset = Subset(dataset, data_ids)
        elif cfg_data.partition == "none":  # Replicate on all users for a sanity check!
            pass
        else:
            raise ValueError(f"Partition scheme {cfg_data.partition} not implemented.")
    return dataset


def _torchvision_collate(batch):
    """Small hack around the pytorch default collator to return a dictionary"""
    transposed = list(zip(*batch))

    def _stack_tensor(tensor_list):
        elem = tensor_list[0]
        elem_type = type(elem)
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in tensor_list)
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(tensor_list, 0, out=out)

    return dict(inputs=_stack_tensor(transposed[0]), labels=torch.tensor(transposed[1]))


class Subset(torch.utils.data.Subset):
    """Overwrite subset class to provide class methods of main class."""

    def __getattr__(self, name):
        """Call this only if all attributes of Subset are exhausted."""
        return getattr(self.dataset, name)


def _get_meanstd(dataset):
    print("Computing dataset mean and std manually ... ")
    # Run parallelized Wellford:
    current_mean = 0
    current_M2 = 0
    n = 0
    for data, _ in dataset:
        datapoint = data.view(3, -1)
        ds, dm = torch.std_mean(datapoint, dim=1)
        n_a, n_b = n, datapoint.shape[1]
        n += n_b
        delta = dm.to(dtype=torch.double) - current_mean
        current_mean += delta * n_b / n
        current_M2 += ds.to(dtype=torch.double) / (n_b - 1) + delta ** 2 * n_a * n_b / n
        # print(current_mean, (current_M2 / (n - 1)).sqrt())

    data_mean = current_mean.tolist()
    data_std = (current_M2 / (n - 1)).sqrt().tolist()
    print(f"Mean: {data_mean}. Standard deviation: {data_std}")
    return data_mean, data_std


def _parse_data_augmentations(cfg_data, split, PIL_only=False, ATS=False):
    def _parse_cfg_dict(cfg_dict):
        list_of_transforms = []
        if hasattr(cfg_dict, "keys"):
            for key in cfg_dict.keys():
                try:  # ducktype iterable
                    transform = getattr(torchvision.transforms, key)(*cfg_dict[key])
                except TypeError:
                    transform = getattr(torchvision.transforms, key)(cfg_dict[key])
                list_of_transforms.append(transform)
        return list_of_transforms

    def _parse_ats_cfg_dict(cfg_dict):
        list_of_transforms = []
        if hasattr(cfg_dict, "keys"):
            policy_list = split_policy(cfg_dict["policy"])
            list_of_transforms.append(construct_policy(policy_list=policy_list)) 
        return list_of_transforms

    if split == "train":
        transforms = _parse_cfg_dict(cfg_data.augmentations_train)
    else:
        transforms = _parse_cfg_dict(cfg_data.augmentations_val)
    # add ATS defense 
    if ATS:
        if cfg_data.augmentations_ats.policy is None:
           log.warning("User ATS defense but provide None policy") 
        else:
            policy = _parse_ats_cfg_dict(cfg_dict=cfg_data.augmentations_ats)
            transforms.extend(policy)
    # breakpoint()
    if not PIL_only:
        transforms.append(torchvision.transforms.ToTensor())
        if cfg_data.normalize:
            transforms.append(torchvision.transforms.Normalize(cfg_data.mean, cfg_data.std))
    return torchvision.transforms.Compose(transforms)


class TinyImageNet(torch.utils.data.Dataset):
    """Tiny ImageNet data set available from `http://cs231n.stanford.edu/tiny-imagenet-200.zip`.

    This is a TinyImageNet variant to the code of Meng Lee, mnicnc404 / Date: 2018/06/04
    References:
        - https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel.html
    Parameters
    ----------
    root: string
        Root directory including `train`, `test` and `val` subdirectories.
    split: string
        Indicating which split to return as a data set.
        Valid option: [`train`, `test`, `val`]
    transform: torchvision.transforms
        A (series) of valid transformation(s).
    cached: bool
        Set to True if there is enough memory (about 5G) and want to minimize disk IO overhead.
    download: bool
        Set to true to automatically download the dataset in to the root folder.
    """

    EXTENSION = "JPEG"
    NUM_IMAGES_PER_CLASS = 500
    CLASS_LIST_FILE = "wnids.txt"
    VAL_ANNOTATION_FILE = "val_annotations.txt"
    CLASSES = "words.txt"

    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    archive = "tiny-imagenet-200.zip"
    folder = "tiny-imagenet-200"
    train_md5 = "c77c61d662a966d2fcae894d82df79e4"
    val_md5 = "cef44e3f1facea2ea8cd5e5a7a46886c"
    test_md5 = "bc72ebd5334b12e3a7ba65506c0f8bc0"

    def __init__(self, root, split="train", transform=None, target_transform=None, cached=True, download=True):
        """Init with split, transform, target_transform."""
        self.root = os.path.expanduser(root)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.cached = cached

        self.split_dir = os.path.join(root, self.folder, self.split)
        self.image_paths = sorted(
            glob.iglob(os.path.join(self.split_dir, "**", "*.%s" % self.EXTENSION), recursive=True)
        )
        self.labels = {}  # fname - label number mapping

        if download:
            self.download()

        self._parse_labels()

        if self.cached:
            self._build_cache()

    def _check_integrity(self):
        """This only checks if all files are there."""
        string_rep = "".join(self.image_paths).encode("utf-8")
        hash = hashlib.md5(string_rep)
        if self.split == "train":
            return hash.hexdigest() == self.train_md5
        elif self.split == "val":
            return hash.hexdigest() == self.val_md5
        else:
            return hash.hexdigest() == self.test_md5

    def download(self):
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        download_and_extract_archive(self.url, self.root, filename=self.archive)

    def _parse_labels(self):
        with open(os.path.join(self.root, self.folder, self.CLASS_LIST_FILE), "r") as fp:
            self.label_texts = sorted([text.strip() for text in fp.readlines()])
        self.label_text_to_number = {text: i for i, text in enumerate(self.label_texts)}

        if self.split == "train":
            for label_text, i in self.label_text_to_number.items():
                for cnt in range(self.NUM_IMAGES_PER_CLASS):
                    self.labels["%s_%d.%s" % (label_text, cnt, self.EXTENSION)] = i
        elif self.split == "val":
            with open(os.path.join(self.split_dir, self.VAL_ANNOTATION_FILE), "r") as fp:
                for line in fp.readlines():
                    terms = line.split("\t")
                    file_name, label_text = terms[0], terms[1]
                    self.labels[file_name] = self.label_text_to_number[label_text]

        # Build class names
        label_text_to_word = dict()
        with open(os.path.join(self.root, self.folder, self.CLASSES), "r") as file:
            for line in file:
                label_text, word = line.split("\t")
                label_text_to_word[label_text] = word.split(",")[0].rstrip("\n")
        self.classes = [label_text_to_word[label] for label in self.label_texts]

        # Prepare index - label mapping
        self.targets = [self.labels[os.path.basename(file_path)] for file_path in self.image_paths]

    def _build_cache(self):
        """Cache images in RAM."""
        self.cache = []
        for index in range(len(self)):
            img = Image.open(self.image_paths[index])
            img = img.convert("RGB")
            self.cache.append(img)

    def __len__(self):
        """Return length via image paths."""
        return len(self.image_paths)

    def __getitem__(self, index):
        """Return image, label."""
        if self.cached:
            img = self.cache[index]
        else:
            img = Image.open(self.image_paths[index])
            img = img.convert("RGB")
        target = self.targets[index]

        img = self.transform(img) if self.transform else img
        target = self.target_transform(target) if self.target_transform else target
        if self.split == "test":
            return img, None
        else:
            return img, target


class Birdsnap(torch.utils.data.Dataset):
    """This is the BirdSnap dataset presented in
    - Berg et al., "Birdsnap: Large-scale Fine-grained Visual Categorization of Birds"
    It contains a lot of classes of birds and can be used as a replacement for ImageNet validation images
    with similar image fidelity but less of the baggage, given that all subjects are in fact birds.

    This is too small to train on though and hence not even partitioned into train/test.
    Several images are missing from flickr (in 2021), these will be discarded automatically.
    """

    METADATA_URL = "http://thomasberg.org/datasets/birdsnap/1.1/birdsnap.tgz"
    METADATA_ARCHIVE = "birdsnap.tgz"
    META_MD5 = "1788158175f6ae794aebf27bcd7a3f5d"
    BASE_FOLDER = "birdsnap"

    def __init__(self, root, split="train", transform=None, target_transform=None, download=True, crop_to_bbx=False):
        """Init with split, transform, target_transform."""
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        self.crop_to_bbx = crop_to_bbx  # Crop to dataset default bounding boxes

        if download:
            self.download()
        if not self.check_integrity():
            raise ValueError("Dataset Birdsnap not downloaded completely or possibly corrupted.")

        self._purge_missing_data()

    def _check_integrity_of_metadata(self, chunk_size=8192):
        """This only checks if all files are there."""
        try:
            with open(os.path.join(self.root, self.METADATA_ARCHIVE), "rb") as f:
                archive_hash = hashlib.md5()
                while chunk := f.read(chunk_size):
                    archive_hash.update(chunk)
            return self.META_MD5 == archive_hash.hexdigest()
        except FileNotFoundError:
            return False

    def check_integrity(self):
        """Full integrity check."""
        if not self._check_integrity_of_metadata():
            return False
        else:
            self._parse_metadata()
            missing_images = 0
            for idx, file in enumerate(self.meta):
                if not self._verify_image(idx):
                    missing_images += 1
            if missing_images > 0:
                print(f"{missing_images} images could not be downloaded.")
            return True

    def download(self):
        # Metadata:
        if self._check_integrity_of_metadata():
            print("Metadata already downloaded and verified")
        else:
            download_and_extract_archive(self.METADATA_URL, self.root, filename=self.METADATA_ARCHIVE)
        # Actual files:
        self._parse_metadata()

        missing_ids = []
        for idx, file in enumerate(self.meta):
            if not self._verify_image(idx):
                missing_ids.append(idx)
        if len(missing_ids) > 0:
            print(f"Downloading {len(missing_ids)} missing files now...")
            self.scrape_images(missing_ids)

    def __len__(self):
        """Return length via metainfo."""
        return len(self.meta)

    def __getitem__(self, index):
        """Return image, label."""
        img = Image.open(self.paths[index])
        if self.crop_to_bbx:
            img = img.crop(
                (
                    self.meta[index]["bb_x1"],
                    self.meta[index]["bb_y1"],
                    self.meta[index]["bb_x2"],
                    self.meta[index]["bb_y2"],
                )
            )
        img = img.convert("RGB")
        label = self.labels[index]

        img = self.transform(img) if self.transform else img
        label = self.target_transform(label) if self.target_transform else label
        return img, label

    def _parse_metadata(self):
        """Metadata keys are
        dict_keys(['url', 'md5', 'path', 'species_id', 'bb_x1', 'bb_y1', 'bb_x2', 'bb_y2', 'back_x', 'back_y', 'beak_x',
        'beak_y', 'belly_x', 'belly_y', 'breast_x', 'breast_y', 'crown_x', 'crown_y', 'forehead_x', 'forehead_y',
        'left_cheek_x', 'left_cheek_y', 'left_eye_x', 'left_eye_y', 'left_leg_x', 'left_leg_y', 'left_wing_x',
        'left_wing_y', 'nape_x', 'nape_y', 'right_cheek_x', 'right_cheek_y', 'right_eye_x', 'right_eye_y',
        'right_leg_x', 'right_leg_y', 'right_wing_x', 'right_wing_y', 'tail_x', 'tail_y', 'throat_x', 'throat_y']
        """
        with open(os.path.join(self.root, self.BASE_FOLDER, "images.txt"), "r") as f:
            reader = csv.DictReader(f, delimiter="\t")
            self.meta = list(reader)  # List of dictionaries.
        self.labels = [int(entry["species_id"]) for entry in self.meta]
        self.paths = [os.path.join(self.root, self.BASE_FOLDER, entry["path"]) for entry in self.meta]
        with open(os.path.join(self.root, self.BASE_FOLDER, "species.txt"), "r") as f:
            reader = csv.DictReader(f, delimiter="\t")
            self.classes_metadata = list(reader)
        self.classes = [str(entry["common"]) for entry in self.classes_metadata]

    def _verify_image(self, idx):
        try:
            # Do this if you want to check in detail:
            # with open(os.path.join(self.root, self.BASE_FOLDER, self.meta[idx]['path']), 'rb') as fin:
            #     return (hashlib.md5(fin.read()).hexdigest() == self.meta[idx]['md5'])
            # In the mean time, just check if everything is there:
            return os.path.exists(os.path.join(self.root, self.BASE_FOLDER, self.meta[idx]["path"]))
        except FileNotFoundError:
            return False

    def scrape_images(self, missing_ids, chunk_size=8196):
        """Scrape images using the python default ThreadPool example."""
        import requests

        def _load_url_and_save_image(idx, timeout):
            full_path = os.path.join(self.root, self.BASE_FOLDER, self.meta[idx]["path"])
            os.makedirs(os.path.split(full_path)[0], exist_ok=True)
            r = requests.get(self.meta[idx]["url"], stream=True)
            with open(full_path, "wb") as write_file:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    write_file.write(chunk)
            return True

        # We can use a with statement to ensure threads are cleaned up promptly
        with concurrent.futures.ThreadPoolExecutor(max_workers=None) as executor:  # Choose max_workers dynamically
            # Start the load operations and mark each future with its URL
            future_to_url = {
                executor.submit(_load_url_and_save_image, idx, 600): self.meta[idx]["url"] for idx in missing_ids
            }
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    data = future.result()
                except Exception as exc:
                    print(f"{url} generated exception: {exc}")
                else:
                    print(f"{url} downloaded successfully.")

    def _purge_missing_data(self):
        """Iterate over all data and throw out missing images."""
        JPG = b"\xff\xd8\xff"

        clean_meta = []
        invalid_files = 0
        for entry in self.meta:
            full_path = os.path.join(self.root, self.BASE_FOLDER, entry["path"])
            with open(full_path, "rb") as file_handle:
                if file_handle.read(3) == JPG:
                    clean_meta.append(entry)
                else:
                    invalid_files += 1
        print(f"Discarded {invalid_files} invalid files.")
        self.meta = clean_meta

        self.labels = [int(entry["species_id"]) for entry in self.meta]
        self.paths = [os.path.join(self.root, self.BASE_FOLDER, entry["path"]) for entry in self.meta]




class CelebaHQ_Gender(VisionDataset):
    """`Large-scale CelebFaces Attributes (CelebA) Dataset <http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        split (string): One of {'train', 'valid', 'test', 'all'}.
            Accordingly dataset is selected.
        target_type (string or list, optional): Type of target to use, ``attr``, ``identity``, ``bbox``,
            or ``landmarks``. Can also be a list to output a tuple with all specified target types.
            The targets represent:

                - ``attr`` (np.array shape=(40,) dtype=int): binary (0, 1) labels for attributes
                - ``identity`` (int): label for each person (data points with the same identity are the same person)
                - ``bbox`` (np.array shape=(4,) dtype=int): bounding box (x, y, width, height)
                - ``landmarks`` (np.array shape=(10,) dtype=int): landmark points (lefteye_x, lefteye_y, righteye_x,
                  righteye_y, nose_x, nose_y, leftmouth_x, leftmouth_y, rightmouth_x, rightmouth_y)

            Defaults to ``attr``. If empty, ``None`` will be returned as target.

        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    base_folder = "celeba_hq"
    # There currently does not appear to be a easy way to extract 7z in python (without introducing additional
    # dependencies). The "in-the-wild" (not aligned+cropped) images are only in 7z, so they are not available
    # right now.
    file_list = [
        # File ID                                      MD5 Hash                            Filename
        # ("0B7EVK8r0v71pZjFTYXZWM3FlRnM", "00d2c5bc6d35e252742224ab0c1e8fcb", "img_align_celeba.zip"),
        # ("0B7EVK8r0v71pbWNEUjJKdDQ3dGc","b6cd7e93bc7a96c2dc33f819aa3ac651", "img_align_celeba_png.7z"),
        # ("0B7EVK8r0v71peklHb0pGdDl6R28", "b6cd7e93bc7a96c2dc33f819aa3ac651", "img_celeba.7z"),
        ("0B7EVK8r0v71pblRyaVFSWGxPY0U", "75e246fa4810816ffd6ee81facbd244c", "list_attr_celeba.txt"),
        # ("1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS", "32bd1bd63d3c78cd57e08160ec5ed1e2", "identity_CelebA.txt"),
        # ("0B7EVK8r0v71pbThiMVRxWXZ4dU0", "00566efa6fedff7a56946cd1c10f1c16", "list_bbox_celeba.txt"),
        # ("0B7EVK8r0v71pd0FJY3Blby1HUTQ", "cc24ecafdb5b50baae59b03474781f8c", "list_landmarks_align_celeba.txt"),
        # ("0B7EVK8r0v71pTzJIdlJWdHczRlU", "063ee6ddb681f96bc9ca28c6febb9d1a", "list_landmarks_celeba.txt"),
        ("0B7EVK8r0v71pY0NSMzRuSXJEVkk", "d32c9cbf5e040fd4025c592c306e6668", "list_eval_partition.txt"),
    ]

    def __init__(
        self,
        root: str,
        split: str = "train",
        target_type: Union[List[str], str] = "attr",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        subdir: str = "data256"
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.split = split
        self.subdir = subdir
        self.classes = [0, 1]
        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]

        if not self.target_type and self.target_transform is not None:
            raise RuntimeError("target_transform is specified but target_type is empty")

        # if download:
        #     self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }
        split_ = split_map[verify_str_arg(split.lower(), "split", ("train", "valid", "test", "all"))]
        splits = self._load_csv("list_eval_partition.txt")
        # identity = self._load_csv("identity_CelebA.txt")
        # bbox = self._load_csv("list_bbox_celeba.txt", header=1)
        # landmarks_align = self._load_csv("list_landmarks_align_celeba.txt", header=1)
        attr = self._load_csv("list_attr_celeba.txt", header=1)

        mask = slice(None) if split_ is None else (splits.data == split_).squeeze()

        if mask == slice(None):  # if split == "all"
            self.filename = splits.index
        else:
            self.filename = [splits.index[i] for i in torch.squeeze(torch.nonzero(mask))]
        # self.identity = identity.data[mask]
        # self.bbox = bbox.data[mask]
        # self.landmarks_align = landmarks_align.data[mask]
        self.attr = attr.data[mask]
        # map from {-1, 1} to {0, 1}
        self.attr = torch.div(self.attr + 1, 2, rounding_mode="floor")
        self.attr_names = attr.header

    def _load_csv(
        self,
        filename: str,
        header: Optional[int] = None,
    ) -> CSV:
        with open(os.path.join(self.root, self.base_folder, filename)) as csv_file:
            data = list(csv.reader(csv_file, delimiter=" ", skipinitialspace=True))

        if header is not None:
            headers = data[header]
            data = data[header + 1 :]
        else:
            headers = []

        indices = [row[0] for row in data]
        data = [row[1:] for row in data]
        data_int = [list(map(int, i)) for i in data]

        return CSV(headers, indices, torch.tensor(data_int))

    def _check_integrity(self) -> bool:
        return os.path.isdir(os.path.join(self.root, self.base_folder, self.subdir))


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        X = PIL.Image.open(os.path.join(self.root, self.base_folder, self.subdir, self.filename[index]))

        target: Any = []
        for t in self.target_type:
            if t == "attr":
                target.append(self.attr[index, :])
            elif t == "identity":
                target.append(self.identity[index, 0])
            elif t == "bbox":
                target.append(self.bbox[index, :])
            elif t == "landmarks":
                target.append(self.landmarks_align[index, :])
            else:
                # TODO: refactor with utils.verify_str_arg
                raise ValueError(f'Target type "{t}" is not recognized.')

        if self.transform is not None:
            X = self.transform(X)

        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None

        return X, target[20].item()


    def __len__(self) -> int:
        return len(self.attr)

    def extra_repr(self) -> str:
        lines = ["Target type: {target_type}", "Split: {split}"]
        return "\n".join(lines).format(**self.__dict__)
    
    
    
class bFFHQ_Gender(torch.utils.data.Dataset):
    base_folder = 'bffhq'
    target_attr_index = 0
    bias_attr_index = 1

    def __init__(self, root, split, transform=None):
        super(bFFHQ_Gender, self).__init__()
        self.transform = transform
        root = os.path.join(root, self.base_folder)

        self.root = root

        if split == 'train':
            self.align = glob.glob(os.path.join(root, split, 'align', "*", "*"))
            self.conflict = glob.glob(os.path.join(root, split, 'conflict', "*", "*"))
            self.data = self.align + self.conflict

        elif split == 'valid':
            self.data = glob.glob(os.path.join(root, split, "*"))

        elif split == 'test':
            self.data =glob.glob(os.path.join(root, split, "*"))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        fpath = self.data[index]
        age_attr = int(fpath.split('_')[-2])
        gender_attr = int(fpath.split('_')[-1].split('.')[0])
        # attr = torch.LongTensor([first_attr, second_attr])
        image = Image.open(fpath).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, gender_attr

class CelebAHQ_Recognition(torchvision.datasets.ImageFolder):
    def __init__(self, root, split="train", transform=None) -> None:
        dataset_dir = os.path.join(root, split)
        super().__init__(root=dataset_dir, transform=transform)


class LFWA_Gender(torchvision.datasets.VisionDataset):
    base_folder = 'lfw'
    def __init__(self,root, split, transform) -> None:
        super().__init__(os.path.join(root, self.base_folder), transform=transform)
        self._get_split(split=split)

    def _get_split(self, split):
        self.data = []
        
        figname_gender_path = os.path.join(self.root, "lfw_figname_gender.txt")        
        figname_gender_f = open(figname_gender_path, 'r')
        figname_label = [figname.strip() for figname in figname_gender_f.readlines()]
        self.data = figname_label
        
        
        if split == "train":
            self.data = self.data[:int(len(self.data)*0.8)]
        elif split == "valid":
            self.data = self.data[int(len(self.data)*0.8):int(len(self.data)*0.9)]
        elif split == "test":
            self.data = self.data[int(len(self.data)*0.9):]
        else:
            raise ValueError("split must be train, valid or test")

    def __getitem__(self, index):
        figname_label = self.data[index].strip()
        name_split = figname_label.split('_')
        subdir = '_'.join(name_split[:-2])
        figname = '_'.join(name_split[:-1]) 
        target = int(name_split[-1])
        
        img = self._loader(os.path.join(self.root, self.base_folder, subdir, figname))
        
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def _loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def __len__(self):
        return len(self.data)

