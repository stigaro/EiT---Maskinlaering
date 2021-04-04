import contextlib
import sys
import os
import unittest
from unittest import mock
import numpy as np
import PIL
from PIL import Image
from torch._utils_internal import get_file_path_2
import torchvision
from torchvision.datasets import utils
from common_utils import get_tmp_dir
from fakedata_generation import svhn_root, places365_root, widerface_root, stl10_root
import xml.etree.ElementTree as ET
from urllib.request import Request, urlopen
import itertools
import datasets_utils
import pathlib
import pickle
from torchvision import datasets
import torch
import shutil
import json
import random
import bz2
import torch.nn.functional as F
import string
import io
import zipfile


try:
    import scipy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import av
    HAS_PYAV = True
except ImportError:
    HAS_PYAV = False


class DatasetTestcase(unittest.TestCase):
    def generic_classification_dataset_test(self, dataset, num_images=1):
        self.assertEqual(len(dataset), num_images)
        img, target = dataset[0]
        self.assertTrue(isinstance(img, PIL.Image.Image))
        self.assertTrue(isinstance(target, int))

    def generic_segmentation_dataset_test(self, dataset, num_images=1):
        self.assertEqual(len(dataset), num_images)
        img, target = dataset[0]
        self.assertTrue(isinstance(img, PIL.Image.Image))
        self.assertTrue(isinstance(target, PIL.Image.Image))


class Tester(DatasetTestcase):
    @mock.patch('torchvision.datasets.SVHN._check_integrity')
    @unittest.skipIf(not HAS_SCIPY, "scipy unavailable")
    def test_svhn(self, mock_check):
        mock_check.return_value = True
        with svhn_root() as root:
            dataset = torchvision.datasets.SVHN(root, split="train")
            self.generic_classification_dataset_test(dataset, num_images=2)

            dataset = torchvision.datasets.SVHN(root, split="test")
            self.generic_classification_dataset_test(dataset, num_images=2)

            dataset = torchvision.datasets.SVHN(root, split="extra")
            self.generic_classification_dataset_test(dataset, num_images=2)

    def test_places365(self):
        for split, small in itertools.product(("train-standard", "train-challenge", "val"), (False, True)):
            with places365_root(split=split, small=small) as places365:
                root, data = places365

                dataset = torchvision.datasets.Places365(root, split=split, small=small, download=True)
                self.generic_classification_dataset_test(dataset, num_images=len(data["imgs"]))

    def test_places365_transforms(self):
        expected_image = "image"
        expected_target = "target"

        def transform(image):
            return expected_image

        def target_transform(target):
            return expected_target

        with places365_root() as places365:
            root, data = places365

            dataset = torchvision.datasets.Places365(
                root, transform=transform, target_transform=target_transform, download=True
            )
            actual_image, actual_target = dataset[0]

            self.assertEqual(actual_image, expected_image)
            self.assertEqual(actual_target, expected_target)

    def test_places365_devkit_download(self):
        for split in ("train-standard", "train-challenge", "val"):
            with self.subTest(split=split):
                with places365_root(split=split) as places365:
                    root, data = places365

                    dataset = torchvision.datasets.Places365(root, split=split, download=True)

                    with self.subTest("classes"):
                        self.assertSequenceEqual(dataset.classes, data["classes"])

                    with self.subTest("class_to_idx"):
                        self.assertDictEqual(dataset.class_to_idx, data["class_to_idx"])

                    with self.subTest("imgs"):
                        self.assertSequenceEqual(dataset.imgs, data["imgs"])

    def test_places365_devkit_no_download(self):
        for split in ("train-standard", "train-challenge", "val"):
            with self.subTest(split=split):
                with places365_root(split=split) as places365:
                    root, data = places365

                    with self.assertRaises(RuntimeError):
                        torchvision.datasets.Places365(root, split=split, download=False)

    def test_places365_images_download(self):
        for split, small in itertools.product(("train-standard", "train-challenge", "val"), (False, True)):
            with self.subTest(split=split, small=small):
                with places365_root(split=split, small=small) as places365:
                    root, data = places365

                    dataset = torchvision.datasets.Places365(root, split=split, small=small, download=True)

                    assert all(os.path.exists(item[0]) for item in dataset.imgs)

    def test_places365_images_download_preexisting(self):
        split = "train-standard"
        small = False
        images_dir = "data_large_standard"

        with places365_root(split=split, small=small) as places365:
            root, data = places365
            os.mkdir(os.path.join(root, images_dir))

            with self.assertRaises(RuntimeError):
                torchvision.datasets.Places365(root, split=split, small=small, download=True)

    def test_places365_repr_smoke(self):
        with places365_root() as places365:
            root, data = places365

            dataset = torchvision.datasets.Places365(root, download=True)
            self.assertIsInstance(repr(dataset), str)


class STL10Tester(DatasetTestcase):
    @contextlib.contextmanager
    def mocked_root(self):
        with stl10_root() as (root, data):
            yield root, data

    @contextlib.contextmanager
    def mocked_dataset(self, pre_extract=False, download=True, **kwargs):
        with self.mocked_root() as (root, data):
            if pre_extract:
                utils.extract_archive(os.path.join(root, data["archive"]))
            dataset = torchvision.datasets.STL10(root, download=download, **kwargs)
            yield dataset, data

    def test_not_found(self):
        with self.assertRaises(RuntimeError):
            with self.mocked_dataset(download=False):
                pass

    def test_splits(self):
        for split in ('train', 'train+unlabeled', 'unlabeled', 'test'):
            with self.mocked_dataset(split=split) as (dataset, data):
                num_images = sum([data["num_images_in_split"][part] for part in split.split("+")])
                self.generic_classification_dataset_test(dataset, num_images=num_images)

    def test_folds(self):
        for fold in range(10):
            with self.mocked_dataset(split="train", folds=fold) as (dataset, data):
                num_images = data["num_images_in_folds"][fold]
                self.assertEqual(len(dataset), num_images)

    def test_invalid_folds1(self):
        with self.assertRaises(ValueError):
            with self.mocked_dataset(folds=10):
                pass

    def test_invalid_folds2(self):
        with self.assertRaises(ValueError):
            with self.mocked_dataset(folds="0"):
                pass

    def test_transforms(self):
        expected_image = "image"
        expected_target = "target"

        def transform(image):
            return expected_image

        def target_transform(target):
            return expected_target

        with self.mocked_dataset(transform=transform, target_transform=target_transform) as (dataset, _):
            actual_image, actual_target = dataset[0]

            self.assertEqual(actual_image, expected_image)
            self.assertEqual(actual_target, expected_target)

    def test_unlabeled(self):
        with self.mocked_dataset(split="unlabeled") as (dataset, _):
            labels = [dataset[idx][1] for idx in range(len(dataset))]
            self.assertTrue(all([label == -1 for label in labels]))

    @unittest.mock.patch("torchvision.datasets.stl10.download_and_extract_archive")
    def test_download_preexisting(self, mock):
        with self.mocked_dataset(pre_extract=True) as (dataset, data):
            mock.assert_not_called()

    def test_repr_smoke(self):
        with self.mocked_dataset() as (dataset, _):
            self.assertIsInstance(repr(dataset), str)


class Caltech101TestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.Caltech101
    FEATURE_TYPES = (PIL.Image.Image, (int, np.ndarray, tuple))

    ADDITIONAL_CONFIGS = datasets_utils.combinations_grid(
        target_type=("category", "annotation", ["category", "annotation"])
    )
    REQUIRED_PACKAGES = ("scipy",)

    def inject_fake_data(self, tmpdir, config):
        root = pathlib.Path(tmpdir) / "caltech101"
        images = root / "101_ObjectCategories"
        annotations = root / "Annotations"

        categories = (("Faces", "Faces_2"), ("helicopter", "helicopter"), ("ying_yang", "ying_yang"))
        num_images_per_category = 2

        for image_category, annotation_category in categories:
            datasets_utils.create_image_folder(
                root=images,
                name=image_category,
                file_name_fn=lambda idx: f"image_{idx + 1:04d}.jpg",
                num_examples=num_images_per_category,
            )
            self._create_annotation_folder(
                root=annotations,
                name=annotation_category,
                file_name_fn=lambda idx: f"annotation_{idx + 1:04d}.mat",
                num_examples=num_images_per_category,
            )

        # This is included in the original archive, but is removed by the dataset. Thus, an empty directory suffices.
        os.makedirs(images / "BACKGROUND_Google")

        return num_images_per_category * len(categories)

    def _create_annotation_folder(self, root, name, file_name_fn, num_examples):
        root = pathlib.Path(root) / name
        os.makedirs(root)

        for idx in range(num_examples):
            self._create_annotation_file(root, file_name_fn(idx))

    def _create_annotation_file(self, root, name):
        mdict = dict(obj_contour=torch.rand((2, torch.randint(3, 6, size=())), dtype=torch.float64).numpy())
        datasets_utils.lazy_importer.scipy.io.savemat(str(pathlib.Path(root) / name), mdict)

    def test_combined_targets(self):
        target_types = ["category", "annotation"]

        individual_targets = []
        for target_type in target_types:
            with self.create_dataset(target_type=target_type) as (dataset, _):
                _, target = dataset[0]
                individual_targets.append(target)

        with self.create_dataset(target_type=target_types) as (dataset, _):
            _, combined_targets = dataset[0]

        actual = len(individual_targets)
        expected = len(combined_targets)
        self.assertEqual(
            actual,
            expected,
            f"The number of the returned combined targets does not match the the number targets if requested "
            f"individually: {actual} != {expected}",
        )

        for target_type, combined_target, individual_target in zip(target_types, combined_targets, individual_targets):
            with self.subTest(target_type=target_type):
                actual = type(combined_target)
                expected = type(individual_target)
                self.assertIs(
                    actual,
                    expected,
                    f"Type of the combined target does not match the type of the corresponding individual target: "
                    f"{actual} is not {expected}",
                )


class Caltech256TestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.Caltech256

    def inject_fake_data(self, tmpdir, config):
        tmpdir = pathlib.Path(tmpdir) / "caltech256" / "256_ObjectCategories"

        categories = ((1, "ak47"), (127, "laptop-101"), (257, "clutter"))
        num_images_per_category = 2

        for idx, category in categories:
            datasets_utils.create_image_folder(
                tmpdir,
                name=f"{idx:03d}.{category}",
                file_name_fn=lambda image_idx: f"{idx:03d}_{image_idx + 1:04d}.jpg",
                num_examples=num_images_per_category,
            )

        return num_images_per_category * len(categories)


class WIDERFaceTestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.WIDERFace
    FEATURE_TYPES = (PIL.Image.Image, (dict, type(None)))  # test split returns None as target
    ADDITIONAL_CONFIGS = datasets_utils.combinations_grid(split=('train', 'val', 'test'))

    def inject_fake_data(self, tmpdir, config):
        widerface_dir = pathlib.Path(tmpdir) / 'widerface'
        annotations_dir = widerface_dir / 'wider_face_split'
        os.makedirs(annotations_dir)

        split_to_idx = split_to_num_examples = {
            "train": 1,
            "val": 2,
            "test": 3,
        }

        # We need to create all folders regardless of the split in config
        for split in ('train', 'val', 'test'):
            split_idx = split_to_idx[split]
            num_examples = split_to_num_examples[split]

            datasets_utils.create_image_folder(
                root=tmpdir,
                name=widerface_dir / f'WIDER_{split}' / 'images' / '0--Parade',
                file_name_fn=lambda image_idx: f"0_Parade_marchingband_1_{split_idx + image_idx}.jpg",
                num_examples=num_examples,
            )

            annotation_file_name = {
                'train': annotations_dir / 'wider_face_train_bbx_gt.txt',
                'val': annotations_dir / 'wider_face_val_bbx_gt.txt',
                'test': annotations_dir / 'wider_face_test_filelist.txt',
            }[split]

            annotation_content = {
                "train": "".join(
                    f"0--Parade/0_Parade_marchingband_1_{split_idx + image_idx}.jpg\n1\n449 330 122 149 0 0 0 0 0 0\n"
                    for image_idx in range(num_examples)
                ),
                "val": "".join(
                    f"0--Parade/0_Parade_marchingband_1_{split_idx + image_idx}.jpg\n1\n501 160 285 443 0 0 0 0 0 0\n"
                    for image_idx in range(num_examples)
                ),
                "test": "".join(
                    f"0--Parade/0_Parade_marchingband_1_{split_idx + image_idx}.jpg\n"
                    for image_idx in range(num_examples)
                ),
            }[split]

            with open(annotation_file_name, "w") as annotation_file:
                annotation_file.write(annotation_content)

        return split_to_num_examples[config["split"]]


class CityScapesTestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.Cityscapes
    TARGET_TYPES = (
        "instance",
        "semantic",
        "polygon",
        "color",
    )
    ADDITIONAL_CONFIGS = (
        *datasets_utils.combinations_grid(
            mode=("fine",), split=("train", "test", "val"), target_type=TARGET_TYPES
        ),
        *datasets_utils.combinations_grid(
            mode=("coarse",),
            split=("train", "train_extra", "val"),
            target_type=TARGET_TYPES,
        ),
    )
    FEATURE_TYPES = (PIL.Image.Image, (dict, PIL.Image.Image))

    def inject_fake_data(self, tmpdir, config):

        tmpdir = pathlib.Path(tmpdir)

        mode_to_splits = {
            "Coarse": ["train", "train_extra", "val"],
            "Fine": ["train", "test", "val"],
        }

        if config["split"] == "train":  # just for coverage of the number of samples
            cities = ["bochum", "bremen"]
        else:
            cities = ["bochum"]

        polygon_target = {
            "imgHeight": 1024,
            "imgWidth": 2048,
            "objects": [
                {
                    "label": "sky",
                    "polygon": [
                        [1241, 0],
                        [1234, 156],
                        [1478, 197],
                        [1611, 172],
                        [1606, 0],
                    ],
                },
                {
                    "label": "road",
                    "polygon": [
                        [0, 448],
                        [1331, 274],
                        [1473, 265],
                        [2047, 605],
                        [2047, 1023],
                        [0, 1023],
                    ],
                },
            ],
        }

        for mode in ["Coarse", "Fine"]:
            gt_dir = tmpdir / f"gt{mode}"
            for split in mode_to_splits[mode]:
                for city in cities:
                    def make_image(name, size=10):
                        datasets_utils.create_image_folder(
                            root=gt_dir / split,
                            name=city,
                            file_name_fn=lambda _: name,
                            size=size,
                            num_examples=1,
                        )
                    make_image(f"{city}_000000_000000_gt{mode}_instanceIds.png")
                    make_image(f"{city}_000000_000000_gt{mode}_labelIds.png")
                    make_image(f"{city}_000000_000000_gt{mode}_color.png", size=(4, 10, 10))

                    polygon_target_name = gt_dir / split / city / f"{city}_000000_000000_gt{mode}_polygons.json"
                    with open(polygon_target_name, "w") as outfile:
                        json.dump(polygon_target, outfile)

        # Create leftImg8bit folder
        for split in ['test', 'train_extra', 'train', 'val']:
            for city in cities:
                datasets_utils.create_image_folder(
                    root=tmpdir / "leftImg8bit" / split,
                    name=city,
                    file_name_fn=lambda _: f"{city}_000000_000000_leftImg8bit.png",
                    num_examples=1,
                )

        info = {'num_examples': len(cities)}
        if config['target_type'] == 'polygon':
            info['expected_polygon_target'] = polygon_target
        return info

    def test_combined_targets(self):
        target_types = ['semantic', 'polygon', 'color']

        with self.create_dataset(target_type=target_types) as (dataset, _):
            output = dataset[0]
            self.assertTrue(isinstance(output, tuple))
            self.assertTrue(len(output) == 2)
            self.assertTrue(isinstance(output[0], PIL.Image.Image))
            self.assertTrue(isinstance(output[1], tuple))
            self.assertTrue(len(output[1]) == 3)
            self.assertTrue(isinstance(output[1][0], PIL.Image.Image))  # semantic
            self.assertTrue(isinstance(output[1][1], dict))  # polygon
            self.assertTrue(isinstance(output[1][2], PIL.Image.Image))  # color

    def test_feature_types_target_color(self):
        with self.create_dataset(target_type='color') as (dataset, _):
            color_img, color_target = dataset[0]
            self.assertTrue(isinstance(color_img, PIL.Image.Image))
            self.assertTrue(np.array(color_target).shape[2] == 4)

    def test_feature_types_target_polygon(self):
        with self.create_dataset(target_type='polygon') as (dataset, info):
            polygon_img, polygon_target = dataset[0]
            self.assertTrue(isinstance(polygon_img, PIL.Image.Image))
            self.assertEqual(polygon_target, info['expected_polygon_target'])


class ImageNetTestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.ImageNet
    REQUIRED_PACKAGES = ('scipy',)
    ADDITIONAL_CONFIGS = datasets_utils.combinations_grid(split=('train', 'val'))

    def inject_fake_data(self, tmpdir, config):
        tmpdir = pathlib.Path(tmpdir)

        wnid = 'n01234567'
        if config['split'] == 'train':
            num_examples = 3
            datasets_utils.create_image_folder(
                root=tmpdir,
                name=tmpdir / 'train' / wnid / wnid,
                file_name_fn=lambda image_idx: f"{wnid}_{image_idx}.JPEG",
                num_examples=num_examples,
            )
        else:
            num_examples = 1
            datasets_utils.create_image_folder(
                root=tmpdir,
                name=tmpdir / 'val' / wnid,
                file_name_fn=lambda image_ifx: "ILSVRC2012_val_0000000{image_idx}.JPEG",
                num_examples=num_examples,
            )

        wnid_to_classes = {wnid: [1]}
        torch.save((wnid_to_classes, None), tmpdir / 'meta.bin')
        return num_examples


class CIFAR10TestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.CIFAR10
    ADDITIONAL_CONFIGS = datasets_utils.combinations_grid(train=(True, False))

    _VERSION_CONFIG = dict(
        base_folder="cifar-10-batches-py",
        train_files=tuple(f"data_batch_{idx}" for idx in range(1, 6)),
        test_files=("test_batch",),
        labels_key="labels",
        meta_file="batches.meta",
        num_categories=10,
        categories_key="label_names",
    )

    def inject_fake_data(self, tmpdir, config):
        tmpdir = pathlib.Path(tmpdir) / self._VERSION_CONFIG["base_folder"]
        os.makedirs(tmpdir)

        num_images_per_file = 1
        for name in itertools.chain(self._VERSION_CONFIG["train_files"], self._VERSION_CONFIG["test_files"]):
            self._create_batch_file(tmpdir, name, num_images_per_file)

        categories = self._create_meta_file(tmpdir)

        return dict(
            num_examples=num_images_per_file
            * len(self._VERSION_CONFIG["train_files"] if config["train"] else self._VERSION_CONFIG["test_files"]),
            categories=categories,
        )

    def _create_batch_file(self, root, name, num_images):
        data = datasets_utils.create_image_or_video_tensor((num_images, 32 * 32 * 3))
        labels = np.random.randint(0, self._VERSION_CONFIG["num_categories"], size=num_images).tolist()
        self._create_binary_file(root, name, {"data": data, self._VERSION_CONFIG["labels_key"]: labels})

    def _create_meta_file(self, root):
        categories = [
            f"{idx:0{len(str(self._VERSION_CONFIG['num_categories'] - 1))}d}"
            for idx in range(self._VERSION_CONFIG["num_categories"])
        ]
        self._create_binary_file(
            root, self._VERSION_CONFIG["meta_file"], {self._VERSION_CONFIG["categories_key"]: categories}
        )
        return categories

    def _create_binary_file(self, root, name, content):
        with open(pathlib.Path(root) / name, "wb") as fh:
            pickle.dump(content, fh)

    def test_class_to_idx(self):
        with self.create_dataset() as (dataset, info):
            expected = {category: label for label, category in enumerate(info["categories"])}
            actual = dataset.class_to_idx
            self.assertEqual(actual, expected)


class CIFAR100(CIFAR10TestCase):
    DATASET_CLASS = datasets.CIFAR100

    _VERSION_CONFIG = dict(
        base_folder="cifar-100-python",
        train_files=("train",),
        test_files=("test",),
        labels_key="fine_labels",
        meta_file="meta",
        num_categories=100,
        categories_key="fine_label_names",
    )


class CelebATestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.CelebA
    FEATURE_TYPES = (PIL.Image.Image, (torch.Tensor, int, tuple, type(None)))

    ADDITIONAL_CONFIGS = datasets_utils.combinations_grid(
        split=("train", "valid", "test", "all"),
        target_type=("attr", "identity", "bbox", "landmarks", ["attr", "identity"]),
    )
    REQUIRED_PACKAGES = ("pandas",)

    _SPLIT_TO_IDX = dict(train=0, valid=1, test=2)

    def inject_fake_data(self, tmpdir, config):
        base_folder = pathlib.Path(tmpdir) / "celeba"
        os.makedirs(base_folder)

        num_images, num_images_per_split = self._create_split_txt(base_folder)

        datasets_utils.create_image_folder(
            base_folder, "img_align_celeba", lambda idx: f"{idx + 1:06d}.jpg", num_images
        )
        attr_names = self._create_attr_txt(base_folder, num_images)
        self._create_identity_txt(base_folder, num_images)
        self._create_bbox_txt(base_folder, num_images)
        self._create_landmarks_txt(base_folder, num_images)

        return dict(num_examples=num_images_per_split[config["split"]], attr_names=attr_names)

    def _create_split_txt(self, root):
        num_images_per_split = dict(train=3, valid=2, test=1)

        data = [
            [self._SPLIT_TO_IDX[split]] for split, num_images in num_images_per_split.items() for _ in range(num_images)
        ]
        self._create_txt(root, "list_eval_partition.txt", data)

        num_images_per_split["all"] = num_images = sum(num_images_per_split.values())
        return num_images, num_images_per_split

    def _create_attr_txt(self, root, num_images):
        header = ("5_o_Clock_Shadow", "Young")
        data = torch.rand((num_images, len(header))).ge(0.5).int().mul(2).sub(1).tolist()
        self._create_txt(root, "list_attr_celeba.txt", data, header=header, add_num_examples=True)
        return header

    def _create_identity_txt(self, root, num_images):
        data = torch.randint(1, 4, size=(num_images, 1)).tolist()
        self._create_txt(root, "identity_CelebA.txt", data)

    def _create_bbox_txt(self, root, num_images):
        header = ("x_1", "y_1", "width", "height")
        data = torch.randint(10, size=(num_images, len(header))).tolist()
        self._create_txt(
            root, "list_bbox_celeba.txt", data, header=header, add_num_examples=True, add_image_id_to_header=True
        )

    def _create_landmarks_txt(self, root, num_images):
        header = ("lefteye_x", "rightmouth_y")
        data = torch.randint(10, size=(num_images, len(header))).tolist()
        self._create_txt(root, "list_landmarks_align_celeba.txt", data, header=header, add_num_examples=True)

    def _create_txt(self, root, name, data, header=None, add_num_examples=False, add_image_id_to_header=False):
        with open(pathlib.Path(root) / name, "w") as fh:
            if add_num_examples:
                fh.write(f"{len(data)}\n")

            if header:
                if add_image_id_to_header:
                    header = ("image_id", *header)
                fh.write(f"{' '.join(header)}\n")

            for idx, line in enumerate(data, 1):
                fh.write(f"{' '.join((f'{idx:06d}.jpg', *[str(value) for value in line]))}\n")

    def test_combined_targets(self):
        target_types = ["attr", "identity", "bbox", "landmarks"]

        individual_targets = []
        for target_type in target_types:
            with self.create_dataset(target_type=target_type) as (dataset, _):
                _, target = dataset[0]
                individual_targets.append(target)

        with self.create_dataset(target_type=target_types) as (dataset, _):
            _, combined_targets = dataset[0]

        actual = len(individual_targets)
        expected = len(combined_targets)
        self.assertEqual(
            actual,
            expected,
            f"The number of the returned combined targets does not match the the number targets if requested "
            f"individually: {actual} != {expected}",
        )

        for target_type, combined_target, individual_target in zip(target_types, combined_targets, individual_targets):
            with self.subTest(target_type=target_type):
                actual = type(combined_target)
                expected = type(individual_target)
                self.assertIs(
                    actual,
                    expected,
                    f"Type of the combined target does not match the type of the corresponding individual target: "
                    f"{actual} is not {expected}",
                )

    def test_no_target(self):
        with self.create_dataset(target_type=[]) as (dataset, _):
            _, target = dataset[0]

        self.assertIsNone(target)

    def test_attr_names(self):
        with self.create_dataset() as (dataset, info):
            self.assertEqual(tuple(dataset.attr_names), info["attr_names"])


class VOCSegmentationTestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.VOCSegmentation
    FEATURE_TYPES = (PIL.Image.Image, PIL.Image.Image)

    ADDITIONAL_CONFIGS = (
        *datasets_utils.combinations_grid(
            year=[f"20{year:02d}" for year in range(7, 13)], image_set=("train", "val", "trainval")
        ),
        dict(year="2007", image_set="test"),
        dict(year="2007-test", image_set="test"),
    )

    def inject_fake_data(self, tmpdir, config):
        year, is_test_set = (
            ("2007", True)
            if config["year"] == "2007-test" or config["image_set"] == "test"
            else (config["year"], False)
        )
        image_set = config["image_set"]

        base_dir = pathlib.Path(tmpdir)
        if year == "2011":
            base_dir /= "TrainVal"
        base_dir = base_dir / "VOCdevkit" / f"VOC{year}"
        os.makedirs(base_dir)

        num_images, num_images_per_image_set = self._create_image_set_files(base_dir, "ImageSets", is_test_set)
        datasets_utils.create_image_folder(base_dir, "JPEGImages", lambda idx: f"{idx:06d}.jpg", num_images)

        datasets_utils.create_image_folder(base_dir, "SegmentationClass", lambda idx: f"{idx:06d}.png", num_images)
        annotation = self._create_annotation_files(base_dir, "Annotations", num_images)

        return dict(num_examples=num_images_per_image_set[image_set], annotation=annotation)

    def _create_image_set_files(self, root, name, is_test_set):
        root = pathlib.Path(root) / name
        src = pathlib.Path(root) / "Main"
        os.makedirs(src, exist_ok=True)

        idcs = dict(train=(0, 1, 2), val=(3, 4), test=(5,))
        idcs["trainval"] = (*idcs["train"], *idcs["val"])

        for image_set in ("test",) if is_test_set else ("train", "val", "trainval"):
            self._create_image_set_file(src, image_set, idcs[image_set])

        shutil.copytree(src, root / "Segmentation")

        num_images = max(itertools.chain(*idcs.values())) + 1
        num_images_per_image_set = dict([(image_set, len(idcs_)) for image_set, idcs_ in idcs.items()])
        return num_images, num_images_per_image_set

    def _create_image_set_file(self, root, image_set, idcs):
        with open(pathlib.Path(root) / f"{image_set}.txt", "w") as fh:
            fh.writelines([f"{idx:06d}\n" for idx in idcs])

    def _create_annotation_files(self, root, name, num_images):
        root = pathlib.Path(root) / name
        os.makedirs(root)

        for idx in range(num_images):
            annotation = self._create_annotation_file(root, f"{idx:06d}.xml")

        return annotation

    def _create_annotation_file(self, root, name):
        def add_child(parent, name, text=None):
            child = ET.SubElement(parent, name)
            child.text = text
            return child

        def add_name(obj, name="dog"):
            add_child(obj, "name", name)
            return name

        def add_bndbox(obj, bndbox=None):
            if bndbox is None:
                bndbox = {"xmin": "1", "xmax": "2", "ymin": "3", "ymax": "4"}

            obj = add_child(obj, "bndbox")
            for name, text in bndbox.items():
                add_child(obj, name, text)

            return bndbox

        annotation = ET.Element("annotation")
        obj = add_child(annotation, "object")
        data = dict(name=add_name(obj), bndbox=add_bndbox(obj))

        with open(pathlib.Path(root) / name, "wb") as fh:
            fh.write(ET.tostring(annotation))

        return data


class VOCDetectionTestCase(VOCSegmentationTestCase):
    DATASET_CLASS = datasets.VOCDetection
    FEATURE_TYPES = (PIL.Image.Image, dict)

    def test_annotations(self):
        with self.create_dataset() as (dataset, info):
            _, target = dataset[0]

            self.assertIn("annotation", target)
            annotation = target["annotation"]

            self.assertIn("object", annotation)
            objects = annotation["object"]

            self.assertEqual(len(objects), 1)
            object = objects[0]

            self.assertEqual(object, info["annotation"])


class CocoDetectionTestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.CocoDetection
    FEATURE_TYPES = (PIL.Image.Image, list)

    REQUIRED_PACKAGES = ("pycocotools",)

    _IMAGE_FOLDER = "images"
    _ANNOTATIONS_FOLDER = "annotations"
    _ANNOTATIONS_FILE = "annotations.json"

    def dataset_args(self, tmpdir, config):
        tmpdir = pathlib.Path(tmpdir)
        root = tmpdir / self._IMAGE_FOLDER
        annotation_file = tmpdir / self._ANNOTATIONS_FOLDER / self._ANNOTATIONS_FILE
        return root, annotation_file

    def inject_fake_data(self, tmpdir, config):
        tmpdir = pathlib.Path(tmpdir)

        num_images = 3
        num_annotations_per_image = 2

        files = datasets_utils.create_image_folder(
            tmpdir, name=self._IMAGE_FOLDER, file_name_fn=lambda idx: f"{idx:012d}.jpg", num_examples=num_images
        )
        file_names = [file.relative_to(tmpdir / self._IMAGE_FOLDER) for file in files]

        annotation_folder = tmpdir / self._ANNOTATIONS_FOLDER
        os.makedirs(annotation_folder)
        info = self._create_annotation_file(
            annotation_folder, self._ANNOTATIONS_FILE, file_names, num_annotations_per_image
        )

        info["num_examples"] = num_images
        return info

    def _create_annotation_file(self, root, name, file_names, num_annotations_per_image):
        image_ids = [int(file_name.stem) for file_name in file_names]
        images = [dict(file_name=str(file_name), id=id) for file_name, id in zip(file_names, image_ids)]

        annotations, info = self._create_annotations(image_ids, num_annotations_per_image)
        self._create_json(root, name, dict(images=images, annotations=annotations))

        return info

    def _create_annotations(self, image_ids, num_annotations_per_image):
        annotations = datasets_utils.combinations_grid(
            image_id=image_ids, bbox=([1.0, 2.0, 3.0, 4.0],) * num_annotations_per_image
        )
        for id, annotation in enumerate(annotations):
            annotation["id"] = id
        return annotations, dict()

    def _create_json(self, root, name, content):
        file = pathlib.Path(root) / name
        with open(file, "w") as fh:
            json.dump(content, fh)
        return file


class CocoCaptionsTestCase(CocoDetectionTestCase):
    DATASET_CLASS = datasets.CocoCaptions

    def _create_annotations(self, image_ids, num_annotations_per_image):
        captions = [str(idx) for idx in range(num_annotations_per_image)]
        annotations = datasets_utils.combinations_grid(image_id=image_ids, caption=captions)
        for id, annotation in enumerate(annotations):
            annotation["id"] = id
        return annotations, dict(captions=captions)

    def test_captions(self):
        with self.create_dataset() as (dataset, info):
            _, captions = dataset[0]
            self.assertEqual(tuple(captions), tuple(info["captions"]))


class UCF101TestCase(datasets_utils.VideoDatasetTestCase):
    DATASET_CLASS = datasets.UCF101

    ADDITIONAL_CONFIGS = datasets_utils.combinations_grid(fold=(1, 2, 3), train=(True, False))

    _VIDEO_FOLDER = "videos"
    _ANNOTATIONS_FOLDER = "annotations"

    def dataset_args(self, tmpdir, config):
        tmpdir = pathlib.Path(tmpdir)
        root = tmpdir / self._VIDEO_FOLDER
        annotation_path = tmpdir / self._ANNOTATIONS_FOLDER
        return root, annotation_path

    def inject_fake_data(self, tmpdir, config):
        tmpdir = pathlib.Path(tmpdir)

        video_folder = tmpdir / self._VIDEO_FOLDER
        os.makedirs(video_folder)
        video_files = self._create_videos(video_folder)

        annotations_folder = tmpdir / self._ANNOTATIONS_FOLDER
        os.makedirs(annotations_folder)
        num_examples = self._create_annotation_files(annotations_folder, video_files, config["fold"], config["train"])

        return num_examples

    def _create_videos(self, root, num_examples_per_class=3):
        def file_name_fn(cls, idx, clips_per_group=2):
            return f"v_{cls}_g{(idx // clips_per_group) + 1:02d}_c{(idx % clips_per_group) + 1:02d}.avi"

        video_files = [
            datasets_utils.create_video_folder(root, cls, lambda idx: file_name_fn(cls, idx), num_examples_per_class)
            for cls in ("ApplyEyeMakeup", "YoYo")
        ]
        return [path.relative_to(root) for path in itertools.chain(*video_files)]

    def _create_annotation_files(self, root, video_files, fold, train):
        current_videos = random.sample(video_files, random.randrange(1, len(video_files) - 1))
        current_annotation = self._annotation_file_name(fold, train)
        self._create_annotation_file(root, current_annotation, current_videos)

        other_videos = set(video_files) - set(current_videos)
        other_annotations = [
            self._annotation_file_name(fold, train) for fold, train in itertools.product((1, 2, 3), (True, False))
        ]
        other_annotations.remove(current_annotation)
        for name in other_annotations:
            self._create_annotation_file(root, name, other_videos)

        return len(current_videos)

    def _annotation_file_name(self, fold, train):
        return f"{'train' if train else 'test'}list{fold:02d}.txt"

    def _create_annotation_file(self, root, name, video_files):
        with open(pathlib.Path(root) / name, "w") as fh:
            fh.writelines(f"{file}\n" for file in sorted(video_files))


class LSUNTestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.LSUN

    REQUIRED_PACKAGES = ("lmdb",)
    ADDITIONAL_CONFIGS = datasets_utils.combinations_grid(
        classes=("train", "test", "val", ["bedroom_train", "church_outdoor_train"])
    )

    _CATEGORIES = (
        "bedroom",
        "bridge",
        "church_outdoor",
        "classroom",
        "conference_room",
        "dining_room",
        "kitchen",
        "living_room",
        "restaurant",
        "tower",
    )

    def inject_fake_data(self, tmpdir, config):
        root = pathlib.Path(tmpdir)

        num_images = 0
        for cls in self._parse_classes(config["classes"]):
            num_images += self._create_lmdb(root, cls)

        return num_images

    @contextlib.contextmanager
    def create_dataset(
        self,
        *args, **kwargs
    ):
        with super().create_dataset(*args, **kwargs) as output:
            yield output
            # Currently datasets.LSUN caches the keys in the current directory rather than in the root directory. Thus,
            # this creates a number of unique _cache_* files in the current directory that will not be removed together
            # with the temporary directory
            for file in os.listdir(os.getcwd()):
                if file.startswith("_cache_"):
                    os.remove(file)

    def _parse_classes(self, classes):
        if not isinstance(classes, str):
            return classes

        split = classes
        if split == "test":
            return [split]

        return [f"{category}_{split}" for category in self._CATEGORIES]

    def _create_lmdb(self, root, cls):
        lmdb = datasets_utils.lazy_importer.lmdb
        hexdigits_lowercase = string.digits + string.ascii_lowercase[:6]

        folder = f"{cls}_lmdb"

        num_images = torch.randint(1, 4, size=()).item()
        format = "png"
        files = datasets_utils.create_image_folder(root, folder, lambda idx: f"{idx}.{format}", num_images)

        with lmdb.open(str(root / folder)) as env, env.begin(write=True) as txn:
            for file in files:
                key = "".join(random.choice(hexdigits_lowercase) for _ in range(40)).encode()

                buffer = io.BytesIO()
                Image.open(file).save(buffer, format)
                buffer.seek(0)
                value = buffer.read()

                txn.put(key, value)

                os.remove(file)

        return num_images

    def test_not_found_or_corrupted(self):
        # LSUN does not raise built-in exception, but a custom one. It is expressive enough to not 'cast' it to
        # RuntimeError or FileNotFoundError that are normally checked by this test.
        with self.assertRaises(datasets_utils.lazy_importer.lmdb.Error):
            super().test_not_found_or_corrupted()


class Kinetics400TestCase(datasets_utils.VideoDatasetTestCase):
    DATASET_CLASS = datasets.Kinetics400

    def inject_fake_data(self, tmpdir, config):
        classes = ("Abseiling", "Zumba")
        num_videos_per_class = 2

        digits = string.ascii_letters + string.digits + "-_"
        for cls in classes:
            datasets_utils.create_video_folder(
                tmpdir,
                cls,
                lambda _: f"{datasets_utils.create_random_string(11, digits)}.avi",
                num_videos_per_class,
            )

        return num_videos_per_class * len(classes)


class HMDB51TestCase(datasets_utils.VideoDatasetTestCase):
    DATASET_CLASS = datasets.HMDB51

    ADDITIONAL_CONFIGS = datasets_utils.combinations_grid(fold=(1, 2, 3), train=(True, False))

    _VIDEO_FOLDER = "videos"
    _SPLITS_FOLDER = "splits"
    _CLASSES = ("brush_hair", "wave")

    def dataset_args(self, tmpdir, config):
        tmpdir = pathlib.Path(tmpdir)
        root = tmpdir / self._VIDEO_FOLDER
        annotation_path = tmpdir / self._SPLITS_FOLDER
        return root, annotation_path

    def inject_fake_data(self, tmpdir, config):
        tmpdir = pathlib.Path(tmpdir)

        video_folder = tmpdir / self._VIDEO_FOLDER
        os.makedirs(video_folder)
        video_files = self._create_videos(video_folder)

        splits_folder = tmpdir / self._SPLITS_FOLDER
        os.makedirs(splits_folder)
        num_examples = self._create_split_files(splits_folder, video_files, config["fold"], config["train"])

        return num_examples

    def _create_videos(self, root, num_examples_per_class=3):
        def file_name_fn(cls, idx, clips_per_group=2):
            return f"{cls}_{(idx // clips_per_group) + 1:d}_{(idx % clips_per_group) + 1:d}.avi"

        return [
            (
                cls,
                datasets_utils.create_video_folder(
                    root,
                    cls,
                    lambda idx: file_name_fn(cls, idx),
                    num_examples_per_class,
                ),
            )
            for cls in self._CLASSES
        ]

    def _create_split_files(self, root, video_files, fold, train):
        num_videos = num_train_videos = 0

        for cls, videos in video_files:
            num_videos += len(videos)

            train_videos = set(random.sample(videos, random.randrange(1, len(videos) - 1)))
            num_train_videos += len(train_videos)

            with open(pathlib.Path(root) / f"{cls}_test_split{fold}.txt", "w") as fh:
                fh.writelines(f"{file.name} {1 if file in train_videos else 2}\n" for file in videos)

        return num_train_videos if train else (num_videos - num_train_videos)


class OmniglotTestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.Omniglot

    ADDITIONAL_CONFIGS = datasets_utils.combinations_grid(background=(True, False))

    def inject_fake_data(self, tmpdir, config):
        target_folder = (
            pathlib.Path(tmpdir) / "omniglot-py" / f"images_{'background' if config['background'] else 'evaluation'}"
        )
        os.makedirs(target_folder)

        num_images = 0
        for name in ("Alphabet_of_the_Magi", "Tifinagh"):
            num_images += self._create_alphabet_folder(target_folder, name)

        return num_images

    def _create_alphabet_folder(self, root, name):
        num_images_total = 0
        for idx in range(torch.randint(1, 4, size=()).item()):
            num_images = torch.randint(1, 4, size=()).item()
            num_images_total += num_images

            datasets_utils.create_image_folder(
                root / name, f"character{idx:02d}", lambda image_idx: f"{image_idx:02d}.png", num_images
            )

        return num_images_total


class SBUTestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.SBU
    FEATURE_TYPES = (PIL.Image.Image, str)

    def inject_fake_data(self, tmpdir, config):
        num_images = 3

        dataset_folder = pathlib.Path(tmpdir) / "dataset"
        images = datasets_utils.create_image_folder(tmpdir, "dataset", self._create_file_name, num_images)

        self._create_urls_txt(dataset_folder, images)
        self._create_captions_txt(dataset_folder, num_images)

        return num_images

    def _create_file_name(self, idx):
        part1 = datasets_utils.create_random_string(10, string.digits)
        part2 = datasets_utils.create_random_string(10, string.ascii_lowercase, string.digits[:6])
        return f"{part1}_{part2}.jpg"

    def _create_urls_txt(self, root, images):
        with open(root / "SBU_captioned_photo_dataset_urls.txt", "w") as fh:
            for image in images:
                fh.write(
                    f"http://static.flickr.com/{datasets_utils.create_random_string(4, string.digits)}/{image.name}\n"
                )

    def _create_captions_txt(self, root, num_images):
        with open(root / "SBU_captioned_photo_dataset_captions.txt", "w") as fh:
            for _ in range(num_images):
                fh.write(f"{datasets_utils.create_random_string(10)}\n")


class SEMEIONTestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.SEMEION

    def inject_fake_data(self, tmpdir, config):
        num_images = 3

        images = torch.rand(num_images, 256)
        labels = F.one_hot(torch.randint(10, size=(num_images,)))
        with open(pathlib.Path(tmpdir) / "semeion.data", "w") as fh:
            for image, one_hot_labels in zip(images, labels):
                image_columns = " ".join([f"{pixel.item():.4f}" for pixel in image])
                labels_columns = " ".join([str(label.item()) for label in one_hot_labels])
                fh.write(f"{image_columns} {labels_columns}\n")

        return num_images


class USPSTestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.USPS

    ADDITIONAL_CONFIGS = datasets_utils.combinations_grid(train=(True, False))

    def inject_fake_data(self, tmpdir, config):
        num_images = 2 if config["train"] else 1

        images = torch.rand(num_images, 256) * 2 - 1
        labels = torch.randint(1, 11, size=(num_images,))

        with bz2.open(pathlib.Path(tmpdir) / f"usps{'.t' if not config['train'] else ''}.bz2", "w") as fh:
            for image, label in zip(images, labels):
                line = " ".join((str(label.item()), *[f"{idx}:{pixel:.6f}" for idx, pixel in enumerate(image, 1)]))
                fh.write(f"{line}\n".encode())

        return num_images


class SBDatasetTestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.SBDataset
    FEATURE_TYPES = (PIL.Image.Image, (np.ndarray, PIL.Image.Image))

    REQUIRED_PACKAGES = ("scipy.io", "scipy.sparse")

    ADDITIONAL_CONFIGS = datasets_utils.combinations_grid(
        image_set=("train", "val", "train_noval"), mode=("boundaries", "segmentation")
    )

    _NUM_CLASSES = 20

    def inject_fake_data(self, tmpdir, config):
        num_images, num_images_per_image_set = self._create_split_files(tmpdir)

        sizes = self._create_target_folder(tmpdir, "cls", num_images)

        datasets_utils.create_image_folder(
            tmpdir, "img", lambda idx: f"{self._file_stem(idx)}.jpg", num_images, size=lambda idx: sizes[idx]
        )

        return num_images_per_image_set[config["image_set"]]

    def _create_split_files(self, root):
        root = pathlib.Path(root)

        splits = dict(train=(0, 1, 2), train_noval=(0, 2), val=(3,))

        for split, idcs in splits.items():
            self._create_split_file(root, split, idcs)

        num_images = max(itertools.chain(*splits.values())) + 1
        num_images_per_split = dict([(split, len(idcs)) for split, idcs in splits.items()])
        return num_images, num_images_per_split

    def _create_split_file(self, root, name, idcs):
        with open(root / f"{name}.txt", "w") as fh:
            fh.writelines(f"{self._file_stem(idx)}\n" for idx in idcs)

    def _create_target_folder(self, root, name, num_images):
        io = datasets_utils.lazy_importer.scipy.io

        target_folder = pathlib.Path(root) / name
        os.makedirs(target_folder)

        sizes = [torch.randint(1, 4, size=(2,)).tolist() for _ in range(num_images)]
        for idx, size in enumerate(sizes):
            content = dict(
                GTcls=dict(Boundaries=self._create_boundaries(size), Segmentation=self._create_segmentation(size))
            )
            io.savemat(target_folder / f"{self._file_stem(idx)}.mat", content)

        return sizes

    def _create_boundaries(self, size):
        sparse = datasets_utils.lazy_importer.scipy.sparse
        return [
            [sparse.csc_matrix(torch.randint(0, 2, size=size, dtype=torch.uint8).numpy())]
            for _ in range(self._NUM_CLASSES)
        ]

    def _create_segmentation(self, size):
        return torch.randint(0, self._NUM_CLASSES + 1, size=size, dtype=torch.uint8).numpy()

    def _file_stem(self, idx):
        return f"2008_{idx:06d}"


class FakeDataTestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.FakeData
    FEATURE_TYPES = (PIL.Image.Image, int)

    def dataset_args(self, tmpdir, config):
        return ()

    def inject_fake_data(self, tmpdir, config):
        return config["size"]

    def test_not_found_or_corrupted(self):
        self.skipTest("The data is generated at creation and thus cannot be non-existent or corrupted.")


class PhotoTourTestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.PhotoTour

    # The PhotoTour dataset returns examples with different features with respect to the 'train' parameter. Thus,
    # we overwrite 'FEATURE_TYPES' with a dummy value to satisfy the initial checks of the base class. Furthermore, we
    # overwrite the 'test_feature_types()' method to select the correct feature types before the test is run.
    FEATURE_TYPES = ()
    _TRAIN_FEATURE_TYPES = (torch.Tensor,)
    _TEST_FEATURE_TYPES = (torch.Tensor, torch.Tensor, torch.Tensor)

    datasets_utils.combinations_grid(train=(True, False))

    _NAME = "liberty"

    def dataset_args(self, tmpdir, config):
        return tmpdir, self._NAME

    def inject_fake_data(self, tmpdir, config):
        tmpdir = pathlib.Path(tmpdir)

        # In contrast to the original data, the fake images injected here comprise only a single patch. Thus,
        # num_images == num_patches.
        num_patches = 5

        image_files = self._create_images(tmpdir, self._NAME, num_patches)
        point_ids, info_file = self._create_info_file(tmpdir / self._NAME, num_patches)
        num_matches, matches_file = self._create_matches_file(tmpdir / self._NAME, num_patches, point_ids)

        self._create_archive(tmpdir, self._NAME, *image_files, info_file, matches_file)

        return num_patches if config["train"] else num_matches

    def _create_images(self, root, name, num_images):
        # The images in the PhotoTour dataset comprises of multiple grayscale patches of 64 x 64 pixels. Thus, the
        # smallest fake image is 64 x 64 pixels and comprises a single patch.
        return datasets_utils.create_image_folder(
            root, name, lambda idx: f"patches{idx:04d}.bmp", num_images, size=(1, 64, 64)
        )

    def _create_info_file(self, root, num_images):
        point_ids = torch.randint(num_images, size=(num_images,)).tolist()

        file = root / "info.txt"
        with open(file, "w") as fh:
            fh.writelines([f"{point_id} 0\n" for point_id in point_ids])

        return point_ids, file

    def _create_matches_file(self, root, num_patches, point_ids):
        lines = [
            f"{patch_id1} {point_ids[patch_id1]} 0 {patch_id2} {point_ids[patch_id2]} 0\n"
            for patch_id1, patch_id2 in itertools.combinations(range(num_patches), 2)
        ]

        file = root / "m50_100000_100000_0.txt"
        with open(file, "w") as fh:
            fh.writelines(lines)

        return len(lines), file

    def _create_archive(self, root, name, *files):
        archive = root / f"{name}.zip"
        with zipfile.ZipFile(archive, "w") as zip:
            for file in files:
                zip.write(file, arcname=file.relative_to(root))

        return archive

    @datasets_utils.test_all_configs
    def test_feature_types(self, config):
        feature_types = self.FEATURE_TYPES
        self.FEATURE_TYPES = self._TRAIN_FEATURE_TYPES if config["train"] else self._TEST_FEATURE_TYPES
        try:
            super().test_feature_types.__wrapped__(self, config)
        finally:
            self.FEATURE_TYPES = feature_types


class Flickr8kTestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.Flickr8k

    FEATURE_TYPES = (PIL.Image.Image, list)

    _IMAGES_FOLDER = "images"
    _ANNOTATIONS_FILE = "captions.html"

    def dataset_args(self, tmpdir, config):
        tmpdir = pathlib.Path(tmpdir)
        root = tmpdir / self._IMAGES_FOLDER
        ann_file = tmpdir / self._ANNOTATIONS_FILE
        return str(root), str(ann_file)

    def inject_fake_data(self, tmpdir, config):
        num_images = 3
        num_captions_per_image = 3

        tmpdir = pathlib.Path(tmpdir)

        images = self._create_images(tmpdir, self._IMAGES_FOLDER, num_images)
        self._create_annotations_file(tmpdir, self._ANNOTATIONS_FILE, images, num_captions_per_image)

        return dict(num_examples=num_images, captions=self._create_captions(num_captions_per_image))

    def _create_images(self, root, name, num_images):
        return datasets_utils.create_image_folder(root, name, self._image_file_name, num_images)

    def _image_file_name(self, idx):
        id = datasets_utils.create_random_string(10, string.digits)
        checksum = datasets_utils.create_random_string(10, string.digits, string.ascii_lowercase[:6])
        size = datasets_utils.create_random_string(1, "qwcko")
        return f"{id}_{checksum}_{size}.jpg"

    def _create_annotations_file(self, root, name, images, num_captions_per_image):
        with open(root / name, "w") as fh:
            fh.write("<table>")
            for image in (None, *images):
                self._add_image(fh, image, num_captions_per_image)
            fh.write("</table>")

    def _add_image(self, fh, image, num_captions_per_image):
        fh.write("<tr>")
        self._add_image_header(fh, image)
        fh.write("</tr><tr><td><ul>")
        self._add_image_captions(fh, num_captions_per_image)
        fh.write("</ul></td></tr>")

    def _add_image_header(self, fh, image=None):
        if image:
            url = f"http://www.flickr.com/photos/user/{image.name.split('_')[0]}/"
            data = f'<a href="{url}">{url}</a>'
        else:
            data = "Image Not Found"
        fh.write(f"<td>{data}</td>")

    def _add_image_captions(self, fh, num_captions_per_image):
        for caption in self._create_captions(num_captions_per_image):
            fh.write(f"<li>{caption}")

    def _create_captions(self, num_captions_per_image):
        return [str(idx) for idx in range(num_captions_per_image)]

    def test_captions(self):
        with self.create_dataset() as (dataset, info):
            _, captions = dataset[0]
            self.assertSequenceEqual(captions, info["captions"])


class Flickr30kTestCase(Flickr8kTestCase):
    DATASET_CLASS = datasets.Flickr30k

    FEATURE_TYPES = (PIL.Image.Image, list)

    _ANNOTATIONS_FILE = "captions.token"

    def _image_file_name(self, idx):
        return f"{idx}.jpg"

    def _create_annotations_file(self, root, name, images, num_captions_per_image):
        with open(root / name, "w") as fh:
            for image, (idx, caption) in itertools.product(
                images, enumerate(self._create_captions(num_captions_per_image))
            ):
                fh.write(f"{image.name}#{idx}\t{caption}\n")


class MNISTTestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.MNIST

    ADDITIONAL_CONFIGS = datasets_utils.combinations_grid(train=(True, False))

    _MAGIC_DTYPES = {
        torch.uint8: 8,
        torch.int8: 9,
        torch.int16: 11,
        torch.int32: 12,
        torch.float32: 13,
        torch.float64: 14,
    }

    _IMAGES_SIZE = (28, 28)
    _IMAGES_DTYPE = torch.uint8

    _LABELS_SIZE = ()
    _LABELS_DTYPE = torch.uint8

    def inject_fake_data(self, tmpdir, config):
        raw_dir = pathlib.Path(tmpdir) / self.DATASET_CLASS.__name__ / "raw"
        os.makedirs(raw_dir, exist_ok=True)

        num_images = self._num_images(config)
        self._create_binary_file(
            raw_dir, self._images_file(config), (num_images, *self._IMAGES_SIZE), self._IMAGES_DTYPE
        )
        self._create_binary_file(
            raw_dir, self._labels_file(config), (num_images, *self._LABELS_SIZE), self._LABELS_DTYPE
        )
        return num_images

    def _num_images(self, config):
        return 2 if config["train"] else 1

    def _images_file(self, config):
        return f"{self._prefix(config)}-images-idx3-ubyte"

    def _labels_file(self, config):
        return f"{self._prefix(config)}-labels-idx1-ubyte"

    def _prefix(self, config):
        return "train" if config["train"] else "t10k"

    def _create_binary_file(self, root, filename, size, dtype):
        with open(pathlib.Path(root) / filename, "wb") as fh:
            for meta in (self._magic(dtype, len(size)), *size):
                fh.write(self._encode(meta))

            # If ever an MNIST variant is added that uses floating point data, this should be adapted.
            data = torch.randint(0, torch.iinfo(dtype).max + 1, size, dtype=dtype)
            fh.write(data.numpy().tobytes())

    def _magic(self, dtype, dims):
        return self._MAGIC_DTYPES[dtype] * 256 + dims

    def _encode(self, v):
        return torch.tensor(v, dtype=torch.int32).numpy().tobytes()[::-1]


class FashionMNISTTestCase(MNISTTestCase):
    DATASET_CLASS = datasets.FashionMNIST


class KMNISTTestCase(MNISTTestCase):
    DATASET_CLASS = datasets.KMNIST


class EMNISTTestCase(MNISTTestCase):
    DATASET_CLASS = datasets.EMNIST

    DEFAULT_CONFIG = dict(split="byclass")
    ADDITIONAL_CONFIGS = datasets_utils.combinations_grid(
        split=("byclass", "bymerge", "balanced", "letters", "digits", "mnist"), train=(True, False)
    )

    def _prefix(self, config):
        return f"emnist-{config['split']}-{'train' if config['train'] else 'test'}"


class QMNISTTestCase(MNISTTestCase):
    DATASET_CLASS = datasets.QMNIST

    ADDITIONAL_CONFIGS = datasets_utils.combinations_grid(what=("train", "test", "test10k", "nist"))

    _LABELS_SIZE = (8,)
    _LABELS_DTYPE = torch.int32

    def _num_images(self, config):
        if config["what"] == "nist":
            return 3
        elif config["what"] == "train":
            return 2
        elif config["what"] == "test50k":
            # The split 'test50k' is defined as the last 50k images beginning at index 10000. Thus, we need to create
            # more than 10000 images for the dataset to not be empty. Since this takes significantly longer than the
            # creation of all other splits, this is excluded from the 'ADDITIONAL_CONFIGS' and is tested only once in
            # 'test_num_examples_test50k'.
            return 10001
        else:
            return 1

    def _labels_file(self, config):
        return f"{self._prefix(config)}-labels-idx2-int"

    def _prefix(self, config):
        if config["what"] == "nist":
            return "xnist"

        if config["what"] is None:
            what = "train" if config["train"] else "test"
        elif config["what"].startswith("test"):
            what = "test"
        else:
            what = config["what"]

        return f"qmnist-{what}"

    def test_num_examples_test50k(self):
        with self.create_dataset(what="test50k") as (dataset, info):
            # Since the split 'test50k' selects all images beginning from the index 10000, we subtract the number of
            # created examples by this.
            self.assertEqual(len(dataset), info["num_examples"] - 10000)


class DatasetFolderTestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.DatasetFolder

    # The dataset has no fixed return type since it is defined by the loader parameter. For testing, we use a loader
    # that simply returns the path as type 'str' instead of loading anything. See the 'dataset_args()' method.
    FEATURE_TYPES = (str, int)

    _IMAGE_EXTENSIONS = ("jpg", "png")
    _VIDEO_EXTENSIONS = ("avi", "mp4")
    _EXTENSIONS = (*_IMAGE_EXTENSIONS, *_VIDEO_EXTENSIONS)

    # DatasetFolder has two mutually exclusive parameters: 'extensions' and 'is_valid_file'. One of both is required.
    # We only iterate over different 'extensions' here and handle the tests for 'is_valid_file' in the
    # 'test_is_valid_file()' method.
    DEFAULT_CONFIG = dict(extensions=_EXTENSIONS)
    ADDITIONAL_CONFIGS = (
        *datasets_utils.combinations_grid(extensions=[(ext,) for ext in _IMAGE_EXTENSIONS]),
        dict(extensions=_IMAGE_EXTENSIONS),
        *datasets_utils.combinations_grid(extensions=[(ext,) for ext in _VIDEO_EXTENSIONS]),
        dict(extensions=_VIDEO_EXTENSIONS),
    )

    def dataset_args(self, tmpdir, config):
        return tmpdir, lambda x: x

    def inject_fake_data(self, tmpdir, config):
        extensions = config["extensions"] or self._is_valid_file_to_extensions(config["is_valid_file"])

        num_examples_total = 0
        classes = []
        for ext, cls in zip(self._EXTENSIONS, string.ascii_letters):
            if ext not in extensions:
                continue

            create_example_folder = (
                datasets_utils.create_image_folder
                if ext in self._IMAGE_EXTENSIONS
                else datasets_utils.create_video_folder
            )

            num_examples = torch.randint(1, 3, size=()).item()
            create_example_folder(tmpdir, cls, lambda idx: self._file_name_fn(cls, ext, idx), num_examples)

            num_examples_total += num_examples
            classes.append(cls)

        return dict(num_examples=num_examples_total, classes=classes)

    def _file_name_fn(self, cls, ext, idx):
        return f"{cls}_{idx}.{ext}"

    def _is_valid_file_to_extensions(self, is_valid_file):
        return {ext for ext in self._EXTENSIONS if is_valid_file(f"foo.{ext}")}

    @datasets_utils.test_all_configs
    def test_is_valid_file(self, config):
        extensions = config.pop("extensions")
        # We need to explicitly pass extensions=None here or otherwise it would be filled by the value from the
        # DEFAULT_CONFIG.
        with self.create_dataset(
                config, extensions=None, is_valid_file=lambda file: pathlib.Path(file).suffix[1:] in extensions
        ) as (dataset, info):
            self.assertEqual(len(dataset), info["num_examples"])

    @datasets_utils.test_all_configs
    def test_classes(self, config):
        with self.create_dataset(config) as (dataset, info):
            self.assertSequenceEqual(dataset.classes, info["classes"])


class ImageFolderTestCase(datasets_utils.ImageDatasetTestCase):
    DATASET_CLASS = datasets.ImageFolder

    def inject_fake_data(self, tmpdir, config):
        num_examples_total = 0
        classes = ("a", "b")
        for cls in classes:
            num_examples = torch.randint(1, 3, size=()).item()
            num_examples_total += num_examples

            datasets_utils.create_image_folder(tmpdir, cls, lambda idx: f"{cls}_{idx}.png", num_examples)

        return dict(num_examples=num_examples_total, classes=classes)

    @datasets_utils.test_all_configs
    def test_classes(self, config):
        with self.create_dataset(config) as (dataset, info):
            self.assertSequenceEqual(dataset.classes, info["classes"])


if __name__ == "__main__":
    unittest.main()
