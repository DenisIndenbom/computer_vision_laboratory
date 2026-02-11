import os
import zipfile
import tarfile

from torch.utils.data import Dataset
from urllib import request
from typing import Callable
from PIL import Image


class BaseImageFolderDataset(Dataset):
    URL: str = ''
    ARCHIVE_NAME: str = ''
    EXTRACTED_FOLDER: str = ''

    def __init__(
        self,
        root: str,
        transform: Callable | None = None,
        download: bool = False,
    ):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.dataset_dir = os.path.join(self.root, self.EXTRACTED_FOLDER)

        if download:
            self._download_and_extract()

        if not os.path.isdir(self.dataset_dir):
            raise RuntimeError(
                'Dataset not found. Set download=True to download it.'
            )

        self.classes = self._find_classes()
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.samples = self._make_dataset()

    def _find_classes(self) -> list[str]:
        return sorted(
            d.name for d in os.scandir(self.dataset_dir) if d.is_dir()
        )

    def _make_dataset(self) -> list[tuple[str, int]]:
        samples = []

        for cls in self.classes:
            cls_dir = os.path.join(self.dataset_dir, cls)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    samples.append(
                        (os.path.join(cls_dir, fname), self.class_to_idx[cls])
                    )

        return samples

    @staticmethod
    def _bytes_to_human(bytes_value: float) -> str:
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_value < 1024 or unit == 'TB':
                return f'{bytes_value:.2f} {unit}'
            bytes_value /= 1024

    @classmethod
    def _reporthook(cls, count, block_size, total_size):
        downloaded = count * block_size
        downloaded_str = cls._bytes_to_human(downloaded)

        if total_size <= 0:
            print(f'\rDownloaded: {downloaded_str}', end='', flush=True)
            return

        total_str = cls._bytes_to_human(total_size)
        percent = min(100, (downloaded * 100) / total_size)

        print(f'\rProgress: {percent:.1f}% ({downloaded_str} / {total_str})',
              end='',
              flush=True)

        if percent >= 100:
            print(f'\nDownload complete! Total: {total_str}')

    def _extract_archive(self, archive_path: str):
        if archive_path.endswith('.zip'):
            with zipfile.ZipFile(archive_path, 'r') as zf:
                zf.extractall(self.root)
        elif archive_path.endswith(('.tar', '.tar.gz', '.tgz', '.tar.bz2')):
            with tarfile.open(archive_path, 'r:*') as tf:
                tf.extractall(self.root)
        else:
            raise RuntimeError(
                f'Unsupported archive format: {archive_path}'
            )

    def _download_and_extract(self):
        if not self.URL:
            raise NotImplementedError('URL must be defined in subclass')

        os.makedirs(self.root, exist_ok=True)
        archive_path = os.path.join(self.root, self.ARCHIVE_NAME)

        if not os.path.exists(self.dataset_dir):

            if not os.path.exists(archive_path):
                print(f'Downloading {self.__class__.__name__}...')
                request.urlretrieve(self.URL, archive_path, self._reporthook)

            print(f'Extracting {self.__class__.__name__}...')
            self._extract_archive(archive_path)
            print('Extracted!')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Image.Image, int]:
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label
