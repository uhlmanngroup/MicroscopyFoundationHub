# src/dino_peft/datasets/flat_image_folder.py
# Python file for handling flat image folder datasets for unsupervised learning.

from pathlib import Path
from typing import Callable, List, Optional, Tuple

from torch.utils.data import Dataset

from dino_peft.utils.image_loading import center_crop_image, load_rgb_image, parse_center_crop_size


class FlatImageFolder(Dataset):
    """
    Simple dataset for a flat folder of images.

    - root_dir: Path to the folder containing images.
    - transform: any torchvision-style transformations to apply to the images.
    - valid_extensions: list of extensions to keep (case-insensitive).
    """

    def __init__(
        self,
        root_dir: str | Path,
        transform: Optional[Callable] = None,
        valid_extensions: Optional[List[str]] = None,
        center_crop_size=None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.center_crop_size = parse_center_crop_size(center_crop_size)

        if valid_extensions is None:
            valid_extensions = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]

        # store normalized extensions
        self.valid_exts = {e.lower() for e in valid_extensions}

        if not self.root_dir.is_dir():
            raise ValueError(f"Provided root_dir '{root_dir}' is not a valid directory.")

        # Gather all valid image file paths
        self.image_paths: List[Path] = []
        for p in sorted(self.root_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in self.valid_exts:
                self.image_paths.append(p)

        if not self.image_paths:
            raise ValueError(
                f"No valid image files found in '{root_dir}' "
                f"with extensions {sorted(self.valid_exts)}."
            )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple:
        image_path = self.image_paths[index]
        image = load_rgb_image(image_path)
        if self.center_crop_size is not None:
            image = center_crop_image(image, self.center_crop_size)

        if self.transform is not None:
            image = self.transform(image)

        return image, str(image_path)
