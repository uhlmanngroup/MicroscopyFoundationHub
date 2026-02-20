from .paired_dirs_seg import PairedDirsSegDataset

class DrosoSegDataset(PairedDirsSegDataset):
    """
    Thin wrapper around PairedDirsSegDataset for the drosophila stack dataset.
    Images live under `raw/NN.tif` and masks in `mitochondria/NN.png`, so a
    simple stem match (with recursion) is sufficient.
    """
    def __init__(
        self,
        image_dir,
        mask_dir,
        img_size=(308, 308),
        to_rgb=True,
        transform=None,
        binarize=False,
        binarize_threshold=128,
        recursive=True,
        mask_prefix="",
        mask_suffix="",
    ):
        super().__init__(
            image_dir=image_dir,
            mask_dir=mask_dir,
            img_size=img_size,
            to_rgb=to_rgb,
            transform=transform,
            binarize=binarize,
            binarize_threshold=binarize_threshold,
            pair_mode="stem",
            mask_prefix=mask_prefix,
            mask_suffix=mask_suffix,
            recursive=recursive,
        )
