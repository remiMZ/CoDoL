import glob
import os.path as osp

from dassl.datasets.build import DATASET_REGISTRY
from dassl.datasets.base_dataset import Datum, SingleDatasetBase

from dassl.utils import listdir_nohidden

@DATASET_REGISTRY.register()
class OfficeHome(SingleDatasetBase):
    """Office-Home.

    Statistics:
        - Around 15,500 images.
        - 65 classes related to office and home objects.
        - 4 domains: Art, Clipart, Product, Real World.
        - URL: http://hemanthdv.org/OfficeHome-Dataset/.

    Reference:
        - Venkateswara et al. Deep Hashing Network for Unsupervised
        Domain Adaptation. CVPR 2017.
    """
    dataset_dir = "office_home"
    domains = ["art", "clipart", "product", "real_world"]
    data_url = "https://drive.google.com/uc?id=1gkbf_KaxoBws-GWT3XIPZ7BnkqbAxIFa"

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)

        if not osp.exists(self.dataset_dir):
            dst = osp.join(root, "office_home.zip")
            self.download_data(self.data_url, dst, from_gdrive=True)

        self.check_input_domains(
            cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS
        )

        train_0 = self._read_data(cfg.DATASET.SOURCE_DOMAINS_0, "train")
        train_1 = self._read_data(cfg.DATASET.SOURCE_DOMAINS_1, "train")
        train_2 = self._read_data(cfg.DATASET.SOURCE_DOMAINS_2, "train")
        val = self._read_data(cfg.DATASET.SOURCE_DOMAINS, "val")
        test = self._read_data(cfg.DATASET.TARGET_DOMAINS, "val")

        super().__init__(train_0=train_0, train_1=train_1, train_2=train_2, val=val, test=test)
        
    def _read_data(self, input_domains, split):

        def _load_data_from_directory(directory):
            folders = listdir_nohidden(directory)
            folders.sort()
            items_ = []

            for label, folder in enumerate(folders):
                impaths = glob.glob(osp.join(directory, folder, "*.jpg"))

                for impath in impaths:
                    items_.append((impath, label))

            return items_

        items = []
        for domain, dname in enumerate(input_domains):
            if split == "all":
                train_dir = osp.join(self.dataset_dir, dname, "train")
                impath_label_list = _load_data_from_directory(train_dir)
                val_dir = osp.join(self.dataset_dir, dname, "val")
                impath_label_list += _load_data_from_directory(val_dir)
            else:
                split_dir = osp.join(self.dataset_dir, dname, split)
                impath_label_list = _load_data_from_directory(split_dir)

            for impath, label in impath_label_list:
                class_name = impath.split("/")[-2].lower()
                item = Datum(
                    impath=impath,
                    label=label,
                    domain=domain,
                    classname=class_name
                )
                items.append(item)
                
        return items

