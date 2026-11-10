"""Dataset entry point for CMCA-Net."""
from .model import CMCANet as _CMCANet


class CMCANet(_CMCANet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, dataset='voc', **kwargs)
