"""Model layer — re-exports the proven backbones from the sibling
``autoresearch`` package so we do not maintain two copies of the architecture
zoo. The equity-index variant only forks the *data* / *features* / *splits*
layer, not the model code itself.
"""

from autoresearch.model.backbone import (  # noqa: F401
    BACKBONE_SEQ_LEN,
    GBMWrapper,
    create_model,
    get_seq_len,
)
from autoresearch.model.train import (  # noqa: F401
    create_contiguous_datasets,
    find_contiguous_segments,
    train_one_epoch,
)
