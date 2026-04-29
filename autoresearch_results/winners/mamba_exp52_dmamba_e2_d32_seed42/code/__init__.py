"""autoresearchindexstock — equity-index variant of the FX autoresearch loop.

Mirrors the architecture of the sibling ``autoresearch`` package but targets
the QQQ ETF (Nasdaq-100) with equity-index-native features. Reuses the shared
``autoresearch.model`` backbones and training loop; everything in
``data/``, ``evaluation/``, and ``run_autoresearch.py`` is purpose-built for
equity index forecasting.
"""

__version__ = "0.1.0"
