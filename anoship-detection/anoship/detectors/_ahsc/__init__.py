"""Faithful AHSC building blocks.

Internal package implementing the *Anti-Drosophila Habituation Stream Clustering*
(AHSC) algorithm of:

    Hancheng Xiao, WeiFu Zhu, Zhipeng Qiu, Zhixia Zeng, Shi Zhang, Ruliang Xiao.
    "Anti-Drosophila Habituation Clustering for Enhanced Anomaly Detection in
    Data Streams." IEEE ISCIPT 2025.

The pieces are split to mirror the paper:

* :mod:`projection`   -- Drosophila olfactory sparse projection + winner-take-all
* :mod:`enhancement`  -- anti-habituation similarity enhancement (Eq. 8-9)
* :mod:`microcluster` -- evolving micro/macro-cluster structure
* :mod:`ahsc`         -- the assembled online stream clusterer
"""

from __future__ import annotations
