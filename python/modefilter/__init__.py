"""
Mode filter

The mode filter is an edge-preserving smoothing filter by taking the local mode
of the empirical density. The empirical null filter uses the mode filter (aka
the null mean) and the null std filter to normalise the image for hypothesis
testing

In this python implementation, only the CUDA GPU version is implemented, thus
an NVIDIA GPU is required

The filters are implemented in the classes `ModeFilter` and
`EmpiricalNullFilter`. They are also available as napari plugins in
`ModeFilterContainer` and `EmpiricalNullFilterContainer`

For these containers to be usable in napari, see the file `napari.yaml`
"""

from modefilter.modefilter import ModeFilter
from modefilter.modefilter import EmpiricalNullFilter

from modefilter._widget import ModeFilterContainer
from modefilter._widget import EmpiricalNullFilterContainer

__all__ = ["ModeFilter", "EmpiricalNullFilter"]
