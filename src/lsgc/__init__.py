"""Longitudinal Spatial Graph Convolution (LSGC).

A continuous-filter graph convolution whose filter is jointly parameterized by
3D position difference (Delta x, y, z in mm) and visit-time difference (Delta t
in imaging visits). Applied to a single unified spatio-temporal supervoxel graph
so that spatial context and longitudinal evolution are handled by one operator.
"""
from .lsgc_layer import LSGCConv, LSGCNet
from .graph_builder import build_spatiotemporal_graph

__all__ = ["LSGCConv", "LSGCNet", "build_spatiotemporal_graph"]
