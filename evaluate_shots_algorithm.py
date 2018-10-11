import os
from shot_segmentor_pkg.Shotify import VideoToShotConverter,PlotShotSegmentationParams
from sys import argv
from argparse_pkg import argparse_fns as af


parameters = af.getopts(argv)
parse_dict = af.parse_arguments(parameters)

pathToVideo = parse_dict['video']
pathToShots = parse_dict['target_folder']

vtsc = VideoToShotConverter(pathToVideo,pathToShots,slidingWindowLength=None)

vtsc.plotOpticalFlowSamplingWindow()
vtsc.segmentVideoToShots()

plotter = PlotShotSegmentationParams(vtsc)
plotter.plotOF()
plotter.plotSlopes()
plotter.plotRatios()