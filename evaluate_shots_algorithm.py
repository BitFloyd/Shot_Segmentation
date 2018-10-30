import matplotlib as mpl
mpl.use('Agg')
import os
from shot_segmentor_pkg.Shotify import VideoToShotConverter,PlotShotSegmentationParams
from sys import argv
from argparse_pkg import argparse_fns as af


parameters = af.getopts(argv)
parse_dict = af.parse_arguments(parameters)

pathToVideo = parse_dict['video']
pathToShots = parse_dict['target_folder']
debug = parse_dict['debug']

vtsc = VideoToShotConverter(pathToVideo,pathToShots,slidingWindowLength=None,debug_mode=debug)

vtsc.segmentVideoToShots()

# plotter = PlotShotSegmentationParams(vtsc)
# plotter.plotOF()
# plotter.plotSlopes()
# plotter.plotRatios()