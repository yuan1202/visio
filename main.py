import argparse
from visio import Visio
from config import pipeline_config

parser = argparse.ArgumentParser()
parser.add_argument('video', type=str)
parser.add_argument('-r', '--record', type=str, help="Path to video file to be used for recorded video.")
parser.add_argument('-bt', '--bluetooth', action="store_true", help="Enable bluetooth.")

args = parser.parse_args()


if __name__ == '__main__':

    # get arguments
    args = vars(args)
    # include configurations
    args.update(pipeline_config._asdict())
    # create app and run
    app = Visio(**args)
    app.run()