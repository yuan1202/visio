import argparse
from visio import Visio


parser = argparse.ArgumentParser()
parser.add_argument('-v', '--video', type=str)
parser.add_argument('-r', '--record', action="store_true", help="Switch to record video.")
parser.add_argument('-bt', '--bluetooth', action="store_true", help="Enable bluetooth.")

args = parser.parse_args()


if __name__ == '__main__':

    # get arguments
    args = vars(args)
    # create app and run
    app = Visio(**args)
    app.run()