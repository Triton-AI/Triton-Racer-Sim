"""
Script to visualize data

Usage:
    data_plotter.py (--y=<y_data_name>) [--x=<x_data_name>]
"""

from os import path
import matplotlib.pyplot as plt
import json
import docopt

# Plot the data in a folder

count = 1

def get_record_path(idx, parent):
    return path.join(parent, f'record_{idx}.json')

def load_data(data_folder, x_axis, y_axis):
    data_folder = path.abspath(data_folder)
    x = []
    y = []
    try:
        while True:
            with open(get_record_path(count, data_folder), 'r') as data_file:
                data = json.load(data_file)
                if x_axis == None:
                    x.append(count)
                else:
                    x.append(data[x_axis])
                y.append(data[y_axis])
            count += 1

    except FileNotFoundError:
        pass
    print(f"Found {count - 1} records")
    return x, y

def plot(x, y, x_label, y_label):
    plt.plot(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

if __name__ == '__main__':
    args = docopt(__doc__)
    y_axis = args['--y']
    x_axis = None
    if '--x' in args:
        x_axis = args['--x']
    x, y = 