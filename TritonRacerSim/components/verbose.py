
from TritonRacerSim.components.component import Component


class Verbose(Component):
    """ Print out a list of variables for debugging """
    def __init__(self, cfg):
        self.arg_list = cfg['verbose']
        super().__init__(inputs=self.arg_list)

    def step(self, *args):
        display = ''
        for i in range(len(args)):
            arg = args[i]
            if isinstance(arg, float): # Beautify printing float
                arg = "{0:0.4f}".format(arg)
            display += f"{self.arg_list[i]}: {arg} "
        print(f"\r{display}", end='')

    def getName(self):
        return "Verbose"