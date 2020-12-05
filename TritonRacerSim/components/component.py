from abc import ABC

class Component(ABC):

    def __init__(self, inputs = [], outputs = [], threaded = False):
        """The name of input and output values must be provided as strings (e.g. 'speed', 'throttle')"""
        self.step_inputs = inputs.copy()
        self.step_outputs = outputs.copy()
        self.threaded = threaded

    def onStart(self):
        """Called right before the main loop begins"""
        pass

    def step(self, *args):
        """The component's behavior in the main loop"""
        pass

    def thread_step(self):
        """The component's behavior in its own thread"""
        pass

    def onShutdown(self):
        """Shutdown"""
        pass

    def getName(self):
        return 'Generic Component'

"""Template of a component"""
# from TritonRacerSim.components.component import Component
class CustomComponentTemplate(Component):
    def __init__(self, *some_args):
        super().__init__(inputs=[], outputs=[], threaded=False)
        # Other initializations

    def onStart(self):
        """What should I do right at the launch of the car?"""
        pass

    def step(self, *args):
        """What should I do each time the component is iterated?"""
        pass

    def thread_step(self):
        """The component's behavior in its own thread"""
        pass

    def onShutdown(self):
        """What to do at shutdown?"""
        pass

    def getName(self):
        return 'Generic Custom Component'
    
