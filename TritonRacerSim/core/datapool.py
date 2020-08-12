from TritonRacerSim.components.component import Component

class DataPool:
    def __init__(self):
        self.pool = dict()

    def add(self, component=Component()):
        for data_name in component.step_inputs:
            self.pool[data_name] = 0.0

        for data_name in component.step_outputs:
            self.pool[data_name] = 0.0

    def get_inputs_for(self, component=Component()):
        inputs = [self.pool[data_name] for data_name in component.step_inputs]
        inputs = tuple(inputs)
        return inputs

    def store_outputs_for(self, component=Component(), output_values=None):
        if output_values is not None:
            for i in range(len(component.step_outputs)):
                self.pool[component.step_outputs[i]] = output_values[i]

    def get_value(self, name):
        return self.pool[name]

    def set_value(self, name, value):
        self.pool[name] = value