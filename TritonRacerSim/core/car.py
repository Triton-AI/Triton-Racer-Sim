from TritonRacerSim.core.datapool import DataPool
from TritonRacerSim.components.component import Component
from threading import Thread
import time

class Car:
    def __init__(self, loop_hz=30):
        self.pool = DataPool()
        self.components = []
        self.component_threads = []
        self.loop_hz = loop_hz

    def addComponent(self, component=Component()):
        assert issubclass(type(component), Component)
        
        self.components.append(component)
        self.pool.add(component)
        print(f'Added component: {component.getName()}')
        if component.threaded:
            component_thread = Thread(target=component.thread_step, args=(), daemon=True)
            self.component_threads.append(component_thread)
            print (f'   Seperate thread added')
        
    def start(self):
        #Ready
        print('\n[Initiating Start Sequence]')
        for component in self.components:
            component.onStart() 

        print('\n[Arming threaded components]')
        for t in self.component_threads:
            t.start()

        #Go
        print ('\n[Launch!]')
        try:
            loop_time = 1.0 / self.loop_hz
            compromised = False            
            while True:
                begin_time = time.time()
                for component in self.components:
                    args = self.pool.get_inputs_for(component)
                    returns = component.step(*args)
                    self.pool.store_outputs_for(component, returns)
                duration = time.time() - begin_time

                if duration > loop_time:
                    if not compromised:
                        compromised = True
                        print(f'Loop frequency compromised! Actual time: {duration * 1000} ms')
                else:
                    time_left = loop_time - duration
                    time.sleep(time_left)

        except KeyboardInterrupt:
            pass
        finally:
            self.stop()


    def stop(self):
        print('[Stopping car]')
        for component in self.components:
            component.onShutdown()

