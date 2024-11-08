class EpochCallback:
    def __init__(self):
        self.callbacks = {}

    def add_callback(self, epoch, callback):
        if epoch not in self.callbacks:
            self.callbacks[epoch] = []
        self.callbacks[epoch].append(callback)

    def execute(self, epoch):
        if epoch in self.callbacks:
            for callback in self.callbacks[epoch]:
                callback()