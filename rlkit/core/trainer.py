import abc


class Trainer(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def train(self, data):
        pass

    def end_epoch(self, epoch):
        pass

    def get_snapshot(self):
        return {}

    def get_diagnostics(self):
        return {}

class IQTrainer(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def train(self, policy_data, expert_data):
        pass

    def end_epoch(self, epoch):
        pass

    def get_snapshot(self):
        return {}

    def get_diagnostics(self):
        return {}
