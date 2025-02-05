from abc import ABC, abstractmethod

class Model(ABC):
    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def update_model(self, data):
        pass

    @abstractmethod
    def save_model(self):
        pass

    @abstractmethod
    def predict(self, audio_file):
        pass

class DummyModel(Model):
    def load_model(self):
        print("Loading model... (dummy)")

    def update_model(self, data):
        print("Updating model with data... (dummy)")

    def save_model(self):
        print("Saving model... (dummy)")

    def predict(self, audio_file):
        print("Predicting... (dummy)")