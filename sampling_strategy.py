from abc import ABC, abstractmethod
import random

class SampleStrategy(ABC):
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def sample_audio_files(self, audio_files):
        pass

class RandomSampleStrategy(SampleStrategy):
    def __init__(self, batch_size, model):
        super().__init__(model)
        self.batch_size = batch_size

    def sample_audio_files(self, audio_files):
        return random.sample(audio_files, self.batch_size)