from abc import ABC, abstractmethod
import librosa

class LabelingStrategy(ABC):
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def segment_audio(self, audio_file):
        pass

class FixedLengthLabelingStrategy(LabelingStrategy):
    def __init__(self, model, segment_length):
        super().__init__(model)
        self.segment_length = segment_length

    def segment_audio(self, audio_file):
        y, sr = librosa.load(audio_file)
        segment_length_samples = int(self.segment_length * sr)
        num_segments = len(y) // segment_length_samples
        segments = [(i * self.segment_length, (i + 1) * self.segment_length) for i in range(num_segments)]
        return segments