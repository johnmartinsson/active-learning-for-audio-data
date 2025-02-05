from abc import ABC, abstractmethod
import wave

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
        with wave.open(audio_file, 'rb') as wf:
            frame_rate = wf.getframerate()
            num_frames = wf.getnframes()
            audio = wf.readframes(num_frames)
        
        segment_size = self.segment_length * frame_rate * wf.getsampwidth()
        segments = [audio[i:i + segment_size] for i in range(0, len(audio), segment_size)]
        return segments