from sampling_strategy import RandomSampleStrategy
from labeling_strategy import FixedLengthLabelingStrategy
from model import DummyModel

import glob

# Initialize the model (placeholder)
model = DummyModel()

# Initialize strategies
sample_strategy = RandomSampleStrategy(batch_size=5, model=model)
labeling_strategy = FixedLengthLabelingStrategy(segment_length=10, model=model)

# Placeholder for audio files
audio_files = glob.glob("./data/baby_cries/*.wav")

# Main loop
while True:
    # 1. Sample audio files using the sampling strategy
    sampled_files = sample_strategy.sample_audio_files(audio_files)

    # 2. Label the audio files using the labeling strategy
    for audio_file in sampled_files:
        # a. Split audio file into segments
        segments = labeling_strategy.segment_audio(audio_file)
        
        # b. Visualize the audio waveform, spectrogram, and the segments
        # (Placeholder for visualization code)
        
        # c. Allow the user to give the segments a presence/absence label
        labels = []
        for segment in segments:
            # (Placeholder for user input code)
            label = input(f"Label for segment (presence/absence): ")
            labels.append(label)
        
        # d. Save the labels for the audio file
        # (Placeholder for saving labels code)
        print(f"Labels for {audio_file}: {labels}")
    
    # 3. Update the model with the labeled audio files
    model.update_model({'audio_files': sampled_files, 'labels': labels})
    
    # 4. Save the model
    model.save_model()
    
    # 5. Repeat from 1 until stopping criterion is met
    # (Placeholder for stopping criterion)
    if input("Continue? (y/n): ") != 'y':
        break