from sampling_strategy import RandomSampleStrategy
from labeling_strategy import FixedLengthLabelingStrategy
from model import DummyModel

import glob
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button  # Import the Button widget

# Initialize the model (placeholder)
model = DummyModel()

# Initialize strategies
sample_strategy = RandomSampleStrategy(batch_size=1, model=model)
labeling_strategy = FixedLengthLabelingStrategy(segment_length=5, model=model)

# Placeholder for audio files
audio_files = glob.glob("./data/baby_cries/*.wav")
print(audio_files)

# Main loop
running = True  # Control variable for the main loop
while running:
    # 1. Sample audio files using the sampling strategy
    sampled_files = sample_strategy.sample_audio_files(audio_files)

    # 2. Label the audio files using the labeling strategy
    for audio_file in sampled_files:
        # Define a global flag so that the program runs
        global flag
        flag = True

        # a. Split audio file into segments
        segments = labeling_strategy.segment_audio(audio_file)
        print(f"Segments for {audio_file}: {segments}")

        # Load audio data
        y, sr = librosa.load(audio_file)

        # Prepare for visualization
        num_segments = len(segments)
        labels = [0] * num_segments  # Initialize labels (0 = absence, 1 = presence)

        # b. Visualize the spectrogram of the audio waveform (top) the audio waveform (bottom), and the segments as bounding boxes that spann overlay both the spectrogram and the audio waveform
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True) # sharex to align time axes

        # Spectrogram
        D = librosa.stft(y)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log', ax=ax1)
        ax1.set_title('Spectrogram')
        ax1.set_xlabel('')  # Remove x-axis label
        #fig.colorbar(img, ax=ax1, format="%+2.f dB")  # Fix colorbar for the spectrogram


        # Waveform
        librosa.display.waveshow(y, sr=sr, ax=ax2)
        ax2.set_title('Waveform')

        # Store vertical lines and text for updating
        lines1 = []
        lines2 = []
        text_labels = []
        rects1 = [] # Bounding boxes for spectrogram
        rects2 = [] # Bounding boxes for waveform

        # Overlay vertical lines and segment labels
        for i, (start, end) in enumerate(segments):
            line1_start = ax1.axvline(x=start, color='r', linestyle='--', alpha=0.7)
            line1_end = ax1.axvline(x=end, color='r', linestyle='--', alpha=0.7)
            line2_start = ax2.axvline(x=start, color='r', linestyle='--', alpha=0.7)
            line2_end = ax2.axvline(x=end, color='r', linestyle='--', alpha=0.7)

            # Draw bounding box for the spectrogram (invisible by default)
            rect1 = ax1.axvspan(start, end, facecolor='r', alpha=0.3, visible=False, zorder=3) #set the rect on a separate layer
            rect2 = ax2.axvspan(start, end, facecolor='r', alpha=0.3) # Bounding box in wavefile

            # Initialize segment labels (absence)
            segment_label = "Absence Indicated"
            text1 = ax1.text((start + end) / 2, -0.10, segment_label, ha='center', va='center', transform=ax1.get_xaxis_transform(), color='black', fontsize=8, zorder=4, bbox={'facecolor': 'white', 'alpha': 0.7, 'edgecolor': 'none'})
            text2 = ax2.text((start + end) / 2, -0.10, segment_label, ha='center', va='center', transform=ax2.get_xaxis_transform(), color='black', fontsize=8, bbox={'facecolor': 'white', 'alpha': 0.7, 'edgecolor': 'none'})

            lines1.append((line1_start, line1_end))
            lines2.append((line2_start, line2_end))
            text_labels.append((text1, text2))
            rects1.append(rect1) # Store rectangle object
            rects2.append(rect2) # Store rectangle object

        fig.tight_layout(pad=2.0) # To make sure annotations do not overlap

        # c. Allow the user to toggle presence/absence in the visualized segments/bounding boxes
        def onclick(event):
            # Find which segment was clicked
            ax = event.inaxes
            if ax in [ax1, ax2]: #Make sure click happened on the spectrogram or waveform, otherwise ignore it
                for i, (start, end) in enumerate(segments):
                    if start <= event.xdata <= end:
                        labels[i] = 1 - labels[i]  # Toggle label

                        # Update colors
                        update_display(i)

                        fig.canvas.draw_idle() # redraw


        def update_display(i): # function to update the display for both plots
            if labels[i] == 1:
                #Text indicator
                text_labels[i][0].set_text("Presence Indicated") # Spectrogram
                text_labels[i][1].set_text("Presence Indicated") # Waveform

                # Update bounding box color
                rects1[i].set_facecolor('g')
                rects1[i].set_visible(True) # Make bounding box in spectrogram visible

                lines1[i][0].set_visible(False) # Start time line spectrogram
                lines1[i][1].set_visible(False) # End time line spectrogram

                rects2[i].set_facecolor('g') # Waveform
                text_labels[i][0].set_color('white') # Spectrogram
                text_labels[i][1].set_color('white') # Waveform
                text_labels[i][0].set_bbox({'facecolor': 'g', 'alpha': 0.7, 'edgecolor': 'none'}) # Spectrogram
                text_labels[i][1].set_bbox({'facecolor': 'g', 'alpha': 0.7, 'edgecolor': 'none'}) # Waveform
                text_labels[i][0].set_color('black') # Spectrogram
                text_labels[i][1].set_color('black') # Waveform
            else:
                #Text indicator
                text_labels[i][0].set_text("Absence Indicated") # Spectrogram
                text_labels[i][1].set_text("Absence Indicated") # Waveform

                rects1[i].set_visible(False) # Make bounding box invisible
                lines1[i][0].set_visible(True) # Start time line spectrogram
                lines1[i][1].set_visible(True) # End time line spectrogram

                rects2[i].set_facecolor('r') # Waveform
                text_labels[i][0].set_color('black') # Spectrogram
                text_labels[i][1].set_color('black') # Waveform
                text_labels[i][0].set_bbox({'facecolor': 'white', 'alpha': 0.7, 'edgecolor': 'none'}) # Spectrogram
                text_labels[i][1].set_bbox({'facecolor': 'white', 'alpha': 0.7, 'edgecolor': 'none'}) # Waveform

        # Add a "Save & Continue" button
        ax_button = plt.axes([0.70, 0.05, 0.15, 0.075])  # [left, bottom, width, height]
        save_button = Button(ax_button, 'Save & Continue')

        # Add a "Quit" button
        ax_quit_button = plt.axes([0.85, 0.05, 0.10, 0.075])  # [left, bottom, width, height]
        quit_button = Button(ax_quit_button, 'Quit')

        # Define the button's callback function for save button
        def on_save_clicked(event):
            #Access to the global variable `flag`
            global flag
            flag = False

            # d. Save the labels for the audio file
            labels_dict = dict(zip([(a[0],a[-1]) for a in segments], labels))

            # Placeholder for saving labels code (replace with your saving logic)
            print(f"Labels for {audio_file}: {labels_dict}")  # Print before saving

            plt.close(fig)  # Important: Close the figure to free memory and prevent warnings

        # Define the button's callback function for quit button
        def on_quit_clicked(event):
            #Access to the global variable `running`
            global running
            running = False

            global flag
            flag = False

            plt.close(fig)  # Important: Close the figure to free memory and prevent warnings

        # Assign the callback function to the button
        save_button.on_clicked(on_save_clicked)

        #Assign quit function
        quit_button.on_clicked(on_quit_clicked)

        fig.canvas.mpl_connect('button_press_event', onclick)  # Connect the click event

        plt.show() # display plot

        # d. Save the labels for the audio file

        while flag:
            plt.pause(0.001) # Let the GUI be drawn
        labels = dict(zip([(a[0],a[-1]) for a in segments], labels))

        
        # 3. Update the model with the labeled audio files
        model.update_model({'audio_files': sampled_files, 'labels': labels})
        
        # 4. Save the model
        model.save_model()
        
        # 5. Delete file from the audio files
        audio_files.remove(audio_file)
    

    # 5. Repeat from 1 until stopping criterion is met
    #The program ended due to quit button
    if not running:
        break # Break the outer annotation process

print("Quitting the program.")