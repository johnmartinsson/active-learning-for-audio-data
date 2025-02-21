// components/AnnotationTool.js
import React, { useState } from 'react';
import Spectrogram from './Spectrogram';
import Waveform from './Waveform';
import SegmentLabeler from './SegmentLabeler';

const AnnotationTool = ({ file, onLabelsSubmitted, labelingStrategyChoice, numSegments }) => {
    const [currentTime, setCurrentTime] = useState(0);
    const [isPlaying, setIsPlaying] = useState(false);
    const [segments, setSegments] = useState([]);
    const [labels, setLabels] = useState([]);

    // Define a resetState function
    const resetState = () => {
        setIsPlaying(false);
        setCurrentTime(0);
        setSegments([]);
        setLabels([]);
    };

    const handlePlayPause = () => {
        setIsPlaying(!isPlaying);
    };

    const toggleLabel = (time) => {
        if (!segments || segments.length === 0) {
            console.error('Segments are not initialized');
            return;
        }

        const updatedLabels = labels.map((label, index) => {
            const segment = segments[index];
            if (segment && time >= segment.start && time < segment.end) {
                return label === 'absence' ? 'presence' : 'absence';
            }
            return label;
        });
        setLabels(updatedLabels);
    };

    const handleSubmit = async () => {
        const data = segments.map((segment, index) => ({
          start_time: segment.start,
          end_time: segment.end,
          label: labels[index]
        }));

        try {
          const response = await fetch('http://localhost:5000/api/audio/submit_labels', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              filename: file.filename,
              labels: data
            }),
          });

          const result = await response.json();
          if (response.ok) {
            console.log('Labels submitted successfully:', result.message);

            // After successful submission, tell App.js to fetch a fresh batch and reset state
            onLabelsSubmitted();
            resetState();
          } else {
            console.error('Failed to submit labels:', result.message);
          }
        } catch (error) {
          console.error('Error submitting labels:', error);
        }
      };

    return (
        <div className="annotation-tool">
            <h3>{file.filename}</h3>

            <div className="media-container">
                <Spectrogram
                    src={`http://localhost:5000${file.spectrogram_path}`}
                    currentTime={currentTime}
                    segments={segments}
                    labels={labels}
                    toggleLabel={toggleLabel}
                    duration={file.audio_length}
                />
                <SegmentLabeler
                    file={file}
                    numSegments={numSegments} // Pass numSegments prop down
                    strategyChoice={labelingStrategyChoice} // Pass labelingStrategyChoice as strategyChoice prop (for now, rename in SegmentLabeler if needed)
                    setSegments={setSegments}
                    setLabels={setLabels}
                />
                <Waveform
                    src={`http://localhost:5000${file.audio_path}`}
                    currentTime={currentTime}
                    setCurrentTime={setCurrentTime}
                    isPlaying={isPlaying}
                    duration={file.audio_length}
                />
                <button onClick={handlePlayPause}>{isPlaying ? 'Pause' : 'Play'}</button>
                <button onClick={handleSubmit}>Submit Labels</button>
            </div>
        </div>
    );
};

export default AnnotationTool;