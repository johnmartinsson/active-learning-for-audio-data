// ./frontend/src/components/AnnotationTool.js
import React, { useState, useEffect } from 'react';
import Waveform from './Waveform';
import Spectrogram from './Spectrogram';
import ProbabilityChart from './ProbabilityChart';

const AnnotationTool = ({ file, onLabelsSubmitted, labelingStrategyChoice, numSegments }) => {
  const [currentTime, setCurrentTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  const [segments, setSegments] = useState([]);
  const [labels, setLabels] = useState([]);

  // Probability data (only relevant if labelingStrategyChoice is "active")
  const [probabilities, setProbabilities] = useState([]);
  const [timings, setTimings] = useState([]);

  // On first load or whenever file/strategy/numSegments change, fetch from the backend
  useEffect(() => {
    const fetchSegments = async () => {
      try {
        const response = await fetch('http://localhost:5000/api/audio/segments', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            filename: file.filename,
            labelingStrategyChoice,
            numSegments
          })
        });
        if (!response.ok) {
          throw new Error(`Failed to fetch segments: ${response.statusText}`);
        }
        const data = await response.json();

        // data.segments, data.suggestedLabels, data.probabilities, data.timings
        setSegments(data.segments || []);
        setLabels(data.suggestedLabels || []);
        setProbabilities(data.probabilities || []);
        setTimings(data.timings || []);
      } catch (err) {
        console.error('Error fetching segments from server:', err);
      }
    };

    fetchSegments();
  }, [file, labelingStrategyChoice, numSegments]);

  const handlePlayPause = () => {
    setIsPlaying(!isPlaying);
  };

  const toggleLabel = (time) => {
    // Example: flip the label in whichever segment the user clicked
    const updatedLabels = labels.map((label, i) => {
      const seg = segments[i];
      if (time >= seg.start && time < seg.end) {
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
        onLabelsSubmitted();
        // reset if you wish
        setIsPlaying(false);
        setCurrentTime(0);
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
        {labelingStrategyChoice === 'active' && (
          <ProbabilityChart
            probabilities={probabilities}
            timings={timings}
            audioLength={file.audio_length}
          />
        )}
        <Waveform
          src={`http://localhost:5000${file.audio_path}`}
          currentTime={currentTime}
          setCurrentTime={setCurrentTime}
          isPlaying={isPlaying}
          duration={file.audio_length}
        />
      </div>
      <button onClick={handlePlayPause}>{isPlaying ? 'Pause' : 'Play'}</button>
      <button onClick={handleSubmit}>Submit Labels</button>
    </div>
  );
};

export default AnnotationTool;
