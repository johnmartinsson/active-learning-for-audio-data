// ./frontend/src/components/AnnotationTool.js
import React, { useState, useEffect } from 'react';
import Waveform from './Waveform';
import Spectrogram from './Spectrogram';
import ProbabilityChart from './ProbabilityChart';
import ClassSelector from "./ClassSelector";

const AnnotationTool = ({
  file,
  onLabelsSubmitted,
  labelingStrategyChoice,
  numSegments,
  availableClasses,
  onClassesChange
}) => {
  const [currentTime, setCurrentTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  const [segments, setSegments] = useState([]);
  const [labels, setLabels] = useState([]);
  const [classes, setClasses] = useState(availableClasses && availableClasses.length > 0 ? availableClasses : ['background']);
  const [currentClass, setCurrentClass] = useState('background');

  const sameClassList = (a, b) => {
    if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) {
      return false;
    }
    return a.every((value, idx) => value === b[idx]);
  };

  useEffect(() => {
    if (!Array.isArray(availableClasses) || availableClasses.length === 0) {
      return;
    }

    setClasses(availableClasses);
    setCurrentClass((prev) => (availableClasses.includes(prev) ? prev : (availableClasses[1] || 'background')));
  }, [availableClasses]);

  // Probability data (only relevant if labelingStrategyChoice is "active")
  const [probabilities, setProbabilities] = useState([]);
  const [timings, setTimings] = useState([]);

  // On first load or whenever file/strategy/numSegments change, fetch from the backend
  useEffect(() => {
    const fetchSegments = async () => {
      try {
        // e.g. GET /api/audio/myfile/segments?labelingStrategyChoice=active&numSegments=10
        const url = new URL(`http://localhost:5000/api/audio/${file.filename}/segments`);
        url.searchParams.set('labelingStrategyChoice', labelingStrategyChoice);
        url.searchParams.set('numSegments', numSegments.toString());

        const response = await fetch(url, {
          method: 'GET'
        });
        if (!response.ok) {
          throw new Error(`Failed to fetch segments: ${response.statusText}`);
        }
        const data = await response.json();

        const safeSegments = data.segments || [];
        const suggested = data.suggestedLabels || [];

        const normalizedLabels = safeSegments.map((_, idx) => {
          const label = suggested[idx];

          // Legacy backend suggestions are binary; treat them as unlabeled/background.
          if (!label || label === 'presence' || label === 'absence' || label === 'background') {
            return 'background';
          }

          return label;
        });

        const discoveredClasses = Array.from(
          new Set(normalizedLabels.filter((label) => label && label !== 'background'))
        );
        const baseClasses = Array.isArray(availableClasses) && availableClasses.length > 0
          ? availableClasses
          : ['background'];
        const nextClasses = Array.from(new Set(['background', ...baseClasses, ...discoveredClasses]));

        setSegments(safeSegments);
        setLabels(normalizedLabels);
        setClasses(nextClasses);
        if (onClassesChange && !sameClassList(baseClasses, nextClasses)) {
          onClassesChange(nextClasses);
        }
        setCurrentClass((prev) => {
          if (nextClasses.includes(prev)) {
            return prev;
          }
          return nextClasses[1] || 'background';
        });
        setProbabilities(data.probabilities || []);
        setTimings(data.timings || []);
      } catch (err) {
        console.error('Error fetching segments from server:', err);
      }
    };

    fetchSegments();
  }, [file, labelingStrategyChoice, numSegments, availableClasses, onClassesChange]);

  const [selectorState, setSelectorState] = useState({
    visible: false,
    segmentIndex: null,
    x: 0,
    y: 0
  });

  const handlePlayPause = () => {
    setIsPlaying(!isPlaying);
  };

  const openClassSelector = (segmentIndex, x, y) => {
    setSelectorState({
      visible: true,
      segmentIndex,
      x,
      y
    });
  };

  const assignClass = (className) => {
    const newLabels = [...labels];
    newLabels[selectorState.segmentIndex] = className;

    setLabels(newLabels);

    setSelectorState({
      visible: false,
      segmentIndex: null,
      x: 0,
      y: 0
    });
  };

  const assignCurrentClassToSegment = (segmentIndex) => {
    if (segmentIndex < 0 || segmentIndex >= labels.length) {
      return;
    }

    const newLabels = [...labels];
    const existing = newLabels[segmentIndex] || 'background';
    newLabels[segmentIndex] = existing === currentClass ? 'background' : currentClass;
    setLabels(newLabels);
  };

  const createClass = () => {
    const name = prompt('Enter new class name:');
    const cleanedName = (name || '').trim();

    if (!cleanedName) {
      return;
    }

    const isDuplicate = classes.some(
      (existingClass) => existingClass.toLowerCase() === cleanedName.toLowerCase()
    );

    if (!isDuplicate) {
      const updatedClasses = [...classes, cleanedName];
      setClasses(updatedClasses);
      if (onClassesChange) {
        onClassesChange(updatedClasses);
      }
      setCurrentClass(cleanedName);
      assignClass(cleanedName);
      return;
    }

    const existingClass = classes.find(
      (className) => className.toLowerCase() === cleanedName.toLowerCase()
    ) || cleanedName;
    setCurrentClass(existingClass);
    assignClass(existingClass);
  };

  const handleSubmit = async () => {
    const data = segments.map((segment, index) => ({
      start_time: segment.start,
      end_time: segment.end,
      label: labels[index]
    }));

    try {
      // e.g. POST /api/audio/myfile/labels
      const response = await fetch(`http://localhost:5000/api/audio/${file.filename}/labels`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          labels: data
        }),
      });

      const result = await response.json();
      if (response.ok) {
        console.log('Labels submitted successfully:', result.message);
        onLabelsSubmitted();
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
      <div style={{ marginBottom: '10px' }}>
        <strong>Current labeling class:</strong>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px', marginTop: '6px' }}>
          {classes.map((className) => (
            <label key={className} style={{ display: 'inline-flex', alignItems: 'center', gap: '4px' }}>
              <input
                type="radio"
                name="current-labeling-class"
                value={className}
                checked={currentClass === className}
                onChange={() => setCurrentClass(className)}
              />
              <span>{className}</span>
            </label>
          ))}
        </div>
      </div>
      <div className="media-container">
        <Spectrogram
          src={`http://localhost:5000${file.spectrogram_path}`}
          currentTime={currentTime}
          segments={segments}
          labels={labels}
          onAssignSegment={assignCurrentClassToSegment}
          openClassSelector={openClassSelector}
          duration={file.audio_length}
        />
        {labelingStrategyChoice === 'active' && (
          <ProbabilityChart
            probabilities={probabilities}
            timings={timings}
            audioLength={file.audio_length}
            segments={segments}
            currentTime={currentTime}
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
      <button onClick={handlePlayPause}>
        {isPlaying ? 'Pause' : 'Play'}
      </button>
      <button onClick={handleSubmit}>
        Submit Labels
      </button>
      {selectorState.visible && (
        <ClassSelector
          x={selectorState.x}
          y={selectorState.y}
          classes={classes}
          onSelect={assignClass}
          onNewClass={createClass}
        />
      )}
    </div>
  );
};

export default AnnotationTool;
