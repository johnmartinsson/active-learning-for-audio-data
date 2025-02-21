// ./frontend/src/App.js
import React, { useState, useEffect, useCallback } from 'react';
import AnnotationTool from './components/AnnotationTool';
import './styles.css';

const App = () => {
  const [batch, setBatch] = useState([]);

  // Sample strategy settings
  const [sampleStrategyChoice, setSampleStrategyChoice] = useState('random');
  const [batchSize, setBatchSize] = useState(1);

  // Labeling strategy settings
  const [labelingStrategyChoice, setLabelingStrategyChoice] = useState('fixed');
  const [numSegments, setNumSegments] = useState(10);

  // --- 1) Load from localStorage in its own effect ---
  useEffect(() => {
    const storedState = localStorage.getItem('appState');
    if (storedState) {
      const parsedState = JSON.parse(storedState);
      if (parsedState.sampleStrategyChoice !== undefined) {
        setSampleStrategyChoice(parsedState.sampleStrategyChoice);
      }
      if (parsedState.batchSize !== undefined) {
        setBatchSize(parsedState.batchSize);
      }
      if (parsedState.labelingStrategyChoice !== undefined) {
        setLabelingStrategyChoice(parsedState.labelingStrategyChoice);
      }
      if (parsedState.numSegments !== undefined) {
        setNumSegments(parsedState.numSegments);
      }
    }
  }, []);

  // Use callback for saving to localStorage
  const saveStateToLocalStorage = useCallback(() => {
    const appState = {
      sampleStrategyChoice,
      batchSize,
      labelingStrategyChoice,
      numSegments,
    };
    localStorage.setItem('appState', JSON.stringify(appState));
  }, [sampleStrategyChoice, batchSize, labelingStrategyChoice, numSegments]);

  // 2) fetchBatch depends on sampleStrategyChoice, batchSize, etc.
  //    Now using GET with query params
  const fetchBatch = useCallback(() => {
    const url = new URL('http://localhost:5000/api/audio/batch');
    url.searchParams.set('strategy', sampleStrategyChoice);
    url.searchParams.set('batchSize', batchSize.toString());

    fetch(url, {
      method: 'GET'
    })
      .then(response => response.json())
      .then(data => {
        setBatch(data.batch);
      })
      .catch(error => console.error('Error fetching batch:', error));
  }, [sampleStrategyChoice, batchSize]);

  // --- 3) Trigger fetchBatch any time these settings change ---
  useEffect(() => {
    fetchBatch();
  }, [sampleStrategyChoice, batchSize, fetchBatch]);

  // Save settings to localStorage whenever they change
  useEffect(() => {
    saveStateToLocalStorage();
  }, [sampleStrategyChoice, batchSize, labelingStrategyChoice, numSegments, saveStateToLocalStorage]);

  const handleSampleStrategyChange = (event) => {
    setSampleStrategyChoice(event.target.value);
  };

  const handleBatchSizeChange = (event) => {
    setBatchSize(parseInt(event.target.value, 10));
  };

  const handleLabelingStrategyChange = (event) => {
    setLabelingStrategyChoice(event.target.value);
  }

  const handleNumSegmentsChange = (event) => {
    setNumSegments(parseInt(event.target.value, 10));
  };

  return (
    <div className="app">
      <h1>Audio Labeling Interface</h1>

      <div className="input-group">
        <label htmlFor="sampleStrategy" className="input-label">Sampling Strategy:</label>
        <select
          id="sampleStrategy"
          value={sampleStrategyChoice}
          onChange={handleSampleStrategyChange}
          className="select-input"
        >
          <option value="uncertainty">Uncertainty Sampling</option>
          <option value="random">Random Sampling</option>
          <option value="certainty">Certainty Sampling</option>
          <option value="high_probability">High Probability Sampling</option>
        </select>
      </div>

      <div className="input-group">
        <label htmlFor="batchSize" className="input-label">Batch Size:</label>
        <input
          type="number"
          id="batchSize"
          value={batchSize}
          onChange={handleBatchSizeChange}
          min="1"
          className="number-input"
        />
      </div>

      <div className="input-group">
        <label htmlFor="labelingStrategy" className="input-label">Labeling Strategy:</label>
        <select
          id="labelingStrategy"
          value={labelingStrategyChoice}
          onChange={handleLabelingStrategyChange}
          className="select-input"
        >
          <option value="active">Adapted Segments</option>
          <option value="fixed">Fixed Segments</option>
        </select>
      </div>

      <div className="input-group">
        <label htmlFor="numSegments" className="input-label">Number of Segments:</label>
        <input
          type="number"
          id="numSegments"
          value={numSegments}
          onChange={handleNumSegmentsChange}
          min="1"
          className="number-input"
        />
      </div>

      {batch.map((file, index) => (
        <AnnotationTool
          key={index}
          file={file}
          onLabelsSubmitted={fetchBatch}
          labelingStrategyChoice={labelingStrategyChoice}
          numSegments={numSegments}
        />
      ))}
    </div>
  );
};

export default App;
