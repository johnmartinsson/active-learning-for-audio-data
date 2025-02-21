// App.js
import React, { useState, useEffect } from 'react';
import AnnotationTool from './components/AnnotationTool';

const App = () => {
    const [batch, setBatch] = useState([]);
    // Sample strategy settings
    const [sampleStrategyChoice, setSampleStrategyChoice] = useState('random');
    const [batchSize, setBatchSize] = useState(1);
    // Labeling strategy settings (typo corrected to labelingStrategyChoice)
    const [labelingStrategyChoice, setLabelingStrategyChoice] = useState('fixed');
    const [numSegments, setNumSegments] = useState(10);

    function fetchBatch() {
        fetch('http://localhost:5000/api/audio/batch', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            strategy: sampleStrategyChoice, // Using sampleStrategyChoice for fetching batch
            batchSize: batchSize,
          }),
        })
          .then((response) => response.json())
          .then((data) => {
            setBatch(data.batch);
          })
          .catch((error) => console.error('Error fetching batch:', error));
      }

    useEffect(() => {
        fetchBatch(); // Fetch initial batch based on default strategy and size
    }, [sampleStrategyChoice, batchSize]); // Re-fetch when sampleStrategyChoice or batchSize changes

    const handleSampleStrategyChange = (event) => {
        setSampleStrategyChoice(event.target.value);
    };

    const handleBatchSizeChange = (event) => {
        setBatchSize(parseInt(event.target.value, 10)); // Parse to integer
    };

    const handleLabelingStrategyChange = (event) => {
        setLabelingStrategyChoice(event.target.value);
    }

    const handleNumSegmentsChange = (event) => {
        setNumSegments(parseInt(event.target.value, 10)); // Parse to integer
    };

    return (
        <div className="app">
            <h1>Audio Labeling Interface</h1>

            <div className="input-group"> {/* Apply input-group CSS class */}
                <label htmlFor="sampleStrategy" className="input-label">Sampling Strategy:</label> {/* Apply input-label CSS class */}
                <select
                    id="sampleStrategy"
                    value={sampleStrategyChoice}
                    onChange={handleSampleStrategyChange}
                    className="select-input" // Optional CSS class for select
                >
                    <option value="uncertainty">Uncertainty Sampling</option>
                    <option value="random">Random Sampling</option>
                    {/* Add more strategies as needed */}
                </select>
            </div>

            <div className="input-group"> {/* Apply input-group CSS class */}
                <label htmlFor="batchSize" className="input-label">Batch Size:</label> {/* Apply input-label CSS class */}
                <input
                    type="number"
                    id="batchSize"
                    value={batchSize}
                    onChange={handleBatchSizeChange}
                    min="1"
                    className="number-input" // Optional CSS class for number input
                />
            </div>

            <div className="input-group"> {/* Apply input-group CSS class */}
                <label htmlFor="labelingStrategy" className="input-label">Labeling Strategy:</label> {/* Apply input-label CSS class */}
                <select
                    id="labelingStrategy"
                    value={labelingStrategyChoice}
                    onChange={handleLabelingStrategyChange}
                    className="select-input" // Optional CSS class for select
                >
                    <option value="active">Adapted Segments</option>
                    <option value="fixed">Fixed Segments</option>
                    {/* Add more strategies as needed */}
                </select>
            </div>

            <div className="input-group"> {/* Apply input-group CSS class */}
                <label htmlFor="numSegments" className="input-label">Number of Segments:</label> {/* Apply input-label CSS class */}
                <input
                    type="number"
                    id="numSegments"
                    value={numSegments}
                    onChange={handleNumSegmentsChange}
                    min="1"
                    className="number-input" // Optional CSS class for number input
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