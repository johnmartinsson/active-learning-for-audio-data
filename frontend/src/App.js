import React, { useState, useEffect } from 'react';
import AnnotationTool from './components/AnnotationTool';

const App = () => {
    const [batch, setBatch] = useState([]);

    useEffect(() => {
        fetch('http://localhost:5000/api/audio/batch')
            .then(response => response.json())
            .then(data => setBatch(data.batch));
    }, []);

    return (
        <div className="app">
            <h1>Audio Labeling Interface</h1>
            {batch.map((file, index) => (
                <AnnotationTool key={index} file={file} />
            ))}
        </div>
    );
};

export default App;