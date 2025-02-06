import React, { useState } from 'react';
import Spectrogram from './Spectrogram';
import Waveform from './Waveform';
import SegmentLabeler from './SegmentLabeler';

const AnnotationTool = ({ file }) => {
    const [currentTime, setCurrentTime] = useState(0);
    const [isPlaying, setIsPlaying] = useState(false);
    const [segments, setSegments] = useState([]);
    const [labels, setLabels] = useState([]);
    const [numSegments, setNumSegments] = useState(10); // Default number of segments

    const handlePlayPause = () => {
        setIsPlaying(!isPlaying);
    };

    const handleNumSegmentsChange = (event) => {
        setNumSegments(Number(event.target.value));
    };

    const toggleLabel = (time) => {
        console.log(`Toggle label at time: ${time}`);
        const updatedLabels = labels.map((label, index) => {
            const segment = segments[index];
            if (time >= segment.start && time < segment.end) {
                return label === 'absence' ? 'presence' : 'absence';
            }
            return label;
        });
        setLabels(updatedLabels);
    };

    return (
        <div className="annotation-tool">
            <h3>{file.filename}</h3>
            <div className="num-segments-control">
                <label htmlFor="numSegments">Number of Segments: </label>
                <input
                    type="range"
                    id="numSegments"
                    name="numSegments"
                    min="1"
                    max={Math.floor(file.audio_length * 1)}
                    value={numSegments}
                    onChange={handleNumSegmentsChange}
                />
                <span>{numSegments}</span>
            </div>
            <div className="media-container">
                <Spectrogram
                    src={`http://localhost:5000${file.spectrogram_path}`}
                    currentTime={currentTime}
                    segments={segments}
                    labels={labels}
                    toggleLabel={toggleLabel}
                    duration={file.audio_length}
                />
                <Waveform
                    src={`http://localhost:5000${file.audio_path}`}
                    currentTime={currentTime}
                    setCurrentTime={setCurrentTime}
                    isPlaying={isPlaying}
                    duration={file.audio_length}
                />
                <button onClick={handlePlayPause}>{isPlaying ? 'Pause' : 'Play'}</button>
            </div>
            <SegmentLabeler
                file={file}
                numSegments={numSegments}
                setSegments={setSegments}
                setLabels={setLabels}
            />
        </div>
    );
};

export default AnnotationTool;