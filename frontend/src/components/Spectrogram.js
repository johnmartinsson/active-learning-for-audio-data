import React, { useRef } from 'react';

const Spectrogram = ({ src, currentTime, segments, labels, toggleLabel, duration }) => {
    const containerRef = useRef(null);

    const handleClick = (event) => {
        const rect = containerRef.current.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const time = (x / rect.width) * duration;
        console.log(`Click position: ${x}, Time: ${time}`);
        toggleLabel(time);
    };

    return (
        <div className="spectrogram" onClick={handleClick} ref={containerRef}>
            <img src={src} alt="Spectrogram" />
            <div className="current-time-line" style={{ left: `${(currentTime / duration) * 100}%` }}></div>
            {/* Draw lines at the beginning and end of the file */}
            <div className="segment-line" style={{ left: '0%' }}></div>
            <div className="segment-line" style={{ left: '100%' }}></div>
            {segments.map((segment, index) => (
                <React.Fragment key={index}>
                    {/* Draw line at the start of each segment */}
                    <div
                        className="segment-line"
                        style={{
                            left: `${(segment.start / duration) * 100}%`,
                        }}
                    ></div>
                    {/* Draw line at the end of each segment */}
                    <div
                        className="segment-line"
                        style={{
                            left: `${(segment.end / duration) * 100}%`,
                        }}
                    ></div>
                    {/* Draw box for presence label */}
                    {labels[index] === 'presence' && (
                        <div
                            className="segment-box"
                            style={{
                                left: `${(segment.start / duration) * 100}%`,
                                width: `${((segment.end - segment.start) / duration) * 100}%`,
                            }}
                        ></div>
                    )}
                </React.Fragment>
            ))}
        </div>
    );
};

export default Spectrogram;