import React, { useRef } from 'react';

const classColors = [
  "rgba(0,255,0,0.3)",
  "rgba(255,0,0,0.3)",
  "rgba(0,0,255,0.3)",
  "rgba(255,255,0,0.3)",
  "rgba(255,0,255,0.3)"
];

const getColorForLabel = (label) => {
    if (!label || label === 'background') {
        return 'transparent';
    }

    // Deterministic hash so each label gets a stable color across segments/files.
    let hash = 0;
    for (let i = 0; i < label.length; i++) {
        hash = ((hash << 5) - hash) + label.charCodeAt(i);
        hash |= 0;
    }

    const idx = Math.abs(hash) % classColors.length;
    return classColors[idx];
};

const Spectrogram = ({ src, currentTime, segments, labels, onAssignSegment, openClassSelector, duration }) => {
    const containerRef = useRef(null);

    const getSegmentIndexFromEvent = (event) => {
        const rect = containerRef.current.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const time = (x / rect.width) * duration;
        return segments.findIndex(
            seg => time >= seg.start && time < seg.end
        );
    };

    const handleLeftClick = (event) => {
        const segmentIndex = getSegmentIndexFromEvent(event);

        if (segmentIndex !== -1 && onAssignSegment) {
            onAssignSegment(segmentIndex);
        }
    };

    const handleRightClick = (event) => {
        event.preventDefault();
        const segmentIndex = getSegmentIndexFromEvent(event);

        if (segmentIndex === -1) {
            return;
        }

        if (openClassSelector) {
            openClassSelector(segmentIndex, event.clientX, event.clientY);
        }
    };

    return (
        <div className="spectrogram" onClick={handleLeftClick} onContextMenu={handleRightClick} ref={containerRef}>
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
                    {labels[index] && labels[index] !== "background" && (
                        <div
                            className="segment-box"
                            style={{
                                left: `${(segment.start / duration) * 100}%`,
                                width: `${((segment.end - segment.start) / duration) * 100}%`,
                                backgroundColor: getColorForLabel(labels[index]),
                            }}
                        ></div>
                    )}
                </React.Fragment>
            ))}
        </div>
    );
};

export default Spectrogram;