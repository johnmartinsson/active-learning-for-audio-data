import { useEffect } from 'react';

class LabelingStrategy {
    segment(audioLength, numSegments) {
        throw new Error('segment() must be implemented by subclass');
    }
}

class FixedLengthLabelingStrategy extends LabelingStrategy {
    segment(audioLength, numSegments) {
        const segmentLength = audioLength / numSegments;
        const segments = [];

        for (let i = 0; i < numSegments; i++) {
            segments.push({
                start: i * segmentLength,
                end: (i + 1) * segmentLength
            });
        }

        return segments;
    }
}

const SegmentLabeler = ({ file, numSegments, setSegments, setLabels }) => {
    useEffect(() => {
        const strategy = new FixedLengthLabelingStrategy();
        const segments = strategy.segment(file.audio_length, numSegments);
        setSegments(segments);
        setLabels(new Array(segments.length).fill('absence'));
    }, [file.audio_length, numSegments, setSegments, setLabels]);

    return null;
};

export default SegmentLabeler;