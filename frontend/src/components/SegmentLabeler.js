import React, { useEffect } from 'react';
import msgpack from 'msgpack-lite';

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
        const fetchEmbeddings = async () => {
            try {
                const response = await fetch(`http://localhost:5000${file.embeddings_path}`);
                const buffer = await response.arrayBuffer();
                const embeddings = msgpack.decode(new Uint8Array(buffer));

                // Use embeddings to decide on how to segment the data
                // For now, we'll use the FixedLengthLabelingStrategy as an example
                const strategy = new FixedLengthLabelingStrategy();
                const segments = strategy.segment(file.audio_length, numSegments);
                setSegments(segments);
                setLabels(new Array(segments.length).fill('absence'));
            } catch (error) {
                console.error('Error fetching embeddings:', error);
            }
        };

        fetchEmbeddings();
    }, [file, numSegments, setSegments, setLabels]);

    return null;
};

export default SegmentLabeler;