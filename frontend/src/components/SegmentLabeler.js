import React, { useEffect, useState } from 'react';
import msgpack from 'msgpack-lite';
import ProbabilityChart from './ProbabilityChart';

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

class ActiveLabelingStrategy extends LabelingStrategy {
    constructor(presencePrototype, absencePrototype) {
        super();
        this.presencePrototype = presencePrototype;
        this.absencePrototype = absencePrototype;
    }

    euclideanDistance(a, b) {
        return Math.sqrt(a.reduce((sum, val, idx) => sum + Math.pow(val - b[idx], 2), 0));
    }

    softmax(values) {
        const max = Math.max(...values);
        const exps = values.map(v => Math.exp(v - max));
        const sum = exps.reduce((sum, val) => sum + val, 0);
        return exps.map(v => v / sum);
    }

    predict(embeddings) {
        return embeddings.map(embedding => {
            const presenceDist = this.euclideanDistance(embedding, this.presencePrototype);
            const absenceDist = this.euclideanDistance(embedding, this.absencePrototype);
            const [absenceProb, presenceProb] = this.softmax([-absenceDist, -presenceDist]);
            return presenceProb;
        });
    }

    detectChangePoints(probabilities, numSegments) {
        const gradients = probabilities.slice(1).map((prob, index) => Math.abs(prob - probabilities[index]));
        const changePoints = gradients
            .map((grad, index) => ({ grad, index }))
            .sort((a, b) => b.grad - a.grad)
            .slice(0, numSegments - 1)
            .map(item => item.index + 1)
            .sort((a, b) => a - b);
        return changePoints;
    }

    segment(audioLength, numSegments, probabilities, timings) {
        const changePoints = this.detectChangePoints(probabilities, numSegments);
        const segments = [];
        let start = 0;

        changePoints.forEach(point => {
            const end = timings[point];
            segments.push({ start, end });
            start = end;
        });

        segments.push({ start, end: audioLength });
        return segments;
    }
}

const SegmentLabeler = ({ file, numSegments, setSegments, setLabels }) => {
    const [probabilities, setProbabilities] = useState([]);
    const [segments, setLocalSegments] = useState([]);
    const [timings, setTimings] = useState([]);

    useEffect(() => {
        const fetchEmbeddingsAndPrototypes = async () => {
            try {
                const [embeddingsResponse, prototypesResponse] = await Promise.all([
                    fetch(`http://localhost:5000${file.embeddings_path}`),
                    fetch('http://localhost:5000/api/audio/prototypes')
                ]);

                const embeddingsBuffer = await embeddingsResponse.arrayBuffer();
                const embeddings = msgpack.decode(new Uint8Array(embeddingsBuffer));

                const prototypes = await prototypesResponse.json();
                const { presence_prototype, absence_prototype } = prototypes;

                // Use embeddings and prototypes to decide on how to segment the data
                const strategy = new ActiveLabelingStrategy(presence_prototype, absence_prototype);
                const probabilities = strategy.predict(embeddings.embeddings);
                const probabilities_timings = embeddings.timings;
                // compute average timing for each embedding
                const average_probabilities_timings = probabilities_timings.map(timing => (timing[0] + timing[1]) / 2);

                setTimings(average_probabilities_timings);
                const segments = strategy.segment(file.audio_length, numSegments, probabilities, average_probabilities_timings);
                setSegments(segments);

                // Determine suggested labels for each segment
                const suggested_segment_labels = segments.map(segment => {
                    const hasPresence = average_probabilities_timings.some((time, index) => 
                        time >= segment.start && time < segment.end && probabilities[index] > 0.5
                    );
                    return hasPresence ? 'presence' : 'absence';
                });

                setLabels(suggested_segment_labels);
                setProbabilities(probabilities);
                setLocalSegments(segments);
            } catch (error) {
                console.error('Error fetching embeddings or prototypes:', error);
            }
        };

        fetchEmbeddingsAndPrototypes();
    }, [file, numSegments, setSegments, setLabels]);

    return (
        <div style={{ width: '100%' }}>
            <ProbabilityChart probabilities={probabilities} timings={timings} audioLength={file.audio_length} />
        </div>
    );
};

export default SegmentLabeler;