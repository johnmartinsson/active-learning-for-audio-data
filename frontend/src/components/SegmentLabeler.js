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

    isBiModal(probabilities) {
        const lowThreshold = 0.3;
        const highThreshold = 0.7;
        const lowCount = probabilities.filter(p => p <= lowThreshold).length;
        const highCount = probabilities.filter(p => p >= highThreshold).length;
        const totalCount = probabilities.length;

        // Check if there are significant clusters around 0 and 1
        return (lowCount / totalCount > 0.1) && (highCount / totalCount > 0.1);
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

const SegmentLabeler = ({ file, numSegments, strategyChoice, setSegments, setLabels }) => {
    const [probabilities, setProbabilities] = useState([]);
    const [timings, setTimings] = useState([]);

    useEffect(() => {
        // If using FixedLengthLabelingStrategy, we do NOT need prototypes or embeddings
        if (strategyChoice === 'fixed') {
            const strategy = new FixedLengthLabelingStrategy();
            const segments = strategy.segment(file.audio_length, numSegments);
            setSegments(segments);
            // For instance, default them all to 'absence':
            setLabels(new Array(numSegments).fill('absence'));
            setProbabilities([]); 
            setTimings([]);
            return; // Done, so we can exit early
        }

        // Otherwise, use the existing ActiveLabelingStrategy
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
    
                const strategy = new ActiveLabelingStrategy(presence_prototype, absence_prototype);
                const probabilities = strategy.predict(embeddings.embeddings);
                const probabilitiesTimings = embeddings.timings.map(
                  timing => (timing[0] + timing[1]) / 2
                );
                setTimings(probabilitiesTimings);
    
                const segments = strategy.segment(file.audio_length, numSegments, probabilities, probabilitiesTimings);
                setSegments(segments);
    
                // Possibly compute labels from presence threshold
                const suggestedLabels = segments.map(segment => {
                  const hasPresence = probabilitiesTimings.some((time, index) =>
                    time >= segment.start && time < segment.end && probabilities[index] > 0.5
                  );
                  return hasPresence ? 'presence' : 'absence';
                });
    
                // Check for bi-modality
                const isBiModal = strategy.isBiModal(probabilities);
                if (!isBiModal) {
                    setLabels(new Array(numSegments).fill('absence'));
                } else {
                    setLabels(suggestedLabels);
                }
    
                setProbabilities(probabilities);
            } catch (error) {
                console.error('Error fetching embeddings/prototypes:', error);
            }
        };
    
        fetchEmbeddingsAndPrototypes();
    }, [file, numSegments, strategyChoice, setSegments, setLabels]);

    return (
        <div style={{ width: '100%' }}>
            {strategyChoice === 'active' && (
            <ProbabilityChart
                probabilities={probabilities}
                timings={timings}
                audioLength={file.audio_length}
            />
            )}
      </div>
    );
};

export default SegmentLabeler;
