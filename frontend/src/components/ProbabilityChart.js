import React, { useState, useEffect } from 'react';
import {
    LineChart,
    Line,
    Tooltip,
    ResponsiveContainer,
    XAxis,
    YAxis,
    ReferenceLine,
} from 'recharts';

const ProbabilityChart = ({ probabilities, timings, audioLength, segments = [], currentTime = 0 }) => {
    const [data, setData] = useState([]);

    useEffect(() => {
        const safeProbabilities = Array.isArray(probabilities) ? probabilities : [];
        const safeTimings = Array.isArray(timings) ? timings : [];
        const n = Math.min(safeProbabilities.length, safeTimings.length);

        if (n > 0) {
            const chartData = [];

            // Anchor the curve to 0 and audioLength for easier visual alignment.
            chartData.push({ time: 0, probability: safeProbabilities[0] });
            for (let i = 0; i < n; i++) {
                chartData.push({
                    time: Number(safeTimings[i]),
                    probability: Number(safeProbabilities[i]),
                });
            }
            chartData.push({ time: audioLength, probability: safeProbabilities[n - 1] });

            setData(chartData);
            return;
        }

        // Fallback if timings are unavailable.
        if (safeProbabilities.length > 0) {
            const step = safeProbabilities.length > 1 ? audioLength / (safeProbabilities.length - 1) : audioLength;
            const chartData = safeProbabilities.map((prob, index) => ({
                time: index * step,
                probability: Number(prob),
            }));
            setData(chartData);
            return;
        }

        setData([]);
    }, [probabilities, timings, audioLength]);

    return (
        <div style={{ width: '100%', height: 200, marginBottom: 10 }}>
        <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 0, right: 0, bottom: 0, left: 0 }}>
                <XAxis
                    dataKey="time"
                    type="number"
                    domain={[0, audioLength]}
                    hide
                    allowDataOverflow
                />
                <YAxis domain={[0, 1]} hide allowDataOverflow />
                <Tooltip
                    formatter={(value) => Number(value).toFixed(3)}
                    labelFormatter={(label) => `t=${Number(label).toFixed(2)}s`}
                />

                {segments.map((segment, idx) => (
                    <ReferenceLine
                        key={`seg-start-${idx}`}
                        x={segment.start}
                        stroke="#b0b0b0"
                        strokeDasharray="3 3"
                    />
                ))}
                <ReferenceLine x={currentTime} stroke="#ff6f00" />

                <Line
                    type="monotone"
                    dataKey="probability"
                    stroke="#1f77b4"
                    strokeWidth={2}
                    dot={false}
                    isAnimationActive={false}
                />
            </LineChart>
        </ResponsiveContainer>
        </div>
    );
};

export default ProbabilityChart;