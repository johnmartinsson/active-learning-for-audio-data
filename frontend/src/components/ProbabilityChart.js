import React, { useState, useEffect } from 'react';
import { LineChart, Line, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const ProbabilityChart = ({ probabilities, timings, audioLength }) => {
    const [data, setData] = useState([]);

    useEffect(() => {
        // Add 3 dummy probabilities to the beginning of the array to make the chart start at time 0
        const modifiedProbabilities = [...probabilities];
        for (let i = 0; i < 3; i++) {
            modifiedProbabilities.unshift(0);
        }
        // Add 1 dummy probability to the end of the array to make the chart end at the audio length
        modifiedProbabilities.push(0);

        const chartData = modifiedProbabilities.map((prob, index) => ({
            probability: prob
        }));

        setData(chartData);
    }, [probabilities]);

    return (
        <ResponsiveContainer width="100%" height={200}>
            <LineChart data={data} margin={{ top: 0, right: 0, bottom: 0, left: 0 }} xDomain={[0, audioLength]} yDomain={[0, 1]}>
                <CartesianGrid strokeDasharray="3 3" />
                <Tooltip />
                <Line type="monotone" dataKey="probability" stroke="#8884d8" dot={false} />
            </LineChart>
        </ResponsiveContainer>
    );
};

export default ProbabilityChart;