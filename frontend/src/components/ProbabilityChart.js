import React from 'react';
import { LineChart, Line, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const ProbabilityChart = ({ probabilities, timings, audioLength }) => {
    const data = probabilities.map((prob, index) => ({
        time: (timings[index][0] + timings[index][1]) / 2,
        probability: prob
    }));

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