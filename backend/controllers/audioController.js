const path = require('path');
const fs = require('fs');
const msgpack = require('msgpack-lite');
const { RandomSamplingStrategy } = require('../models/samplingStrategy');
require('dotenv').config();

const metadataPath = path.join(process.env.DATA_DIR, process.env.DATASET_NAME, process.env.METADATA_FILE);
const metadata = JSON.parse(fs.readFileSync(metadataPath, 'utf8'));

const audioFiles = metadata.files.audio_files;

const getBatch = (req, res) => {
    const strategy = new RandomSamplingStrategy(audioFiles, 5);
    const sampledFiles = strategy.sample();
    const batch = sampledFiles.map(file => ({
        filename: file,
        audio_length: metadata.files.audio_lengths[file],
        audio_path: `/data/${process.env.DATASET_NAME}/audio/${file}`,
        spectrogram_path: `/data/${process.env.DATASET_NAME}/spectrograms/${file.replace('.wav', '.png')}`,
        embeddings_path: `/data/${process.env.DATASET_NAME}/embeddings/${file.replace('.wav', '.birdnet.embeddings.msgpack')}`
    }));
    res.status(200).json({ batch });
};

const submitLabels = (req, res) => {
    const { filename, labels } = req.body;

    const outputDir = path.join(process.env.DATA_DIR, process.env.DATASET_NAME, 'labels');
    if (!fs.existsSync(outputDir)) {
        fs.mkdirSync(outputDir, { recursive: true });
    }

    const outputPath = path.join(outputDir, `${path.parse(filename).name}.txt`);

    const fileContent = labels.map(label => `${label.start_time},${label.end_time},${label.label}`).join('\n');
    const header = 'start_time,end_time,label\n';

    fs.writeFile(outputPath, header + fileContent, (err) => {
        if (err) {
            console.error('Error writing file:', err);
            return res.status(500).json({ message: 'Failed to save labels' });
        }

        // Process embeddings
        const embeddingsPath = path.join(process.env.DATA_DIR, process.env.DATASET_NAME, 'embeddings', `${path.parse(filename).name}.birdnet.embeddings.msgpack`);
        fs.readFile(embeddingsPath, (err, data) => {
            if (err) {
                console.error('Error reading embeddings file:', err);
                return res.status(500).json({ message: 'Failed to read embeddings file' });
            }

            const embeddingsData = msgpack.decode(data);
            const { timings, embeddings } = embeddingsData;
            const presence_embeddings = [];
            const absence_embeddings = [];

            labels.forEach(label => {
                const labelCenter = (label.start_time + label.end_time) / 2;
                timings.forEach((timing, index) => {
                    const timingCenter = (timing[0] + timing[1]) / 2;
                    if (timingCenter >= label.start_time && timingCenter <= label.end_time) {
                        if (label.label === 'presence') {
                            presence_embeddings.push(embeddings[index]);
                        } else if (label.label === 'absence') {
                            absence_embeddings.push(embeddings[index]);
                        }
                    }
                });
            });

            embeddingsData.presence_embeddings = presence_embeddings;
            embeddingsData.absence_embeddings = absence_embeddings;

            const updatedData = msgpack.encode(embeddingsData);
            fs.writeFile(embeddingsPath, updatedData, (err) => {
                if (err) {
                    console.error('Error writing updated embeddings file:', err);
                    return res.status(500).json({ message: 'Failed to save updated embeddings file' });
                }
                res.status(200).json({ message: 'Labels and embeddings updated successfully' });
            });
        });
    });
};

const getPrototypes = (req, res) => {
    const labelsDir = path.join(process.env.DATA_DIR, process.env.DATASET_NAME, 'labels');
    const embeddingsDir = path.join(process.env.DATA_DIR, process.env.DATASET_NAME, 'embeddings');

    let presence_embeddings = [];
    let absence_embeddings = [];

    fs.readdir(labelsDir, (err, files) => {
        if (err) {
            console.error('Error reading labels directory:', err);
            return res.status(500).json({ message: 'Failed to read labels directory' });
        }

        files.forEach(file => {
            const labelPath = path.join(labelsDir, file);
            const embeddingsPath = path.join(embeddingsDir, `${path.parse(file).name}.birdnet.embeddings.msgpack`);

            const labels = fs.readFileSync(labelPath, 'utf8').split('\n').slice(1).map(line => {
                const [start_time, end_time, label] = line.split(',');
                return { start_time: parseFloat(start_time), end_time: parseFloat(end_time), label };
            });

            const data = fs.readFileSync(embeddingsPath);
            const embeddingsData = msgpack.decode(data);
            const { timings, embeddings } = embeddingsData;

            labels.forEach(label => {
                const labelCenter = (label.start_time + label.end_time) / 2;
                timings.forEach((timing, index) => {
                    const timingCenter = (timing[0] + timing[1]) / 2;
                    if (timingCenter >= label.start_time && timingCenter <= label.end_time) {
                        if (label.label === 'presence') {
                            presence_embeddings.push(embeddings[index]);
                        } else if (label.label === 'absence') {
                            absence_embeddings.push(embeddings[index]);
                        }
                    }
                });
            });
        });

        const average = (embeddings) => {
            const sum = embeddings.reduce((acc, embedding) => {
                return acc.map((val, idx) => val + embedding[idx]);
            }, new Array(1024).fill(0));
            return sum.map(val => val / embeddings.length);
        };

        const presence_prototype = average(presence_embeddings);
        const absence_prototype = average(absence_embeddings);

        res.status(200).json({ presence_prototype, absence_prototype });
    });
};

module.exports = { getBatch, submitLabels, getPrototypes };