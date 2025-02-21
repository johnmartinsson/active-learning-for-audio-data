const fetch = require('node-fetch');
const path = require('path');
const fs = require('fs');
const msgpack = require('msgpack-lite');
const { RandomSamplingStrategy, UncertaintyBasedSamplingStrategy } = require('../models/samplingStrategy');
require('dotenv').config();

const metadataPath = path.join(process.env.DATA_DIR, process.env.DATASET_NAME, process.env.METADATA_FILE);
const metadata = JSON.parse(fs.readFileSync(metadataPath, 'utf8'));

const getLabeledFileNames = () => {
    const labelsDir = path.join(process.env.DATA_DIR, process.env.DATASET_NAME, 'labels');
    if (!fs.existsSync(labelsDir)) {
        return [];
    }
    return fs.readdirSync(labelsDir).map(file => path.parse(file).name);
};

const getUnlabeledFileNames = () => {
    const labeledFileNames = getLabeledFileNames();
    const allFileNames = metadata.files.audio_files.map(file => path.parse(file).name);
    return allFileNames.filter(file => !labeledFileNames.includes(file));
};

const getBatch = async (req, res) => { // Make getBatch async
    // Read from JSON body instead of query
    const { strategy = 'random', batchSize = 1 } = req.body;

    // e.g. get the list of unlabeled files
    const unlabeledFiles = getUnlabeledFileNames();

    // Decide how to pick
    let sampledFiles;
    if (strategy === 'uncertainty') {
        try {
            const prototypesResponse = await fetch('http://localhost:5000/api/audio/prototypes'); // Fetch prototypes from your API endpoint
            if (!prototypesResponse.ok) {
                console.error(`Failed to fetch prototypes: ${prototypesResponse.status} ${prototypesResponse.statusText}`);
                return res.status(500).json({ message: 'Failed to fetch prototypes for uncertainty sampling' }); // Return error response
            }
            const prototypes = await prototypesResponse.json();
            const strategyObj = new UncertaintyBasedSamplingStrategy(unlabeledFiles, batchSize, prototypes);
            sampledFiles = await strategyObj.sample(); // Await the async sample method
            console.log('sampledFiles (uncertainty):', sampledFiles);
        } catch (error) {
            console.error('Error fetching prototypes or during uncertainty sampling:', error);
            return res.status(500).json({ message: 'Error during uncertainty sampling' }); // Return error response
        }
    } else {
        // random fallback
        const strategyObj = new RandomSamplingStrategy(unlabeledFiles, batchSize);
        sampledFiles = strategyObj.sample();
        console.log('sampledFiles (random):', sampledFiles);
    }

    // Build response like usual
    const batch = sampledFiles.map((filename) => ({
        filename,
        audio_length: metadata.files.audio_lengths[`${filename}.wav`],
        audio_path: `/data/${process.env.DATASET_NAME}/audio/${filename}.wav`,
        spectrogram_path: `/data/${process.env.DATASET_NAME}/spectrograms/${filename}.png`,
        embeddings_path: `/data/${process.env.DATASET_NAME}/embeddings/${filename}.birdnet.embeddings.msgpack`
    }));

    // Return JSON
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

        if (files.length === 0) {
            // Return random prototypes if there are no labeled files
            const randomPrototype = () => Array.from({ length: 1024 }, () => Math.random());
            const presence_prototype = randomPrototype();
            const absence_prototype = randomPrototype();
            return res.status(200).json({ presence_prototype, absence_prototype });
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

        const presence_prototype = presence_embeddings.length > 0 ? average(presence_embeddings) : Array.from({ length: 1024 }, () => Math.random());
        const absence_prototype = absence_embeddings.length > 0 ? average(absence_embeddings) : Array.from({ length: 1024 }, () => Math.random());

        res.status(200).json({ presence_prototype, absence_prototype });
    });
};

module.exports = { getBatch, submitLabels, getPrototypes, getLabeledFileNames, getUnlabeledFileNames };