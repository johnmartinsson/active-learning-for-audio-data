const path = require('path');
const fs = require('fs');
const msgpack = require('msgpack-lite');
const { RandomSamplingStrategy } = require('../models/samplingStrategy');

const configPath = path.join(__dirname, '../config.json');
const config = JSON.parse(fs.readFileSync(configPath, 'utf8'));

const metadataPath = path.join(__dirname, `../../data/${config.dataset_name}`, config.metadata_file);
const metadata = JSON.parse(fs.readFileSync(metadataPath, 'utf8'));

const audioFiles = metadata.files.audio_files;

const getBatch = (req, res) => {
    const strategy = new RandomSamplingStrategy(audioFiles, 5);
    const sampledFiles = strategy.sample();
    const batch = sampledFiles.map(file => ({
        filename: file,
        audio_length: metadata.files.audio_lengths[file],
        audio_path: `/data/${config.dataset_name}/audio/${file}`,
        spectrogram_path: `/data/${config.dataset_name}/spectrograms/${file.replace('.wav', '.png')}`,
        embeddings_path: `/data/${config.dataset_name}/embeddings/${file.replace('.wav', '.birdnet.embeddings.msgpack')}`
    }));
    res.status(200).json({ batch });
};

const submitLabels = (req, res) => {
    const { filename, labels } = req.body;

    const outputDir = path.join(__dirname, '../../data', config.dataset_name, 'labels');
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
        const embeddingsPath = path.join(__dirname, '../../data', config.dataset_name, 'embeddings', `${path.parse(filename).name}.birdnet.embeddings.msgpack`);
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

module.exports = { getBatch, submitLabels };