const path = require('path');
const fs = require('fs');
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
        embeddings_path: `/data/${config.dataset_name}/embeddings/${file.replace('.wav', '.json')}`
    }));
    res.status(200).json({ batch });
};

module.exports = { getBatch };