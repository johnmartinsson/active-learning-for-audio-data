// ./backend/controllers/audioController.js
const fetch = require('node-fetch');
const path = require('path');
const fs = require('fs');
const msgpack = require('msgpack-lite');
const { RandomSamplingStrategy, UncertaintySamplingStrategy, CertaintySamplingStrategy, HighProbabilitySamplingStrategy } = require('../models/samplingStrategy');
const PrototypicalNetwork = require('../models/prototypicalNetwork');
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

const getSegments = async (req, res) => {
    try {
      const { filename, labelingStrategyChoice, numSegments } = req.body;
      const fileBase = path.parse(filename).name;
      const audioLength = metadata.files.audio_lengths[`${fileBase}.wav`];
  
      // Helper for checking bimodality
      function isBiModal(probabilities) {
        const lowThreshold = 0.3;
        const highThreshold = 0.7;
        const lowCount = probabilities.filter(p => p <= lowThreshold).length;
        const highCount = probabilities.filter(p => p >= highThreshold).length;
        const totalCount = probabilities.length;
        return (lowCount / totalCount > 0.1) && (highCount / totalCount > 0.1);
      }
  
      if (labelingStrategyChoice === 'fixed') {
        // ...existing fixed code...
        const segmentLength = audioLength / numSegments;
        const segments = [];
        for (let i = 0; i < numSegments; i++) {
          segments.push({ start: i * segmentLength, end: (i + 1) * segmentLength });
        }
        const suggestedLabels = Array(numSegments).fill('absence');
        return res.status(200).json({
          segments,
          probabilities: [],
          timings: [],
          suggestedLabels
        });
      }
  
      // 1) Load embeddings
      const embeddingsPath = path.join(
        process.env.DATA_DIR,
        process.env.DATASET_NAME,
        'embeddings',
        `${fileBase}.birdnet.embeddings.msgpack`
      );
      const buffer = fs.readFileSync(embeddingsPath);
      const embeddingsData = msgpack.decode(buffer);
      const { embeddings, timings } = embeddingsData;
  
      // 2) Fetch prototypes
      const prototypesResponse = await fetch('http://localhost:5000/api/audio/prototypes');
      if (!prototypesResponse.ok) {
        return res.status(500).json({ message: 'Failed to fetch prototypes' });
      }
      const { presence_prototype, absence_prototype } = await prototypesResponse.json();
  
      // 3) Predict probabilities
      const net = new PrototypicalNetwork(presence_prototype, absence_prototype);
      const probabilities = net.predict(embeddings);  // array of presence probabilities [0..1]
  
      // 4) Detect change points for adaptive segmentation
      const gradients = probabilities.slice(1)
        .map((prob, idx) => Math.abs(prob - probabilities[idx]));
      const changePoints = gradients
        .map((g, idx) => ({ grad: g, index: idx }))
        .sort((a, b) => b.grad - a.grad)
        .slice(0, numSegments - 1)
        .map(item => item.index + 1)
        .sort((a, b) => a - b);
  
      const segments = [];
      let start = 0;
      changePoints.forEach(point => {
        // pick a middle-of-frame time for splitting
        const splitTime = (timings[point][0] + timings[point][1]) / 2;
        segments.push({ start, end: splitTime });
        start = splitTime;
      });
      segments.push({ start, end: audioLength });
  
      // 5) Generate suggested labels
      //    By default, do presence if any probability > 0.5 in that segment
      let suggestedLabels = segments.map(segment => {
        const hasPresence = timings.some((time, idx) => {
          const center = (time[0] + time[1]) / 2;
          return center >= segment.start && center < segment.end && probabilities[idx] > 0.5;
        });
        return hasPresence ? 'presence' : 'absence';
      });
  
      // *** Bimodality check *** 
      const bimodal = isBiModal(probabilities);
      if (!bimodal) {
        // If not bimodal, override everything to absence
        suggestedLabels = suggestedLabels.map(() => 'absence');
      }
  
      // 6) Return results
      return res.status(200).json({
        segments,
        probabilities,
        timings: timings.map(t => (t[0] + t[1]) / 2),
        suggestedLabels
      });
    } catch (error) {
      console.error('Error in getSegments:', error);
      return res.status(500).json({ message: 'Failed to compute segments' });
    }
  };

const getBatch = async (req, res) => {
    // 1) Parse inputs
    const { strategy = 'random', batchSize = 1 } = req.body;
    const unlabeledFiles = getUnlabeledFileNames();
    console.log('strategy:', strategy);
    console.log('batchSize:', batchSize);
  
    // 2) Decide which strategy class to use, and whether prototypes are needed
    let StrategyClass;
    let needsPrototypes = false;
  
    switch (strategy) {
      case 'uncertainty':
        StrategyClass = UncertaintySamplingStrategy;
        needsPrototypes = true;
        break;
      case 'certainty':
        StrategyClass = CertaintySamplingStrategy;
        needsPrototypes = true;
        break;
      case 'high_probability':
        StrategyClass = HighProbabilitySamplingStrategy;
        needsPrototypes = true;
        break;
      default:
        // random fallback
        StrategyClass = RandomSamplingStrategy;
        needsPrototypes = false;
        break;
    }
  
    // 3) Fetch prototypes only if required, then sample
    try {
      let prototypes;
      if (needsPrototypes) {
        const prototypesResponse = await fetch('http://localhost:5000/api/audio/prototypes');
        if (!prototypesResponse.ok) {
          console.error(`Failed to fetch prototypes: ${prototypesResponse.status} ${prototypesResponse.statusText}`);
          return res
            .status(500)
            .json({ message: `Failed to fetch prototypes for ${strategy} sampling` });
        }
        prototypes = await prototypesResponse.json();
      }
  
      let sampledFiles;
      if (needsPrototypes) {
        const strategyObj = new StrategyClass(unlabeledFiles, batchSize, prototypes);
        sampledFiles = await strategyObj.sample();
      } else {
        const strategyObj = new StrategyClass(unlabeledFiles, batchSize);
        sampledFiles = strategyObj.sample();
      }
      console.log(`sampledFiles (${strategy}):`, sampledFiles);
  
      // 4) Build response
      const batch = sampledFiles.map((filename) => ({
        filename,
        audio_length: metadata.files.audio_lengths[`${filename}.wav`],
        audio_path: `/data/${process.env.DATASET_NAME}/audio/${filename}.wav`,
        spectrogram_path: `/data/${process.env.DATASET_NAME}/spectrograms/${filename}.png`,
        embeddings_path: `/data/${process.env.DATASET_NAME}/embeddings/${filename}.birdnet.embeddings.msgpack`
      }));
  
      // 5) Return JSON
      res.status(200).json({ batch });
    } catch (error) {
      console.error(`Error fetching prototypes or during ${strategy} sampling:`, error);
      return res.status(500).json({ message: `Error during ${strategy} sampling` });
    }
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

module.exports = { getBatch, submitLabels, getPrototypes, getLabeledFileNames, getUnlabeledFileNames, getSegments };