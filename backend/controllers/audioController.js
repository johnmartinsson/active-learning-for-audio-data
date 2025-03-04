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

// At the top of audioController.js, or put this in a separate utils file
const computePrototypes = () => {
  const labelsDir = path.join(process.env.DATA_DIR, process.env.DATASET_NAME, 'labels');
  const embeddingsDir = path.join(process.env.DATA_DIR, process.env.DATASET_NAME, 'embeddings');

  let presence_embeddings = [];
  let absence_embeddings = [];

  // If labelsDir doesn’t exist or is empty, return random prototypes
  if (!fs.existsSync(labelsDir)) {
    // Return random prototypes if there's no labels directory
    const randomPrototype = () => Array.from({ length: 1024 }, () => Math.random());
    return {
      presence_prototype: randomPrototype(),
      absence_prototype: randomPrototype(),
    };
  }

  const files = fs.readdirSync(labelsDir);

  if (files.length === 0) {
    // Return random prototypes if there are no labeled files
    const randomPrototype = () => Array.from({ length: 1024 }, () => Math.random());
    return {
      presence_prototype: randomPrototype(),
      absence_prototype: randomPrototype(),
    };
  }

  // Gather presence and absence embeddings from all labeled files
  files.forEach((file) => {
    const labelPath = path.join(labelsDir, file);
    const embeddingsPath = path.join(embeddingsDir, `${path.parse(file).name}.birdnet.embeddings.msgpack`);
    console.log('embeddingsPath:', embeddingsPath);

    if (!fs.existsSync(embeddingsPath)) {
      return; // skip if no embeddings file
    }

    const labelLines = fs.readFileSync(labelPath, 'utf8')
      .split('\n')
      .slice(1) // skip header
      .filter((line) => line.trim().length > 0);

    const data = fs.readFileSync(embeddingsPath);
    const embeddingsData = msgpack.decode(data);
    const { timings, embeddings } = embeddingsData;

    labelLines.forEach((line) => {
      const [start_time, end_time, label] = line.split(',');
      const st = parseFloat(start_time);
      const et = parseFloat(end_time);
      // For each labeled region, find embeddings whose center is in [st, et]
      timings.forEach((timing, index) => {
        const timingCenter = (timing[0] + timing[1]) / 2;
        if (timingCenter >= st && timingCenter <= et) {
          if (label === 'presence') {
            presence_embeddings.push(embeddings[index]);
          } else if (label === 'absence') {
            absence_embeddings.push(embeddings[index]);
          }
        }
      });
    });
  });

  // If we have any embeddings, compute their average. Otherwise, return random prototypes.
  const average = (vectors) => {
    if (vectors.length === 0) {
      return Array.from({ length: 1024 }, () => Math.random());
    }
    const dim = vectors[0].length;
    const sum = new Array(dim).fill(0);
    vectors.forEach((vec) => {
      for (let i = 0; i < dim; i++) {
        sum[i] += vec[i];
      }
    });
    return sum.map((val) => val / vectors.length);
  };

  const presence_prototype = average(presence_embeddings);
  const absence_prototype = average(absence_embeddings);

  return { presence_prototype, absence_prototype };
};

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
      // 1) read from route param
      const { filename } = req.params;

      // 2) read from query string for "labelingStrategyChoice" and "numSegments"
      const labelingStrategyChoice = req.query.labelingStrategyChoice || 'fixed';
      const numSegments = parseInt(req.query.numSegments, 10) || 10;
      const audioLength = metadata.files.audio_lengths[`${filename}.wav`];
  
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
      console.log('filename:', filename);
      const embeddingsPath = path.join(
        process.env.DATA_DIR,
        process.env.DATASET_NAME,
        'embeddings',
        `${filename}.birdnet.embeddings.msgpack`
      );
      const buffer = fs.readFileSync(embeddingsPath);
      const embeddingsData = msgpack.decode(buffer);
      const { embeddings, timings } = embeddingsData;
  
      // 2) Fetch prototypes
      const { presence_prototype, absence_prototype } = computePrototypes();

  
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
    try {
      // 1) Parse inputs from query params instead of the request body
      const strategy = req.query.strategy || 'random';
      const batchSize = parseInt(req.query.batchSize, 10) || 1;
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
      let prototypes;
      if (needsPrototypes) {
        prototypes = computePrototypes();
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
      console.error(`Error fetching prototypes or during ${req.query.strategy || 'random'} sampling:`, error);
      return res.status(500).json({ message: `Error during batch retrieval` });
    }
  };
  
  
const submitLabels = (req, res) => {
    const filename = req.params.filename;
    console.log('submitting labels filename:', filename);
    const { labels } = req.body;

    const outputDir = path.join(process.env.DATA_DIR, process.env.DATASET_NAME, 'labels');
    if (!fs.existsSync(outputDir)) {
        fs.mkdirSync(outputDir, { recursive: true });
    }

    const outputPath = path.join(outputDir, `${filename}.txt`);

    const fileContent = labels.map(label => `${label.start_time},${label.end_time},${label.label}`).join('\n');
    const header = 'start_time,end_time,label\n';

    fs.writeFile(outputPath, header + fileContent, (err) => {
        if (err) {
            console.error('Error writing file:', err);
            return res.status(500).json({ message: 'Failed to save labels' });
        }

        // Process embeddings
        const embeddingsPath = path.join(process.env.DATA_DIR, process.env.DATASET_NAME, 'embeddings', `${filename}.birdnet.embeddings.msgpack`);
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

module.exports = { getBatch, submitLabels, getLabeledFileNames, getUnlabeledFileNames, getSegments };