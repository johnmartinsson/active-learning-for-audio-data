// ./backend/controllers/audioController.js
const fetch = require('node-fetch');
const path = require('path');
const fs = require('fs');
const msgpack = require('msgpack-lite');
const { spawn } = require('child_process');

const { RandomSamplingStrategy, UncertaintySamplingStrategy, CertaintySamplingStrategy, HighProbabilitySamplingStrategy } = require('../models/samplingStrategy');
const PrototypicalNetwork = require('../models/prototypicalNetwork');
require('dotenv').config();

const metadataPath = path.join(process.env.DATA_DIR, process.env.DATASET_NAME, process.env.METADATA_FILE);
const metadata = JSON.parse(fs.readFileSync(metadataPath, 'utf8'));
const backendRoot = path.join(__dirname, '..');

const normalizeLabel = (label) => String(label || '').trim().toLowerCase();

const isBackgroundLikeLabel = (label) => {
  const normalized = normalizeLabel(label);
  return normalized === '' || normalized === 'background' || normalized === 'absence';
};

const isLegacyPresenceLabel = (label) => normalizeLabel(label) === 'presence';

const runPythonJsonScript = (scriptName, inputData) => {
  return new Promise((resolve, reject) => {
    const pythonExecutable = process.env.PYTHON_BIN || 'python3';
    const scriptPath = path.join(backendRoot, 'scripts', scriptName);
    const pythonProcess = spawn(pythonExecutable, [scriptPath], { cwd: backendRoot });

    let stdout = '';
    let stderr = '';

    pythonProcess.stdout.on('data', (chunk) => {
      stdout += chunk.toString();
    });

    pythonProcess.stderr.on('data', (chunk) => {
      stderr += chunk.toString();
    });

    pythonProcess.on('error', (error) => {
      reject(error);
    });

    pythonProcess.on('close', (code) => {
      if (code !== 0) {
        return reject(new Error(`Python script ${scriptName} exited with code ${code}: ${stderr.trim()}`));
      }

      try {
        resolve(JSON.parse(stdout));
      } catch (error) {
        reject(new Error(`Failed to parse JSON from ${scriptName}: ${error.message}. Output: ${stdout}`));
      }
    });

    pythonProcess.stdin.write(JSON.stringify(inputData));
    pythonProcess.stdin.end();
  });
};

const runPythonJsonModule = (moduleName, inputData) => {
  return new Promise((resolve, reject) => {
    const pythonExecutable = process.env.PYTHON_BIN || 'python3';
    const pythonProcess = spawn(pythonExecutable, ['-m', moduleName], { cwd: backendRoot });

    let stdout = '';
    let stderr = '';

    pythonProcess.stdout.on('data', (chunk) => {
      stdout += chunk.toString();
    });

    pythonProcess.stderr.on('data', (chunk) => {
      stderr += chunk.toString();
    });

    pythonProcess.on('error', (error) => {
      reject(error);
    });

    pythonProcess.on('close', (code) => {
      if (code !== 0) {
        return reject(new Error(`Python module ${moduleName} exited with code ${code}: ${stderr.trim()}`));
      }

      try {
        resolve(JSON.parse(stdout));
      } catch (error) {
        reject(new Error(`Failed to parse JSON from module ${moduleName}: ${error.message}. Output: ${stdout}`));
      }
    });

    pythonProcess.stdin.write(JSON.stringify(inputData));
    pythonProcess.stdin.end();
  });
};

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
    // console.log('embeddingsPath:', embeddingsPath);

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
          if (isBackgroundLikeLabel(label)) {
            absence_embeddings.push(embeddings[index]);
          } else {
            presence_embeddings.push(embeddings[index]);
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

const detectChangePoints = (probabilities, numSegments) => {
  return new Promise((resolve, reject) => {
      const pythonProcess = spawn('python', ['scripts/change_point_detection.py']);

      const inputData = { probabilities, num_segments: numSegments };
      // console.log("Sending input data:", inputData);  // Log input data to console

      pythonProcess.stdin.write(JSON.stringify(inputData));
      pythonProcess.stdin.end();

      let data = '';
      pythonProcess.stdout.on('data', (chunk) => {
          data += chunk.toString();
      });

      pythonProcess.stderr.on('data', (chunk) => {
          console.error(`Python error: ${chunk.toString()}`);
      });

      pythonProcess.on('close', (code) => {
          if (code !== 0) {
              return reject(new Error(`Python script exited with code ${code}`));
          }
          resolve(JSON.parse(data));
      });
  });
};

const getSegments = async (req, res) => {
  try {
      const { filename } = req.params;
      const labelingStrategyChoice = req.query.labelingStrategyChoice || 'fixed';
      const numSegments = parseInt(req.query.numSegments, 10) || 10;
      const negativeClusteringMethod = req.query.negativeClusteringMethod || 'none';
      const numNegativeClusters = parseInt(req.query.numNegativeClusters, 10) || 1;
      const audioLength = metadata.files.audio_lengths[`${filename}.wav`];
      const embeddingsPath = path.join(
        process.env.DATA_DIR,
        process.env.DATASET_NAME,
        'embeddings',
        `${filename}.birdnet.embeddings.msgpack`
      );
      const labelsDir = path.join(process.env.DATA_DIR, process.env.DATASET_NAME, 'labels');
      const embeddingsDir = path.join(process.env.DATA_DIR, process.env.DATASET_NAME, 'embeddings');

      const segmentResponse = await runPythonJsonModule('python.acpd.get_segments_cli', {
        filename,
        labeling_strategy_choice: labelingStrategyChoice,
        requested_num_segments: numSegments,
        audio_length: audioLength,
        embeddings_path: embeddingsPath,
        labels_dir: labelsDir,
        embeddings_dir: embeddingsDir,
        negative_clustering_method: negativeClusteringMethod,
        num_negative_clusters: numNegativeClusters,
      });

      return res.status(200).json(segmentResponse);
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
            const class_embeddings = {};

            labels.forEach(label => {
              const normalizedLabel = normalizeLabel(label.label) || 'background';
                timings.forEach((timing, index) => {
                    const timingCenter = (timing[0] + timing[1]) / 2;
                    if (timingCenter >= label.start_time && timingCenter <= label.end_time) {
                  if (isBackgroundLikeLabel(normalizedLabel)) {
                    absence_embeddings.push(embeddings[index]);
                  } else {
                            presence_embeddings.push(embeddings[index]);

                    if (!isLegacyPresenceLabel(normalizedLabel)) {
                      if (!class_embeddings[normalizedLabel]) {
                        class_embeddings[normalizedLabel] = [];
                      }
                      class_embeddings[normalizedLabel].push(embeddings[index]);
                    }
                        }
                    }
                });
            });

            embeddingsData.presence_embeddings = presence_embeddings;
            embeddingsData.absence_embeddings = absence_embeddings;
            embeddingsData.class_embeddings = class_embeddings;

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