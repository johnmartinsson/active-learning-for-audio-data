const fetch = require('node-fetch');
const PrototypicalNetwork = require('./prototypicalNetwork');
const msgpack = require('msgpack-lite');
require('dotenv').config();

/**
 * Base class for file-level sampling strategies.
 */
class SamplingStrategy {
    /**
     * @param {string[]} audioFileNames - Candidate file basenames.
     */
    constructor(audioFileNames) {
        this.audioFileNames = audioFileNames;
    }

    /**
     * Sample file names according to strategy-specific criteria.
     *
     * @abstract
     * @returns {string[]|Promise<string[]>} Selected file basenames.
     */
    sample() {
        throw new Error('sample() must be implemented by subclass');
    }
}

/**
 * Uniform random sampling over unlabeled files.
 */
class RandomSamplingStrategy extends SamplingStrategy {
    /**
     * @param {string[]} audioFileNames - Candidate file basenames.
     * @param {number} batchSize - Number of files to return.
     */
    constructor(audioFileNames, batchSize) {
        super(audioFileNames);
        this.batchSize = batchSize;
    }

    /**
     * Return a random subset of files.
     *
     * @returns {string[]} Randomly sampled file basenames.
     */
    sample() {
        const shuffled = this.audioFileNames.sort(() => 0.5 - Math.random());
        console.log('random sampling');
        return shuffled.slice(0, this.batchSize);
    }
}

/**
 * Entropy-based uncertainty sampling using a binary prototypical model.
 */
class UncertaintySamplingStrategy extends SamplingStrategy {
    /**
     * @param {string[]} audioFileNames - Candidate file basenames.
     * @param {number} batchSize - Number of files to return.
     * @param {{presence_prototype:number[], absence_prototype:number[]}} prototypes - Binary prototypes.
     */
    constructor(audioFileNames, batchSize, prototypes) {
        super(audioFileNames);
        this.batchSize = batchSize;
        this.prototypes = prototypes;
        this.protoNet = new PrototypicalNetwork(prototypes.presence_prototype, prototypes.absence_prototype);
    }

    /**
     * Rank files by average predictive entropy (descending).
     *
     * @returns {Promise<string[]>} Most uncertain file basenames.
     */
    async sample() {
        console.log('uncertainty sampling');
        const filesWithUncertainty = [];

        for (const filename of this.audioFileNames) {

            try {
                const embeddings_path = `/data/${process.env.DATASET_NAME}/embeddings/${filename}.birdnet.embeddings.msgpack`
                const embeddingsResponse = await fetch(`http://localhost:5000${embeddings_path}`); // Assuming your server is also running on localhost:5000 for backend calls
                if (!embeddingsResponse.ok) {
                    console.error(`Failed to fetch embeddings for ${filename}: ${embeddingsResponse.status} ${embeddingsResponse.statusText}`);
                    continue; // Skip to the next file if fetching embeddings fails
                }
                const embeddingsBuffer = await embeddingsResponse.arrayBuffer();
                const embeddingsData = msgpack.decode(new Uint8Array(embeddingsBuffer));
                const embeddings = embeddingsData.embeddings; // Assuming embeddings are under 'embeddings' key

                if (!embeddings || embeddings.length === 0) {
                    console.warn(`No embeddings found for ${filename}, skipping for uncertainty calculation.`);
                    continue; // Skip if no embeddings are found
                }

                const probabilities = this.protoNet.predict(embeddings);

                // Calculate average entropy as uncertainty score using ProtoNet's entropy method
                const entropies = this.protoNet.entropy(probabilities);
                const averageEntropy = entropies.reduce((sum, entropy) => sum + entropy, 0) / entropies.length;

                // Debug print average entropy for each file
                console.log(`Average entropy for ${filename
                }:`, averageEntropy);

                filesWithUncertainty.push({ filename, uncertainty: averageEntropy });

            } catch (error) {
                console.error(`Error processing embeddings for ${filename}:`, error);
                // Consider how to handle errors - skip file, or fail the whole batch? For now, skip file.
            }
        }

        // Sort files by uncertainty in descending order (higher uncertainty first)
        filesWithUncertainty.sort((a, b) => b.uncertainty - a.uncertainty);

        const bestFiles = filesWithUncertainty.slice(0, this.batchSize).map(item => item.filename); // Just return filenames
        return bestFiles;
    }
}

/**
 * Low-entropy certainty sampling using a binary prototypical model.
 */
class CertaintySamplingStrategy extends SamplingStrategy {
    /**
     * @param {string[]} audioFileNames - Candidate file basenames.
     * @param {number} batchSize - Number of files to return.
     * @param {{presence_prototype:number[], absence_prototype:number[]}} prototypes - Binary prototypes.
     */
    constructor(audioFileNames, batchSize, prototypes) {
        super(audioFileNames);
        this.batchSize = batchSize;
        this.prototypes = prototypes;
        this.protoNet = new PrototypicalNetwork(prototypes.presence_prototype, prototypes.absence_prototype);
    }

    /**
     * Rank files by average predictive entropy (ascending).
     *
     * @returns {Promise<string[]>} Most certain file basenames.
     */
    async sample() {
        console.log('certainty sampling');
        const filesWithUncertainty = [];

        for (const filename of this.audioFileNames) {

            try {
                const embeddings_path = `/data/${process.env.DATASET_NAME}/embeddings/${filename}.birdnet.embeddings.msgpack`
                const embeddingsResponse = await fetch(`http://localhost:5000${embeddings_path}`); // Assuming your server is also running on localhost:5000 for backend calls
                if (!embeddingsResponse.ok) {
                    console.error(`Failed to fetch embeddings for ${filename}: ${embeddingsResponse.status} ${embeddingsResponse.statusText}`);
                    continue; // Skip to the next file if fetching embeddings fails
                }
                const embeddingsBuffer = await embeddingsResponse.arrayBuffer();
                const embeddingsData = msgpack.decode(new Uint8Array(embeddingsBuffer));
                const embeddings = embeddingsData.embeddings; // Assuming embeddings are under 'embeddings' key

                if (!embeddings || embeddings.length === 0) {
                    console.warn(`No embeddings found for ${filename}, skipping for uncertainty calculation.`);
                    continue; // Skip if no embeddings are found
                }

                const probabilities = this.protoNet.predict(embeddings);

                // Calculate average entropy as uncertainty score using ProtoNet's entropy method
                const entropies = this.protoNet.entropy(probabilities);
                const averageEntropy = entropies.reduce((sum, entropy) => sum + entropy, 0) / entropies.length;

                // Debug print average entropy for each file
                console.log(`Average entropy for ${filename
                }:`, averageEntropy);

                filesWithUncertainty.push({ filename, uncertainty: averageEntropy });

            } catch (error) {
                console.error(`Error processing embeddings for ${filename}:`, error);
                // Consider how to handle errors - skip file, or fail the whole batch? For now, skip file.
            }
        }

        // Sort files by uncertainty in ascending order (lower uncertainty first)
        filesWithUncertainty.sort((a, b) => a.uncertainty - b.uncertainty);

        const bestFiles = filesWithUncertainty.slice(0, this.batchSize).map(item => item.filename); // Just return filenames
        return bestFiles;
    }
}

/**
 * High-probability sampling using mean foreground probability.
 */
class HighProbabilitySamplingStrategy extends SamplingStrategy {
    /**
     * @param {string[]} audioFileNames - Candidate file basenames.
     * @param {number} batchSize - Number of files to return.
     * @param {{presence_prototype:number[], absence_prototype:number[]}} prototypes - Binary prototypes.
     */
    constructor(audioFileNames, batchSize, prototypes) {
        super(audioFileNames);
        this.batchSize = batchSize;
        this.prototypes = prototypes;
        this.protoNet = new PrototypicalNetwork(prototypes.presence_prototype, prototypes.absence_prototype);
    }

    /**
     * Rank files by average foreground probability (descending).
     *
     * @returns {Promise<string[]>} File basenames with strongest predicted presence.
     */
    async sample() {
        console.log('certainty sampling');
        const filesWithProbability = [];

        for (const filename of this.audioFileNames) {

            try {
                const embeddings_path = `/data/${process.env.DATASET_NAME}/embeddings/${filename}.birdnet.embeddings.msgpack`
                const embeddingsResponse = await fetch(`http://localhost:5000${embeddings_path}`); // Assuming your server is also running on localhost:5000 for backend calls
                if (!embeddingsResponse.ok) {
                    console.error(`Failed to fetch embeddings for ${filename}: ${embeddingsResponse.status} ${embeddingsResponse.statusText}`);
                    continue; // Skip to the next file if fetching embeddings fails
                }
                const embeddingsBuffer = await embeddingsResponse.arrayBuffer();
                const embeddingsData = msgpack.decode(new Uint8Array(embeddingsBuffer));
                const embeddings = embeddingsData.embeddings; // Assuming embeddings are under 'embeddings' key

                if (!embeddings || embeddings.length === 0) {
                    console.warn(`No embeddings found for ${filename}, skipping for uncertainty calculation.`);
                    continue; // Skip if no embeddings are found
                }

                const probabilities = this.protoNet.predict(embeddings);
                const averageProbability = probabilities.reduce((sum, prob) => sum + prob, 0) / probabilities.length;

                // Debug print average entropy for each file
                // console.log(`Average probability for ${filename
                // }:`, averageProbability);

                filesWithProbability.push({ filename, probability: averageProbability });

            } catch (error) {
                console.error(`Error processing embeddings for ${filename}:`, error);
                // Consider how to handle errors - skip file, or fail the whole batch? For now, skip file.
            }
        }

        // Sort files by probability in descending order (higher probability first)
        filesWithProbability.sort((a, b) => b.probability - a.probability);

        const bestFiles = filesWithProbability.slice(0, this.batchSize).map(item => item.filename); // Just return filenames
        return bestFiles;
    }
}

module.exports = { SamplingStrategy, RandomSamplingStrategy, UncertaintySamplingStrategy, CertaintySamplingStrategy, HighProbabilitySamplingStrategy };