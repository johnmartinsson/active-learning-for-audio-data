// ./backend/models/prototypicalNetwork.js

/**
 * Simple binary prototypical classifier.
 *
 * This class is used by legacy sampling strategies to score candidate files.
 */
class PrototypicalNetwork {
    /**
     * @param {number[]} presencePrototype - Prototype embedding for foreground/presence.
     * @param {number[]} absencePrototype - Prototype embedding for background/absence.
     */
    constructor(presencePrototype, absencePrototype) {
        this.presencePrototype = presencePrototype;
        this.absencePrototype = absencePrototype;
    }

    /**
     * Compute Euclidean distance between two vectors.
     *
     * @param {number[]} a - First vector.
     * @param {number[]} b - Second vector.
     * @returns {number} L2 distance.
     */
    euclideanDistance(a, b) {
        return Math.sqrt(a.reduce((sum, val, idx) => sum + Math.pow(val - b[idx], 2), 0));
    }

    /**
     * Compute a numerically stable softmax.
     *
     * @param {number[]} values - Unnormalized scores.
     * @returns {number[]} Probability distribution.
     */
    softmax(values) {
        const max = Math.max(...values);
        const exps = values.map(v => Math.exp(v - max));
        const sum = exps.reduce((sum, val) => sum + val, 0);
        return exps.map(v => v / sum);
    }

    /**
     * Predict foreground probability for each embedding.
     *
     * @param {number[][]} embeddings - Sequence of embedding vectors.
     * @returns {number[]} Foreground probability per embedding.
     */
    predict(embeddings) {
        return embeddings.map(embedding => {
            const presenceDist = this.euclideanDistance(embedding, this.presencePrototype);
            const absenceDist = this.euclideanDistance(embedding, this.absencePrototype);
            const [absenceProb, presenceProb] = this.softmax([-absenceDist, -presenceDist]);
            return presenceProb;
        });
    }

    /**
     * Compute binary entropy for each probability value.
     *
     * @param {number[]} probabilities - Foreground probabilities in [0, 1].
     * @returns {number[]} Entropy values.
     */
    entropy(probabilities) {
        return probabilities.map(p => {
            if (p <= 0 || p >= 1) return 0; // Entropy is 0 for probabilities 0 and 1
            return -p * Math.log2(p) - (1 - p) * Math.log2(1 - p);
        });
    }
}

module.exports = PrototypicalNetwork;