// ./backend/models/prototypicalNetwork.js

class PrototypicalNetwork {
    constructor(presencePrototype, absencePrototype) {
        this.presencePrototype = presencePrototype;
        this.absencePrototype = absencePrototype;
    }

    euclideanDistance(a, b) {
        return Math.sqrt(a.reduce((sum, val, idx) => sum + Math.pow(val - b[idx], 2), 0));
    }

    softmax(values) {
        const max = Math.max(...values);
        const exps = values.map(v => Math.exp(v - max));
        const sum = exps.reduce((sum, val) => sum + val, 0);
        return exps.map(v => v / sum);
    }

    predict(embeddings) {
        return embeddings.map(embedding => {
            const presenceDist = this.euclideanDistance(embedding, this.presencePrototype);
            const absenceDist = this.euclideanDistance(embedding, this.absencePrototype);
            const [absenceProb, presenceProb] = this.softmax([-absenceDist, -presenceDist]);
            return presenceProb;
        });
    }

    entropy(probabilities) {
        return probabilities.map(p => {
            if (p <= 0 || p >= 1) return 0; // Entropy is 0 for probabilities 0 and 1
            return -p * Math.log2(p) - (1 - p) * Math.log2(1 - p);
        });
    }
}

module.exports = PrototypicalNetwork;