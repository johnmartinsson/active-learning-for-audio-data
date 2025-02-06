class SamplingStrategy {
    constructor(audioFiles) {
        this.audioFiles = audioFiles;
    }

    sample() {
        throw new Error('sample() must be implemented by subclass');
    }
}

class RandomSamplingStrategy extends SamplingStrategy {
    constructor(audioFiles, batchSize) {
        super(audioFiles);
        this.batchSize = batchSize;
    }

    sample() {
        const shuffled = this.audioFiles.sort(() => 0.5 - Math.random());
        return shuffled.slice(0, this.batchSize);
    }
}

module.exports = { SamplingStrategy, RandomSamplingStrategy };