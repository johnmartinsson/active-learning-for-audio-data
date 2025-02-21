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
        console.log('random sampling');
        return shuffled.slice(0, this.batchSize);
    }
}

class UncertaintyBasedSamplingStrategy extends SamplingStrategy {
    constructor(audioFiles, batchSize, prototypes) {
        super(audioFiles);
        this.batchSize = batchSize;
    }

    // TODO: Now I just need to implement the sample() method. 
    // I should probably implement a shared ProtoNet module that is used by both the backend and frontend, or I do everything in the backend.
    // Computing the segments has the advantage that I do not need to send the embeddings to the frontend, which is a good thing.
    // I can just send the audio file and the segment start and end times for the different labeling strategies. This makes a lot of sense.
    // However, it does require the backend to do some more work. I think I will go with this approach.

    sample() {
        const shuffled = this.audioFiles.sort(() => 0.5 - Math.random());
        console.log('uncertainty based sampling');
        return shuffled.slice(0, this.batchSize);
    }
}

module.exports = { SamplingStrategy, RandomSamplingStrategy, UncertaintyBasedSamplingStrategy};