# How to use the Annotation Tool?

Clone the annotation tool code

      git clone https://github.com/johnmartinsson/active-learning-for-audio-data.git

## Prepare the Audio Data
Clone a fork of the BitdNET-Analyzer

      git clone https://github.com/johnmartinsson/BirdNET-Analyzer.git

Create an environment for the BirdNET-Analyzer

      conda create -n birdnet
      conda activate birdnet
      pip install -r requirements.txt

Run the data preparation script

      python prepare_audio_data.py birdnet_analyzer/example/ <directory>/active-learning-for-audio-data/backend/data/example 

where you set the directory to that where you cloned the repository. This will pre-process the data by

- splitting audio into 30 second segments (audio),
- pre-compute Mel spectrograms for each segment (spectrogram), and
- pre-compute the BirdNET embeddings for each segment (embeddings).

## Prepare the backend

The node.js backend does run some python code, and this will setup the environment for that.

      cd backend
      conda create -n audio-labeling-test python=3.13
      pip install -r requirements.txt

Next install the node.js dependencies and start the backend

      npm install
      npm start

## Prepare the frontend

      cd frontend
      npm install
      npm start

# Project Guidelines

1. **Backend and Frontend Separation**:
   - The backend is developed using Node.js.
   - The frontend is developed using React.

2. **Machine Learning Integration**:
   - Machine learning models are implemented in Python.
   - The backend should be able to call Python scripts for tasks such as change point detection.

3. **Self-Hosting**:
   - Users should be able to self-host the backend and frontend for their projects.
   - Provide detailed documentation and scripts (e.g., Docker) for self-hosting.

4. **Scalability and Performance**:
   - The tool should handle CPU/GPU intensive tasks efficiently.
   - Implement load balancing and caching mechanisms to improve performance.
   - Use asynchronous processing for background jobs.

5. **Data Handling**:
   - The tool should handle the submission of labels for audio recordings.
   - For large datasets, uploading audio data should be done through a dedicated upload application/script.

6. **Security**:
   - Implement robust authentication and authorization mechanisms.
   - Ensure data is encrypted both in transit and at rest.

7. **Collaboration**:
   - Multiple users should be able to connect to the backend and annotate the same dataset.
   - Implement real-time collaboration features using WebSockets or similar technologies.

8. **Version Control**:
   - Implement version control for annotations to track changes and allow users to revert to previous versions.

9. **Monitoring and Logging**:
   - Use monitoring tools to keep track of the performance and health of the services.
   - Implement centralized logging to collect and analyze logs from the backend and microservices.

10. **Modularity**:
    - The project should be designed in a modular way to allow for easy integration of new labeling strategies and machine learning models.

These constraints should guide the development and deployment of the audio labeling tool, ensuring it is scalable, secure, and user-friendly.

## Example Architecture:
```
+-------------------+       +-------------------+       +-------------------+
|                   |       |                   |       |                   |
|     Frontend      | <---> |   API Gateway     | <---> |  ML Microservices |
|    (React)        |       | (Node.js Backend) |       |   (Python)        |
|                   |       |                   |       |                   |
+-------------------+       +-------------------+       +-------------------+
        |                           |                           |
        |                           |                           |
        |                           |                           |
        +---------------------------+---------------------------+
                                    |
                                    |
                                    |
                            +---------------------+
                            |                     |
                            | Cloud/Local Storage |
                            |                     |
                            +---------------------+
```
