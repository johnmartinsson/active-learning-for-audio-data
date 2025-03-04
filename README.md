# Active learning


## Active sampling

## Active labeling

# Embeddings

- [ ] embed audio using https://github.com/bioacoustic-ai/bacpipe

Collecting workspace information

Sure, here is a summary of the project constraints for your audio labeling tool:

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
