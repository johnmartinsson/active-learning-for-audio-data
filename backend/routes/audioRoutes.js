// ./backend/routes/audioRoutes.js
const express = require('express');
const {
  getBatch,
  getSegments,
  submitLabels
} = require('../controllers/audioController');

const router = express.Router();

// Return a new batch of unlabeled files
// e.g. GET /audio/batch?strategy=random&batchSize=5
router.get('/batch', getBatch);

// Return computed segments for a given file
// e.g. GET /audio/myfile/segments?labelingStrategyChoice=active&numSegments=10
router.get('/:filename/segments', getSegments);

// Submit labels for a given file
// e.g. POST /audio/myfile/labels
router.post('/:filename/labels', submitLabels);

module.exports = router;
