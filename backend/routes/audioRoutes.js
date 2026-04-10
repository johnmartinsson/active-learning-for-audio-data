// ./backend/routes/audioRoutes.js
/**
 * Audio API route declarations.
 *
 * This router exposes the backend endpoints used by the annotation UI for:
 * - selecting unlabeled files,
 * - retrieving available classes,
 * - computing segmentation proposals,
 * - persisting user labels.
 */
const express = require('express');
const {
  getBatch,
  getClasses,
  getSegments,
  submitLabels
} = require('../controllers/audioController');

const router = express.Router();

// Return a new batch of unlabeled files
// e.g. GET /audio/batch?strategy=random&batchSize=5
router.get('/batch', getBatch);

// Return all available positive class labels inferred from saved label files
// e.g. GET /audio/classes
router.get('/classes', getClasses);

// Return computed segments for a given file
// e.g. GET /audio/myfile/segments?labelingStrategyChoice=active&numSegments=10
router.get('/:filename/segments', getSegments);

// Submit labels for a given file
// e.g. POST /audio/myfile/labels
router.post('/:filename/labels', submitLabels);

module.exports = router;
