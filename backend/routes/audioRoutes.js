// ./backend/routes/audioRoutes.js
const express = require('express');
const {
  getBatch,
  submitLabels,
  getPrototypes,
  getSegments
} = require('../controllers/audioController');

const router = express.Router();

// Getters
router.get('/batch', getBatch);
router.get('/prototypes', getPrototypes);
router.post('/batch', getBatch);
router.post('/segments', getSegments);

// Setters
router.post('/submit_labels', submitLabels);

module.exports = router;
