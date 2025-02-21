const express = require('express');
const { getBatch, submitLabels, getPrototypes } = require('../controllers/audioController');

const router = express.Router();

// get routes
router.get('/batch', getBatch);
router.get('/prototypes', getPrototypes);

// post routes
router.post('/batch', getBatch);
router.post('/submit_labels', submitLabels);


module.exports = router;