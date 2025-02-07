const express = require('express');
const { getBatch, submitLabels } = require('../controllers/audioController');

const router = express.Router();

router.get('/batch', getBatch);
router.post('/submit_labels', submitLabels);

module.exports = router;