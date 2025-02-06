const express = require('express');
const { getBatch } = require('../controllers/audioController');

const router = express.Router();

router.get('/batch', getBatch);

module.exports = router;