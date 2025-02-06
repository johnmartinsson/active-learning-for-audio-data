const express = require('express');
const path = require('path');
const cors = require('cors');

const app = express();
app.use(cors());
app.use(express.json());

// Serve static files
app.use('/data', express.static(path.join(__dirname, '../data')));

// Routes
const audioRoutes = require('./routes/audioRoutes');
app.use('/api/audio', audioRoutes);

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));