const express = require('express');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3000;

// Serve static files from the public directory
app.use(express.static(path.join(__dirname, 'src/public')));

// Serve the main page
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'src/public/index.html'));
});

// Start the server
app.listen(PORT, () => {
    console.log(`Gomoku game server is running on http://localhost:${PORT}`);
    console.log('Press Ctrl+C to stop the server');
});
