const express = require('express');
const path = require('path');

const app = express();
const PORT = 3000;

// Serve the public directory as static
app.use(express.static('public'));

// Start the server
app.listen(PORT, () => {
    console.log(`Server is listening on port ${PORT}`);
});
