const express = require('express');
const app = express();
const port = 3000;

app.set('view engine', 'ejs');
app.use(express.static('public'));

let todos = [
    { id: 1, task: 'Learn Node.js' },
    { id: 2, task: 'Use Cline in a project' }
];

app.get('/', (req, res) => {
    res.render('index', { todos });
});

app.listen(port, () => {
    console.log(`Server running on http://localhost:${port}`);
});
