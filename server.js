require("babel-core/register");
require("babel-polyfill");

const express = require('express');
const ejs = require('ejs');

import * as tf from './layers.js';

const routes = require('./controllers/routes.js');

const app = express();
const port = '8080';

app.set('view engine', 'ejs');
app.use(express.static('public'));

app.listen(port, () => {
  console.log('listening to port ' + port);
});

routes(app);
