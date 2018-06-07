require("babel-polyfill");

const bodyParser = require('body-parser');
import * as tf from '../layers.js';

const urlencodedParser = bodyParser.urlencoded({ extended: false })
module.exports = function(app) {
  app.get('/', function(req,res) {
    res.render('index.ejs');
  });

  app.post('/', urlencodedParser,function(req,res) {
    if(req.body) {
      if(req.body.status == "train") {

        var alpha = req.body.rate;
        var epoch = req.body.epoch;
        var num_iter = req.body.iter;

        tf.TrainNetwork(alpha, epoch, num_iter);

        res.send("Training Complete");

      }
    }
  });
}
