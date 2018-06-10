require("babel-polyfill");

var tf = require('@tensorflow/tfjs');

// Create an empty neural network architecture
const model = tf.sequential();

var losses = []; // Array for losses

export function CreateNetwork(rate) {
  // Learning Rate for the network
  const alpha = rate;
  // Create an stochastic gradient descent optimizer
  const sgdOpt = tf.train.sgd(alpha);

  // Configuration for the network
  const config = {
    optimizer: sgdOpt,
    loss: tf.losses.meanSquaredError
  }

  // Create a hidden and output layer
  // These are fully connected layers
  const hidden = tf.layers.dense({
    units: 1, // No of nodes
    inputShape: [2], // No of inputs recieved
    activation: 'sigmoid' // Activation function (1/1+e^-z)
  });

  const output = tf.layers.dense({
    units: 1,
    activation: 'sigmoid'
  });

  // Connect the layers with each other
  model.add(hidden);
  model.add(output);

  // Compile and prepare the network
  model.compile(config, function() {
    console.log("neural network generated");
  });

}

export function TrainNetwork(epoch, num_iter) {

  const inputs = tf.tensor2d([
    [0.25, 0.25],
    [0.12,0.12],
    [0.4,0.4],
    [0.3, 0.3],
    [0.4,0.4],
    [0.42,0.42],
    [0.21, 0.21],
    [0.43,0.43],
    [0.22,0.22],
    [0.56, 0.56],
    [0.11,0.11],
    [0.05,0.05],

  ]);

  const outputs = tf.tensor2d([
     [0.50],
     [0.24],
     [0.8],
     [0.6],
     [0.8],
     [0.82],
     [0.42],
     [0.86],
     [0.44],
     [1.02],
     [0.22],
     [0.1],
   ]);

   tf.tidy(() => {
     train().then(() => {
       console.log('training complete');
       console.log('Tensors in memory: ' + tf.memory().numTensors);
       console.log(tf.memory().numBytes + ' bytes occupied');
     });
     async function train() {
       for(let i = 0; i < num_iter; i++) {
         const response = await model.fit(inputs, outputs, {shuffle: true, epoch: epoch});
         console.log('current loss: ' + response.history.loss[0]);

         let obj = {
           index: i,
           loss: response.history.loss[0]
         }
         losses.push(obj);
       }
     }
  });

}

export function getLosses() {
  return losses;
}
