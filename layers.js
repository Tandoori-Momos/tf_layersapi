  require("babel-polyfill");

  var tf = require('@tensorflow/tfjs');
  const model = tf.sequential(); // Create an empty neural network architecture


  export function TrainNetwork(rate, epoch, num_iter) {
    // Choosing an learning rate for the optimizer algo
    const alpha = rate;
    // Create an stochastic gradient descent optimizer
    const sgdOpt = tf.train.sgd(alpha);

    // Configuration for the neural network model
    const config = {
      optimizer: sgdOpt,
      loss: tf.losses.meanSquaredError
    }

    // Create 'dense' fully connected layers
    const hidden = tf.layers.dense({
      units: 1, // Number of nodes
      inputShape: [2], // Number of inputs
      activation: 'sigmoid' // Activation Function
    });
    const output = tf.layers.dense({
      units: 1, // Number of nodes
      inputShape: [4], // Number of inputs
      activation: 'sigmoid' // Activation Function
    });

    // Connect the empty layers with the empty architecture
    model.add(hidden);
    model.add(output);

    // Configure and prepare the model for training
    model.compile(config);

    // Training the neural network
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
        console.log(tf.memory().numTensors +  ' tensors in memory');
        console.log(tf.memory().numBytes + ' bytes occupied');
      });

      async function train() {
        for(let i = 0; i < num_iter; i++) {
          const response = await model.fit(inputs, outputs, {shuffle: true, epochs: epoch});
          console.log('current loss: ' + response.history.loss[0]);
        }
      }
    });
  }


    //   function predict() {
    //     const xs = tf.tensor([
    //       [0.3,0.3],
    //       [0.2,0.2],
    //       [0.1,0.1]
    //     ]);
    //     const ys = model.predict(xs);
    //     ys.print();
    //   }
    // });
