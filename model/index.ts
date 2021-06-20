import * as tf from '@tensorflow/tfjs-node';
const diceData = require('./dice-data.json');

// Shuffles 2 arrays at the same time.
function shuffleCombo(array: any[], array2: any[]) {
  let counter = array.length;
  console.assert(array.length === array2.length);
  let temp, temp2;
  let index = 0;
  // While there are elements in the array
  while (counter > 0) {
    // Pick a random index
    index = (Math.random() * counter) | 0;
    // Decrease counter by 1
    counter--;
    // And swap the last element with it
    temp = array[counter];
    temp2 = array2[counter];
    array[counter] = array[index];
    array2[counter] = array2[index];
    array[index] = temp;
    array2[index] = temp2;
  }
}

const doStuff = async () => {
  const [trainX, trainY, testX, testY] = tf.tidy(() => {
    const diceImages = [].concat(
      diceData['0'],
      diceData['1'],
      diceData['2'],
      diceData['3'],
      diceData['4'],
      diceData['5'],
      diceData['6'],
      diceData['7'],
      diceData['8'],
    );

    const answers = ([] as any[]).concat(
      new Array(diceData['0'].length).fill(0),
      new Array(diceData['1'].length).fill(1),
      new Array(diceData['2'].length).fill(2),
      new Array(diceData['3'].length).fill(3),
      new Array(diceData['4'].length).fill(4),
      new Array(diceData['5'].length).fill(5),
      new Array(diceData['6'].length).fill(6),
      new Array(diceData['7'].length).fill(7),
      new Array(diceData['8'].length).fill(8),
    );

    shuffleCombo(diceImages, answers);

    // Group into train/test split
    const testCount = Math.round(diceImages.length * 0.2);
    const trainCount = diceImages.length - testCount;

    const testImgData = diceImages.slice(trainCount);
    const testAnswerData = answers.slice(trainCount);
    diceImages.splice(trainCount);
    answers.splice(trainCount);

    // Convert to tensors
    const numOptions = Object.keys(diceData).length;
    const trainX = tf.tensor(diceImages).expandDims(3);
    const trainY = tf.oneHot(answers, numOptions);

    const testX = tf.tensor(testImgData).expandDims(3);
    const testY = tf.oneHot(testAnswerData, numOptions);

    console.log('trainX ', trainX.shape);
    console.log('trainY ', trainY.shape);
    console.log('testX ', testX.shape);
    console.log('testY ', testY.shape);

    return [trainX, trainY, testX, testY];
  });

  const model = tf.sequential();

  // Flatten the image so it can be deeply connected to the next layer.
  model.add(tf.layers.flatten({ inputShape: [12, 12, 1] }));
  model.add(
    tf.layers.dense({
      units: 64,
      activation: 'relu',
    }),
  );
  model.add(
    tf.layers.dense({
      units: 8,
      activation: 'relu',
    }),
  );
  model.add(
    tf.layers.dense({
      units: 9,
      kernelInitializer: 'varianceScaling',
      activation: 'softmax',
    }),
  );
  const learningRate = 0.005;
  model.compile({
    optimizer: tf.train.adam(learningRate),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  await model.fit(trainX, trainY, {
    epochs: 20,
    validationSplit: 0.1,
    shuffle: true,
  });

  console.log('Done Training');
  const result: any = model.evaluate(testX, testY);
  console.log('Test Loss', result[0].dataSync());
  console.log('Test Accuracy', result[1].dataSync());
  tf.dispose(result);
  model.save(`file://${__dirname}/model`);
};

doStuff();
