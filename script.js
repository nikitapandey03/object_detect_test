// let model;
// const video = document.getElementById('webcam');
// const canvas = document.getElementById('canvas');
// const ctx = canvas.getContext('2d');

// // Load model
// async function loadModel() {
//   model = await cocoSsd.load();
//   console.log("Model loaded.");
//   detectFrame();
// }

// // Setup webcam
// async function setupWebcam() {
//   return new Promise((resolve, reject) => {
//     navigator.mediaDevices.getUserMedia({
//       video: true,
//       audio: false
//     }).then(stream => {
//       video.srcObject = stream;
//       video.onloadedmetadata = () => {
//         video.play();
//         resolve();
//       };
//     }).catch(err => {
//       reject(err);
//     });
//   });
// }


// async function detectFrame() {
//   const predictions = await model.detect(video);

 
//   ctx.clearRect(0, 0, canvas.width, canvas.height);


//   predictions.forEach(pred => {
//     ctx.strokeStyle = "#00FF00";
//     ctx.lineWidth = 2;
//     ctx.strokeRect(...pred.bbox);
//     ctx.font = "18px Arial";
//     ctx.fillStyle = "#00FF00";
//     ctx.fillText(pred.class + " - " + Math.round(pred.score * 100) + "%", pred.bbox[0], pred.bbox[1] > 10 ? pred.bbox[1] - 5 : 10);
//   });

//   requestAnimationFrame(detectFrame);
// }

// // Start everything
// setupWebcam().then(() => {
//   loadModel();
// });


// Step 8 — Constants & Listeners
const STATUS = document.getElementById('status');
const VIDEO = document.getElementById('webcam');
const ENABLE_CAM_BUTTON = document.getElementById('enableCam');
const RESET_BUTTON = document.getElementById('reset');
const TRAIN_BUTTON = document.getElementById('train');
const MOBILE_NET_INPUT_WIDTH = 224;
const MOBILE_NET_INPUT_HEIGHT = 224;
const STOP_DATA_GATHER = -1;
const CLASS_NAMES = [];

ENABLE_CAM_BUTTON.addEventListener('click', enableCam);
TRAIN_BUTTON.addEventListener('click', trainAndPredict);
RESET_BUTTON.addEventListener('click', reset);

let dataCollectorButtons = document.querySelectorAll('button.dataCollector');
for (let i = 0; i < dataCollectorButtons.length; i++) {
  dataCollectorButtons[i].addEventListener('mousedown', gatherDataForClass);
  dataCollectorButtons[i].addEventListener('mouseup', gatherDataForClass);
  CLASS_NAMES.push(dataCollectorButtons[i].getAttribute('data-name'));
}

let mobilenet = undefined;
let gatherDataState = STOP_DATA_GATHER;
let videoPlaying = false;
let trainingDataInputs = [];
let trainingDataOutputs = [];
let examplesCount = [];
let predict = false;

// Step 9 — Load MobileNet Feature Model
async function loadMobileNetFeatureModel() {
  const URL =
    'https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v3_small_100_224/feature_vector/5/default/1';
  mobilenet = await tf.loadGraphModel(URL, { fromTFHub: true });
  STATUS.innerText = 'MobileNet v3 loaded successfully!';
  tf.tidy(function () {
    let answer = mobilenet.predict(tf.zeros([1, MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH, 3]));
    console.log(answer.shape);
  });
}
loadMobileNetFeatureModel();

// Step 10 — Define Model Head
let model = tf.sequential();
model.add(tf.layers.dense({ inputShape: [1024], units: 128, activation: 'relu' }));
model.add(tf.layers.dense({ units: CLASS_NAMES.length, activation: 'softmax' }));
model.summary();

model.compile({
  optimizer: 'adam',
  loss: (CLASS_NAMES.length === 2) ? 'binaryCrossentropy' : 'categoricalCrossentropy',
  metrics: ['accuracy']
});

// Step 11 — Enable Webcam
function hasGetUserMedia() {
  return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

function enableCam() {
  if (hasGetUserMedia()) {
    const constraints = { video: true, width: 640, height: 480 };
    navigator.mediaDevices.getUserMedia(constraints).then(function (stream) {
      VIDEO.srcObject = stream;
      VIDEO.addEventListener('loadeddata', function () {
        videoPlaying = true;
        ENABLE_CAM_BUTTON.classList.add('removed');
      });
    });
  } else {
    console.warn('getUserMedia() is not supported by your browser');
  }
}

// Step 12 — Gather Data Event Handler
function gatherDataForClass() {
  let classNumber = parseInt(this.getAttribute('data-1hot'));
  gatherDataState = (gatherDataState === STOP_DATA_GATHER) ? classNumber : STOP_DATA_GATHER;
  dataGatherLoop();
}

// Step 13 — Data Gathering
function dataGatherLoop() {
  if (videoPlaying && gatherDataState !== STOP_DATA_GATHER) {
    let imageFeatures = tf.tidy(function () {
      let videoFrameAsTensor = tf.browser.fromPixels(VIDEO);
      let resizedTensorFrame = tf.image.resizeBilinear(
        videoFrameAsTensor,
        [MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH],
        true
      );
      let normalizedTensorFrame = resizedTensorFrame.div(255);
      return mobilenet.predict(normalizedTensorFrame.expandDims()).squeeze();
    });
    trainingDataInputs.push(imageFeatures);
    trainingDataOutputs.push(gatherDataState);

    if (examplesCount[gatherDataState] === undefined) {
      examplesCount[gatherDataState] = 0;
    }
    examplesCount[gatherDataState]++;
    
    STATUS.innerText = '';
    for (let n = 0; n < CLASS_NAMES.length; n++) {
      STATUS.innerText += CLASS_NAMES[n] + ' data count: ' + examplesCount[n] + '. ';
    }
    window.requestAnimationFrame(dataGatherLoop);
  }
}

// Step 14 — Train & Predict
async function trainAndPredict() {
  predict = false;
  tf.util.shuffleCombo(trainingDataInputs, trainingDataOutputs);
  let outputsAsTensor = tf.tensor1d(trainingDataOutputs, 'int32');
  let oneHotOutputs = tf.oneHot(outputsAsTensor, CLASS_NAMES.length);
  let inputsAsTensor = tf.stack(trainingDataInputs);

  let results = await model.fit(inputsAsTensor, oneHotOutputs, {
    shuffle: true,
    batchSize: 5,
    epochs: 10,
    callbacks: { onEpochEnd: logProgress }
  });
  outputsAsTensor.dispose();
  oneHotOutputs.dispose();
  inputsAsTensor.dispose();
  predict = true;
  predictLoop();
}

function logProgress(epoch, logs) {
  console.log('Data for epoch ' + epoch, logs);
}

// Core Prediction Loop
function predictLoop() {
  if (predict) {
    tf.tidy(function () {
      let videoFrameAsTensor = tf.browser.fromPixels(VIDEO).div(255);
      let resizedTensorFrame = tf.image.resizeBilinear(
        videoFrameAsTensor,
        [MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH],
        true
      );
      let imageFeatures = mobilenet.predict(resizedTensorFrame.expandDims());
      let prediction = model.predict(imageFeatures).squeeze();
      let highestIndex = prediction.argMax().arraySync();
      let predictionArray = prediction.arraySync();

      STATUS.innerText =
        'Prediction: ' +
        CLASS_NAMES[highestIndex] +
        ' with ' +
        Math.floor(predictionArray[highestIndex] * 100) +
        '% confidence';
    });

    window.requestAnimationFrame(predictLoop);
  }
}

// Step 15 — Reset Button
function reset() {
  predict = false;
  examplesCount.length = 0;
  for (let i = 0; i < trainingDataInputs.length; i++) {
    trainingDataInputs[i].dispose();
  }
  trainingDataInputs.length = 0;
  trainingDataOutputs.length = 0;
  STATUS.innerText = 'No data collected';
  console.log('Tensors in memory: ' + tf.memory().numTensors);
}