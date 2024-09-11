import * as tf from '@tensorflow/tfjs';
import { makePrediction } from './utils';

let model: tf.LayersModel | null = null;

async function loadModel(modelUrl: string) {
  if (!model) {
    const response = await fetch(modelUrl);
    const modelJson = await response.json();
    model = await tf.loadLayersModel(tf.io.fromMemory(modelJson));
    console.log('Model loaded in worker');
  }
  return model;
}

self.onmessage = async (event) => {
  const { framesData, videoWidth, videoHeight, modelUrl, action } = event.data;

  if (action === 'load_model') {
    await loadModel(modelUrl);
    self.postMessage({ action: 'model_loaded' });
    return;
  }

  if (!model) {
    console.error('Model not loaded');
    return;
  }

  const tensor = tf.tensor(framesData);
  const expanded = tensor.expandDims(0);
  const startTime = performance.now();

  const scores = model.predict(expanded) as tf.Tensor;
  
  const label = await makePrediction(scores, 0.8, videoWidth, videoHeight);
  const endTime = performance.now();
  const latency = endTime - startTime;

  tf.dispose([tensor, expanded, scores]);

  self.postMessage({ action: 'prediction', label, latency });
};