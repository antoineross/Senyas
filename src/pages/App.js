import Webcam from 'react-webcam';
import { drawConnectors, drawLandmarks } from '@mediapipe/drawing_utils';
import { labelMap, makePrediction } from "../components/utilities"; 

import * as tf from "@tensorflow/tfjs";
import { nextFrame } from "@tensorflow/tfjs";

import React, { useState, useEffect, useRef, Suspense } from 'react';

import * as cam from '@mediapipe/camera_utils';
import * as controls from '@mediapipe/control_utils';
import * as holistics from '@mediapipe/holistic';
import * as drawingUtils from '@mediapipe/drawing_utils';
import './App.css';

import ExecutionEnvironment from '@docusaurus/ExecutionEnvironment';

let Webcam2;
if (ExecutionEnvironment.canUseDOM) {
  Webcam2 = require('react-webcam').default;
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}


let frameCounter = 0;
const framesData = [];
let predictioncount = 0;
let totalLatency = 0;

function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const netRef = useRef(null);
  const [lastPrediction, setLastPrediction] = useState(null);
  const [averageLatency, setAverageLatency] = useState(0);
  const [predictionCount, setPredictionCount] = useState(0);
  const [camHeight, setCamHeight] = useState(0);
  const [camWidth, setCamWidth] = useState(0);

  useEffect(() => {
    // Load TensorFlow Model in the client-side environment
    const loadModel = async () => {
      netRef.current = await tf.loadLayersModel('https://senyasfsltranslatorv2.s3.jp-tok.cloud-object-storage.appdomain.cloud/model.json');
      netRef.current.summary();
    };

    if (typeof window !== 'undefined') {
      loadModel();
      // Setup window error handlers
      window.onerror = function (message, source, lineno, colno, error) {
        console.log('A global error was caught:', message);
        return true;
      };

      window.addEventListener('unhandledrejection', event => {
        event.preventDefault();
        console.log('Caught unhandled rejection:', event.reason);
      });
    }
  }, []);


  var camera = null;



  const onResults = async (model) => {
    if (typeof webcamRef.current !== "undefined" &&  webcamRef.current !== null && webcamRef.current.video.readyState === 4) {
      const video = webcamRef.current.video;

      const videoWidth = webcamRef.current.video.videoWidth;
      const videoHeight = webcamRef.current.video.videoHeight;
      
      setCamHeight(videoHeight);
      setCamWidth(videoWidth);

      canvasRef.current.width = videoWidth;
      canvasRef.current.height = videoHeight;

      const ctx = canvasRef.current.getContext("2d");
      ctx.save();

      ctx.clearRect(0, 0, videoWidth, videoHeight);
      //ctx.drawImage(model.segmentationMask, 0, 0, videoWidth, videoHeight);
      ctx.globalCompositeOperation = "source-in";
      ctx.fillStyle = "#00FF00";
      ctx.fillRect(0, 0, videoWidth, videoHeight);

      ctx.globalCompositeOperation = "destination-atop";
      ctx.drawImage(model.image, 0, 0, videoWidth, videoHeight);

      ctx.globalCompositeOperation = "source-over";
      drawConnectors(ctx, model.poseLandmarks, holistics.POSE_CONNECTIONS, { color: '#C0C0C070', lineWidth: 4 });
      drawLandmarks(ctx, model.poseLandmarks, { color: '#FF0000', lineWidth: 2 });
      drawConnectors(ctx, model.faceLandmarks, holistics.FACEMESH_TESSELATION,
        { color: '#C0C0C070', lineWidth: 1 });
      drawConnectors(ctx, model.leftHandLandmarks, holistics.HAND_CONNECTIONS,
        { color: '#CC0000', lineWidth: 5 });
      drawLandmarks(ctx, model.leftHandLandmarks,
        { color: '#00FF00', lineWidth: 1 });
      drawConnectors(ctx, model.rightHandLandmarks, holistics.HAND_CONNECTIONS,
        { color: '#00CC00', lineWidth: 5 });
      drawLandmarks(ctx, model.rightHandLandmarks,
        { color: '#FF0000', lineWidth: 1 });

      // console.log(model.poseLandmarks)
      // Call the frameCount function to track frames and collect data
      if (framesData.length === 30){
        frameCount(model, ctx, videoWidth, videoHeight);
      } else{
        const pose = model.poseLandmarks ? model.poseLandmarks.map(landmark => [landmark.x, landmark.y, landmark.z, landmark.visibility]).flat() : new Array(33*4).fill(0);
        const lh = model.leftHandLandmarks ? model.leftHandLandmarks.map(landmark => [landmark.x, landmark.y, landmark.z]).flat() : new Array(21*3).fill(0);
        const rh = model.rightHandLandmarks ? model.rightHandLandmarks.map(landmark => [landmark.x, landmark.y, landmark.z]).flat() : new Array(21*3).fill(0);
        
        const frameData = [...pose, ...lh, ...rh];
        
        framesData.push(frameData);
      }
      
      ctx.restore();
    }
  };



  useEffect(() => {
    // Camera stream and other browser-specific operations
    if (typeof window !== 'undefined') {
      const getCameraStream = async () => {
        try {
          const constraints = { video: true };
          const stream = await navigator.mediaDevices.getUserMedia(constraints);
          if (webcamRef.current) {
            webcamRef.current.srcObject = stream;
          }
        } catch (err) {
          console.error("Error accessing the camera: ", err);
        }
      };

      getCameraStream();
    }

  
    const holistic = new holistics.Holistic({locateFile: (file) => {
      return `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`;
    }});
  
    holistic.setOptions({
      modelComplexity: 1,
      smoothLandmarks: false,
      enableSegmentation: false,
      smoothSegmentation: false,
      refineFaceLandmarks: false,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5
    });
  
    holistic.onResults(onResults);
    let localCamera = null;
  
    if (typeof window !== 'undefined' && typeof webcamRef.current !== "undefined" && webcamRef.current !== null) {
      localCamera = new cam.Camera(webcamRef.current.video, {
        onFrame: async () => {
          try {
            frameCounter++;
            await holistic.send({ image: webcamRef.current.video });
          } catch (error) {
            // Do nothing
          }
        },
        width: window.innerWidth,
        height: window.innerHeight
      });
      localCamera.start();
    }
    // Return a cleanup function
    return () => {
        // Stop the camera
        if (localCamera) {
            localCamera.stop();
        }
      };
  }, []);
  
function getArrayShape(array) {
    return [array.length, array[0].length];
}

function frameCount(results, ctx, videoWidth, videoHeight) {
    if (framesData.length === 30) {
      processFramesData(framesData, ctx, videoWidth, videoHeight);
      framesData.length = 0;
    }
  }

async function processFramesData(framesData, ctx, videoWidth, videoHeight) {
  
  const tensor = tf.tensor(framesData);
  const expanded = tensor.expandDims(0);
  // console.log(expanded.shape);
  // Start time
  const startTime = performance.now();

  const scores = netRef.current.predict(expanded);
  const label = await makePrediction(scores, 0.8, videoWidth, videoHeight, ctx)

  // End time
  const endTime = performance.now();

  // Calculate and log latency
  const latency = endTime - startTime;
  totalLatency += latency;
  predictioncount++;
  setPredictionCount(predictioncount);
  setAverageLatency(totalLatency / predictioncount);

  setLastPrediction(label);
  tf.dispose(tensor);
  tf.dispose(expanded);
  tf.dispose(scores);

  // Sleep for 500 ms
  await sleep(500);
  
}



  return (
    <div className = "flex flex-col items-center justify-center">
    <div className="flex justify-center items-center h-screen">
      <h6 style={{
    fontWeight: 'medium',
    fontSize: '1em',
    textAlign: 'center',
  }}
  >Last prediction: {lastPrediction} </h6>
<h6 style={{
    fontWeight: 'medium',
    fontSize: '0.8em',
    textAlign: 'center',
  }}>Avg. Prediction Latency: {averageLatency.toFixed(2)} ms @ {predictionCount} predicts</h6>

    </div>      
    <div> 
    <div>
    <Webcam
        ref={webcamRef}
        audio={false}
        id="img"
        style={{
          position: "absolute",
          marginLeft: "auto",
          marginRight: "auto",
          left: 0,
          right: 0,
          textAlign: "center",
          zindex: 9,
          width: {camWidth}, // 9:16 aspect ratio
          height: {camHeight},
          '@media': { // Media query for smartphones
              width: {camWidth}, // 9:16 aspect ratio
              height: {camHeight},
          }
        }}
      />

      <canvas
        ref={canvasRef}
        style={{
          position: "absolute",
          marginLeft: "auto",
          marginRight: "auto",
          left: 0,
          right: 0,
          textAlign: "center",
          zindex: 9,
          width: {camWidth}, // 9:16 aspect ratio
          height: {camHeight},
        }}
        id="myCanvas"
      />
    </div>
    </div>
    </div>
  );
}

export default App;