import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import Webcam from 'react-webcam';
import * as cam from '@mediapipe/camera_utils';
import * as controls from '@mediapipe/control_utils';
import { drawConnectors, drawLandmarks } from '@mediapipe/drawing_utils';
import * as holistics from '@mediapipe/holistic';
import { makePrediction } from "./utilities"; 

import * as tf from "@tensorflow/tfjs";
import { nextFrame } from "@tensorflow/tfjs";

const labelMap = {
  1:{name:'Ako', color:'red'},
  2:{name:'Bakit', color:'yellow'},
  3:{name:'F', color:'lime'},
  4:{name:'Hi', color:'blue'},
  5:{name:'Hindi', color:'purple'},
  6:{name:'Ikaw', color:'red'},
  7:{name:'Kamusta', color:'yellow'},
  8:{name:'L', color:'lime'},
  9:{name:'Maganda', color:'blue'},
  10:{name:'Magandang Umaga', color:'purple'},
  11:{name:'N', color:'red'},
  12:{name:'O', color:'yellow'},
  13:{name:'Oo', color:'lime'},
  14:{name:'P', color:'blue'},
  15:{name:'Salamat', color:'purple'},
}

// Global error handler
window.onerror = function (message, source, lineno, colno, error) {
  console.log('A global error was caught:', message);
  return true; // Prevents the default browser error handling
};

window.addEventListener('unhandledrejection', event => {
  event.preventDefault();
  console.log('Caught unhandled rejection:', event.reason);
});

const net = await tf.loadLayersModel('https://senyasfsltranslator.s3.jp-tok.cloud-object-storage.appdomain.cloud/model.json')
net.summary();
let frameCounter = 0;
const framesData = [];
let predictioncount = 0;
let totalLatency = 0;

const Sidebar = ({ onAppModeChange, onDetectionConfidenceChange, onTrackingConfidenceChange, detectionConfidence, trackingConfidence }) => {
  // State of the application
  const [isExpanded, setIsExpanded] = useState(true);
  const [appMode, setAppMode] = useState('About App');

  const handleDetectionConfidenceChange = (e) => {
    onDetectionConfidenceChange(parseFloat(e.target.value));
  };

  // Function to handle tracking confidence change
  const handleTrackingConfidenceChange = (e) => {
    onTrackingConfidenceChange(parseFloat(e.target.value));
  };
  const toggleSidebar = () => {
    setIsExpanded(!isExpanded);
  };

  const handleAppModeChange = (event) => {
    onAppModeChange(event.target.value);
    setAppMode(event.target.value); // Update selected app mode
  };
  
  const sidebarClass = isExpanded ? "sidebar" : "sidebar-hidden";

  return (
    <div>
    <div>
      <button className ='expand-button' onClick={toggleSidebar}>
        {isExpanded ? "Hide" : "Show"}
      </button>
    </div>
    <div className={sidebarClass}>
      <div className="dropdown">
        <h2>Senyas: Filipino Sign Language Translator solution</h2>
        <div className="centered-content">
          <select value={appMode} onChange={handleAppModeChange}>
            <option value="About App">About App</option>
            <option value="Run on Video">Run on Video</option>
            <option value="Add a video">Add a video</option>
          </select>
        </div>
      </div>
      <h4>Parameters</h4>
      <div className="slider-container">
        <h3>Detection Confidence</h3>
        <input
          type="range"
          min="0"
          max="1"
          step="0.1"
          defaultValue="0.5"
          onChange={handleDetectionConfidenceChange}
        />
      </div>
      <div className="slider-container">
        <p>Min Detection Confidence: {detectionConfidence}</p>
        <h3>Tracking Confidence</h3>
        <input
          type="range"
          min="0"
          max="1"
          step="0.1"
          defaultValue="0.5"
          onChange={handleTrackingConfidenceChange}
        />
        <p>Min Tracking Confidence: {trackingConfidence}</p>
      </div>
    </div>
    </div>
  );
};

function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);

  var camera = null;
  const [lastPrediction, setLastPrediction] = useState(null);
  const [averageLatency, setAverageLatency] = useState(0);
  const [predictionCount, setPredictionCount] = useState(0);
  
  const [camHeight, setCamHeight] = useState(0);
  const [camWidth, setCamWidth] = useState(0);

  const [appMode, setAppMode] = useState('About App');
  const [detectionConfidence, setDetectionConfidence] = useState(0.5);
  const [trackingConfidence, setTrackingConfidence] = useState(0.5);

  const handleAppModeChange = (selectedMode) => {
    setAppMode(selectedMode); // Update the app mode based on sidebar selection
  };

  // Function to handle detection confidence change
  const handleDetectionConfidenceChange = (value) => {
    setDetectionConfidence(value);
    // Add logic here for further actions on change if needed
  };

  // Function to handle tracking confidence change
  const handleTrackingConfidenceChange = (value) => {
    setTrackingConfidence(value);
    // Add logic here for further actions on change if needed
  };

  
  const onResults = async (model) => {
    if (typeof webcamRef.current !== "undefined" &&  webcamRef.current !== null && webcamRef.current.video.readyState === 4) {
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
    if (appMode === 'Run on Video') {
    const getCameraStream = async () => {
      try {
        // Use generic media constraints
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
  
    const holistic = new holistics.Holistic({locateFile: (file) => {
      return `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`;
    }});
  
    holistic.setOptions({
      modelComplexity: 1,
      smoothLandmarks: false,
      enableSegmentation: false,
      smoothSegmentation: false,
      refineFaceLandmarks: false,
      minDetectionConfidence: detectionConfidence,
      minTrackingConfidence: trackingConfidence
    });
  
    holistic.onResults(onResults);
    if (typeof webcamRef.current !== "undefined" && webcamRef.current !== null && webcamRef.current.video) {
      try{
        camera = new cam.Camera(webcamRef.current.video, {
          onFrame: async () => {
            frameCounter++;
            await holistic.send({ image: webcamRef.current.video });
          },
          width: window.innerWidth,
          height: window.innerHeight
        });
        camera.start();
    } catch (err) {
      console.error("Error starting the camera: ", err);
      }
    } 
  }}, [appMode]);

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

    const scores = net.predict(expanded);
    const label = await makePrediction(scores, 0.8, videoWidth, videoHeight, ctx)

    // End time
    const endTime = performance.now();

    // Calculate and log latency
    const latency = endTime - startTime;
    totalLatency += latency;
    predictioncount++;
    setPredictionCount(predictioncount);
    setAverageLatency(totalLatency / predictioncount);

    //console.log(`Prediction latency: ${latency} milliseconds`);
    //console.log(`Average Prediction latency: ${totalLatency/predictioncount} milliseconds`);

    setLastPrediction(label);
    tf.dispose(tensor)
    tf.dispose(expanded)
    tf.dispose(scores)
  }

  return (
    <div className = "App-header">
      <Sidebar 
              onAppModeChange={handleAppModeChange}
              onDetectionConfidenceChange={handleDetectionConfidenceChange}
              onTrackingConfidenceChange={handleTrackingConfidenceChange} 
              detectionConfidence={detectionConfidence}
              trackingConfidence={trackingConfidence}
          />
      <div className="content">
          {appMode === 'About App' && (
            <div>
              <h1>Senyas: Filipino Sign Language Translator Solution</h1>
              <p>In this application we are using <b>MediaPipe</b> for to recognize Filipino sign language static and dynamic hand gestures. 
              <b>ReactJS</b> is used to build the Web Graphical User Interface (GUI)</p>
            </div>
          )}
          {appMode === 'Run on Video' && (
          <div>
          <div>
          <h4>Avg. Prediction Latency: {averageLatency.toFixed(2)} ms @ {predictionCount} predicts</h4>
          </div>
          <div className='content-wrapper'>
          <Webcam
            ref={webcamRef}
            audio={false}
            id="img"
            className="centered-element"
            style={{
              backgroundColor: "000000",
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
            className="centered-element"
            style={{
              backgroundColor: "000000",
              width: {camWidth}, // 9:16 aspect ratio
              height: {camHeight},
              '@media': { // Media query for smartphones
                  width: {camWidth}, // 9:16 aspect ratio
                  height: {camHeight},
              }
            }}
            id="myCanvas"
          />
          </div>
          <div className='below-video' style={{marginTop: `${camHeight}px`}}>
            <p> video </p>
            <h4> Last prediction: {lastPrediction} </h4>
          </div>
        </div>
          )}
          {appMode === 'Add a video' && (
            <div>
              <p>Add a video: Details here</p>
            </div>
          )}
      </div>
    </div>
  );
}

export default App;