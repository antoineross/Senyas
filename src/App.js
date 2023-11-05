import React, { useState, useEffect, useRef } from 'react';
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

const net = await tf.loadLayersModel('https://senyasfsltranslator.s3.jp-tok.cloud-object-storage.appdomain.cloud/model.json')
net.summary();
let frameCounter = 0;
const framesData = [];

function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const connect = window.drawConnectors;
  var camera = null;


  const onResults = async (model) => {
    if (typeof webcamRef.current !== "undefined" && webcamRef.current !== null && webcamRef.current.video.readyState === 4) {
      const video = webcamRef.current.video;

      const videoWidth = webcamRef.current.video.videoWidth;
      const videoHeight = webcamRef.current.video.videoHeight;

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
        frameCount(model, ctx);
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
    const holistic = new holistics.Holistic({locateFile: (file) => {
      return `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`;
    }});

    holistic.setOptions({
      modelComplexity: 1,
      smoothLandmarks: true,
      enableSegmentation: true,
      smoothSegmentation: true,
      refineFaceLandmarks: true,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5
    });

    holistic.onResults(onResults);

    if (typeof webcamRef.current !== "undefined" && webcamRef.current !== null) {
      camera = new cam.Camera(webcamRef.current.video, {
        onFrame: async () => {
          frameCounter++;
          await holistic.send({ image: webcamRef.current.video })
          console.log("frames ", frameCounter);
        },
        width: 1280,
        height: 900
      });
      camera.start();
    }
  });

  const [lastPrediction, setLastPrediction] = useState(null);

function getArrayShape(array) {
    return [array.length, array[0].length];
}

function frameCount(results, ctx) {
    if (framesData.length === 30) {
      // Process the collected 30 frames data
      // console.log(getArrayShape(framesData));
      processFramesData(framesData, ctx);
      // Reset the framesData array
      // console.log(framesData);
      framesData.length = 0;
      // console.log(framesData);

    }
  }

async function processFramesData(framesData, ctx) {
  const tensor = tf.tensor(framesData);
  const expanded = tensor.expandDims(0);
  // console.log(expanded.shape);
  const scores = net.predict(expanded);
  const label = await makePrediction(scores, 0.8, 1280, 900, ctx)
  setLastPrediction(label);
  tf.dispose(tensor)
  tf.dispose(expanded)
  tf.dispose(scores)
}

  return (
    <div>Last prediction: {lastPrediction}
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
          width: 1280,
          height: 900
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
          width: 1280,
          height: 900
        }}
        id="myCanvas"
      />
    </div>
    </div>
    

  );
}

export default App;