'use client'

import Webcam from 'react-webcam';
import { drawConnectors, drawLandmarks } from '@mediapipe/drawing_utils';
import { labelMap, makePrediction } from "./utils"; 
import * as tf from "@tensorflow/tfjs";
import React, { useState, useEffect, useRef, useCallback } from 'react';
import * as cam from '@mediapipe/camera_utils';
import * as holistics from '@mediapipe/holistic';
import { debounce } from 'lodash';

const FRAMES_PER_PREDICTION = 30;
const DEBOUNCE_DELAY = 500; // ms

function App() {
  const webcamRef = useRef<Webcam | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const netRef = useRef<tf.LayersModel | null>(null);
  const [lastPrediction, setLastPrediction] = useState<string | null>(null);
  const [averageLatency, setAverageLatency] = useState<number>(0);
  const [predictionCountState, setPredictionCount] = useState<number>(0);
  const [camHeight, setCamHeight] = useState<number>(0);
  const [camWidth, setCamWidth] = useState<number>(0);
  const [isModelLoaded, setIsModelLoaded] = useState(false);

  const framesDataRef = useRef<number[][]>([]);
  const predictionCountRef = useRef<number>(0);
  const totalLatencyRef = useRef<number>(0);

  const debouncedPrediction = useCallback(
    debounce(async (framesData, videoWidth, videoHeight) => {
      if (!netRef.current) return;

      const tensor = tf.tensor(framesData);
      const expanded = tensor.expandDims(0);
      const startTime = performance.now();

      const scores = netRef.current.predict(expanded) as tf.Tensor;
      
      const label = await makePrediction(scores, 0.05, videoWidth, videoHeight);
      const endTime = performance.now();
      const latency = endTime - startTime;
      
      totalLatencyRef.current += latency;
      predictionCountRef.current++;
      setPredictionCount(predictionCountRef.current);
      setAverageLatency(totalLatencyRef.current / predictionCountRef.current);
      setLastPrediction(label);

      tf.dispose([tensor, expanded, scores]);
    }, DEBOUNCE_DELAY),
    []
  );

  useEffect(() => {
    const loadModel = async () => {
      try {
        await tf.ready();
        const response = await fetch('/api/get_model');
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const modelJson = await response.json();
        
        // Progressive loading
        const loadedModel = await tf.loadLayersModel(tf.io.fromMemory(modelJson), {
          onProgress: (fraction) => {
            console.log(`Model loading progress: ${(fraction * 100).toFixed(2)}%`);
            // You can update a loading state here if you want to show a progress bar
          }
        });
        
        netRef.current = loadedModel;
        setIsModelLoaded(true);
        console.log('Model loaded successfully.');
      } catch (error) {
        console.error('Error loading model:', error);
      }
    };

    loadModel();
  }, []);
  const onResults = useCallback((model: any) => {
    const video = webcamRef.current?.video;
    if (video?.readyState === 4) {
      const videoWidth = video.videoWidth;
      const videoHeight = video.videoHeight;

      setCamHeight(videoHeight);
      setCamWidth(videoWidth);

      if (canvasRef.current) {
        canvasRef.current.width = videoWidth;
        canvasRef.current.height = videoHeight;

        const ctx = canvasRef.current.getContext("2d");
        if (ctx) {
          ctx.save();
          ctx.clearRect(0, 0, videoWidth, videoHeight);
          ctx.drawImage(model.image, 0, 0, videoWidth, videoHeight);
          
          // Draw all landmarks
          drawConnectors(ctx, model.poseLandmarks, holistics.POSE_CONNECTIONS, { color: '#00FF00', lineWidth: 4 });
          drawLandmarks(ctx, model.poseLandmarks, { color: '#FF0000', lineWidth: 2 });
          
          drawConnectors(ctx, model.faceLandmarks, holistics.FACEMESH_TESSELATION, { color: '#C0C0C070', lineWidth: 1 });
          drawConnectors(ctx, model.faceLandmarks, holistics.FACEMESH_RIGHT_EYE, { color: '#FF3030' });
          drawConnectors(ctx, model.faceLandmarks, holistics.FACEMESH_RIGHT_EYEBROW, { color: '#FF3030' });
          drawConnectors(ctx, model.faceLandmarks, holistics.FACEMESH_LEFT_EYE, { color: '#30FF30' });
          drawConnectors(ctx, model.faceLandmarks, holistics.FACEMESH_LEFT_EYEBROW, { color: '#30FF30' });
          drawConnectors(ctx, model.faceLandmarks, holistics.FACEMESH_FACE_OVAL, { color: '#E0E0E0' });
          drawConnectors(ctx, model.faceLandmarks, holistics.FACEMESH_LIPS, { color: '#E0E0E0' });

          drawConnectors(ctx, model.leftHandLandmarks, holistics.HAND_CONNECTIONS, { color: '#CC0000', lineWidth: 5 });
          drawLandmarks(ctx, model.leftHandLandmarks, { color: '#00FF00', lineWidth: 2 });

          drawConnectors(ctx, model.rightHandLandmarks, holistics.HAND_CONNECTIONS, { color: '#00CC00', lineWidth: 5 });
          drawLandmarks(ctx, model.rightHandLandmarks, { color: '#FF0000', lineWidth: 2 });

          const pose = model.poseLandmarks ? model.poseLandmarks.map((landmark: any) => [landmark.x, landmark.y, landmark.z, landmark.visibility]).flat() : new Array(33 * 4).fill(0);
          const lh = model.leftHandLandmarks ? model.leftHandLandmarks.map((landmark: any) => [landmark.x, landmark.y, landmark.z]).flat() : new Array(21 * 3).fill(0);
          const rh = model.rightHandLandmarks ? model.rightHandLandmarks.map((landmark: any) => [landmark.x, landmark.y, landmark.z]).flat() : new Array(21 * 3).fill(0);
          const frameData = [...pose, ...lh, ...rh];
          
          framesDataRef.current.push(frameData);

          if (framesDataRef.current.length === FRAMES_PER_PREDICTION) {
            debouncedPrediction(framesDataRef.current, videoWidth, videoHeight);
            framesDataRef.current = [];
          }

          ctx.restore();
        }
      }
    }
  }, [debouncedPrediction]);

  useEffect(() => {
    const holistic = new holistics.Holistic({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`
    });
  
    holistic.setOptions({
      modelComplexity: 1,
      smoothLandmarks: true,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5
    });
  
    holistic.onResults(onResults);
  
    if (typeof window !== 'undefined' && webcamRef.current && webcamRef.current.video) {
      const camera = new cam.Camera(webcamRef.current.video, {
        onFrame: async () => {
          if (webcamRef.current && webcamRef.current.video) {
            await holistic.send({ image: webcamRef.current.video });
          }
        },
        width: 640,
        height: 480
      });
      camera.start();
  
      return () => {
        camera.stop();
      };
    }
  }, [onResults]);

  return (
    <div className="flex flex-col items-center justify-center">
      {!isModelLoaded && <div>Loading model...</div>}
      <div className="flex justify-center items-center max-w-3xl max-h-screen relative">
        <Webcam
          ref={webcamRef}
          audio={false}
          id="img"
          style={{
            position: "relative",
            width: '100%',
            height: 'auto',
          }}
        />
        <canvas
          ref={canvasRef}
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
          }}
          id="myCanvas"
        />
      </div>
      <div className="flex flex-col items-center mt-4">
        <h6 className="font-medium text-center">{`Last prediction: ${lastPrediction}`}</h6>
        <h6 className="font-medium text-center">{`Avg. Prediction Latency: ${averageLatency.toFixed(2)} ms @ ${predictionCountState} predicts`}</h6>
      </div>
    </div>
  );
}

export default App;