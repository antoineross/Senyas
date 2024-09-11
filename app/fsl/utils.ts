import * as tf from '@tensorflow/tfjs';

// Define our labelmap
// array(['ako',  'bakit', 'F', 'hi', 'hindi', 'ikaw',  'kamusta', 'L', 'maganda', 'magandang umaga', 'N', 'O', 'oo', 'P', 'salamat'])
export const labelMap = {
  1:{name:'Ako', color:'red'},
  2:{name:'Bakit', color:'yellow'},
  3:{name:'Hi', color:'lime'},
  4:{name:'Hindi', color:'blue'},
  5:{name:'Ikaw', color:'purple'},
  6:{name:'Kamusta', color:'red'},
  7:{name:'Maganda', color:'yellow'},
  8:{name:'Magandang Umaga', color:'lime'},
  9:{name:'Oo', color:'blue'},
  10:{name:'Salamat', color:'purple'},
  11:{name:'F', color:'red'},
  12:{name:'L', color:'yellow'},
  13:{name:'P', color:'lime'},
  14:{name:'N', color:'blue'},
  15:{name:'O', color:'purple'},
  16:{name:'None', color:'red'},
}

export async function makePrediction(
  scores: tf.Tensor,
  threshold: number,
  videoWidth: number,
  videoHeight: number
): Promise<string> {
  console.log('Scores tensor shape:', scores.shape);
  console.log('Scores tensor rank:', scores.rank);
  
  const probabilities = await scores.data();
  console.log('Probabilities:', Array.from(probabilities));
  
  const maxProbIndex = probabilities.indexOf(Math.max(...Array.from(probabilities)));
  const maxProb = probabilities[maxProbIndex];
  
  console.log('Max probability:', maxProb);
  console.log('Max probability index:', maxProbIndex);
  console.log('Threshold:', threshold);

  if (maxProb > threshold) {
    const prediction = labelMap[maxProbIndex + 1]?.name || 'Unknown';
    console.log('Prediction:', prediction);
    return prediction;
  } else {
    console.log('No confident prediction');
    return 'No confident prediction';
  }
}