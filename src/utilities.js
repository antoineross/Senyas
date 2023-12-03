// Define our labelmap
// array(['ako',  'bakit', 'F', 'hi', 'hindi', 'ikaw',  'kamusta', 'L', 'maganda', 'magandang umaga', 'N', 'O', 'oo', 'P', 'salamat'])
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

// Define a drawing function
export const makePrediction = async (scores, threshold, imgWidth, imgHeight, ctx) => {
    // Clear the previous drawings/text
    ctx.clearRect(0, 0, imgWidth, imgHeight);

    const scoresArray = await scores.array();

    const flattenedScores = [].concat(...scoresArray);
    // Find the maximum score and its index
    const maxScore = Math.max(...flattenedScores);
    const index = flattenedScores.indexOf(maxScore) + 1; // adding 1 because labelMap indices start from 1

    const label = labelMap[index]['name'];
    console.log("Prediction Label: ", label)
    // Check if the maximum score is above the threshold
    // if (maxScore > threshold) {
    //     console.log(labelMap[index]['name']);
    //     // Retrieve the corresponding label from labelMap
    //     const label = labelMap[index]['name'];
    //     const color = labelMap[index]['color'];
    
    //     // Set the text and color
    //     const text = `${label} - ${Math.round(maxScore * 100) / 100}`; // round to 2 decimal places
    
    //     // Set styling for the text
    //     ctx.fillStyle = color;
    //     ctx.font = '30px Arial';
    
    //     // Set text alignment to center and middle for both horizontal and vertical alignment
    //     ctx.textAlign = 'center';
    //     ctx.textBaseline = 'middle';
    
    //     // Write the text in the middle of the canvas
    //     ctx.fillText(text, imgWidth / 4, imgHeight / 4);
    // }   
    return label;
}
