import React from 'react'
import { createBoard } from '@wixc3/react-board';

export default createBoard({
    name: 'Senyas',
    Board: () => <div></div>,
    isSnippet: true,
    environmentProps: {
        windowWidth: 1280
    }
});
