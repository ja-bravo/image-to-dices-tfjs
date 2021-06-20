import Head from 'next/head';
import { useEffect, useRef, useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import { cutIntoGrid, predictResults, turnIntoBlackAndWhite, diceData } from '../utils/utils';

export default function Home() {
  const inputRef = useRef<HTMLInputElement>();
  const model = useRef<tf.LayersModel>();
  useEffect(() => {
    tf.loadLayersModel('/model/model.json')
      .then((m) => {
        console.log('Model loaded');
        model.current = m;
      })
      .catch((e) => {
        console.log(e);
      });
  }, []);

  const [numOfDices, setNumOfDices] = useState(64);
  const [loading, setLoading] = useState(false);

  return (
    <div className="w-full max-w-screen-xl flex flex-col justify-center m-auto mt-4">
      <Head>
        <title>Dicify an image using Tensorflow</title>
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <h1 className="text-4xl text-center text-bold">Dicify an image using Tensorflow</h1>
      <h2 className="text-xl text-center">
        Capstone project from the{' '}
        <a
          className="cursor-pointer text-green-500 underline"
          target="_blank"
          href="https://www.oreilly.com/library/view/learning-tensorflowjs/9781492090786/">
          Learning TensorFlow.js book
        </a>{' '}
        by{' '}
        <a className="cursor-pointer text-green-500 underline" target="_blank" href="https://twitter.com/GantLaborde">
          Gant Laborde
        </a>
      </h2>

      <p>Dices per row: {numOfDices}</p>
      <input
        type="range"
        id="vol"
        name="vol"
        min="32"
        max="128"
        value={numOfDices}
        onChange={(e) => setNumOfDices(Number.parseInt(e.target.value))}
      />

      <div className="w-full flex flex-row h-[75vh]  mt-8">
        <div className="flex-1 h-full flex flex-col justify-center items-center">
          <canvas id="display" className={'w-full h-full'} style={{ imageRendering: 'pixelated' }} />
          <button
            onClick={() => {
              inputRef.current!.click();
            }}
            className="mt-4 bg-white text-gray-800 font-bold rounded border-b-2 border-green-400 hover:border-green-500 hover:bg-green-400 hover:text-white shadow-md py-2 px-6 inline-flex items-center">
            <span className="mr-2">{loading ? 'Processing...' : 'Select Image'}</span>
          </button>

          <input
            ref={(r) => (inputRef.current = r!)}
            type="file"
            accept="image/*"
            className="fixed -top-1/2 w-0.5 h-0.5"
            onChange={(e) => {
              if (!model.current) {
                return alert("The Tensorflow model hasn't loaded yet");
              }
              const files = e.target.files;

              if (files) {
                setLoading(true);
                const reader = new FileReader();
                reader.onload = () => {
                  const img = new Image();
                  img.src = reader.result as any;
                  img.onload = async () => {
                    let imgTensor = await tf.browser.fromPixelsAsync(img, 1);
                    imgTensor = turnIntoBlackAndWhite(imgTensor);

                    const grid = cutIntoGrid(imgTensor, numOfDices);
                    const predictions = predictResults(model.current!, grid);

                    tf.tidy(() => {
                      const diceTensors = diceData.map((dt) => tf.tensor(dt));
                      // Get the top predictions indices
                      // @ts-ignore
                      const { indices } = tf.topk(predictions);
                      const answerIndices = indices.dataSync();

                      const tColumns = [];
                      for (let y = 0; y < numOfDices; y++) {
                        const tRow = [];
                        for (let x = 0; x < numOfDices; x++) {
                          // Multiple by the number of dices to simulate a 2D array.
                          const curIndex = y * numOfDices + x;
                          tRow.push(diceTensors[answerIndices[curIndex]]);
                        }
                        const oneRow = tf.concat(tRow, 1);
                        tColumns.push(oneRow);
                      }
                      const diceConstruct: tf.Tensor<tf.Rank.R3> = tf.concat(tColumns) as tf.Tensor<tf.Rank.R3>;
                      // Print the reconstruction to the canvas
                      const canvas = document.getElementById('display') as HTMLCanvasElement;
                      tf.browser.toPixels(diceConstruct, canvas);
                      tf.dispose(imgTensor);
                      setLoading(false);
                    });
                  };
                };
                reader.readAsDataURL(files[0]);
              }

              inputRef.current!.value = '';
              e.target.value = null as any;
            }}
          />
        </div>
      </div>
    </div>
  );
}
