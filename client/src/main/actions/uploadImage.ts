import axios from 'axios';
import { Blob, File } from 'buffer';
import { BinaryLike } from 'crypto';
import { createReadStream } from 'fs';

function streamToBuffer(stream) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    stream.on('data', (chunk: never) => chunks.push(chunk));
    stream.on('end', () => resolve(Buffer.concat(chunks)));
    stream.on('error', reject);
  });
}

export async function handleUploadImage(file: string) {
  const formData = new FormData();
  const fileStream = createReadStream(file);
  const fileData = (await streamToBuffer(fileStream)) as BinaryLike | Blob;

  const outputFile = new File([fileData], file);

  // @ts-ignore
  formData.append('file', outputFile);
  formData.append('name', 'file');

  const response = await axios.post('http://127.0.0.1:8000/api/upload', formData);

  return response.data;
}
