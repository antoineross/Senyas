import { NextResponse } from 'next/server';
import { Storage } from '@google-cloud/storage';

export async function GET() {
  try {
    const encodedCredentials = process.env.GOOGLE_CLOUD_CREDENTIALS;
    if (!encodedCredentials) {
      throw new Error('GOOGLE_CLOUD_CREDENTIALS is not set');
    }

    // Decode the Base64 encoded credentials
    const credentialsString = Buffer.from(encodedCredentials, 'base64').toString('utf-8');
    const credentials = JSON.parse(credentialsString);
    
    // Ensure the private key is properly formatted
    if (credentials.private_key) {
      credentials.private_key = credentials.private_key.replace(/\\n/g, '\n');
    }

    const storage = new Storage({
      credentials: credentials,
      projectId: credentials.project_id,
    });

    const bucketName = 'senyas';
    const fileName = 'model.json';
    
    const bucket = storage.bucket(bucketName);
    const file = bucket.file(fileName);

    const [fileContents] = await file.download();
    const modelJson = JSON.parse(fileContents.toString());

    console.log("modelJson", modelJson);

    return NextResponse.json(modelJson);
  } catch (error) {
    console.error('Error loading model:', error);
    return NextResponse.json({ error: 'Failed to load model' }, { status: 500 });
  }
}