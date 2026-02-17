const mongoose = require('mongoose');

async function connectDb (uri) {
  if (!uri) {
    throw new Error('Missing MONGODB_URI environment variable');
  }

  mongoose.set('strictQuery', true);

  await mongoose.connect(uri, {
    dbName: process.env.MONGODB_DB_NAME || 'talent_bridge',
    serverSelectionTimeoutMS: 5000
  });

  console.log('✅ Connected to MongoDB Atlas');
}

module.exports = connectDb;

