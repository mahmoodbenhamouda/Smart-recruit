require('dotenv').config({ path: process.env.CONFIG_PATH || '.env' });
const http = require('http');
const path = require('path');
const fs = require('fs');

const connectDb = require('./config/db');
const app = require('./app');

const PORT = process.env.PORT || 5000;

async function bootstrap () {
  try {
    await connectDb(process.env.MONGODB_URI);

    const uploadsDir = path.join(process.cwd(), 'server', 'uploads');
    if (!fs.existsSync(uploadsDir)) {
      fs.mkdirSync(uploadsDir, { recursive: true });
    }

    const server = http.createServer(app);
    server.listen(PORT, () => {
      console.log(`🚀 API server running on port ${PORT}`);
    });
  } catch (error) {
    console.error('Failed to start server:', error);
    process.exit(1);
  }
}

bootstrap();

