const { spawn } = require('child_process');
const fsPromises = require('fs/promises');
const fs = require('fs');
const os = require('os');
const path = require('path');
const axios = require('axios');
const FormData = require('form-data');

async function runAtsPipeline (resumePath, jobDescription) {
  const serviceUrl = process.env.ATS_SERVICE_URL;
  if (serviceUrl) {
    return runPipelineViaHttp(resumePath, jobDescription, serviceUrl);
  }

  return runPipelineLocally(resumePath, jobDescription);
}

async function runPipelineViaHttp (resumePath, jobDescription, serviceUrl, targetRole = null) {
  const form = new FormData();
  form.append('resume', fs.createReadStream(resumePath), {
    filename: path.basename(resumePath)
  });
  form.append('job_description', jobDescription);
  
  if (targetRole) {
    form.append('target_role', targetRole);
  }

  const url = new URL('/analyze', serviceUrl).toString();

  try {
    const response = await axios.post(url, form, {
      headers: form.getHeaders(),
      timeout: Number(process.env.ATS_SERVICE_TIMEOUT || 300000), // Increased to 5 minutes
      maxContentLength: Infinity,
      maxBodyLength: Infinity
    });
    return response.data;
  } catch (error) {
    const message = error.response?.data?.detail ||
      error.response?.data?.message ||
      error.message;
    throw new Error(`ATS service request failed: ${message}`);
  }
}

async function runPipelineLocally (resumePath, jobDescription) {
  const scriptPath = path.join(process.cwd(), 'server', 'scripts', 'run_ats_pipeline.py');
  const jobDescFile = path.join(os.tmpdir(), `job-desc-${Date.now()}.txt`);

  await fsPromises.writeFile(jobDescFile, jobDescription, { encoding: 'utf8' });

  return new Promise((resolve, reject) => {
    const pythonPath = process.env.PYTHON_PATH || 'python';
    const args = [scriptPath, resumePath, jobDescFile];
    const proc = spawn(pythonPath, args, {
      env: {
        ...process.env,
        PYTHONIOENCODING: 'utf-8'
      }
    });

    let stdout = '';
    let stderr = '';

    proc.stdout.on('data', (data) => {
      stdout += data.toString();
    });

    proc.stderr.on('data', (data) => {
      stderr += data.toString();
    });

    proc.on('close', async (code) => {
      await fsPromises.unlink(jobDescFile).catch(() => {});
      if (code !== 0) {
        return reject(new Error(stderr || `ATS pipeline exited with code ${code}`));
      }
      try {
        const parsed = JSON.parse(stdout);
        resolve(parsed);
      } catch (error) {
        reject(new Error(`Failed to parse ATS output: ${error.message}; stdout: ${stdout}; stderr: ${stderr}`));
      }
    });
  });
}

module.exports = {
  runAtsPipeline
};

