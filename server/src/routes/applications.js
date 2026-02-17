const express = require('express');
const multer = require('multer');
const path = require('path');
const { randomUUID } = require('crypto');
const fs = require('fs/promises');

const { authenticate, authorize } = require('../middleware/auth');
const Job = require('../models/Job');
const Application = require('../models/Application');
const { runAtsPipeline } = require('../services/atsService');

const router = express.Router();

const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, path.join(process.cwd(), 'server', 'uploads'));
  },
  filename: (req, file, cb) => {
    const ext = path.extname(file.originalname) || '.pdf';
    const unique = randomUUID().replace(/-/g, '').slice(0, 10);
    cb(null, `${Date.now()}-${unique}${ext}`);
  }
});

const upload = multer({
  storage,
  limits: { fileSize: 10 * 1024 * 1024 },
  fileFilter: (req, file, cb) => {
    if (file.mimetype !== 'application/pdf') {
      return cb(new Error('Only PDF resumes are supported'));
    }
    cb(null, true);
  }
});

router.post('/:jobId', authenticate, authorize('candidate'), upload.single('resume'), async (req, res) => {
  try {
    const job = await Job.findById(req.params.jobId);
    if (!job) {
      if (req.file) {
        await fs.unlink(req.file.path).catch(() => {});
      }
      return res.status(404).json({ message: 'Job not found' });
    }

    const existing = await Application.findOne({ job: job._id, candidate: req.user._id });
    if (existing) {
      if (req.file) {
        await fs.unlink(req.file.path).catch(() => {});
      }
      return res.status(400).json({ message: 'You already applied to this job' });
    }

    if (!req.file) {
      return res.status(400).json({ message: 'Resume file is required' });
    }

    const jobDescription = job.description;
    console.log('📄 Processing resume:', req.file.filename);
    console.log('🎯 Job description length:', jobDescription.length);

    let atsResults = null;
    let parsedCV = null;
    
    try {
      console.log('🚀 Calling ATS service...');
      atsResults = await runAtsPipeline(req.file.path, jobDescription);
      console.log('✅ ATS analysis completed successfully');
      console.log('📊 Results preview:', {
        hasResults: !!atsResults,
        hasSimilarity: !!atsResults?.similarity_scores,
        overallMatch: atsResults?.similarity_scores?.overall_percentage,
        hasJobPrediction: !!atsResults?.job_prediction,
        predictedRole: atsResults?.job_prediction?.predicted_role
      });

      // Parse CV sections from results
      if (atsResults?.cv_text) {
        parsedCV = {
          rawText: atsResults.cv_text,
          skills: atsResults.similarity_scores?.matched_skills || [],
          personalInfo: {},
          experience: [],
          education: [],
          certifications: [],
          projects: [],
          languages: []
        };

        // Extract sections if available
        if (atsResults.cv_sections) {
          const sections = atsResults.cv_sections;
          
          // Map sections to structured data
          if (sections.summary) parsedCV.summary = sections.summary;
          if (sections.skills) parsedCV.skills = sections.skills.split(',').map(s => s.trim());
          if (sections.experience) {
            // Simple parsing - can be enhanced
            parsedCV.experience = sections.experience.split('\n\n').map(exp => ({
              description: exp.trim()
            }));
          }
          if (sections.education) {
            parsedCV.education = sections.education.split('\n\n').map(edu => ({
              details: edu.trim()
            }));
          }
        }
      }
    } catch (error) {
      console.error('⚠️ ATS analysis error:', error.message);
      console.error('Stack:', error.stack);
      // Don't fail the application, just log the error
    }

    const applicationData = {
      job: job._id,
      candidate: req.user._id,
      resumeFilename: path.basename(req.file.path),
      resumeMimeType: req.file.mimetype,
      parsedCV: parsedCV || undefined
    };

    if (atsResults && atsResults.overall_match !== undefined) {
      applicationData.atsReport = {
        overallMatch: atsResults.overall_match || 0,
        matchLevel: atsResults.match_level || 'Unknown',
        skillsMatchRate: atsResults.detailed_scores?.skills_match_rate || 0,
        matchedSkills: atsResults.matched_skills || [],
        missingSkills: atsResults.missing_skills || [],
        jobPrediction: atsResults.job_prediction
          ? {
              predictedRole: atsResults.job_prediction.predicted_role,
              confidence: atsResults.job_prediction.confidence || 0,
              topPredictions: (atsResults.job_prediction.top_predictions || []).map((entry) => {
                if (Array.isArray(entry)) {
                  const [role, probability] = entry;
                  return { role, probability };
                }
                return {
                  role: entry.role || entry[0],
                  probability: entry.probability || entry[1]
                };
              })
            }
          : undefined,
        xai: {
          shap: atsResults.shap_explanation || null,
          lime: atsResults.lime_explanation || null,
          missingSkills: atsResults.missing_skills_analysis || null
        },
        cvSections: atsResults.cv_sections || {},
        raw: atsResults
      };

      console.log('💾 Saving application with ATS data:', {
        overallMatch: applicationData.atsReport.overallMatch,
        predictedRole: applicationData.atsReport.jobPrediction?.predictedRole,
        matchedSkills: applicationData.atsReport.matchedSkills.length,
        missingSkills: applicationData.atsReport.missingSkills.length
      });
    } else {
      console.log('⚠️ No ATS results available, saving application without analysis');
    }

    const application = await Application.create(applicationData);
    console.log('✅ Application saved successfully with ID:', application._id);

    res.status(201).json(application);
  } catch (error) {
    console.error('❌ Application submission error:', error);
    res.status(500).json({ message: error.message || 'Failed to apply to job' });
  }
});

router.get('/me', authenticate, authorize('candidate'), async (req, res) => {
  const applications = await Application.find({ candidate: req.user._id })
    .sort({ createdAt: -1 })
    .populate('job');
  res.json(applications);
});

router.get('/job/:jobId', authenticate, authorize('recruiter'), async (req, res) => {
  const job = await Job.findById(req.params.jobId);
  if (!job) {
    return res.status(404).json({ message: 'Job not found' });
  }
  if (job.recruiter.toString() !== req.user._id.toString()) {
    return res.status(403).json({ message: 'Not allowed to view applications for this job' });
  }

  const applications = await Application.find({ job: job._id })
    .sort({ createdAt: -1 })
    .populate('candidate', 'name email');
  res.json(applications);
});

router.patch('/:applicationId/status', authenticate, authorize('recruiter'), async (req, res) => {
  const { status } = req.body;
  if (!['submitted', 'reviewed', 'rejected', 'accepted'].includes(status)) {
    return res.status(400).json({ message: 'Invalid status' });
  }

  const application = await Application.findById(req.params.applicationId).populate('job');
  if (!application) {
    return res.status(404).json({ message: 'Application not found' });
  }

  if (application.job.recruiter.toString() !== req.user._id.toString()) {
    return res.status(403).json({ message: 'Not allowed to update this application' });
  }

  application.status = status;
  await application.save();

  res.json(application);
});

router.get('/:applicationId/resume', authenticate, async (req, res) => {
  const application = await Application.findById(req.params.applicationId).populate('job');
  if (!application) {
    return res.status(404).json({ message: 'Application not found' });
  }

  const isCandidate = application.candidate.toString() === req.user._id.toString();
  const isRecruiter = application.job.recruiter.toString() === req.user._id.toString();

  if (!isCandidate && (!isRecruiter || req.user.role !== 'recruiter')) {
    return res.status(403).json({ message: 'Not allowed to access this resume' });
  }

  const resumePath = path.join(process.cwd(), 'server', 'uploads', application.resumeFilename);
  res.sendFile(resumePath, (err) => {
    if (err) {
      console.error('Failed to send resume:', err);
      res.status(err.status || 500).json({ message: 'Failed to download resume' });
    }
  });
});

router.use((err, req, res, next) => {
  if (err instanceof multer.MulterError || err.message.includes('Only PDF')) {
    return res.status(400).json({ message: err.message });
  }
  next(err);
});

module.exports = router;

