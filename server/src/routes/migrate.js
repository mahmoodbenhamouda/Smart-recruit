const express = require('express');
const path = require('path');
const Application = require('../models/Application');
const Job = require('../models/Job');
const { runAtsPipeline } = require('../services/atsService');
const { authenticate, authorize } = require('../middleware/auth');

const router = express.Router();

/**
 * Re-analyze all applications that don't have ATS data
 * Only accessible by admin/recruiter for now
 */
router.post('/reanalyze-applications', authenticate, authorize('recruiter'), async (req, res) => {
  try {
    console.log('🔄 Starting batch re-analysis of applications...');
    
    // Find applications without ATS data
    const applicationsToReanalyze = await Application.find({
      $or: [
        { atsReport: { $exists: false } },
        { 'atsReport.overallMatch': { $exists: false } },
        { 'atsReport.overallMatch': 0 }
      ]
    }).populate('job').populate('candidate');

    console.log(`📊 Found ${applicationsToReanalyze.length} applications to re-analyze`);

    const results = {
      total: applicationsToReanalyze.length,
      processed: 0,
      successful: 0,
      failed: 0,
      errors: []
    };

    for (const application of applicationsToReanalyze) {
      try {
        results.processed++;
        console.log(`\n🔍 Processing application ${results.processed}/${results.total}`);
        console.log(`   Job: ${application.job.title}`);
        console.log(`   Candidate: ${application.candidate.name}`);
        
        const resumePath = path.join(process.cwd(), 'server', 'uploads', application.resumeFilename);
        const jobDescription = application.job.description;

        // Run ATS analysis
        const atsResults = await runAtsPipeline(resumePath, jobDescription);
        
        if (!atsResults || !atsResults.overall_match) {
          throw new Error('No ATS results returned');
        }

        // Parse CV data
        let parsedCV = null;
        if (atsResults.cv_sections) {
          parsedCV = {
            rawText: atsResults.cv_text || '',
            skills: atsResults.matched_skills || [],
            sections: atsResults.cv_sections || {},
            personalInfo: {},
            experience: [],
            education: [],
            certifications: [],
            projects: [],
            languages: []
          };
        }

        // Update application with ATS data
        application.parsedCV = parsedCV;
        application.atsReport = {
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

        await application.save();
        
        results.successful++;
        console.log(`   ✅ Success! ATS Score: ${application.atsReport.overallMatch}%`);
        console.log(`   Role: ${application.atsReport.jobPrediction?.predictedRole}`);
        
      } catch (error) {
        results.failed++;
        const errorMsg = `Failed for ${application.candidate.name} - ${application.job.title}: ${error.message}`;
        console.error(`   ❌ ${errorMsg}`);
        results.errors.push(errorMsg);
      }
    }

    console.log('\n' + '='.repeat(80));
    console.log('📊 Re-analysis Complete!');
    console.log('='.repeat(80));
    console.log(`Total: ${results.total}`);
    console.log(`✅ Successful: ${results.successful}`);
    console.log(`❌ Failed: ${results.failed}`);
    console.log('='.repeat(80));

    res.json({
      message: 'Re-analysis complete',
      results
    });

  } catch (error) {
    console.error('Migration error:', error);
    res.status(500).json({ 
      message: 'Failed to re-analyze applications', 
      error: error.message 
    });
  }
});

/**
 * Re-analyze a single application
 */
router.post('/reanalyze-application/:applicationId', authenticate, async (req, res) => {
  try {
    const application = await Application.findById(req.params.applicationId)
      .populate('job')
      .populate('candidate');

    if (!application) {
      return res.status(404).json({ message: 'Application not found' });
    }

    // Check permission
    const isOwner = application.candidate._id.toString() === req.user._id.toString();
    const isRecruiter = req.user.role === 'recruiter' && 
      application.job.recruiter.toString() === req.user._id.toString();

    if (!isOwner && !isRecruiter && req.user.role !== 'recruiter') {
      return res.status(403).json({ message: 'Not authorized' });
    }

    console.log('🔄 Re-analyzing single application...');
    
    const resumePath = path.join(process.cwd(), 'server', 'uploads', application.resumeFilename);
    const jobDescription = application.job.description;

    const atsResults = await runAtsPipeline(resumePath, jobDescription);
    
    console.log('✅ ATS Results received:', Object.keys(atsResults || {}));
    
    if (!atsResults || !atsResults.overall_match) {
      console.error('❌ Invalid ATS results:', atsResults);
      throw new Error('No ATS results returned');
    }

    // Parse CV data with sections
    let parsedCV = null;
    if (atsResults.cv_sections) {
      parsedCV = {
        rawText: atsResults.cv_text || '',
        skills: atsResults.matched_skills || [],
        sections: atsResults.cv_sections || {},
        personalInfo: {},
        experience: [],
        education: [],
        certifications: [],
        projects: [],
        languages: []
      };
    }

    // Update application with new format
    application.parsedCV = parsedCV;
    application.atsReport = {
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
      xai: atsResults.xai
        ? {
            shap: atsResults.xai.shap,
            lime: atsResults.xai.lime,
            missingSkills: atsResults.xai.missing_skills
          }
        : undefined,
      cvSections: atsResults.cv_sections,
      raw: atsResults
    };

    await application.save();
    
    console.log(`✅ Re-analysis successful! ATS Score: ${application.atsReport.overallMatch}%`);

    res.json({
      message: 'Application re-analyzed successfully',
      application
    });

  } catch (error) {
    console.error('Re-analysis error:', error);
    res.status(500).json({ 
      message: 'Failed to re-analyze application', 
      error: error.message 
    });
  }
});

module.exports = router;
