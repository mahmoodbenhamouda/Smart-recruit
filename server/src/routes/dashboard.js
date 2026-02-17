const express = require('express');
const { authenticate, authorize } = require('../middleware/auth');
const Job = require('../models/Job');
const Application = require('../models/Application');

const router = express.Router();

/**
 * GET /api/dashboard/recruiter/stats
 * Get recruiter dashboard statistics
 */
router.get('/recruiter/stats', authenticate, authorize('recruiter'), async (req, res) => {
  try {
    const recruiterId = req.user._id;

    // Get all jobs posted by this recruiter
    const jobs = await Job.find({ recruiter: recruiterId });
    const jobIds = jobs.map(j => j._id);

    // Get all applications for these jobs
    const applications = await Application.find({ job: { $in: jobIds } })
      .populate('job', 'title')
      .populate('candidate', 'name email');

    // Calculate statistics
    const stats = {
      totalJobs: jobs.length,
      totalApplications: applications.length,
      applicationsByStatus: {
        submitted: applications.filter(a => a.status === 'submitted').length,
        reviewed: applications.filter(a => a.status === 'reviewed').length,
        accepted: applications.filter(a => a.status === 'accepted').length,
        rejected: applications.filter(a => a.status === 'rejected').length
      },
      averageAtsScore: 0,
      topCandidates: []
    };

    // Calculate average ATS score
    const withAts = applications.filter(a => a.atsReport?.overallMatch);
    if (withAts.length > 0) {
      stats.averageAtsScore = withAts.reduce((sum, a) => sum + a.atsReport.overallMatch, 0) / withAts.length;
    }

    // Get top candidates (by ATS score)
    stats.topCandidates = applications
      .filter(a => a.atsReport?.overallMatch)
      .sort((a, b) => (b.atsReport.overallMatch || 0) - (a.atsReport.overallMatch || 0))
      .slice(0, 10)
      .map(app => ({
        id: app._id,
        candidateName: app.candidate.name,
        candidateEmail: app.candidate.email,
        jobTitle: app.job.title,
        atsScore: app.atsReport.overallMatch,
        matchLevel: app.atsReport.matchLevel,
        predictedRole: app.atsReport.jobPrediction?.predictedRole,
        confidence: app.atsReport.jobPrediction?.confidence,
        status: app.status,
        appliedAt: app.createdAt
      }));

    res.json(stats);
  } catch (error) {
    console.error('Dashboard stats error:', error);
    res.status(500).json({ message: 'Failed to load dashboard statistics' });
  }
});

/**
 * GET /api/dashboard/recruiter/jobs
 * Get all jobs with application counts
 */
router.get('/recruiter/jobs', authenticate, authorize('recruiter'), async (req, res) => {
  try {
    const jobs = await Job.find({ recruiter: req.user._id }).sort({ createdAt: -1 });

    const jobsWithStats = await Promise.all(
      jobs.map(async (job) => {
        const applications = await Application.find({ job: job._id });
        const avgAts = applications
          .filter(a => a.atsReport?.overallMatch)
          .reduce((sum, a, _, arr) => sum + a.atsReport.overallMatch / arr.length, 0);

        return {
          id: job._id,
          title: job.title,
          location: job.location,
          employmentType: job.employmentType,
          totalApplications: applications.length,
          newApplications: applications.filter(a => a.status === 'submitted').length,
          averageAtsScore: avgAts || 0,
          createdAt: job.createdAt
        };
      })
    );

    res.json(jobsWithStats);
  } catch (error) {
    console.error('Jobs list error:', error);
    res.status(500).json({ message: 'Failed to load jobs' });
  }
});

/**
 * GET /api/dashboard/candidate/insights
 * Get candidate dashboard insights
 */
router.get('/candidate/insights', authenticate, authorize('candidate'), async (req, res) => {
  try {
    const applications = await Application.find({ candidate: req.user._id })
      .populate('job', 'title location employmentType')
      .sort({ createdAt: -1 });

    const insights = {
      totalApplications: applications.length,
      applicationsByStatus: {
        submitted: applications.filter(a => a.status === 'submitted').length,
        reviewed: applications.filter(a => a.status === 'reviewed').length,
        accepted: applications.filter(a => a.status === 'accepted').length,
        rejected: applications.filter(a => a.status === 'rejected').length
      },
      averageAtsScore: 0,
      topMatchedRole: null,
      commonMissingSkills: [],
      recentApplications: []
    };

    // Calculate average ATS score
    const withAts = applications.filter(a => a.atsReport?.overallMatch);
    if (withAts.length > 0) {
      insights.averageAtsScore = withAts.reduce((sum, a) => sum + a.atsReport.overallMatch, 0) / withAts.length;
    }

    // Find most common predicted role
    const roleCounts = {};
    applications.forEach(app => {
      const role = app.atsReport?.jobPrediction?.predictedRole;
      if (role) {
        roleCounts[role] = (roleCounts[role] || 0) + 1;
      }
    });
    if (Object.keys(roleCounts).length > 0) {
      insights.topMatchedRole = Object.entries(roleCounts).sort((a, b) => b[1] - a[1])[0][0];
    }

    // Find common missing skills across applications
    const missingSkillsMap = {};
    applications.forEach(app => {
      const missingSkills = app.atsReport?.xai?.missingSkills?.topMissingSkills || [];
      missingSkills.forEach(skill => {
        const key = skill.skill;
        if (!missingSkillsMap[key]) {
          missingSkillsMap[key] = {
            skill: key,
            count: 0,
            avgImpact: 0,
            totalImpact: 0
          };
        }
        missingSkillsMap[key].count++;
        missingSkillsMap[key].totalImpact += skill.impactPercentage || 0;
        missingSkillsMap[key].avgImpact = missingSkillsMap[key].totalImpact / missingSkillsMap[key].count;
      });
    });
    
    insights.commonMissingSkills = Object.values(missingSkillsMap)
      .sort((a, b) => b.count - a.count)
      .slice(0, 10)
      .map(s => ({
        skill: s.skill,
        frequency: s.count,
        averageImpact: s.avgImpact.toFixed(2)
      }));

    // Recent applications with key data
    insights.recentApplications = applications.slice(0, 5).map(app => ({
      id: app._id,
      jobTitle: app.job.title,
      jobLocation: app.job.location,
      atsScore: app.atsReport?.overallMatch || 0,
      predictedRole: app.atsReport?.jobPrediction?.predictedRole,
      status: app.status,
      appliedAt: app.createdAt
    }));

    res.json(insights);
  } catch (error) {
    console.error('Candidate insights error:', error);
    res.status(500).json({ message: 'Failed to load insights' });
  }
});

module.exports = router;
