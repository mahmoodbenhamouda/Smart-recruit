const express = require('express');
const { body, validationResult } = require('express-validator');
const Job = require('../models/Job');
const { authenticate, authorize } = require('../middleware/auth');

const router = express.Router();

router.get('/', async (req, res) => {
  const jobs = await Job.find()
    .sort({ createdAt: -1 })
    .populate('recruiter', 'name organization');
  res.json(jobs);
});

router.post('/', authenticate, authorize('recruiter'), [
  body('title').notEmpty().withMessage('Title required'),
  body('description').notEmpty().withMessage('Description required'),
  body('skills').optional().isArray().withMessage('Skills must be an array of strings')
], async (req, res) => {
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    return res.status(400).json({ errors: errors.array() });
  }

  const { title, location, employmentType, description, skills } = req.body;

  const job = await Job.create({
    title,
    location,
    employmentType,
    description,
    skills,
    recruiter: req.user._id
  });

  res.status(201).json(job);
});

router.get('/recruiter/me', authenticate, authorize('recruiter'), async (req, res) => {
  const jobs = await Job.find({ recruiter: req.user._id }).sort({ createdAt: -1 });
  res.json(jobs);
});

router.get('/:id', async (req, res) => {
  const job = await Job.findById(req.params.id)
    .populate('recruiter', 'name organization');
  if (!job) {
    return res.status(404).json({ message: 'Job not found' });
  }
  res.json(job);
});

module.exports = router;

