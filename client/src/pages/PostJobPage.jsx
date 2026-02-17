import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import api from '../services/api.js';
import { Card, Button, Input, Textarea, Select, Alert } from '../components/UI.jsx';

export default function PostJobPage () {
  const navigate = useNavigate();
  const [form, setForm] = useState({
    title: '',
    location: '',
    employmentType: 'Full-time',
    description: '',
    skills: ''
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleChange = (event) => {
    setForm((prev) => ({ ...prev, [event.target.name]: event.target.value }));
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const payload = {
        ...form,
        skills: form.skills.split(',').map((skill) => skill.trim()).filter(Boolean)
      };
      await api.post('/jobs', payload);
      navigate('/recruiter/dashboard');
    } catch (err) {
      setError(err.response?.data?.message || 'Failed to create job');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-4xl mx-auto px-4">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900">Post a New Job</h1>
          <p className="text-gray-600 mt-2">Create a job posting and find the best candidates</p>
        </div>

        {/* Form Card */}
        <Card>
          {error && <Alert type="error" onClose={() => setError(null)}>{error}</Alert>}

          <form onSubmit={handleSubmit} className="space-y-6">
            <Input
              type="text"
              id="title"
              name="title"
              label="Job Title"
              placeholder="e.g. Senior React Developer"
              value={form.title}
              onChange={handleChange}
              required
            />

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <Input
                type="text"
                id="location"
                name="location"
                label="Location"
                placeholder="Remote / City, Country"
                value={form.location}
                onChange={handleChange}
              />

              <Select
                id="employmentType"
                name="employmentType"
                label="Employment Type"
                value={form.employmentType}
                onChange={handleChange}
                required
              >
                <option value="Full-time">Full-time</option>
                <option value="Part-time">Part-time</option>
                <option value="Contract">Contract</option>
                <option value="Internship">Internship</option>
                <option value="Freelance">Freelance</option>
              </Select>
            </div>

            <Textarea
              id="description"
              name="description"
              label="Job Description"
              placeholder="Describe the role, responsibilities, requirements, and what makes this opportunity great..."
              rows={10}
              value={form.description}
              onChange={handleChange}
              required
            />

            <Input
              type="text"
              id="skills"
              name="skills"
              label="Required Skills (comma separated)"
              placeholder="e.g. React, Node.js, AWS, TypeScript"
              value={form.skills}
              onChange={handleChange}
              helpText="Enter skills separated by commas. These will be used for candidate matching."
            />

            <div className="flex gap-4 pt-4">
              <Button
                type="submit"
                variant="primary"
                disabled={loading}
                className="flex-1"
              >
                {loading ? 'Publishing...' : '🚀 Publish Job'}
              </Button>
              <Button
                type="button"
                variant="secondary"
                onClick={() => navigate('/recruiter/dashboard')}
                disabled={loading}
              >
                Cancel
              </Button>
            </div>
          </form>
        </Card>

        {/* Tips Card */}
        <Card className="mt-6 bg-blue-50 border-blue-200">
          <h3 className="text-lg font-semibold text-blue-900 mb-3">💡 Tips for a Great Job Post</h3>
          <ul className="space-y-2 text-sm text-blue-800">
            <li>✓ Use a clear, specific job title</li>
            <li>✓ Include detailed responsibilities and requirements</li>
            <li>✓ List specific skills for better candidate matching</li>
            <li>✓ Mention what makes your company/role unique</li>
            <li>✓ Be transparent about remote work options</li>
          </ul>
        </Card>
      </div>
    </div>
  );
}

