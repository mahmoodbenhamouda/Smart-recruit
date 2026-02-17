import { useEffect, useState } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import api from '../services/api.js';
import { useAuth } from '../state/AuthContext.jsx';
import { Card, Button, Badge, Alert, Spinner } from '../components/UI.jsx';

export default function JobDetailPage () {
  const { id } = useParams();
  const { user } = useAuth();
  const navigate = useNavigate();
  const [job, setJob] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [resumeFile, setResumeFile] = useState(null);
  const [applyMessage, setApplyMessage] = useState(null);
  const [applyLoading, setApplyLoading] = useState(false);
  const [applySuccess, setApplySuccess] = useState(false);

  useEffect(() => {
    setLoading(true);
    api.get(`/jobs/${id}`)
      .then((res) => setJob(res.data))
      .catch((err) => setError(err.response?.data?.message || 'Failed to load job'))
      .finally(() => setLoading(false));
  }, [id]);

  const handleApply = async (event) => {
    event.preventDefault();
    if (!resumeFile) {
      setApplyMessage('Please select a PDF resume to upload.');
      setApplySuccess(false);
      return;
    }
    setApplyLoading(true);
    setApplyMessage(null);
    const formData = new FormData();
    formData.append('resume', resumeFile);
    try {
      await api.post(`/applications/${id}`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setApplyMessage('Application submitted successfully! Check "My Applications" to view your ATS score and feedback.');
      setApplySuccess(true);
      setResumeFile(null);
      // Reset file input
      const fileInput = document.getElementById('resume');
      if (fileInput) fileInput.value = '';
    } catch (err) {
      setApplyMessage(err.response?.data?.message || 'Failed to submit application');
      setApplySuccess(false);
    } finally {
      setApplyLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <Spinner size="lg" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50 px-4">
        <Card className="max-w-md text-center">
          <p className="text-danger-600 text-lg mb-4">❌ {error}</p>
          <Button variant="primary" onClick={() => navigate('/')}>
            Back to Jobs
          </Button>
        </Card>
      </div>
    );
  }

  if (!job) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50 px-4">
        <Card className="max-w-md text-center">
          <p className="text-gray-600 text-lg mb-4">Job not found.</p>
          <Button variant="primary" onClick={() => navigate('/')}>
            Back to Jobs
          </Button>
        </Card>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-7xl mx-auto px-4">
        {/* Back Button */}
        <Button 
          variant="secondary" 
          className="mb-6"
          onClick={() => navigate('/')}
        >
          ← Back to Jobs
        </Button>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Job Details - Left Side (2/3) */}
          <div className="lg:col-span-2">
            <Card>
              {/* Header */}
              <div className="border-b border-gray-200 pb-6 mb-6">
                <div className="flex items-start justify-between mb-4">
                  <h1 className="text-3xl font-bold text-gray-900 flex-1 pr-4">
                    {job.title}
                  </h1>
                  <Badge variant="primary" className="text-sm px-3 py-1">
                    {job.employmentType || 'Full-time'}
                  </Badge>
                </div>

                <div className="flex flex-wrap gap-4 text-gray-600">
                  {job.location && (
                    <div className="flex items-center">
                      <span className="mr-2">📍</span>
                      <span>{job.location}</span>
                    </div>
                  )}
                  <div className="flex items-center">
                    <span className="mr-2">👤</span>
                    <span>Posted by {job.recruiter?.name || 'Recruiter'}</span>
                  </div>
                  <div className="flex items-center">
                    <span className="mr-2">📅</span>
                    <span>{new Date(job.createdAt).toLocaleDateString()}</span>
                  </div>
                </div>
              </div>

              {/* Description */}
              <div className="mb-6">
                <h2 className="text-xl font-semibold text-gray-900 mb-3">
                  Job Description
                </h2>
                <p className="text-gray-700 whitespace-pre-line leading-relaxed">
                  {job.description}
                </p>
              </div>

              {/* Skills */}
              {job.skills?.length > 0 && (
                <div>
                  <h2 className="text-xl font-semibold text-gray-900 mb-3">
                    Required Skills
                  </h2>
                  <div className="flex flex-wrap gap-2">
                    {job.skills.map((skill, idx) => (
                      <Badge key={idx} variant="info" className="text-sm px-3 py-1">
                        {skill}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}
            </Card>
          </div>

          {/* Application Form - Right Side (1/3) */}
          <div className="lg:col-span-1">
            <Card className="sticky top-6">
              <h2 className="text-2xl font-bold text-gray-900 mb-4">
                Apply Now
              </h2>

              {/* Not logged in */}
              {!user && (
                <div className="text-center py-4">
                  <p className="text-gray-600 mb-4">
                    Sign in to apply for this position
                  </p>
                  <div className="space-y-3">
                    <Link to="/login" className="block">
                      <Button variant="primary" className="w-full">
                        Sign In
                      </Button>
                    </Link>
                    <Link to="/register" className="block">
                      <Button variant="outline" className="w-full">
                        Create Account
                      </Button>
                    </Link>
                  </div>
                </div>
              )}

              {/* Recruiter cannot apply */}
              {user?.role === 'recruiter' && (
                <Alert type="warning">
                  Recruiters cannot apply to jobs. This feature is for candidates only.
                </Alert>
              )}

              {/* Candidate application form */}
              {user?.role === 'candidate' && (
                <>
                  {applySuccess ? (
                    <div className="text-center py-6">
                      <div className="text-6xl mb-4">✅</div>
                      <Alert type="success" className="mb-4">
                        {applyMessage}
                      </Alert>
                      <Link to="/candidate/applications">
                        <Button variant="primary" className="w-full">
                          View My Applications
                        </Button>
                      </Link>
                    </div>
                  ) : (
                    <form onSubmit={handleApply} className="space-y-4">
                      {/* File Upload */}
                      <div>
                        <label 
                          htmlFor="resume" 
                          className="block text-sm font-medium text-gray-700 mb-2"
                        >
                          Upload Your Resume (PDF) *
                        </label>
                        <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-primary-500 transition-colors">
                          <input
                            id="resume"
                            type="file"
                            accept="application/pdf"
                            onChange={(event) => setResumeFile(event.target.files[0] || null)}
                            className="hidden"
                          />
                          <label 
                            htmlFor="resume" 
                            className="cursor-pointer"
                          >
                            <div className="text-4xl mb-2">📄</div>
                            {resumeFile ? (
                              <p className="text-sm font-medium text-primary-600">
                                {resumeFile.name}
                              </p>
                            ) : (
                              <>
                                <p className="text-sm font-medium text-gray-700 mb-1">
                                  Click to upload resume
                                </p>
                                <p className="text-xs text-gray-500">
                                  PDF files only
                                </p>
                              </>
                            )}
                          </label>
                        </div>
                        {resumeFile && (
                          <p className="text-xs text-gray-600 mt-2">
                            Size: {(resumeFile.size / 1024).toFixed(1)} KB
                          </p>
                        )}
                      </div>

                      {/* Message */}
                      {applyMessage && !applySuccess && (
                        <Alert type="error" onClose={() => setApplyMessage(null)}>
                          {applyMessage}
                        </Alert>
                      )}

                      {/* Submit Button */}
                      <Button
                        type="submit"
                        variant="primary"
                        className="w-full"
                        disabled={applyLoading || !resumeFile}
                      >
                        {applyLoading ? 'Analyzing Resume...' : '🚀 Submit Application'}
                      </Button>

                      {/* Info Box */}
                      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                        <p className="text-xs text-blue-900">
                          <strong>💡 What happens next:</strong>
                        </p>
                        <ul className="text-xs text-blue-800 mt-2 space-y-1">
                          <li>✓ AI analyzes your resume</li>
                          <li>✓ Get instant ATS score</li>
                          <li>✓ Receive personalized feedback</li>
                          <li>✓ See missing skills & improvements</li>
                        </ul>
                      </div>
                    </form>
                  )}
                </>
              )}
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}

