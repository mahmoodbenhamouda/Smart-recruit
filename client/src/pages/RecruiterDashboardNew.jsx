import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import api from '../services/api.js';
import { Card, Button, Badge, Spinner, Alert, ProgressBar } from '../components/UI.jsx';

export default function RecruiterDashboard () {
  const [stats, setStats] = useState(null);
  const [jobs, setJobs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadDashboard();
  }, []);

  const loadDashboard = async () => {
    setLoading(true);
    try {
      const [statsRes, jobsRes] = await Promise.all([
        api.get('/dashboard/recruiter/stats'),
        api.get('/dashboard/recruiter/jobs')
      ]);
      setStats(statsRes.data);
      setJobs(jobsRes.data);
    } catch (err) {
      setError(err.response?.data?.message || 'Failed to load dashboard');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <Spinner size="lg" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="max-w-7xl mx-auto px-4 py-8">
        <Alert type="error">{error}</Alert>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      {/* Header */}
      <div className="flex justify-between items-center mb-8">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Recruiter Dashboard</h1>
          <p className="text-gray-600 mt-1">Manage your job postings and review candidates</p>
        </div>
        <Link to="/recruiter/jobs/new">
          <Button variant="primary" size="lg">+ Post New Job</Button>
        </Link>
      </div>

      {/* Statistics Cards */}
      {stats && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <Card>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Total Jobs</p>
                <p className="text-3xl font-bold text-gray-900 mt-1">{stats.totalJobs}</p>
              </div>
              <div className="w-12 h-12 bg-primary-100 rounded-lg flex items-center justify-center">
                <svg className="w-6 h-6 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 13.255A23.931 23.931 0 0112 15c-3.183 0-6.22-.62-9-1.745M16 6V4a2 2 0 00-2-2h-4a2 2 0 00-2 2v2m4 6h.01M5 20h14a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                </svg>
              </div>
            </div>
          </Card>

          <Card>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Applications</p>
                <p className="text-3xl font-bold text-gray-900 mt-1">{stats.totalApplications}</p>
              </div>
              <div className="w-12 h-12 bg-success-100 rounded-lg flex items-center justify-center">
                <svg className="w-6 h-6 text-success-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
              </div>
            </div>
          </Card>

          <Card>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Avg ATS Score</p>
                <p className="text-3xl font-bold text-gray-900 mt-1">{stats.averageAtsScore.toFixed(1)}%</p>
              </div>
              <div className="w-12 h-12 bg-warning-100 rounded-lg flex items-center justify-center">
                <svg className="w-6 h-6 text-warning-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
              </div>
            </div>
          </Card>

          <Card>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">New Submissions</p>
                <p className="text-3xl font-bold text-gray-900 mt-1">{stats.applicationsByStatus.submitted}</p>
              </div>
              <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center">
                <svg className="w-6 h-6 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9" />
                </svg>
              </div>
            </div>
          </Card>
        </div>
      )}

      {/* Top Candidates */}
      {stats && stats.topCandidates && stats.topCandidates.length > 0 && (
        <Card className="mb-8">
          <h2 className="text-xl font-bold text-gray-900 mb-4">🏆 Top Candidates</h2>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-gray-200">
                  <th className="text-left py-3 px-4 font-semibold text-gray-700">Candidate</th>
                  <th className="text-left py-3 px-4 font-semibold text-gray-700">Job</th>
                  <th className="text-left py-3 px-4 font-semibold text-gray-700">ATS Score</th>
                  <th className="text-left py-3 px-4 font-semibold text-gray-700">Predicted Role</th>
                  <th className="text-left py-3 px-4 font-semibold text-gray-700">Status</th>
                  <th className="text-left py-3 px-4 font-semibold text-gray-700">Applied</th>
                </tr>
              </thead>
              <tbody>
                {stats.topCandidates.map((candidate) => (
                  <tr key={candidate.id} className="border-b border-gray-100 hover:bg-gray-50">
                    <td className="py-3 px-4">
                      <div>
                        <p className="font-medium text-gray-900">{candidate.candidateName}</p>
                        <p className="text-sm text-gray-500">{candidate.candidateEmail}</p>
                      </div>
                    </td>
                    <td className="py-3 px-4 text-gray-700">{candidate.jobTitle}</td>
                    <td className="py-3 px-4">
                      <div className="flex items-center gap-2">
                        <ProgressBar 
                          value={candidate.atsScore} 
                          showLabel={false} 
                          className="w-24"
                          variant={candidate.atsScore >= 70 ? 'success' : candidate.atsScore >= 40 ? 'warning' : 'danger'}
                        />
                        <span className="text-sm font-semibold">{candidate.atsScore.toFixed(1)}%</span>
                      </div>
                    </td>
                    <td className="py-3 px-4">
                      <div>
                        <p className="text-sm font-medium text-gray-900">{candidate.predictedRole}</p>
                        <p className="text-xs text-gray-500">{(candidate.confidence * 100).toFixed(0)}% confidence</p>
                      </div>
                    </td>
                    <td className="py-3 px-4">
                      <Badge variant={
                        candidate.status === 'accepted' ? 'success' :
                        candidate.status === 'rejected' ? 'danger' :
                        candidate.status === 'reviewed' ? 'warning' : 'default'
                      }>
                        {candidate.status}
                      </Badge>
                    </td>
                    <td className="py-3 px-4 text-sm text-gray-600">
                      {new Date(candidate.appliedAt).toLocaleDateString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* Jobs List */}
      <div>
        <h2 className="text-2xl font-bold text-gray-900 mb-4">📋 Your Job Postings</h2>
        {jobs.length === 0 ? (
          <Card>
            <div className="text-center py-8">
              <p className="text-gray-500 mb-4">You haven't posted any jobs yet.</p>
              <Link to="/recruiter/jobs/new">
                <Button variant="primary">Post Your First Job</Button>
              </Link>
            </div>
          </Card>
        ) : (
          <div className="grid grid-cols-1 gap-6">
            {jobs.map((job) => (
              <Card key={job.id}>
                <div className="flex justify-between items-start">
                  <div className="flex-1">
                    <h3 className="text-xl font-bold text-gray-900 mb-2">{job.title}</h3>
                    <div className="flex flex-wrap gap-2 mb-3">
                      {job.location && (
                        <Badge variant="default">📍 {job.location}</Badge>
                      )}
                      {job.employmentType && (
                        <Badge variant="primary">💼 {job.employmentType}</Badge>
                      )}
                    </div>
                    <div className="flex gap-6 text-sm text-gray-600 mb-4">
                      <div>
                        <span className="font-semibold">{job.totalApplications}</span> applications
                      </div>
                      {job.newApplications > 0 && (
                        <div className="text-primary-600 font-semibold">
                          {job.newApplications} new
                        </div>
                      )}
                      {job.averageAtsScore > 0 && (
                        <div>
                          Avg ATS: <span className="font-semibold">{job.averageAtsScore.toFixed(1)}%</span>
                        </div>
                      )}
                    </div>
                  </div>
                  <Link to={`/recruiter/jobs/${job.id}/applications`}>
                    <Button variant="primary">View Applications</Button>
                  </Link>
                </div>
              </Card>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
