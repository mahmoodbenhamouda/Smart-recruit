import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import api from '../services/api.js';
import { Card, Button, Badge, Spinner, Alert, Modal } from '../components/UI.jsx';
import { ATSResults } from '../components/ATSResults.jsx';

export default function CandidateApplicationsNew () {
  const [insights, setInsights] = useState(null);
  const [applications, setApplications] = useState([]);
  const [selectedApp, setSelectedApp] = useState(null);
  const [showModal, setShowModal] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [reanalyzing, setReanalyzing] = useState(null);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    setLoading(true);
    try {
      const [insightsRes, appsRes] = await Promise.all([
        api.get('/dashboard/candidate/insights'),
        api.get('/applications/me')
      ]);
      setInsights(insightsRes.data);
      setApplications(appsRes.data);
    } catch (err) {
      setError(err.response?.data?.message || 'Failed to load applications');
    } finally {
      setLoading(false);
    }
  };

  const handleViewDetails = (app) => {
    setSelectedApp(app);
    setShowModal(true);
  };

  const handleReanalyze = async (applicationId) => {
    if (!confirm('Re-analyze this application? This will update your ATS score and analysis.')) {
      return;
    }

    setReanalyzing(applicationId);
    try {
      await api.post(`/migrate/reanalyze-application/${applicationId}`);
      alert('✅ Application re-analyzed successfully!');
      await loadData(); // Reload to show updated data
    } catch (err) {
      alert(err.response?.data?.message || 'Failed to re-analyze application');
    } finally {
      setReanalyzing(null);
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
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">My Applications</h1>
        <p className="text-gray-600 mt-1">Track your job applications and get personalized feedback</p>
      </div>

      {/* Insights Dashboard */}
      {insights && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <Card>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Total Applied</p>
                <p className="text-3xl font-bold text-gray-900 mt-1">{insights.totalApplications}</p>
              </div>
              <div className="w-12 h-12 bg-primary-100 rounded-lg flex items-center justify-center">
                <svg className="w-6 h-6 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
              </div>
            </div>
          </Card>

          <Card>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Avg ATS Score</p>
                <p className="text-3xl font-bold text-gray-900 mt-1">{insights.averageAtsScore.toFixed(1)}%</p>
              </div>
              <div className="w-12 h-12 bg-success-100 rounded-lg flex items-center justify-center">
                <svg className="w-6 h-6 text-success-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
              </div>
            </div>
          </Card>

          <Card>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Accepted</p>
                <p className="text-3xl font-bold text-success-600 mt-1">{insights.applicationsByStatus.accepted}</p>
              </div>
              <div className="w-12 h-12 bg-success-100 rounded-lg flex items-center justify-center">
                <svg className="w-6 h-6 text-success-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
            </div>
          </Card>

          <Card>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Under Review</p>
                <p className="text-3xl font-bold text-warning-600 mt-1">
                  {insights.applicationsByStatus.submitted + insights.applicationsByStatus.reviewed}
                </p>
              </div>
              <div className="w-12 h-12 bg-warning-100 rounded-lg flex items-center justify-center">
                <svg className="w-6 h-6 text-warning-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
            </div>
          </Card>
        </div>
      )}

      {/* Career Insights */}
      {insights && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
          {insights.topMatchedRole && (
            <Card>
              <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                <span>🎯</span>
                <span>Your Top Matched Role</span>
              </h3>
              <p className="text-2xl font-bold text-primary-600">{insights.topMatchedRole}</p>
              <p className="text-sm text-gray-600 mt-2">
                This role appears most frequently in your job predictions
              </p>
            </Card>
          )}

          {insights.commonMissingSkills && insights.commonMissingSkills.length > 0 && (
            <Card>
              <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                <span>💡</span>
                <span>Skills to Improve</span>
              </h3>
              <div className="space-y-2">
                {insights.commonMissingSkills.slice(0, 5).map((skill, idx) => (
                  <div key={idx} className="flex items-center justify-between p-2 bg-gray-50 rounded">
                    <span className="text-sm font-medium text-gray-900">{skill.skill}</span>
                    <div className="flex items-center gap-2">
                      <Badge variant="warning">×{skill.frequency}</Badge>
                      <span className="text-xs text-gray-600">+{skill.averageImpact}%</span>
                    </div>
                  </div>
                ))}
              </div>
              <p className="text-xs text-gray-500 mt-3">
                Skills that appear frequently across your applications
              </p>
            </Card>
          )}
        </div>
      )}

      {/* Applications List */}
      <Card>
        <h2 className="text-xl font-bold text-gray-900 mb-4">📋 Application History</h2>
        
        {applications.length === 0 ? (
          <div className="text-center py-8">
            <p className="text-gray-500 mb-4">You haven't applied to any jobs yet.</p>
            <Link to="/">
              <Button variant="primary">Browse Jobs</Button>
            </Link>
          </div>
        ) : (
          <div className="space-y-4">
            {applications.map((app) => (
              <div key={app._id} className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow">
                <div className="flex justify-between items-start mb-3">
                  <div className="flex-1">
                    <h3 className="text-lg font-bold text-gray-900">{app.job.title}</h3>
                    <p className="text-sm text-gray-600">{app.job.location} • {app.job.employmentType}</p>
                  </div>
                  <Badge variant={
                    app.status === 'accepted' ? 'success' :
                    app.status === 'rejected' ? 'danger' :
                    app.status === 'reviewed' ? 'warning' : 'default'
                  }>
                    {app.status}
                  </Badge>
                </div>

                {app.atsReport && (
                  <div className="grid grid-cols-3 gap-4 mb-3 p-3 bg-gray-50 rounded">
                    <div>
                      <p className="text-xs text-gray-600">ATS Score</p>
                      <p className="text-lg font-bold text-primary-600">
                        {app.atsReport.overallMatch?.toFixed(1) || 0}%
                      </p>
                    </div>
                    <div>
                      <p className="text-xs text-gray-600">Predicted Role</p>
                      <p className="text-sm font-semibold text-gray-900">
                        {app.atsReport.jobPrediction?.predictedRole || 'N/A'}
                      </p>
                    </div>
                    <div>
                      <p className="text-xs text-gray-600">Applied</p>
                      <p className="text-sm text-gray-700">
                        {new Date(app.createdAt).toLocaleDateString()}
                      </p>
                    </div>
                  </div>
                )}

                <div className="flex gap-2">
                  {!app.atsReport || !app.atsReport.overallMatch ? (
                    <Button
                      variant="warning"
                      size="sm"
                      onClick={() => handleReanalyze(app._id)}
                      disabled={reanalyzing === app._id}
                    >
                      {reanalyzing === app._id ? '🔄 Analyzing...' : '🔄 Get ATS Analysis'}
                    </Button>
                  ) : (
                    <>
                      <Button
                        variant="primary"
                        size="sm"
                        onClick={() => handleViewDetails(app)}
                      >
                        View Full Analysis
                      </Button>
                      <Link to={`/candidate/feedback/${app._id}`}>
                        <Button variant="success" size="sm">
                          💡 Get Career Guidance
                        </Button>
                      </Link>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => handleReanalyze(app._id)}
                        disabled={reanalyzing === app._id}
                        title="Re-analyze this application"
                      >
                        {reanalyzing === app._id ? '🔄...' : '🔄'}
                      </Button>
                    </>
                  )}
                  <Link to={`/jobs/${app.job._id}`}>
                    <Button variant="outline" size="sm">View Job</Button>
                  </Link>
                </div>
              </div>
            ))}
          </div>
        )}
      </Card>

      {/* Details Modal */}
      {selectedApp && (
        <Modal
          isOpen={showModal}
          onClose={() => setShowModal(false)}
          title={`Analysis: ${selectedApp.job.title}`}
          className="max-w-4xl"
        >
          <div className="max-h-[70vh] overflow-y-auto">
            <ATSResults atsReport={selectedApp.atsReport} />
          </div>
        </Modal>
      )}
    </div>
  );
}
