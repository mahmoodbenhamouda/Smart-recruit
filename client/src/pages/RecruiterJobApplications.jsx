import { useEffect, useState } from 'react';
import { Link, useParams, useNavigate } from 'react-router-dom';
import api from '../services/api.js';
import { Card, Button, Badge, Spinner, Alert, Modal, ProgressBar } from '../components/UI.jsx';
import { ATSResults } from '../components/ATSResults.jsx';

export default function RecruiterJobApplications () {
  const { jobId } = useParams();
  const navigate = useNavigate();
  const [applications, setApplications] = useState([]);
  const [jobTitle, setJobTitle] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedApp, setSelectedApp] = useState(null);
  const [showModal, setShowModal] = useState(false);
  const [actionLoading, setActionLoading] = useState(null);

  useEffect(() => {
    loadData();
  }, [jobId]);

  const loadData = async () => {
    setLoading(true);
    try {
      const [jobRes, appsRes] = await Promise.all([
        api.get(`/jobs/${jobId}`),
        api.get(`/applications/job/${jobId}`)
      ]);
      setJobTitle(jobRes.data.title);
      // Sort by ATS score descending
      const sorted = appsRes.data.sort((a, b) => {
        const scoreA = a.atsReport?.overallMatch || 0;
        const scoreB = b.atsReport?.overallMatch || 0;
        return scoreB - scoreA;
      });
      setApplications(sorted);
    } catch (err) {
      setError(err.response?.data?.message || 'Failed to load applications');
    } finally {
      setLoading(false);
    }
  };

  const handleStatusChange = async (applicationId, newStatus) => {
    setActionLoading(applicationId);
    try {
      await api.patch(`/applications/${applicationId}/status`, { status: newStatus });
      await loadData();
    } catch (err) {
      alert(err.response?.data?.message || 'Failed to update status');
    } finally {
      setActionLoading(null);
    }
  };

  const handleViewDetails = (app) => {
    setSelectedApp(app);
    setShowModal(true);
  };

  const getStatusVariant = (status) => {
    switch (status) {
      case 'accepted': return 'success';
      case 'rejected': return 'danger';
      case 'reviewed': return 'warning';
      default: return 'default';
    }
  };

  const getRankBadge = (index) => {
    if (index === 0) return { emoji: '🥇', variant: 'success', text: '1st' };
    if (index === 1) return { emoji: '🥈', variant: 'info', text: '2nd' };
    if (index === 2) return { emoji: '🥉', variant: 'warning', text: '3rd' };
    return { emoji: '', variant: 'default', text: `${index + 1}th` };
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
        <Button className="mt-4" onClick={() => navigate('/recruiter/dashboard')}>
          Back to Dashboard
        </Button>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-7xl mx-auto px-4">
        {/* Header */}
        <div className="mb-6">
          <Button variant="secondary" onClick={() => navigate('/recruiter/dashboard')}>
            ← Back to Dashboard
          </Button>
        </div>

        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900">Applicants for {jobTitle}</h1>
          <p className="text-gray-600 mt-2">
            {applications.length} {applications.length === 1 ? 'candidate' : 'candidates'} applied
          </p>
        </div>

        {/* Summary Stats */}
        {applications.length > 0 && (
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
            <Card>
              <p className="text-sm font-medium text-gray-600">Total Applicants</p>
              <p className="text-3xl font-bold text-gray-900 mt-1">{applications.length}</p>
            </Card>
            <Card>
              <p className="text-sm font-medium text-gray-600">Avg ATS Score</p>
              <p className="text-3xl font-bold text-primary-600 mt-1">
                {(applications.reduce((sum, app) => sum + (app.atsReport?.overallMatch || 0), 0) / applications.length).toFixed(1)}%
              </p>
            </Card>
            <Card>
              <p className="text-sm font-medium text-gray-600">Under Review</p>
              <p className="text-3xl font-bold text-warning-600 mt-1">
                {applications.filter(a => ['submitted', 'reviewed'].includes(a.status)).length}
              </p>
            </Card>
            <Card>
              <p className="text-sm font-medium text-gray-600">Accepted</p>
              <p className="text-3xl font-bold text-success-600 mt-1">
                {applications.filter(a => a.status === 'accepted').length}
              </p>
            </Card>
          </div>
        )}

        {/* Applications List */}
        {applications.length === 0 ? (
          <Card className="text-center py-12">
            <div className="text-6xl mb-4">📭</div>
            <h3 className="text-xl font-semibold text-gray-900 mb-2">No Applications Yet</h3>
            <p className="text-gray-600">Candidates will appear here once they apply</p>
          </Card>
        ) : (
          <div className="space-y-4">
            {applications.map((app, index) => {
              const rank = getRankBadge(index);
              return (
                <Card key={app._id} className="hover:shadow-lg transition-shadow">
                  <div className="flex items-start justify-between mb-4">
                    {/* Candidate Info */}
                    <div className="flex-1">
                      <div className="flex items-center gap-3 mb-2">
                        {rank.emoji && <span className="text-2xl">{rank.emoji}</span>}
                        <h3 className="text-xl font-bold text-gray-900">{app.candidate?.name}</h3>
                        <Badge variant={getStatusVariant(app.status)} className="capitalize">
                          {app.status}
                        </Badge>
                      </div>
                      <p className="text-sm text-gray-600">✉️ {app.candidate?.email}</p>
                      <p className="text-xs text-gray-500 mt-1">
                        Applied {new Date(app.createdAt).toLocaleDateString()}
                      </p>
                    </div>

                    {/* Rank Badge */}
                    {index < 3 && (
                      <Badge variant={rank.variant} className="text-lg px-4 py-2">
                        #{rank.text}
                      </Badge>
                    )}
                  </div>

                  {/* ATS Summary */}
                  {app.atsReport ? (
                    <div className="bg-gray-50 rounded-lg p-4 mb-4">
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-3">
                        <div>
                          <p className="text-xs text-gray-600 mb-1">ATS Score</p>
                          <p className="text-2xl font-bold text-primary-600">
                            {app.atsReport.overallMatch?.toFixed(1) || 0}%
                          </p>
                          <ProgressBar
                            value={app.atsReport.overallMatch || 0}
                            variant={
                              (app.atsReport.overallMatch || 0) >= 70 ? 'success' :
                              (app.atsReport.overallMatch || 0) >= 40 ? 'warning' : 'danger'
                            }
                            className="mt-2"
                          />
                        </div>
                        <div>
                          <p className="text-xs text-gray-600 mb-1">Predicted Role</p>
                          <p className="text-lg font-semibold text-gray-900">
                            {app.atsReport.jobPrediction?.predictedRole || 'N/A'}
                          </p>
                          <p className="text-xs text-gray-500 mt-1">
                            Confidence: {((app.atsReport.jobPrediction?.confidence || 0) * 100).toFixed(1)}%
                          </p>
                        </div>
                        <div>
                          <p className="text-xs text-gray-600 mb-1">Skills Match</p>
                          <p className="text-lg font-semibold text-success-600">
                            {app.atsReport.matchedSkills?.length || 0} matched
                          </p>
                          <p className="text-lg font-semibold text-danger-600">
                            {app.atsReport.missingSkills?.length || 0} missing
                          </p>
                        </div>
                      </div>

                      {/* Skills Preview */}
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        {app.atsReport.matchedSkills && app.atsReport.matchedSkills.length > 0 && (
                          <div>
                            <p className="text-xs font-medium text-gray-700 mb-2">✅ Matched Skills:</p>
                            <div className="flex flex-wrap gap-1">
                              {app.atsReport.matchedSkills.slice(0, 5).map((skill, idx) => (
                                <Badge key={idx} variant="success" className="text-xs">
                                  {skill}
                                </Badge>
                              ))}
                              {app.atsReport.matchedSkills.length > 5 && (
                                <Badge variant="default" className="text-xs">
                                  +{app.atsReport.matchedSkills.length - 5} more
                                </Badge>
                              )}
                            </div>
                          </div>
                        )}
                        {app.atsReport.missingSkills && app.atsReport.missingSkills.length > 0 && (
                          <div>
                            <p className="text-xs font-medium text-gray-700 mb-2">❌ Missing Skills:</p>
                            <div className="flex flex-wrap gap-1">
                              {app.atsReport.missingSkills.slice(0, 5).map((skill, idx) => (
                                <Badge key={idx} variant="danger" className="text-xs">
                                  {skill}
                                </Badge>
                              ))}
                              {app.atsReport.missingSkills.length > 5 && (
                                <Badge variant="default" className="text-xs">
                                  +{app.atsReport.missingSkills.length - 5} more
                                </Badge>
                              )}
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  ) : (
                    <Alert type="warning" className="mb-4">
                      ATS analysis not available for this application
                    </Alert>
                  )}

                  {/* Actions */}
                  <div className="flex flex-wrap gap-2">
                    <Button
                      variant="primary"
                      size="sm"
                      onClick={() => handleViewDetails(app)}
                    >
                      📊 View Full Analysis
                    </Button>
                    <a
                      href={`/api/applications/${app._id}/resume`}
                      target="_blank"
                      rel="noreferrer"
                    >
                      <Button variant="outline" size="sm">
                        📄 Download Resume
                      </Button>
                    </a>
                    {app.status === 'submitted' && (
                      <>
                        <Button
                          variant="success"
                          size="sm"
                          onClick={() => handleStatusChange(app._id, 'accepted')}
                          disabled={actionLoading === app._id}
                        >
                          ✓ Accept
                        </Button>
                        <Button
                          variant="danger"
                          size="sm"
                          onClick={() => handleStatusChange(app._id, 'rejected')}
                          disabled={actionLoading === app._id}
                        >
                          ✗ Reject
                        </Button>
                      </>
                    )}
                    {app.status === 'reviewed' && (
                      <>
                        <Button
                          variant="success"
                          size="sm"
                          onClick={() => handleStatusChange(app._id, 'accepted')}
                          disabled={actionLoading === app._id}
                        >
                          ✓ Accept
                        </Button>
                        <Button
                          variant="danger"
                          size="sm"
                          onClick={() => handleStatusChange(app._id, 'rejected')}
                          disabled={actionLoading === app._id}
                        >
                          ✗ Reject
                        </Button>
                      </>
                    )}
                  </div>
                </Card>
              );
            })}
          </div>
        )}

        {/* Details Modal */}
        {selectedApp && (
          <Modal
            isOpen={showModal}
            onClose={() => setShowModal(false)}
            title={`Full Analysis: ${selectedApp.candidate?.name}`}
            className="max-w-6xl"
          >
            <div className="max-h-[75vh] overflow-y-auto">
              {/* Candidate Info Header */}
              <Card className="mb-4 bg-gray-50">
                <div className="flex justify-between items-start">
                  <div>
                    <h3 className="text-xl font-bold text-gray-900">{selectedApp.candidate?.name}</h3>
                    <p className="text-gray-600">{selectedApp.candidate?.email}</p>
                  </div>
                  <div className="text-right">
                    <Badge variant={getStatusVariant(selectedApp.status)} className="text-sm capitalize">
                      {selectedApp.status}
                    </Badge>
                    <p className="text-xs text-gray-500 mt-1">
                      Applied {new Date(selectedApp.createdAt).toLocaleDateString()}
                    </p>
                  </div>
                </div>
              </Card>

              {/* Full ATS Analysis */}
              <ATSResults atsReport={selectedApp.atsReport} showFeedback={false} />

              {/* CV Sections if available (LayoutLMv3 Extraction) */}
              {selectedApp.atsReport?.cvSections && Object.keys(selectedApp.atsReport.cvSections).length > 0 && (
                <Card className="mt-4">
                  <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
                    <span>📋</span>
                    <span>Structured Resume Sections</span>
                    <Badge variant="info" className="text-xs ml-2">AI Extraction</Badge>
                  </h3>
                  <p className="text-sm text-gray-600 mb-4">
                    Automatically extracted and structured sections from the resume
                  </p>
                  
                  <div className="space-y-4">
                    {Object.entries(selectedApp.atsReport.cvSections).map(([section, content]) => {
                      // Map section names to icons
                      const sectionIcons = {
                        'HEADER': '👤',
                        'CONTACT': '📧',
                        'SUMMARY': '📝',
                        'EDUCATION': '🎓',
                        'EXPERIENCE': '💼',
                        'SKILLS': '🛠️',
                        'PROJECTS': '🚀',
                        'CERTIFICATIONS': '🏆',
                        'LANGUAGES': '🌐',
                        'PUBLICATIONS': '📚',
                        'REFERENCES': '✉️',
                        'OTHER': '📄'
                      };
                      
                      const icon = sectionIcons[section] || '📌';
                      const displayName = section.replace(/_/g, ' ').toLowerCase()
                        .split(' ')
                        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
                        .join(' ');
                      
                      return (
                        <div key={section} className="bg-gray-50 rounded-lg p-4 border-l-4 border-primary-500">
                          <div className="flex items-center gap-2 mb-2">
                            <span className="text-2xl">{icon}</span>
                            <h4 className="text-base font-semibold text-gray-900">{displayName}</h4>
                          </div>
                          <div className="text-sm text-gray-700 leading-relaxed whitespace-pre-wrap bg-white p-3 rounded border border-gray-200">
                            {typeof content === 'string' ? content : JSON.stringify(content, null, 2)}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                  
                  <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
                    <p className="text-xs text-blue-800">
                      <strong>💡 Tip:</strong> These sections were automatically extracted using intelligent 
                      pattern recognition to identify key resume components like experience, education, and skills.
                    </p>
                  </div>
                </Card>
              )}
            </div>
          </Modal>
        )}
      </div>
    </div>
  );
}

