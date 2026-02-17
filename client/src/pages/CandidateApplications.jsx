import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import api from '../services/api.js';

export default function CandidateApplications () {
  const [applications, setApplications] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    api.get('/applications/me')
      .then((res) => setApplications(res.data))
      .catch((err) => setError(err.response?.data?.message || 'Failed to load applications'))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <p>Loading applications...</p>;
  }

  if (error) {
    return <p style={{ color: 'crimson' }}>{error}</p>;
  }

  if (applications.length === 0) {
    return <p>You have not applied to any jobs yet. Visit the <Link to="/">jobs page</Link> to get started.</p>;
  }

  return (
    <div className="grid">
      {applications.map((application) => (
        <div className="card" key={application._id}>
          <h2>{application.job?.title}</h2>
          <p>Status: <strong>{application.status}</strong></p>
          {application.atsReport && (
            <>
              <p>Match score: {application.atsReport.overallMatch?.toFixed(1)}%</p>
              <p>Predicted role: {application.atsReport.jobPrediction?.predictedRole || 'N/A'}</p>
              {application.atsReport.matchedSkills?.length > 0 && (
                <p>
                  Matched skills: {application.atsReport.matchedSkills.slice(0, 8).join(', ')}
                  {application.atsReport.matchedSkills.length > 8 && ' ...'}
                </p>
              )}
            </>
          )}
          <p style={{ fontSize: '0.9rem', color: '#64748b' }}>
            Applied on {new Date(application.createdAt).toLocaleString()}
          </p>
        </div>
      ))}
    </div>
  );
}

