import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import api from '../services/api.js';

export default function RecruiterDashboard () {
  const [jobs, setJobs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    api.get('/jobs/recruiter/me')
      .then((res) => setJobs(res.data))
      .catch((err) => setError(err.response?.data?.message || 'Failed to fetch jobs'))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <p>Loading your jobs...</p>;
  }

  if (error) {
    return <p style={{ color: 'crimson' }}>{error}</p>;
  }

  return (
    <div className="grid">
      <div className="card" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h1>Your job postings</h1>
          <p>Create new openings and review candidates in one place.</p>
        </div>
        <Link className="primary-button" to="/recruiter/jobs/new">Post a job</Link>
      </div>
      {jobs.length === 0 && (
        <p>You have not posted any jobs yet. Click “Post a job” to create your first opening.</p>
      )}
      {jobs.map((job) => (
        <div className="card" key={job._id}>
          <h2>{job.title}</h2>
          {job.location && <p><strong>Location:</strong> {job.location}</p>}
          {job.employmentType && <p><strong>Type:</strong> {job.employmentType}</p>}
          <p>{job.description.slice(0, 180)}...</p>
          <Link className="primary-button" to={`/recruiter/jobs/${job._id}/applications`}>
            View applications
          </Link>
        </div>
      ))}
    </div>
  );
}

