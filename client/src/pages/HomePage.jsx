import { useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import api from '../services/api.js';
import { useAuth } from '../state/AuthContext.jsx';
import { Card, Badge, Button, Spinner } from '../components/UI.jsx';

export default function HomePage () {
  const [jobs, setJobs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const { user } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    let isMounted = true;
    setLoading(true);
    api.get('/jobs')
      .then((res) => {
        if (isMounted) {
          setJobs(res.data);
        }
      })
      .catch((err) => {
        setError(err.message || 'Failed to load jobs');
      })
      .finally(() => setLoading(false));

    return () => { isMounted = false; };
  }, []);

  const handleApplyClick = (jobId) => {
    if (!user) {
      navigate('/login');
      return;
    }
    navigate(`/jobs/${jobId}`);
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
      <div className="min-h-screen flex items-center justify-center">
        <Card className="max-w-md text-center">
          <p className="text-danger-600 text-lg">❌ {error}</p>
          <Button variant="primary" className="mt-4" onClick={() => window.location.reload()}>
            Try Again
          </Button>
        </Card>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-7xl mx-auto px-4">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            {user?.role === 'candidate' ? 'Find Your Dream Job' : 'Available Positions'}
          </h1>
          <p className="text-gray-600 text-lg">
            {jobs.length} {jobs.length === 1 ? 'position' : 'positions'} available
          </p>
        </div>

        {/* Jobs Grid */}
        {jobs.length === 0 ? (
          <Card className="text-center py-12">
            <div className="text-6xl mb-4">📭</div>
            <h3 className="text-xl font-semibold text-gray-900 mb-2">No Jobs Posted Yet</h3>
            <p className="text-gray-600">
              {user?.role === 'recruiter' 
                ? 'Be the first to post a job!' 
                : 'Check back soon for new opportunities!'}
            </p>
            {user?.role === 'recruiter' && (
              <Button variant="primary" className="mt-4" onClick={() => navigate('/recruiter/jobs/new')}>
                + Post Your First Job
              </Button>
            )}
          </Card>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {jobs.map((job) => (
              <Card key={job._id} className="flex flex-col h-full hover:shadow-xl transition-shadow duration-200">
                {/* Job Header */}
                <div className="flex-1">
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-bold text-gray-900 flex-1 pr-2">
                      {job.title}
                    </h3>
                    <Badge variant="primary">{job.employmentType || 'Full-time'}</Badge>
                  </div>

                  {/* Location */}
                  {job.location && (
                    <div className="flex items-center text-gray-600 mb-3">
                      <span className="mr-2">📍</span>
                      <span className="text-sm">{job.location}</span>
                    </div>
                  )}

                  {/* Description Preview */}
                  <p className="text-gray-700 text-sm leading-relaxed mb-4 line-clamp-3">
                    {job.description.length > 150 
                      ? `${job.description.slice(0, 150)}...` 
                      : job.description}
                  </p>

                  {/* Skills */}
                  {job.skills && job.skills.length > 0 && (
                    <div className="flex flex-wrap gap-2 mb-4">
                      {job.skills.slice(0, 3).map((skill, idx) => (
                        <Badge key={idx} variant="info" className="text-xs">
                          {skill}
                        </Badge>
                      ))}
                      {job.skills.length > 3 && (
                        <Badge variant="default" className="text-xs">
                          +{job.skills.length - 3} more
                        </Badge>
                      )}
                    </div>
                  )}
                </div>

                {/* Footer */}
                <div className="border-t border-gray-200 pt-4 mt-4">
                  <p className="text-xs text-gray-500 mb-3">
                    Posted by <span className="font-medium text-gray-700">{job.recruiter?.name || 'Recruiter'}</span>
                  </p>

                  {user?.role === 'candidate' ? (
                    <Button 
                      variant="primary" 
                      className="w-full"
                      onClick={() => handleApplyClick(job._id)}
                    >
                      📄 Apply Now
                    </Button>
                  ) : user?.role === 'recruiter' ? (
                    <Link to={`/jobs/${job._id}`} className="block">
                      <Button variant="outline" className="w-full">
                        View Details
                      </Button>
                    </Link>
                  ) : (
                    <Link to={`/jobs/${job._id}`} className="block">
                      <Button variant="primary" className="w-full">
                        View Details
                      </Button>
                    </Link>
                  )}
                </div>
              </Card>
            ))}
          </div>
        )}

        {/* CTA for non-authenticated users */}
        {!user && jobs.length > 0 && (
          <Card className="mt-8 bg-primary-50 border-primary-200 text-center">
            <h3 className="text-xl font-semibold text-primary-900 mb-2">
              Ready to Apply?
            </h3>
            <p className="text-primary-800 mb-4">
              Sign up now to apply to these positions and get AI-powered feedback on your resume!
            </p>
            <div className="flex gap-3 justify-center">
              <Link to="/register">
                <Button variant="primary">Create Account</Button>
              </Link>
              <Link to="/login">
                <Button variant="outline">Sign In</Button>
              </Link>
            </div>
          </Card>
        )}
      </div>
    </div>
  );
}

