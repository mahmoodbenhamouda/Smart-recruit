import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../state/AuthContext.jsx';
import { Card, Button, Input, Select, Alert } from '../components/UI.jsx';

export default function RegisterPageNew () {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    password: '',
    role: 'candidate',
    organization: ''
  });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const { register } = useAuth();
  const navigate = useNavigate();

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');

    if (formData.password.length < 6) {
      setError('Password must be at least 6 characters long');
      return;
    }

    setLoading(true);

    try {
      const user = await register(formData);
      
      // Redirect based on role
      if (user.role === 'recruiter') {
        navigate('/recruiter/dashboard');
      } else {
        navigate('/');
      }
    } catch (err) {
      setError(err.response?.data?.message || err.response?.data?.errors?.[0]?.msg || 'Registration failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-primary-50 to-blue-100 flex items-center justify-center px-4 py-12">
      <div className="w-full max-w-md">
        {/* Logo/Header */}
        <div className="text-center mb-8">
          <div className="inline-block w-16 h-16 bg-primary-600 rounded-2xl flex items-center justify-center mb-4">
            <span className="text-white font-bold text-3xl">T</span>
          </div>
          <h1 className="text-3xl font-bold text-gray-900">Create Account</h1>
          <p className="text-gray-600 mt-2">Join Smart-Recruit today</p>
        </div>

        {/* Registration Form */}
        <Card>
          {error && <Alert type="error" onClose={() => setError('')}>{error}</Alert>}

          <form onSubmit={handleSubmit}>
            <Input
              type="text"
              name="name"
              label="Full Name"
              placeholder="John Doe"
              value={formData.name}
              onChange={handleChange}
              required
            />

            <Input
              type="email"
              name="email"
              label="Email Address"
              placeholder="you@example.com"
              value={formData.email}
              onChange={handleChange}
              required
            />

            <Input
              type="password"
              name="password"
              label="Password"
              placeholder="••••••••"
              value={formData.password}
              onChange={handleChange}
              required
            />

            <Select
              name="role"
              label="I am a..."
              value={formData.role}
              onChange={handleChange}
              required
            >
              <option value="candidate">Job Seeker / Candidate</option>
              <option value="recruiter">Recruiter / Employer</option>
            </Select>

            {formData.role === 'recruiter' && (
              <Input
                type="text"
                name="organization"
                label="Organization (Optional)"
                placeholder="Company Name"
                value={formData.organization}
                onChange={handleChange}
              />
            )}

            <Button
              type="submit"
              variant="primary"
              className="w-full"
              disabled={loading}
            >
              {loading ? 'Creating Account...' : 'Sign Up'}
            </Button>
          </form>

          <div className="mt-6 text-center">
            <p className="text-sm text-gray-600">
              Already have an account?{' '}
              <Link to="/login" className="text-primary-600 font-semibold hover:text-primary-700">
                Sign in
              </Link>
            </p>
          </div>
        </Card>

        {/* Features */}
        <div className="mt-6 grid grid-cols-2 gap-4">
          <Card className="bg-white bg-opacity-80 text-center py-4">
            <p className="text-2xl mb-1">🎯</p>
            <p className="text-xs font-semibold text-gray-700">AI-Powered Matching</p>
          </Card>
          <Card className="bg-white bg-opacity-80 text-center py-4">
            <p className="text-2xl mb-1">📊</p>
            <p className="text-xs font-semibold text-gray-700">Detailed Analytics</p>
          </Card>
        </div>
      </div>
    </div>
  );
}
