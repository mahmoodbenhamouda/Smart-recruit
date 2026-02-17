import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../state/AuthContext.jsx';
import { Button } from './UI.jsx';

export default function NavBar () {
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  return (
    <nav className="bg-white shadow-md">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Logo */}
          <Link to="/" className="flex items-center space-x-2">
            <div className="w-8 h-8 bg-primary-600 rounded-lg flex items-center justify-center">
              <span className="text-white font-bold text-xl">S</span>
            </div>
            <span className="text-xl font-bold text-gray-900">Smart-Recruit</span>
          </Link>

          {/* Navigation Links */}
          <div className="flex items-center space-x-4">
            {!user ? (
              <>
                <Link to="/login">
                  <Button variant="outline" size="sm">Login</Button>
                </Link>
                <Link to="/register">
                  <Button variant="primary" size="sm">Sign Up</Button>
                </Link>
              </>
            ) : (
              <>
                <Link to="/">
                  <button className="text-gray-700 hover:text-primary-600 transition-colors px-3 py-2 text-sm font-medium">
                    Jobs
                  </button>
                </Link>
                
                {user.role === 'recruiter' ? (
                  <>
                    <Link to="/recruiter/dashboard">
                      <button className="text-gray-700 hover:text-primary-600 transition-colors px-3 py-2 text-sm font-medium">
                        Dashboard
                      </button>
                    </Link>
                    <Link to="/recruiter/jobs/new">
                      <Button variant="primary" size="sm">+ Post Job</Button>
                    </Link>
                  </>
                ) : (
                  <Link to="/candidate/applications">
                    <button className="text-gray-700 hover:text-primary-600 transition-colors px-3 py-2 text-sm font-medium">
                      My Applications
                    </button>
                  </Link>
                )}

                <div className="flex items-center space-x-3 border-l border-gray-200 pl-4">
                  <div className="text-right">
                    <p className="text-sm font-medium text-gray-900">{user.name}</p>
                    <p className="text-xs text-gray-500 capitalize">{user.role}</p>
                  </div>
                  <Button variant="secondary" size="sm" onClick={handleLogout}>
                    Logout
                  </Button>
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </nav>
  );
}

