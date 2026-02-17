import { Routes, Route, Navigate } from 'react-router-dom';
import { useAuth } from './state/AuthContext.jsx';
import NavBar from './components/NavBar.jsx';
import HomePage from './pages/HomePage.jsx';
import LoginPageNew from './pages/LoginPageNew.jsx';
import RegisterPageNew from './pages/RegisterPageNew.jsx';
import JobDetailPage from './pages/JobDetailPage.jsx';
import RecruiterDashboardNew from './pages/RecruiterDashboardNew.jsx';
import PostJobPage from './pages/PostJobPage.jsx';
import CandidateApplicationsNew from './pages/CandidateApplicationsNew.jsx';
import RecruiterJobApplications from './pages/RecruiterJobApplications.jsx';
import CareerFeedbackPage from './pages/CareerFeedbackPage.jsx';
import AdminMigrationPage from './pages/AdminMigrationPage.jsx';

function ProtectedRoute ({ children, roles }) {
  const { user } = useAuth();

  if (!user) {
    return <Navigate to="/login" replace />;
  }

  if (roles && !roles.includes(user.role)) {
    return <Navigate to="/" replace />;
  }

  return children;
}

export default function App () {
  return (
    <div className="app-shell">
      <NavBar />
      <main className="app-content">
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/login" element={<LoginPageNew />} />
          <Route path="/register" element={<RegisterPageNew />} />
          <Route path="/jobs/:id" element={<JobDetailPage />} />
          <Route
            path="/candidate/applications"
            element={(
              <ProtectedRoute roles={['candidate']}>
                <CandidateApplicationsNew />
              </ProtectedRoute>
            )}
          />
          <Route
            path="/candidate/feedback/:applicationId"
            element={(
              <ProtectedRoute roles={['candidate']}>
                <CareerFeedbackPage />
              </ProtectedRoute>
            )}
          />
          <Route
            path="/recruiter/dashboard"
            element={(
              <ProtectedRoute roles={['recruiter']}>
                <RecruiterDashboardNew />
              </ProtectedRoute>
            )}
          />
          <Route
            path="/recruiter/jobs/:jobId/applications"
            element={(
              <ProtectedRoute roles={['recruiter']}>
                <RecruiterJobApplications />
              </ProtectedRoute>
            )}
          />
          <Route
            path="/recruiter/jobs/new"
            element={(
              <ProtectedRoute roles={['recruiter']}>
                <PostJobPage />
              </ProtectedRoute>
            )}
          />
          <Route
            path="/admin/migrate"
            element={(
              <ProtectedRoute roles={['recruiter']}>
                <AdminMigrationPage />
              </ProtectedRoute>
            )}
          />
        </Routes>
      </main>
    </div>
  );
}

