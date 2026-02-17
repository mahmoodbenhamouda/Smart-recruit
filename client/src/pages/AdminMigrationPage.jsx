import React, { useState } from 'react';
import axios from 'axios';
import { useAuth } from '../hooks/useAuth';

const AdminMigrationPage = () => {
  const { user } = useAuth();
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState('');

  const reanalyzeAll = async () => {
    if (!window.confirm('This will re-analyze ALL applications without ATS data. Continue?')) {
      return;
    }

    setLoading(true);
    setError('');
    setResults(null);

    try {
      const response = await axios.post(
        'http://localhost:5000/api/migrate/reanalyze-applications',
        {},
        { withCredentials: true }
      );
      
      setResults(response.data.results);
      alert(`✅ Complete! ${response.data.results.successful} applications re-analyzed successfully.`);
    } catch (err) {
      console.error('Migration error:', err);
      setError(err.response?.data?.message || 'Failed to re-analyze applications');
    } finally {
      setLoading(false);
    }
  };

  if (user?.role !== 'recruiter') {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="text-center">
          <h1 className="text-2xl font-bold text-gray-900">Access Denied</h1>
          <p className="text-gray-600 mt-2">Only recruiters can access this page</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="bg-white rounded-lg shadow-sm p-8 mb-6">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            🔄 ATS Data Migration
          </h1>
          <p className="text-gray-600">
            Re-analyze applications that were submitted before the Python API was running
          </p>
        </div>

        {/* Info Box */}
        <div className="bg-blue-50 border-l-4 border-blue-500 p-6 mb-6 rounded-r-lg">
          <div className="flex">
            <div className="flex-shrink-0">
              <svg className="h-5 w-5 text-blue-500" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
              </svg>
            </div>
            <div className="ml-3">
              <h3 className="text-sm font-medium text-blue-800">Why do I need this?</h3>
              <div className="mt-2 text-sm text-blue-700">
                <ul className="list-disc list-inside space-y-1">
                  <li>Applications submitted before the Python API was running don't have ATS data</li>
                  <li>This tool will re-process all applications with missing or incomplete ATS analysis</li>
                  <li>The Python API must be running on port 8000 for this to work</li>
                  <li>This may take a few minutes depending on the number of applications</li>
                </ul>
              </div>
            </div>
          </div>
        </div>

        {/* Action Button */}
        <div className="bg-white rounded-lg shadow-sm p-8 mb-6">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-xl font-semibold text-gray-900 mb-2">
                Re-analyze All Applications
              </h2>
              <p className="text-gray-600">
                Process all applications with missing ATS data
              </p>
            </div>
            <button
              onClick={reanalyzeAll}
              disabled={loading}
              className={`
                px-6 py-3 rounded-lg font-medium transition-colors
                ${loading 
                  ? 'bg-gray-300 text-gray-500 cursor-not-allowed' 
                  : 'bg-blue-600 text-white hover:bg-blue-700'
                }
              `}
            >
              {loading ? (
                <span className="flex items-center">
                  <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Processing...
                </span>
              ) : (
                'Start Re-analysis'
              )}
            </button>
          </div>
        </div>

        {/* Error Message */}
        {error && (
          <div className="bg-red-50 border-l-4 border-red-500 p-6 rounded-r-lg mb-6">
            <div className="flex">
              <div className="flex-shrink-0">
                <svg className="h-5 w-5 text-red-500" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                </svg>
              </div>
              <div className="ml-3">
                <h3 className="text-sm font-medium text-red-800">Error</h3>
                <div className="mt-2 text-sm text-red-700">
                  <p>{error}</p>
                  <p className="mt-2">Make sure the Python API is running on port 8000.</p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Results */}
        {results && (
          <div className="bg-white rounded-lg shadow-sm p-8">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">
              📊 Migration Results
            </h2>
            
            <div className="grid grid-cols-3 gap-4 mb-6">
              <div className="bg-blue-50 rounded-lg p-4 text-center">
                <div className="text-3xl font-bold text-blue-600">{results.total}</div>
                <div className="text-sm text-gray-600 mt-1">Total Found</div>
              </div>
              <div className="bg-green-50 rounded-lg p-4 text-center">
                <div className="text-3xl font-bold text-green-600">{results.successful}</div>
                <div className="text-sm text-gray-600 mt-1">Successful</div>
              </div>
              <div className="bg-red-50 rounded-lg p-4 text-center">
                <div className="text-3xl font-bold text-red-600">{results.failed}</div>
                <div className="text-sm text-gray-600 mt-1">Failed</div>
              </div>
            </div>

            {results.errors && results.errors.length > 0 && (
              <div className="mt-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-3">Errors:</h3>
                <div className="bg-gray-50 rounded-lg p-4 max-h-64 overflow-y-auto">
                  {results.errors.map((error, index) => (
                    <div key={index} className="text-sm text-red-600 mb-2">
                      • {error}
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div className="mt-6 p-4 bg-green-50 rounded-lg">
              <p className="text-sm text-green-800">
                ✅ Migration complete! You can now view the updated ATS scores in the recruiter dashboard.
              </p>
            </div>
          </div>
        )}

        {/* Instructions */}
        <div className="bg-white rounded-lg shadow-sm p-8 mt-6">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">
            💡 Alternative: Submit New Applications
          </h2>
          <p className="text-gray-600 mb-4">
            Instead of migrating old applications, you can also:
          </p>
          <ol className="list-decimal list-inside space-y-2 text-gray-700">
            <li>Make sure the Python API is running (check port 8000)</li>
            <li>Log in as a candidate</li>
            <li>Apply to any job with a PDF resume</li>
            <li>The new application will automatically get ATS analysis</li>
            <li>Check "My Applications" to see the results</li>
          </ol>
        </div>
      </div>
    </div>
  );
};

export default AdminMigrationPage;
