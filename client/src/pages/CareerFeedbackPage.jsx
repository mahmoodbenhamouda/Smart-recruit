import { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import api from '../services/api.js';
import { Card, Button, Badge, Spinner, Alert, ProgressBar } from '../components/UI.jsx';

/**
 * Dedicated Career Feedback Page
 * Provides comprehensive career guidance based on application analysis
 */
export default function CareerFeedbackPage() {
  const { applicationId } = useParams();
  const navigate = useNavigate();
  const [application, setApplication] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadApplication();
  }, [applicationId]);

  const loadApplication = async () => {
    setLoading(true);
    try {
      const response = await api.get('/applications/me');
      const app = response.data.find(a => a._id === applicationId);
      if (!app) {
        setError('Application not found');
      } else {
        setApplication(app);
      }
    } catch (err) {
      setError(err.response?.data?.message || 'Failed to load application');
    } finally {
      setLoading(false);
    }
  };

  const generateSkillGap = () => {
    const matched = application.atsReport?.matchedSkills?.length || 0;
    const missing = application.atsReport?.missingSkills?.length || 0;
    const total = matched + missing;
    return total > 0 ? ((matched / total) * 100).toFixed(1) : 0;
  };

  const generateLearningPath = (skills) => {
    const skillCategories = {
      'Programming Languages': ['Python', 'JavaScript', 'Java', 'TypeScript', 'C++', 'Go', 'Rust'],
      'Frameworks & Libraries': ['React', 'Angular', 'Vue', 'Node.js', 'Django', 'Flask', 'Spring'],
      'Cloud & DevOps': ['AWS', 'Azure', 'GCP', 'Docker', 'Kubernetes', 'CI/CD', 'Terraform'],
      'Data & AI': ['Machine Learning', 'Deep Learning', 'NLP', 'SQL', 'MongoDB', 'TensorFlow', 'PyTorch'],
      'Tools & Methodologies': ['Git', 'Agile', 'Scrum', 'JIRA', 'REST API', 'GraphQL', 'Microservices']
    };

    const categorized = {};
    skills.forEach(skill => {
      for (const [category, keywords] of Object.entries(skillCategories)) {
        if (keywords.some(k => skill.toLowerCase().includes(k.toLowerCase()))) {
          if (!categorized[category]) categorized[category] = [];
          categorized[category].push(skill);
          break;
        }
      }
      if (!Object.values(categorized).flat().includes(skill)) {
        if (!categorized['Other Skills']) categorized['Other Skills'] = [];
        categorized['Other Skills'].push(skill);
      }
    });

    return categorized;
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <Spinner size="lg" />
      </div>
    );
  }

  if (error || !application) {
    return (
      <div className="max-w-7xl mx-auto px-4 py-8">
        <Alert type="error">{error || 'Application not found'}</Alert>
        <Button className="mt-4" onClick={() => navigate('/candidate/applications')}>
          Back to Applications
        </Button>
      </div>
    );
  }

  const { atsReport, job } = application;
  const skillGapPercentage = generateSkillGap();
  const missingSkillsCategories = generateLearningPath(atsReport?.missingSkills || []);
  const matchedSkillsCategories = generateLearningPath(atsReport?.matchedSkills || []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-blue-50 py-8">
      <div className="max-w-6xl mx-auto px-4">
        {/* Header */}
        <div className="mb-6">
          <Button variant="secondary" onClick={() => navigate('/candidate/applications')}>
            ← Back to Applications
          </Button>
        </div>

        {/* Title */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">Your Career Roadmap</h1>
          <p className="text-lg text-gray-600">Personalized feedback for: <strong>{job.title}</strong></p>
        </div>

        {/* Overall Assessment */}
        <Card className="mb-6 bg-gradient-to-r from-primary-50 to-blue-50">
          <div className="text-center">
            <h2 className="text-2xl font-bold text-gray-900 mb-4">Your Match Score</h2>
            <div className="flex items-center justify-center gap-8 mb-6">
              <div>
                <p className="text-6xl font-bold text-primary-600">{atsReport?.overallMatch?.toFixed(0) || 0}%</p>
                <p className="text-sm text-gray-600 mt-2">Overall Match</p>
              </div>
              <div>
                <p className="text-6xl font-bold text-success-600">{skillGapPercentage}%</p>
                <p className="text-sm text-gray-600 mt-2">Skills Coverage</p>
              </div>
            </div>
            <ProgressBar 
              value={atsReport?.overallMatch || 0} 
              variant={atsReport?.overallMatch >= 70 ? 'success' : atsReport?.overallMatch >= 40 ? 'warning' : 'danger'}
              className="max-w-2xl mx-auto"
            />
          </div>
        </Card>

        {/* Your Strengths */}
        <Card className="mb-6">
          <h3 className="text-2xl font-bold text-gray-900 mb-4 flex items-center gap-2">
            <span>💪</span>
            <span>Your Strengths</span>
          </h3>
          {Object.entries(matchedSkillsCategories).length > 0 ? (
            <div className="space-y-4">
              {Object.entries(matchedSkillsCategories).map(([category, skills]) => (
                <div key={category} className="border-l-4 border-success-500 pl-4">
                  <p className="font-semibold text-success-900 mb-2">{category}</p>
                  <div className="flex flex-wrap gap-2">
                    {skills.map((skill, idx) => (
                      <Badge key={idx} variant="success">{skill}</Badge>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-gray-600">Keep building your skills! Every journey starts somewhere.</p>
          )}
        </Card>

        {/* Skill Development Plan */}
        {Object.entries(missingSkillsCategories).length > 0 && (
          <Card className="mb-6 border-2 border-warning-300">
            <h3 className="text-2xl font-bold text-gray-900 mb-4 flex items-center gap-2">
              <span>🎯</span>
              <span>Your Skill Development Plan</span>
            </h3>
            <div className="space-y-6">
              {Object.entries(missingSkillsCategories).map(([category, skills], catIdx) => (
                <div key={category} className="bg-warning-50 border border-warning-200 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-3">
                    <h4 className="font-semibold text-warning-900">{category}</h4>
                    <Badge variant="warning">Priority {catIdx + 1}</Badge>
                  </div>
                  <div className="space-y-3">
                    {skills.map((skill, idx) => (
                      <div key={idx} className="bg-white rounded-lg p-3 border border-warning-200">
                        <div className="flex items-center justify-between mb-2">
                          <p className="font-medium text-gray-900">{skill}</p>
                          <Badge variant="info">Learn</Badge>
                        </div>
                        <div className="text-sm text-gray-600">
                          <p className="mb-2">🎓 <strong>Learning Path:</strong></p>
                          <ul className="list-disc list-inside space-y-1 ml-4">
                            <li>Start with beginner tutorials on YouTube or Coursera</li>
                            <li>Build 2-3 small projects to practice</li>
                            <li>Add to your resume with specific examples</li>
                            <li>Estimated time: {idx === 0 ? '2-4 weeks' : '1-2 months'}</li>
                          </ul>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </Card>
        )}

        {/* Career Path Recommendations */}
        {atsReport?.jobPrediction?.topPredictions && (
          <Card className="mb-6 bg-gradient-to-br from-purple-50 to-pink-50">
            <h3 className="text-2xl font-bold text-gray-900 mb-4 flex items-center gap-2">
              <span>🚀</span>
              <span>Recommended Career Paths</span>
            </h3>
            <p className="text-gray-600 mb-4">
              Based on your profile, here are roles you're well-suited for:
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {atsReport.jobPrediction.topPredictions.slice(0, 6).map((pred, idx) => (
                <div key={idx} className="bg-white rounded-lg p-4 border-2 border-purple-200 hover:border-purple-400 transition-colors">
                  <div className="flex items-start justify-between mb-2">
                    <h4 className="font-bold text-gray-900">{pred.role}</h4>
                    <Badge variant={idx === 0 ? 'success' : 'primary'}>
                      {(pred.probability * 100).toFixed(0)}% match
                    </Badge>
                  </div>
                  <ProgressBar value={pred.probability * 100} variant="primary" className="mb-2" />
                  <p className="text-sm text-gray-600">
                    {idx === 0 ? '⭐ Top recommendation based on your skills' :
                     idx < 3 ? '✓ Strong fit - consider exploring this path' :
                     '💡 Worth considering as you develop more skills'}
                  </p>
                </div>
              ))}
            </div>
          </Card>
        )}

        {/* Next Steps */}
        <Card className="mb-6 border-2 border-primary-300">
          <h3 className="text-2xl font-bold text-gray-900 mb-4 flex items-center gap-2">
            <span>✅</span>
            <span>Your Action Plan (Next 30 Days)</span>
          </h3>
          <div className="space-y-4">
            <div className="flex items-start gap-3 p-4 bg-blue-50 rounded-lg">
              <span className="text-2xl">1️⃣</span>
              <div>
                <p className="font-semibold text-gray-900">Week 1-2: Update Your Resume</p>
                <ul className="text-sm text-gray-700 mt-2 space-y-1 list-disc list-inside">
                  <li>Add quantifiable achievements for your matched skills</li>
                  <li>Incorporate keywords from job descriptions naturally</li>
                  <li>Ensure ATS-friendly formatting</li>
                </ul>
              </div>
            </div>

            <div className="flex items-start gap-3 p-4 bg-green-50 rounded-lg">
              <span className="text-2xl">2️⃣</span>
              <div>
                <p className="font-semibold text-gray-900">Week 2-3: Skill Development</p>
                <ul className="text-sm text-gray-700 mt-2 space-y-1 list-disc list-inside">
                  <li>Start online courses for top 2-3 missing skills</li>
                  <li>Complete at least one small project</li>
                  <li>Document your learning on LinkedIn</li>
                </ul>
              </div>
            </div>

            <div className="flex items-start gap-3 p-4 bg-purple-50 rounded-lg">
              <span className="text-2xl">3️⃣</span>
              <div>
                <p className="font-semibold text-gray-900">Week 3-4: Application Strategy</p>
                <ul className="text-sm text-gray-700 mt-2 space-y-1 list-disc list-inside">
                  <li>Reapply to this position with updated resume</li>
                  <li>Apply to similar roles that match your strengthened profile</li>
                  <li>Network with professionals in your target roles</li>
                </ul>
              </div>
            </div>
          </div>
        </Card>

        {/* Resources */}
        <Card className="bg-gradient-to-r from-indigo-50 to-blue-50">
          <h3 className="text-2xl font-bold text-gray-900 mb-4 flex items-center gap-2">
            <span>📚</span>
            <span>Recommended Learning Platforms</span>
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {[
              { name: 'Coursera', icon: '🎓', desc: 'University courses & certifications', color: 'blue' },
              { name: 'Udemy', icon: '💻', desc: 'Practical skill-based courses', color: 'purple' },
              { name: 'LinkedIn Learning', icon: '💼', desc: 'Professional development', color: 'indigo' },
              { name: 'freeCodeCamp', icon: '🆓', desc: 'Free coding bootcamp', color: 'green' },
              { name: 'GitHub', icon: '📦', desc: 'Build portfolio projects', color: 'gray' },
              { name: 'YouTube', icon: '▶️', desc: 'Free tutorials & guides', color: 'red' }
            ].map((platform) => (
              <div key={platform.name} className={`p-4 bg-white rounded-lg border-2 border-${platform.color}-200 hover:shadow-lg transition-shadow`}>
                <div className="text-3xl mb-2">{platform.icon}</div>
                <p className="font-bold text-gray-900">{platform.name}</p>
                <p className="text-xs text-gray-600 mt-1">{platform.desc}</p>
              </div>
            ))}
          </div>
        </Card>

        {/* Motivational Footer */}
        <div className="text-center mt-8 p-6 bg-gradient-to-r from-primary-500 to-blue-500 rounded-xl text-white">
          <h3 className="text-2xl font-bold mb-2">🌟 You've Got This!</h3>
          <p className="text-lg">
            Every expert was once a beginner. Your journey to success starts with these small steps.
          </p>
          <div className="mt-4 flex justify-center gap-4">
            <Button variant="secondary" onClick={() => navigate('/candidate/applications')}>
              View All Applications
            </Button>
            <Button variant="outline" className="bg-white text-primary-600 hover:bg-gray-100" onClick={() => navigate('/')}>
              Browse More Jobs
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
