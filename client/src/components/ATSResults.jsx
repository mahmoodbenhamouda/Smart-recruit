import React from 'react';
import { Card, Badge, ProgressBar } from './UI.jsx';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

/**
 * ATS Results Visualization Component
 * Displays ATS score, SHAP, LIME, and Missing Skills analysis
 * @param {Object} atsReport - The ATS analysis report
 * @param {boolean} showFeedback - Whether to show personalized feedback (default: true, set false for recruiter view)
 */
export function ATSResults ({ atsReport, showFeedback = true }) {
  if (!atsReport) {
    return (
      <Card>
        <p className="text-gray-500 text-center py-8">
          No ATS analysis available
        </p>
      </Card>
    );
  }

  const {
    overallMatch,
    matchLevel,
    skillsMatchRate,
    matchedSkills,
    missingSkills,
    jobPrediction,
    xai
  } = atsReport;

  // Determine match level variant
  const getMatchVariant = (level) => {
    if (!level) return 'default';
    const lower = level.toLowerCase();
    if (lower.includes('excellent') || lower.includes('high')) return 'success';
    if (lower.includes('good') || lower.includes('medium')) return 'warning';
    if (lower.includes('low') || lower.includes('poor')) return 'danger';
    return 'default';
  };

  return (
    <div className="space-y-6">
      {/* Overall ATS Score */}
      <Card>
        <h3 className="text-lg font-semibold mb-4">📊 ATS Score</h3>
        <div className="flex items-center justify-between mb-4">
          <div>
            <p className="text-4xl font-bold text-primary-600">
              {overallMatch?.toFixed(1) || 0}%
            </p>
            <Badge variant={getMatchVariant(matchLevel)} className="mt-2">
              {matchLevel || 'Unknown'}
            </Badge>
          </div>
          <div className="text-right">
            <p className="text-sm text-gray-600">Skills Match</p>
            <p className="text-2xl font-semibold text-gray-800">
              {skillsMatchRate?.toFixed(1) || 0}%
            </p>
          </div>
        </div>
        <ProgressBar
          value={overallMatch || 0}
          variant={overallMatch >= 70 ? 'success' : overallMatch >= 40 ? 'warning' : 'danger'}
        />
      </Card>

      {/* Job Prediction */}
      {jobPrediction && (
        <Card>
          <h3 className="text-lg font-semibold mb-4">🎯 Job Role Prediction</h3>
          <div className="mb-4">
            <p className="text-sm text-gray-600">Predicted Role</p>
            <p className="text-2xl font-bold text-gray-900">{jobPrediction.predictedRole}</p>
            <p className="text-sm text-gray-600 mt-1">
              Confidence: {(jobPrediction.confidence * 100).toFixed(1)}%
            </p>
          </div>
          
          {jobPrediction.topPredictions && jobPrediction.topPredictions.length > 0 && (
            <div className="mt-4">
              <p className="text-sm font-medium text-gray-700 mb-2">Top Predictions:</p>
              <div className="space-y-2">
                {jobPrediction.topPredictions.slice(0, 5).map((pred, idx) => (
                  <div key={idx} className="flex items-center justify-between">
                    <span className="text-sm text-gray-700">{pred.role}</span>
                    <div className="flex items-center gap-2">
                      <ProgressBar
                        value={pred.probability * 100}
                        showLabel={false}
                        className="w-32"
                        variant={idx === 0 ? 'primary' : 'secondary'}
                      />
                      <span className="text-sm text-gray-600 w-12 text-right">
                        {(pred.probability * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </Card>
      )}

      {/* Skills Matched & Missing */}
      <div className="grid md:grid-cols-2 gap-6">
        <Card>
          <h3 className="text-lg font-semibold mb-4 text-success-600">✅ Matched Skills</h3>
          {matchedSkills && matchedSkills.length > 0 ? (
            <div className="flex flex-wrap gap-2">
              {matchedSkills.map((skill, idx) => (
                <Badge key={idx} variant="success">{skill}</Badge>
              ))}
            </div>
          ) : (
            <p className="text-gray-500 text-sm">No matched skills found</p>
          )}
        </Card>

        <Card>
          <h3 className="text-lg font-semibold mb-4 text-danger-600">❌ Missing Skills</h3>
          {missingSkills && missingSkills.length > 0 ? (
            <div className="flex flex-wrap gap-2">
              {missingSkills.map((skill, idx) => (
                <Badge key={idx} variant="danger">{skill}</Badge>
              ))}
            </div>
          ) : (
            <p className="text-gray-500 text-sm">No missing skills identified</p>
          )}
        </Card>
      </div>

      {/* XAI - SHAP Explanation */}
      {xai?.shap && (
        <Card>
          <h3 className="text-lg font-semibold mb-4">🔍 SHAP Feature Importance</h3>
          <p className="text-sm text-gray-600 mb-4">{xai.shap.summary}</p>
          
          {xai.shap.topFeatures && xai.shap.topFeatures.length > 0 && (
            <div className="space-y-2">
              {xai.shap.topFeatures.slice(0, 10).map((feature, idx) => (
                <div key={idx} className="flex items-center justify-between py-2 border-b border-gray-100">
                  <span className="text-sm font-medium text-gray-700 flex-1">{feature.feature}</span>
                  <div className="flex items-center gap-2">
                    <span className={`text-sm font-semibold ${feature.impact === 'positive' ? 'text-success-600' : 'text-danger-600'}`}>
                      {feature.impact === 'positive' ? '+' : ''}{feature.shapValue.toFixed(4)}
                    </span>
                    <Badge variant={feature.impact === 'positive' ? 'success' : 'danger'}>
                      {feature.impact}
                    </Badge>
                  </div>
                </div>
              ))}
            </div>
          )}
        </Card>
      )}

      {/* XAI - LIME Explanation */}
      {xai?.lime && (
        <Card>
          <h3 className="text-lg font-semibold mb-4">📝 LIME Word-Level Explanation</h3>
          <p className="text-sm text-gray-600 mb-4">{xai.lime.summary}</p>
          
          {xai.lime.features && xai.lime.features.length > 0 && (
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={xai.lime.features.slice(0, 10)} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis dataKey="feature" type="category" width={150} />
                <Tooltip />
                <Legend />
                <Bar dataKey="weight" fill="#3b82f6" name="Influence Weight" />
              </BarChart>
            </ResponsiveContainer>
          )}
        </Card>
      )}

      {/* XAI - Missing Skills Analysis */}
      {xai?.missingSkills && (
        <Card>
          <h3 className="text-lg font-semibold mb-4">💡 Missing Skills Analysis (AI-Powered)</h3>
          <div className="mb-4">
            <p className="text-sm text-gray-600">Target Role: <span className="font-semibold text-gray-900">{xai.missingSkills.targetRole}</span></p>
            <p className="text-sm text-gray-600">Current Match Probability: <span className="font-semibold text-gray-900">{(xai.missingSkills.currentMatchProbability * 100).toFixed(2)}%</span></p>
            {xai.missingSkills.totalPotentialImprovement > 0 && (
              <p className="text-sm text-success-600 font-semibold mt-2">
                ⬆️ Potential Improvement: +{(xai.missingSkills.totalPotentialImprovement * 100).toFixed(2)}%
              </p>
            )}
          </div>

          {xai.missingSkills.topMissingSkills && xai.missingSkills.topMissingSkills.length > 0 && (
            <div className="space-y-3 mb-4">
              {xai.missingSkills.topMissingSkills.map((skill, idx) => (
                <div key={idx} className="p-3 bg-gray-50 rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-medium text-gray-900">{skill.skill}</span>
                    <Badge variant={
                      skill.priority === 'High' ? 'danger' :
                      skill.priority === 'Medium' ? 'warning' : 'info'
                    }>
                      {skill.priority} Priority
                    </Badge>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-sm text-gray-600">Impact if added:</span>
                    <span className="text-sm font-semibold text-success-600">
                      +{skill.impactPercentage.toFixed(2)}%
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}

          {xai.missingSkills.recommendation && (
            <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
              <p className="text-sm font-medium text-blue-900 mb-1">💬 Recommendation:</p>
              <p className="text-sm text-blue-800 whitespace-pre-wrap">{xai.missingSkills.recommendation}</p>
            </div>
          )}
        </Card>
      )}

      {/* Personalized Feedback Section - Only show for candidates */}
      {showFeedback && (
        <Card className="bg-gradient-to-br from-blue-50 to-indigo-50 border-2 border-blue-200">
          <h3 className="text-xl font-semibold mb-4 text-blue-900 flex items-center gap-2">
            <span>💡</span>
            <span>Personalized Feedback & Career Guidance</span>
          </h3>

        {/* Score Assessment */}
        <div className="mb-6">
          <h4 className="text-lg font-semibold text-gray-900 mb-3">📈 Your Application Strength</h4>
          {overallMatch >= 80 ? (
            <div className="p-4 bg-success-50 border border-success-300 rounded-lg">
              <p className="font-semibold text-success-900 mb-2">🌟 Excellent Match!</p>
              <p className="text-success-800 text-sm">
                Your profile is an outstanding fit for this role. Your resume demonstrates strong alignment 
                with the job requirements. Focus on highlighting your achievements in the interview.
              </p>
            </div>
          ) : overallMatch >= 60 ? (
            <div className="p-4 bg-warning-50 border border-warning-300 rounded-lg">
              <p className="font-semibold text-warning-900 mb-2">⚡ Good Match with Room for Improvement</p>
              <p className="text-warning-800 text-sm">
                You have a solid foundation for this role. Review the missing skills below and consider 
                adding relevant keywords or experiences to strengthen your application.
              </p>
            </div>
          ) : (
            <div className="p-4 bg-danger-50 border border-danger-300 rounded-lg">
              <p className="font-semibold text-danger-900 mb-2">🎯 Development Areas Identified</p>
              <p className="text-danger-800 text-sm">
                There's a gap between your current profile and this role's requirements. Focus on acquiring 
                the missing skills and gaining relevant experience before applying to similar positions.
              </p>
            </div>
          )}
        </div>

        {/* Action Items */}
        <div className="mb-6">
          <h4 className="text-lg font-semibold text-gray-900 mb-3">✅ Recommended Actions</h4>
          <div className="space-y-3">
            {/* Skills to Add */}
            {missingSkills && missingSkills.length > 0 && (
              <div className="p-4 bg-white rounded-lg border border-gray-200">
                <p className="font-semibold text-gray-900 mb-2">🛠️ Skills to Acquire</p>
                <ul className="list-disc list-inside space-y-1 text-sm text-gray-700">
                  {missingSkills.slice(0, 5).map((skill, idx) => (
                    <li key={idx}>
                      Learn <strong>{skill}</strong> through online courses (Coursera, Udemy) or hands-on projects
                    </li>
                  ))}
                </ul>
                {missingSkills.length > 5 && (
                  <p className="text-xs text-gray-500 mt-2">
                    +{missingSkills.length - 5} more skills to consider
                  </p>
                )}
              </div>
            )}

            {/* Resume Optimization */}
            <div className="p-4 bg-white rounded-lg border border-gray-200">
              <p className="font-semibold text-gray-900 mb-2">📝 Resume Optimization Tips</p>
              <ul className="list-disc list-inside space-y-1 text-sm text-gray-700">
                <li>Add quantifiable achievements (e.g., "Increased efficiency by 30%")</li>
                <li>Use action verbs at the start of bullet points</li>
                <li>Include relevant keywords from the job description naturally</li>
                <li>Keep format clean and ATS-friendly (avoid tables, images in main content)</li>
                <li>Ensure your resume is 1-2 pages maximum</li>
              </ul>
            </div>

            {/* Interview Preparation */}
            {overallMatch >= 60 && (
              <div className="p-4 bg-white rounded-lg border border-gray-200">
                <p className="font-semibold text-gray-900 mb-2">🎤 Interview Preparation</p>
                <ul className="list-disc list-inside space-y-1 text-sm text-gray-700">
                  <li>Prepare STAR method examples for your matched skills</li>
                  <li>Research the company's recent projects and achievements</li>
                  <li>Be ready to discuss how you'd address the missing skills</li>
                  <li>Practice explaining your career progression and goals</li>
                </ul>
              </div>
            )}
          </div>
        </div>

        {/* Alternative Career Paths */}
        {jobPrediction && jobPrediction.topPredictions && jobPrediction.topPredictions.length > 1 && (
          <div className="mb-6">
            <h4 className="text-lg font-semibold text-gray-900 mb-3">🚀 Alternative Career Paths</h4>
            <p className="text-sm text-gray-600 mb-3">
              Based on your profile, you might also be a strong fit for these roles:
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {jobPrediction.topPredictions.slice(0, 4).map((pred, idx) => (
                <div key={idx} className="p-3 bg-white rounded-lg border border-gray-200">
                  <div className="flex items-center justify-between mb-1">
                    <p className="font-semibold text-gray-900 text-sm">{pred.role}</p>
                    <Badge variant={idx === 0 ? 'success' : 'info'}>
                      {(pred.probability * 100).toFixed(0)}%
                    </Badge>
                  </div>
                  <p className="text-xs text-gray-600">
                    {idx === 0 ? 'Your top match! Consider roles with this title.' : 
                     'Worth exploring - good fit for your skills.'}
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Learning Resources */}
        <div>
          <h4 className="text-lg font-semibold text-gray-900 mb-3">📚 Suggested Learning Resources</h4>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            <div className="p-3 bg-white rounded-lg border border-blue-200">
              <p className="font-semibold text-blue-900 text-sm mb-1">📖 Online Courses</p>
              <p className="text-xs text-gray-600">Coursera, Udemy, LinkedIn Learning</p>
            </div>
            <div className="p-3 bg-white rounded-lg border border-green-200">
              <p className="font-semibold text-green-900 text-sm mb-1">💼 Practice Projects</p>
              <p className="text-xs text-gray-600">GitHub, personal portfolio, open source</p>
            </div>
            <div className="p-3 bg-white rounded-lg border border-purple-200">
              <p className="font-semibold text-purple-900 text-sm mb-1">🎯 Certifications</p>
              <p className="text-xs text-gray-600">Industry-recognized credentials</p>
            </div>
          </div>
        </div>

        {/* Motivational Message */}
        <div className="mt-6 p-4 bg-gradient-to-r from-primary-100 to-blue-100 rounded-lg border-2 border-primary-300">
          <p className="text-center font-semibold text-primary-900">
            🌟 Keep improving! Every application is a learning opportunity.
          </p>
        </div>
        </Card>
      )}
    </div>
  );
}
