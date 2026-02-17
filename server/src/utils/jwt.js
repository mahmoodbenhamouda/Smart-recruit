const jwt = require('jsonwebtoken');

const TOKEN_TTL = process.env.JWT_EXPIRES_IN || '1h';

function signToken (payload, options = {}) {
  const secret = process.env.JWT_SECRET || 'dev-secret-change-me';
  return jwt.sign(payload, secret, { expiresIn: TOKEN_TTL, ...options });
}

function verifyToken (token) {
  const secret = process.env.JWT_SECRET || 'dev-secret-change-me';
  return jwt.verify(token, secret);
}

module.exports = {
  signToken,
  verifyToken
};

