const { verifyToken } = require('../utils/jwt');
const User = require('../models/User');

async function authenticate (req, res, next) {
  try {
    const header = req.headers.authorization;
    let token = null;

    if (header && header.startsWith('Bearer ')) {
      token = header.substring(7);
    } else if (req.cookies?.token) {
      token = req.cookies.token;
    }

    if (!token) {
      return res.status(401).json({ message: 'Authentication required' });
    }

    const decoded = verifyToken(token);
    const user = await User.findById(decoded.sub).select('-password');

    if (!user) {
      return res.status(401).json({ message: 'User not found' });
    }

    req.user = user;
    next();
  } catch (error) {
    console.error('Auth error:', error);
    return res.status(401).json({ message: 'Invalid or expired token' });
  }
}

function authorize (...roles) {
  return (req, res, next) => {
    if (!req.user || !roles.includes(req.user.role)) {
      return res.status(403).json({ message: 'Forbidden' });
    }
    next();
  };
}

module.exports = {
  authenticate,
  authorize
};

