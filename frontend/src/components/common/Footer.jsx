import { Link, useLocation } from 'react-router-dom';
import './Footer.css';

export default function Footer() {
  const location = useLocation();
  if (location.pathname === '/dashboard') return null;

  return (
    <footer className="footer">
      <div className="container">
        <div className="footer__grid">
          <div className="footer__brand">
            <div className="footer__logo-row">
              <svg width="28" height="28" viewBox="0 0 32 32" fill="none">
                <rect width="32" height="32" rx="8" fill="#2563eb" />
                <path d="M16 24s-7-4.5-7-10a4.5 4.5 0 0 1 7-3.5A4.5 4.5 0 0 1 23 14c0 5.5-7 10-7 10z" fill="#fff" />
              </svg>
              <span className="footer__name">HeartLens</span>
            </div>
            <p className="footer__desc">
              An LLM-augmented explainable AI visual analytics dashboard for heart disease prediction.
            </p>
            <p className="footer__university">
              University of Arizona &middot; CSC 696D Visual Analytics &middot; Spring 2026
            </p>
          </div>

          <div className="footer__col">
            <h4 className="footer__col-title">Navigation</h4>
            <Link to="/" className="footer__link">Home</Link>
            <Link to="/dashboard" className="footer__link">Dashboard</Link>
            <Link to="/about" className="footer__link">About</Link>
            <Link to="/contact" className="footer__link">Contact</Link>
          </div>

          <div className="footer__col">
            <h4 className="footer__col-title">Technology</h4>
            <span className="footer__link footer__link--static">React + D3.js</span>
            <span className="footer__link footer__link--static">Python Flask</span>
            <span className="footer__link footer__link--static">XGBoost + TreeSHAP</span>
            <span className="footer__link footer__link--static">Claude API</span>
          </div>

          <div className="footer__col">
            <h4 className="footer__col-title">Resources</h4>
            <a href="https://archive.ics.uci.edu/dataset/45/heart+disease" target="_blank" rel="noopener noreferrer" className="footer__link">UCI Dataset</a>
            <a href="https://shap.readthedocs.io/" target="_blank" rel="noopener noreferrer" className="footer__link">SHAP Docs</a>
            <a href="https://umap-learn.readthedocs.io/" target="_blank" rel="noopener noreferrer" className="footer__link">UMAP Docs</a>
          </div>
        </div>

        <div className="footer__bottom">
          <p>&copy; 2026 HeartLens. Built by Arun Sanyal & Chinmay Mhatre.</p>
        </div>
      </div>
    </footer>
  );
}
