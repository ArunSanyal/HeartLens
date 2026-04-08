import { useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import './Navbar.css';

export default function Navbar() {
  const location = useLocation();
  const [scrolled, setScrolled] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);

  const isDashboard = location.pathname === '/dashboard';

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 20);
    window.addEventListener('scroll', onScroll);
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  // Don't render on dashboard — it has its own header
  if (isDashboard) return null;

  const links = [
    { to: '/', label: 'Home' },
    { to: '/about', label: 'About' },
    { to: '/contact', label: 'Contact' },
  ];

  return (
    <nav className={`navbar ${scrolled ? 'navbar--scrolled' : ''}`}>
      <div className="navbar__inner container">
        <Link to="/" className="navbar__brand">
          <svg className="navbar__logo" width="32" height="32" viewBox="0 0 32 32" fill="none">
            <rect width="32" height="32" rx="8" fill="#2563eb" />
            <path d="M16 24s-7-4.5-7-10a4.5 4.5 0 0 1 7-3.5A4.5 4.5 0 0 1 23 14c0 5.5-7 10-7 10z" fill="#fff" />
          </svg>
          <span className="navbar__name">HeartLens</span>
        </Link>

        <div className={`navbar__links ${mobileOpen ? 'navbar__links--open' : ''}`}>
          {links.map(l => (
            <Link
              key={l.to}
              to={l.to}
              className={`navbar__link ${location.pathname === l.to ? 'navbar__link--active' : ''}`}
              onClick={() => setMobileOpen(false)}
            >
              {l.label}
            </Link>
          ))}
          <Link to="/dashboard" className="btn btn--primary navbar__cta" onClick={() => setMobileOpen(false)}>
            Open Dashboard
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M5 12h14M12 5l7 7-7 7" />
            </svg>
          </Link>
        </div>

        <button className="navbar__hamburger" onClick={() => setMobileOpen(!mobileOpen)} aria-label="Toggle menu">
          <span /><span /><span />
        </button>
      </div>
    </nav>
  );
}
