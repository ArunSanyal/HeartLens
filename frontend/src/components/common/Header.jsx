import { useState, useMemo, useRef, useEffect } from 'react';
import { usePatients } from '../../context/PatientContext';
import './Header.css';

export default function Header() {
  const { allPatients, selectedPatientIds, activePatient, clearSelection, selectPatient } = usePatients();
  const selected = selectedPatientIds.size;

  const [query, setQuery]       = useState('');
  const [open, setOpen]         = useState(false);
  const searchRef               = useRef(null);

  // Search by ID, site, risk level, or age
  const results = useMemo(() => {
    const q = query.trim().toLowerCase().replace(/^#/, '');
    if (!q) return [];
    return allPatients.filter(p => {
      if (String(p.id).includes(q)) return true;
      if (p.site.toLowerCase().includes(q)) return true;
      const risk = p.riskProb >= 0.7 ? 'high' : p.riskProb >= 0.4 ? 'moderate' : 'low';
      if (risk.startsWith(q)) return true;
      if (String(parseInt(p.features.age)) === q) return true;
      return false;
    }).slice(0, 7);
  }, [allPatients, query]);

  // Close dropdown when clicking outside
  useEffect(() => {
    function handleClick(e) {
      if (searchRef.current && !searchRef.current.contains(e.target)) setOpen(false);
    }
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, []);

  function handleSelect(patient) {
    selectPatient(patient);
    setQuery('');
    setOpen(false);
  }

  function handleKey(e) {
    if (e.key === 'Escape') { setQuery(''); setOpen(false); }
  }

  const riskColor = p =>
    p.riskProb >= 0.7 ? '#ef4444' : p.riskProb >= 0.4 ? '#f59e0b' : '#22c55e';
  const riskWord  = p =>
    p.riskProb >= 0.7 ? 'High' : p.riskProb >= 0.4 ? 'Moderate' : 'Low';

  return (
    <header className="hl-header">
      <div className="hl-header__brand">
        <div className="hl-header__logo">
          <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
            <circle cx="16" cy="16" r="15" fill="#EF4444" opacity="0.12" />
            <path d="M16 26s-9-5.5-9-12a5.5 5.5 0 0 1 9-4.24A5.5 5.5 0 0 1 25 14c0 6.5-9 12-9 12z"
              fill="#EF4444" />
          </svg>
        </div>
        <div>
          <h1 className="hl-header__title">HeartLens</h1>
          <p className="hl-header__subtitle">Explainable AI Visual Analytics for Heart Disease Prediction</p>
        </div>
      </div>

      {/* ── Patient search ── */}
      <div className="hl-search" ref={searchRef}>
        <div className="hl-search__box">
          <svg className="hl-search__icon" width="14" height="14" viewBox="0 0 16 16" fill="none">
            <circle cx="6.5" cy="6.5" r="5" stroke="#94a3b8" strokeWidth="1.6"/>
            <path d="M10.5 10.5L14 14" stroke="#94a3b8" strokeWidth="1.6" strokeLinecap="round"/>
          </svg>
          <input
            className="hl-search__input"
            type="text"
            placeholder="Search by ID, site, age, or risk…"
            value={query}
            onChange={e => { setQuery(e.target.value); setOpen(true); }}
            onFocus={() => setOpen(true)}
            onKeyDown={handleKey}
          />
          {query && (
            <button className="hl-search__clear" onClick={() => { setQuery(''); setOpen(false); }}>
              ×
            </button>
          )}
        </div>

        {open && results.length > 0 && (
          <ul className="hl-search__dropdown">
            {results.map(p => (
              <li key={p.id} className="hl-search__result" onMouseDown={() => handleSelect(p)}>
                <span className="hl-search__result-dot" style={{ background: riskColor(p) }} />
                <span className="hl-search__result-id">#{p.id}</span>
                <span className="hl-search__result-meta">
                  {p.features.sex === 1 ? 'Male' : 'Female'}, {parseInt(p.features.age)}
                  &nbsp;·&nbsp;{p.site}
                </span>
                <span className="hl-search__result-risk" style={{ color: riskColor(p) }}>
                  {riskWord(p)}
                </span>
              </li>
            ))}
            <li className="hl-search__hint">
              {results.length === 7 ? 'Showing first 7 matches — type more to narrow down' : `${results.length} patient${results.length > 1 ? 's' : ''} found`}
            </li>
          </ul>
        )}

        {open && query.trim() && results.length === 0 && (
          <ul className="hl-search__dropdown">
            <li className="hl-search__hint">No patients match "{query}"</li>
          </ul>
        )}
      </div>

      <div className="hl-header__stats">
        <div className="hl-header__stat">
          <span className="hl-header__stat-value">{allPatients.length}</span>
          <span className="hl-header__stat-label">Total Patients</span>
        </div>
        <div className="hl-header__stat">
          <span className="hl-header__stat-value hl-header__stat-value--accent">
            {selected || 'All'}
          </span>
          <span className="hl-header__stat-label">Selected</span>
        </div>
        {activePatient && (
          <div className="hl-header__stat">
            <span className="hl-header__stat-value hl-header__stat-value--active">
              #{activePatient.id}
            </span>
            <span className="hl-header__stat-label">Active Patient</span>
          </div>
        )}
        {(selected > 0 || activePatient) && (
          <button className="hl-header__clear" onClick={clearSelection}>
            Clear Selection
          </button>
        )}
      </div>
    </header>
  );
}
