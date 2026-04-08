import { usePatients } from '../../context/PatientContext';
import './Header.css';

export default function Header() {
  const { allPatients, selectedPatientIds, activePatient, clearSelection } = usePatients();
  const selected = selectedPatientIds.size;

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
