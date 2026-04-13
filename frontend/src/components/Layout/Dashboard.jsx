import { usePatients } from '../../context/PatientContext';
import Header from '../common/Header';
import PopulationScatter from '../views/PopulationScatter';
import FeatureImportance from '../views/FeatureImportance';
import PatientDetail from '../views/PatientDetail';
import ChatPanel from '../views/ChatPanel';
import './Dashboard.css';

export default function Dashboard() {
  const { loading, error, activePatient, clearSelection } = usePatients();

  if (loading) {
    return (
      <div className="hl-dashboard hl-dashboard--loading">
        <div className="hl-loading">
          <div className="hl-loading__spinner" />
          <p>Loading 920 patients from HeartLens API...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="hl-dashboard hl-dashboard--loading">
        <div className="hl-loading hl-loading--error">
          <p>Failed to connect to backend: {error}</p>
          <p className="hl-loading__hint">Make sure Flask is running: <code>cd backend && source venv/bin/activate && python app.py</code></p>
        </div>
      </div>
    );
  }

  const riskLevel  = activePatient
    ? activePatient.riskProb >= 0.7 ? 'high' : activePatient.riskProb >= 0.4 ? 'moderate' : 'low'
    : null;
  const riskLabel  = riskLevel === 'high' ? 'High Risk' : riskLevel === 'moderate' ? 'Moderate Risk' : 'Low Risk';
  const riskPct    = activePatient ? `${(activePatient.riskProb * 100).toFixed(0)}%` : '';
  const sex        = activePatient ? (activePatient.features.sex === 1 ? 'Male' : 'Female') : '';
  const age        = activePatient ? parseInt(activePatient.features.age) : '';

  return (
    <div className="hl-dashboard">
      <Header />

      {/* Patient banner — appears when a patient is selected */}
      {activePatient && (
        <div className={`hl-patient-banner hl-patient-banner--${riskLevel}`}>
          <div className="hl-patient-banner__info">
            <span className="hl-patient-banner__dot" />
            <span className="hl-patient-banner__id">Patient #{activePatient.id}</span>
            <span className="hl-patient-banner__sep">·</span>
            <span>{sex}, {age}</span>
            <span className="hl-patient-banner__sep">·</span>
            <span>{activePatient.site}</span>
            <span className="hl-patient-banner__risk-pill">{riskLabel} — {riskPct}</span>
          </div>
          <button className="hl-patient-banner__clear" onClick={clearSelection}>
            Clear
          </button>
        </div>
      )}

      <main className={`hl-dashboard__grid${activePatient ? ' hl-dashboard__grid--expanded' : ''}`}>
        <section className="hl-panel hl-panel--scatter">
          <div className="hl-panel__header">
            <h2 className="hl-panel__title">Population Overview</h2>
            <span className="hl-panel__badge">Patient Similarity Map</span>
          </div>
          <div className="hl-panel__body">
            <PopulationScatter />
          </div>
        </section>

        <section className="hl-panel hl-panel--beeswarm">
          <div className="hl-panel__header">
            <h2 className="hl-panel__title">Feature Importance</h2>
            <span className="hl-panel__badge">What Drives Risk</span>
          </div>
          <div className="hl-panel__body">
            <FeatureImportance />
          </div>
        </section>

        <section className="hl-panel hl-panel--detail">
          <div className="hl-panel__header">
            <h2 className="hl-panel__title">Patient Detail</h2>
            <span className="hl-panel__badge">Risk Breakdown</span>
          </div>
          <div className="hl-panel__body">
            <PatientDetail />
          </div>
        </section>

        <section className="hl-panel hl-panel--chat">
          <div className="hl-panel__header">
            <h2 className="hl-panel__title">Clinical Query</h2>
            <span className="hl-panel__badge">AI Assistant</span>
          </div>
          <div className="hl-panel__body hl-panel__body--chat">
            <ChatPanel />
          </div>
        </section>
      </main>
    </div>
  );
}
