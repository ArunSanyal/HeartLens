import { useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { usePatients } from '../context/PatientContext';
import PatientDetail from '../components/views/PatientDetail';
import ChatPanel from '../components/views/ChatPanel';
import './PatientDeepDive.css';

export default function PatientDeepDive() {
  const { id } = useParams();
  const { allPatients, activePatient, selectPatient } = usePatients();

  useEffect(() => {
    if (!id || allPatients.length === 0) return;
    const patientId = parseInt(id, 10);
    if (!activePatient || activePatient.id !== patientId) {
      const patient = allPatients.find(p => p.id === patientId);
      if (patient) selectPatient(patient);
    }
  }, [id, allPatients.length]);

  if (!activePatient) {
    return (
      <div className="pdp-loading">
        <div className="pdp-loading__spinner" />
        <p>Loading patient data…</p>
      </div>
    );
  }

  const riskLevel = activePatient.riskProb >= 0.7 ? 'high' : activePatient.riskProb >= 0.4 ? 'moderate' : 'low';
  const riskLabel = riskLevel === 'high' ? 'High Risk' : riskLevel === 'moderate' ? 'Moderate Risk' : 'Low Risk';
  const sex = activePatient.features.sex === 1 ? 'Male' : 'Female';
  const age = parseInt(activePatient.features.age);

  return (
    <div className="pdp-container">
      <header className="pdp-header">
        <Link to="/dashboard" className="pdp-back">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <path d="M19 12H5M12 5l-7 7 7 7" />
          </svg>
          Back to Dashboard
        </Link>

        <div className={`pdp-patient-info pdp-patient-info--${riskLevel}`}>
          <span className="pdp-patient-id">Patient #{activePatient.id}</span>
          <span className="pdp-sep">·</span>
          <span>{sex}, {age}</span>
          <span className="pdp-sep">·</span>
          <span>{activePatient.site}</span>
          <span className="pdp-risk-pill">{riskLabel} — {(activePatient.riskProb * 100).toFixed(0)}%</span>
        </div>

        <span className="pdp-page-title">In-Depth Patient Analysis</span>
      </header>

      <div className="pdp-panels">
        <section className="pdp-panel pdp-panel--detail">
          <div className="pdp-panel__header">
            <h2 className="pdp-panel__title">Patient Detail</h2>
            <span className="pdp-badge pdp-badge--detail">Risk Breakdown + What-If</span>
          </div>
          <div className="pdp-panel__body">
            <PatientDetail />
          </div>
        </section>

        <section className="pdp-panel pdp-panel--chat">
          <div className="pdp-panel__header">
            <h2 className="pdp-panel__title">Clinical Query</h2>
            <span className="pdp-badge pdp-badge--chat">AI Assistant</span>
          </div>
          <div className="pdp-panel__body pdp-panel__body--chat">
            <ChatPanel />
          </div>
        </section>
      </div>
    </div>
  );
}
