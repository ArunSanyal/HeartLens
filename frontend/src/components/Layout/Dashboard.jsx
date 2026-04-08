import { usePatients } from '../../context/PatientContext';
import Header from '../common/Header';
import PopulationScatter from '../views/PopulationScatter';
import FeatureImportance from '../views/FeatureImportance';
import PatientDetail from '../views/PatientDetail';
import ChatPanel from '../views/ChatPanel';
import './Dashboard.css';

export default function Dashboard() {
  const { loading, error } = usePatients();

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

  return (
    <div className="hl-dashboard">
      <Header />
      <main className="hl-dashboard__grid">
        <section className="hl-panel hl-panel--scatter">
          <div className="hl-panel__header">
            <h2 className="hl-panel__title">Population Overview</h2>
            <span className="hl-panel__badge">UMAP Projection</span>
          </div>
          <div className="hl-panel__body">
            <PopulationScatter />
          </div>
        </section>

        <section className="hl-panel hl-panel--beeswarm">
          <div className="hl-panel__header">
            <h2 className="hl-panel__title">Feature Importance</h2>
            <span className="hl-panel__badge">SHAP Values</span>
          </div>
          <div className="hl-panel__body">
            <FeatureImportance />
          </div>
        </section>

        <section className="hl-panel hl-panel--detail">
          <div className="hl-panel__header">
            <h2 className="hl-panel__title">Patient Detail</h2>
            <span className="hl-panel__badge">Waterfall + Narrative</span>
          </div>
          <div className="hl-panel__body">
            <PatientDetail />
          </div>
        </section>

        <section className="hl-panel hl-panel--chat">
          <div className="hl-panel__header">
            <h2 className="hl-panel__title">Clinical Query</h2>
            <span className="hl-panel__badge">LLM Assistant</span>
          </div>
          <div className="hl-panel__body hl-panel__body--chat">
            <ChatPanel />
          </div>
        </section>
      </main>
    </div>
  );
}
