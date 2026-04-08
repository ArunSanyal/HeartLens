import { Link } from 'react-router-dom';
import './Home.css';

const FEATURES = [
  {
    icon: (
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="10" /><circle cx="12" cy="12" r="3" />
        <path d="M2 12h7M15 12h7M12 2v7M12 15v7" />
      </svg>
    ),
    title: 'Population Overview',
    desc: 'Explore all 920 patients via UMAP dimensionality reduction. Brush-select subsets, toggle between risk and hospital site color encodings.',
    color: '#0ea5e9',
    bg: '#f0f9ff',
  },
  {
    icon: (
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M3 3v18h18" /><path d="M7 16l4-8 4 4 4-10" />
      </svg>
    ),
    title: 'Feature Importance',
    desc: 'Global SHAP beeswarm plot showing per-feature value distributions and their impact. Click any feature to highlight patients.',
    color: '#8b5cf6',
    bg: '#f5f3ff',
  },
  {
    icon: (
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2" />
        <circle cx="9" cy="7" r="4" /><path d="M22 21v-2a4 4 0 0 0-3-3.87M16 3.13a4 4 0 0 1 0 7.75" />
      </svg>
    ),
    title: 'Patient Detail',
    desc: 'SHAP waterfall chart with LLM-generated clinical narrative. Includes a what-if editor to modify features and see live prediction changes.',
    color: '#f59e0b',
    bg: '#fffbeb',
  },
  {
    icon: (
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
      </svg>
    ),
    title: 'Clinical Query',
    desc: 'Ask natural-language questions about patients, features, or cohorts. The LLM responds with data-grounded answers and triggers view updates.',
    color: '#10b981',
    bg: '#ecfdf5',
  },
];

const STATS = [
  { value: '920', label: 'Patients', suffix: '' },
  { value: '4', label: 'Hospital Sites', suffix: '' },
  { value: '82.6', label: 'CV Accuracy', suffix: '%' },
  { value: '13', label: 'Clinical Features', suffix: '' },
];

const STEPS = [
  { num: '01', title: 'Data Ingestion', desc: 'UCI Heart Disease dataset from 4 international clinical sites is preprocessed with imputation and encoding.' },
  { num: '02', title: 'Model Training', desc: 'XGBoost gradient-boosted classifier trained with 5-fold stratified cross-validation and hyperparameter tuning.' },
  { num: '03', title: 'SHAP Explanations', desc: 'TreeSHAP computes exact per-feature attribution values for every prediction, supporting both global and local interpretability.' },
  { num: '04', title: 'Visual Analytics', desc: 'Four coordinated React + D3 views present visual, quantitative, and conversational explanations in one unified interface.' },
];

export default function Home() {
  return (
    <div className="home">
      {/* ── Hero ── */}
      <section className="hero">
        <div className="hero__bg" />
        <div className="container hero__inner">
          <div className="hero__content">
            <div className="hero__badge">
              <span className="hero__badge-dot" />
              CSC 696D &middot; Visual Analytics &middot; Spring 2026
            </div>
            <h1 className="hero__title">
              Understand Heart Disease <br />
              Predictions with <span className="hero__highlight">Explainable AI</span>
            </h1>
            <p className="hero__subtitle">
              HeartLens combines XGBoost predictions, SHAP explanations, and LLM-generated
              clinical narratives into one interactive visual analytics dashboard — making
              ML interpretability accessible to clinicians.
            </p>
            <div className="hero__actions">
              <Link to="/dashboard" className="btn btn--primary btn--lg">
                Launch Dashboard
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M5 12h14M12 5l7 7-7 7" />
                </svg>
              </Link>
              <Link to="/about" className="btn btn--outline btn--lg">
                Learn More
              </Link>
            </div>
          </div>
          <div className="hero__visual">
            <div className="hero__card hero__card--1">
              <div className="hero__card-icon" style={{ background: '#fee2e2' }}>
                <svg width="20" height="20" viewBox="0 0 24 24" fill="#ef4444">
                  <path d="M12 21s-8-5-8-11a5 5 0 0 1 8-4 5 5 0 0 1 8 4c0 6-8 11-8 11z" />
                </svg>
              </div>
              <div>
                <div className="hero__card-label">Patient #247</div>
                <div className="hero__card-value hero__card-value--high">High Risk — 87%</div>
              </div>
            </div>
            <div className="hero__card hero__card--2">
              <div className="hero__card-bar">
                <span style={{ width: '80%', background: '#ef4444' }} />
                <span style={{ width: '60%', background: '#3b82f6' }} />
                <span style={{ width: '45%', background: '#ef4444' }} />
                <span style={{ width: '30%', background: '#3b82f6' }} />
              </div>
              <div className="hero__card-label">SHAP Waterfall</div>
            </div>
            <div className="hero__card hero__card--3">
              <div className="hero__card-dots">
                {Array.from({ length: 24 }).map((_, i) => (
                  <span key={i} className="hero__dot" style={{
                    background: ['#22c55e', '#eab308', '#f97316', '#ef4444'][Math.floor(Math.random() * 4)],
                    left: `${10 + Math.random() * 80}%`,
                    top: `${10 + Math.random() * 80}%`,
                    opacity: 0.5 + Math.random() * 0.5,
                  }} />
                ))}
              </div>
              <div className="hero__card-label">UMAP Population</div>
            </div>
          </div>
        </div>
      </section>

      {/* ── Stats ── */}
      <section className="stats">
        <div className="container">
          <div className="stats__grid">
            {STATS.map((s, i) => (
              <div key={i} className="stats__item">
                <span className="stats__value">{s.value}<span className="stats__suffix">{s.suffix}</span></span>
                <span className="stats__label">{s.label}</span>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Features ── */}
      <section className="section section--alt" id="features">
        <div className="container">
          <div className="section__label">Dashboard Views</div>
          <h2 className="section__title">Four Coordinated Views, One Story</h2>
          <p className="section__subtitle">
            Interaction in any view propagates to all others in real time through
            shared React context state management.
          </p>
          <div className="features__grid">
            {FEATURES.map((f, i) => (
              <div key={i} className="feature-card">
                <div className="feature-card__icon" style={{ color: f.color, background: f.bg }}>
                  {f.icon}
                </div>
                <h3 className="feature-card__title">{f.title}</h3>
                <p className="feature-card__desc">{f.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── How It Works ── */}
      <section className="section" id="how-it-works">
        <div className="container">
          <div className="section__label">Pipeline</div>
          <h2 className="section__title">How It Works</h2>
          <p className="section__subtitle">
            From raw clinical data to interactive explainable predictions in four steps.
          </p>
          <div className="steps__grid">
            {STEPS.map((s, i) => (
              <div key={i} className="step-card">
                <div className="step-card__num">{s.num}</div>
                <h3 className="step-card__title">{s.title}</h3>
                <p className="step-card__desc">{s.desc}</p>
                {i < STEPS.length - 1 && <div className="step-card__connector" />}
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Tech Stack ── */}
      <section className="section section--alt">
        <div className="container">
          <div className="section__label">Technology</div>
          <h2 className="section__title">Built With</h2>
          <div className="tech__grid">
            {[
              { name: 'React', desc: 'Component-based UI with shared context state' },
              { name: 'D3.js', desc: 'Custom scatter, beeswarm, and waterfall charts' },
              { name: 'Flask', desc: 'REST API serving model, SHAP, and embeddings' },
              { name: 'XGBoost', desc: 'Gradient-boosted classifier with 82.6% accuracy' },
              { name: 'TreeSHAP', desc: 'Exact Shapley values for tree-based models' },
              { name: 'Claude API', desc: 'LLM narratives and conversational query engine' },
            ].map((t, i) => (
              <div key={i} className="tech-card">
                <h4 className="tech-card__name">{t.name}</h4>
                <p className="tech-card__desc">{t.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── CTA ── */}
      <section className="cta-section">
        <div className="container cta__inner">
          <h2 className="cta__title">Ready to Explore?</h2>
          <p className="cta__desc">
            Dive into 920 patient records across four hospital sites with real-time
            SHAP explanations and AI-powered clinical narratives.
          </p>
          <Link to="/dashboard" className="btn btn--white btn--lg">
            Open Dashboard
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M5 12h14M12 5l7 7-7 7" />
            </svg>
          </Link>
        </div>
      </section>
    </div>
  );
}
