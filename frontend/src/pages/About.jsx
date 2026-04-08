import { Link } from 'react-router-dom';
import './About.css';

const TEAM = [
  {
    name: 'Arun Sanyal',
    role: 'Backend & ML Pipeline',
    focus: 'XGBoost training, SHAP computation, Flask API, LLM prompt engineering',
  },
  {
    name: 'Chinmay Mhatre',
    role: 'Frontend & Visualization',
    focus: 'React scaffolding, D3 views, interaction design, cross-view coordination',
  },
];

const DATASET_FEATURES = [
  { name: 'age', type: 'Continuous', desc: 'Age in years' },
  { name: 'sex', type: 'Binary', desc: '1 = male, 0 = female' },
  { name: 'cp', type: 'Categorical (4)', desc: 'Chest pain type' },
  { name: 'trestbps', type: 'Continuous', desc: 'Resting blood pressure (mm Hg)' },
  { name: 'chol', type: 'Continuous', desc: 'Serum cholesterol (mg/dl)' },
  { name: 'fbs', type: 'Binary', desc: 'Fasting blood sugar > 120 mg/dl' },
  { name: 'restecg', type: 'Categorical (3)', desc: 'Resting ECG results' },
  { name: 'thalach', type: 'Continuous', desc: 'Maximum heart rate achieved' },
  { name: 'exang', type: 'Binary', desc: 'Exercise-induced angina' },
  { name: 'oldpeak', type: 'Continuous', desc: 'ST depression induced by exercise' },
  { name: 'slope', type: 'Categorical (3)', desc: 'Slope of peak exercise ST segment' },
  { name: 'ca', type: 'Continuous (0-3)', desc: 'Major vessels colored by fluoroscopy' },
  { name: 'thal', type: 'Categorical (3)', desc: 'Thalassemia type' },
  { name: 'target', type: 'Binary', desc: '0 = no disease, 1 = disease present' },
];

const REFERENCES = [
  { id: 1, text: 'Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. NeurIPS 30.' },
  { id: 2, text: 'Lundberg, S. M. et al. (2020). From local explanations to global understanding with explainable AI for trees. Nature Machine Intelligence, 2(1).' },
  { id: 3, text: 'Spinner, T. et al. (2020). explAIner: A visual analytics framework for interactive and explainable ML. IEEE TVCG, 26(1).' },
  { id: 4, text: 'Detrano, R. et al. (1989). International application of a new probability algorithm for the diagnosis of coronary artery disease. American Journal of Cardiology, 64(5).' },
];

export default function About() {
  return (
    <div className="about">
      {/* Hero */}
      <section className="about-hero">
        <div className="container">
          <div className="section__label">About the Project</div>
          <h1 className="about-hero__title">HeartLens</h1>
          <p className="about-hero__subtitle">
            An LLM-Augmented Explainable AI Visual Analytics Dashboard for Heart Disease Prediction
          </p>
        </div>
      </section>

      {/* Problem */}
      <section className="section">
        <div className="container container--narrow">
          <h2 className="section__title">The Problem</h2>
          <div className="about-text">
            <p>
              Machine learning models like gradient-boosted trees can achieve strong predictive accuracy for
              heart disease diagnosis, but they are essentially black boxes. If a clinician sees a prediction
              that says "high risk," there is no easy way to understand why the model made that decision, which
              features mattered most, or whether the reasoning makes sense from a medical standpoint.
            </p>
            <p>
              Explainability techniques like SHAP help by breaking down predictions into per-feature contributions.
              But SHAP outputs are still technical — waterfall charts and beeswarm plots require statistical
              background to read properly. A busy clinician looking at a SHAP chart might not immediately
              understand what it means that "oldpeak contributed +0.12 to the log-odds."
            </p>
          </div>
        </div>
      </section>

      {/* Research Questions */}
      <section className="section section--alt">
        <div className="container container--narrow">
          <h2 className="section__title">Research Questions</h2>
          <div className="about-rqs">
            <div className="about-rq">
              <span className="about-rq__num">RQ1</span>
              <p>Does combining LLM-generated narratives with SHAP visualizations improve interpretability compared to visual-only explanations?</p>
            </div>
            <div className="about-rq">
              <span className="about-rq__num">RQ2</span>
              <p>Can a conversational LLM interface enable non-technical users to perform meaningful exploratory analysis of ML predictions without requiring them to learn visual encodings directly?</p>
            </div>
          </div>
        </div>
      </section>

      {/* Team */}
      <section className="section">
        <div className="container">
          <div className="section__label">Team</div>
          <h2 className="section__title">Meet the Team</h2>
          <p className="section__subtitle">
            University of Arizona &middot; CSC 696D Visual Analytics &middot; Spring 2026
          </p>
          <div className="team__grid">
            {TEAM.map((t, i) => (
              <div key={i} className="team-card">
                <div className="team-card__avatar">
                  {t.name.split(' ').map(n => n[0]).join('')}
                </div>
                <h3 className="team-card__name">{t.name}</h3>
                <p className="team-card__role">{t.role}</p>
                <p className="team-card__focus">{t.focus}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Dataset */}
      <section className="section section--alt">
        <div className="container">
          <div className="section__label">Data</div>
          <h2 className="section__title">UCI Heart Disease Dataset</h2>
          <p className="section__subtitle">
            920 patient records from four international clinical sites with 14 clinical attributes.
          </p>

          <div className="about-sites">
            {['Cleveland Clinic', 'Hungarian Institute of Cardiology', 'University Hospital Zurich', 'VA Medical Center (Long Beach)'].map((s, i) => (
              <div key={i} className="about-site">
                <span className="about-site__dot" style={{ background: ['#3b82f6', '#8b5cf6', '#f59e0b', '#10b981'][i] }} />
                {s}
              </div>
            ))}
          </div>

          <div className="about-table-wrap">
            <table className="about-table">
              <thead>
                <tr>
                  <th>Feature</th>
                  <th>Type</th>
                  <th>Description</th>
                </tr>
              </thead>
              <tbody>
                {DATASET_FEATURES.map((f, i) => (
                  <tr key={i}>
                    <td><code>{f.name}</code></td>
                    <td>{f.type}</td>
                    <td>{f.desc}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>

      {/* References */}
      <section className="section">
        <div className="container container--narrow">
          <h2 className="section__title">Key References</h2>
          <div className="about-refs">
            {REFERENCES.map(r => (
              <div key={r.id} className="about-ref">
                <span className="about-ref__num">[{r.id}]</span>
                <p>{r.text}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="cta-section">
        <div className="container cta__inner">
          <h2 className="cta__title">Explore the Dashboard</h2>
          <p className="cta__desc">See HeartLens in action with real patient data and interactive SHAP explanations.</p>
          <Link to="/dashboard" className="btn btn--white btn--lg">
            Launch Dashboard
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M5 12h14M12 5l7 7-7 7" />
            </svg>
          </Link>
        </div>
      </section>
    </div>
  );
}
