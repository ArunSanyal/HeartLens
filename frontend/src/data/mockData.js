// Mock data generator for HeartLens
// Generates 920 patients across 4 hospital sites with realistic clinical features,
// SHAP values, UMAP embeddings, and risk predictions.

import { randomNormal, randomUniform, randomInt } from 'd3';

const SITES = ['Cleveland', 'Hungary', 'Switzerland', 'VA Long Beach'];
const SITE_COUNTS = [303, 294, 123, 200];

const FEATURES = [
  'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
  'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
];

const FEATURE_LABELS = {
  age: 'Age',
  sex: 'Sex',
  cp: 'Chest Pain Type',
  trestbps: 'Resting Blood Pressure',
  chol: 'Serum Cholesterol',
  fbs: 'Fasting Blood Sugar',
  restecg: 'Resting ECG',
  thalach: 'Max Heart Rate',
  exang: 'Exercise Angina',
  oldpeak: 'ST Depression (Oldpeak)',
  slope: 'ST Slope',
  ca: 'Major Vessels (Ca)',
  thal: 'Thalassemia'
};

const FEATURE_DESCRIPTIONS = {
  age: 'Age in years',
  sex: '1 = male, 0 = female',
  cp: 'Chest pain type (0: typical angina, 1: atypical angina, 2: non-anginal, 3: asymptomatic)',
  trestbps: 'Resting blood pressure (mm Hg)',
  chol: 'Serum cholesterol (mg/dl)',
  fbs: 'Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)',
  restecg: 'Resting ECG results (0: normal, 1: ST-T abnormality, 2: LV hypertrophy)',
  thalach: 'Maximum heart rate achieved',
  exang: 'Exercise-induced angina (1 = yes, 0 = no)',
  oldpeak: 'ST depression induced by exercise relative to rest',
  slope: 'Slope of peak exercise ST segment (0: upsloping, 1: flat, 2: downsloping)',
  ca: 'Number of major vessels colored by fluoroscopy (0-3)',
  thal: 'Thalassemia (1: normal, 2: fixed defect, 3: reversible defect)'
};

// Seeded pseudo-random for reproducibility
function seededRandom(seed) {
  let s = seed;
  return function () {
    s = (s * 16807 + 0) % 2147483647;
    return (s - 1) / 2147483646;
  };
}

function generatePatients() {
  const rng = seededRandom(42);
  const patients = [];
  let id = 0;

  for (let siteIdx = 0; siteIdx < SITES.length; siteIdx++) {
    const site = SITES[siteIdx];
    const count = SITE_COUNTS[siteIdx];

    for (let i = 0; i < count; i++) {
      const age = Math.round(30 + rng() * 45); // 30-75
      const sex = rng() > 0.32 ? 1 : 0;
      const cp = Math.floor(rng() * 4);
      const trestbps = Math.round(100 + rng() * 80); // 100-180
      const chol = Math.round(120 + rng() * 300); // 120-420
      const fbs = rng() > 0.85 ? 1 : 0;
      const restecg = Math.floor(rng() * 3);
      const thalach = Math.round(80 + rng() * 120); // 80-200
      const exang = rng() > 0.67 ? 1 : 0;
      const oldpeak = Math.round((rng() * 5) * 10) / 10; // 0-5
      const slope = Math.floor(rng() * 3);
      const ca = Math.floor(rng() * 4);
      const thal = Math.floor(1 + rng() * 3);

      // Generate a risk score based on realistic clinical correlations
      let riskLogit = -1.5;
      riskLogit += (cp === 0) ? 0.8 : (cp === 3 ? -0.5 : 0.1);
      riskLogit += exang * 0.7;
      riskLogit += oldpeak * 0.3;
      riskLogit += (ca * 0.5);
      riskLogit += (thal === 3) ? 0.6 : (thal === 2 ? 0.3 : -0.4);
      riskLogit += (age - 54) * 0.02;
      riskLogit += (thalach < 140) ? 0.3 : -0.2;
      riskLogit += sex * 0.2;
      riskLogit += (trestbps > 140) ? 0.2 : -0.1;
      riskLogit += (chol > 250) ? 0.15 : -0.05;
      riskLogit += (rng() - 0.5) * 0.8; // noise

      const riskProb = 1 / (1 + Math.exp(-riskLogit));
      const target = riskProb > 0.5 ? 1 : 0;

      // Generate SHAP values correlated with feature contributions
      const baseValue = 0.42;
      const shapValues = {
        age: ((age - 54) * 0.005 + (rng() - 0.5) * 0.05),
        sex: (sex === 1 ? 0.06 : -0.06) + (rng() - 0.5) * 0.03,
        cp: (cp === 0 ? 0.18 : cp === 3 ? -0.12 : 0.02) + (rng() - 0.5) * 0.04,
        trestbps: ((trestbps - 131) * 0.001 + (rng() - 0.5) * 0.03),
        chol: ((chol - 246) * 0.0004 + (rng() - 0.5) * 0.03),
        fbs: (fbs === 1 ? 0.03 : -0.01) + (rng() - 0.5) * 0.02,
        restecg: (restecg * 0.02 + (rng() - 0.5) * 0.02),
        thalach: ((150 - thalach) * 0.002 + (rng() - 0.5) * 0.04),
        exang: (exang === 1 ? 0.15 : -0.05) + (rng() - 0.5) * 0.03,
        oldpeak: (oldpeak * 0.06 + (rng() - 0.5) * 0.03),
        slope: (slope * 0.04 + (rng() - 0.5) * 0.03),
        ca: (ca * 0.1 + (rng() - 0.5) * 0.04),
        thal: (thal === 3 ? 0.12 : thal === 2 ? 0.05 : -0.08) + (rng() - 0.5) * 0.03,
      };

      // Round SHAP values
      Object.keys(shapValues).forEach(k => {
        shapValues[k] = Math.round(shapValues[k] * 1000) / 1000;
      });

      // UMAP embeddings - cluster by risk with site offsets
      const angle = rng() * Math.PI * 2;
      const radius = 1.5 + rng() * 2.5;
      const riskOffset = riskProb * 6;
      const siteOffsetX = [0, 3, -2, 1][siteIdx] * 0.8;
      const siteOffsetY = [0, -2, 2, -1][siteIdx] * 0.8;
      const umapX = Math.cos(angle) * radius + riskOffset + siteOffsetX + (rng() - 0.5) * 2;
      const umapY = Math.sin(angle) * radius + (rng() - 0.5) * 4 + siteOffsetY;

      patients.push({
        id: id++,
        site,
        features: { age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal },
        riskProb: Math.round(riskProb * 100) / 100,
        target,
        shapValues,
        baseValue,
        umapX: Math.round(umapX * 100) / 100,
        umapY: Math.round(umapY * 100) / 100,
      });

      id++;
    }
  }

  // Re-index IDs sequentially
  patients.forEach((p, i) => { p.id = i; });
  return patients;
}

export const patients = generatePatients();
export { FEATURES, FEATURE_LABELS, FEATURE_DESCRIPTIONS, SITES };

// Compute global feature importance (mean |SHAP|)
export function getGlobalFeatureImportance(patientSubset = patients) {
  const importance = {};
  FEATURES.forEach(f => {
    const vals = patientSubset.map(p => Math.abs(p.shapValues[f]));
    importance[f] = vals.reduce((a, b) => a + b, 0) / vals.length;
  });
  return Object.entries(importance)
    .sort((a, b) => b[1] - a[1])
    .map(([feature, mean]) => ({ feature, mean: Math.round(mean * 1000) / 1000 }));
}

// Generate a mock LLM narrative for a patient
export function generateNarrative(patient) {
  const p = patient;
  const risk = p.riskProb >= 0.7 ? 'high' : p.riskProb >= 0.4 ? 'moderate' : 'low';
  const topShap = Object.entries(p.shapValues)
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    .slice(0, 4);

  const drivers = topShap
    .filter(([, v]) => v > 0)
    .map(([f, v]) => `${FEATURE_LABELS[f]} (contributing +${v.toFixed(2)} to risk)`)
    .join(', ');

  const protectors = topShap
    .filter(([, v]) => v < 0)
    .map(([f, v]) => `${FEATURE_LABELS[f]} (${v.toFixed(2)})`)
    .join(', ');

  const ageText = p.features.age;
  const sexText = p.features.sex === 1 ? 'male' : 'female';

  let narrative = `This ${ageText}-year-old ${sexText} patient was classified as **${risk}-risk** (predicted probability: ${p.riskProb.toFixed(2)}).`;

  if (drivers) {
    narrative += ` The primary risk drivers were ${drivers}.`;
  }
  if (protectors) {
    narrative += ` Protective factors included ${protectors}.`;
  }

  narrative += ` Resting blood pressure was ${p.features.trestbps} mmHg, serum cholesterol ${p.features.chol} mg/dl, and maximum heart rate ${p.features.thalach} bpm.`;

  return narrative;
}
