import { useRef, useEffect, useState, useMemo, useCallback } from 'react';
import * as d3 from 'd3';
import { usePatients } from '../../context/PatientContext';
import { FEATURE_LABELS, FEATURE_DESCRIPTIONS } from '../../data/mockData';
import { predictWhatIf } from '../../api';
import './PatientDetail.css';

const MARGIN = { top: 20, right: 20, bottom: 30, left: 140 };

export default function PatientDetail() {
  const waterfallRef = useRef(null);
  const containerRef = useRef(null);
  const [dims, setDims] = useState({ width: 500, height: 300 });
  const [whatIfValues, setWhatIfValues] = useState(null);
  const [showWhatIf, setShowWhatIf] = useState(false);
  const [whatIfResult, setWhatIfResult] = useState(null);
  const [whatIfLoading, setWhatIfLoading] = useState(false);
  const { activePatient, narrative, narrativeLoading } = usePatients();

  useEffect(() => {
    const observer = new ResizeObserver(entries => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        if (width > 0 && height > 0) setDims({ width, height });
      }
    });
    if (containerRef.current) observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, []);

  // Reset what-if when patient changes
  useEffect(() => {
    if (activePatient) {
      setWhatIfValues({ ...activePatient.features });
      setWhatIfResult(null);
      setShowWhatIf(false);
    }
  }, [activePatient]);

  // Sort SHAP contributions by absolute value (use what-if result if available)
  const sortedShap = useMemo(() => {
    if (whatIfResult) {
      return whatIfResult.shapValues.map(s => [s.feature, s.value]);
    }
    if (!activePatient) return [];
    return Object.entries(activePatient.shapValues)
      .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
  }, [activePatient, whatIfResult]);

  // Debounced what-if API call
  const whatIfTimer = useRef(null);
  const runWhatIf = useCallback((values) => {
    if (whatIfTimer.current) clearTimeout(whatIfTimer.current);
    whatIfTimer.current = setTimeout(() => {
      setWhatIfLoading(true);
      predictWhatIf(values)
        .then(res => {
          setWhatIfResult(res);
          setWhatIfLoading(false);
        })
        .catch(err => {
          console.error('What-if failed:', err);
          setWhatIfLoading(false);
        });
    }, 500);
  }, []);

  const handleWhatIfChange = (feature, value) => {
    const updated = { ...whatIfValues, [feature]: parseFloat(value) || 0 };
    setWhatIfValues(updated);
    runWhatIf(updated);
  };

  // Draw waterfall
  useEffect(() => {
    if (!waterfallRef.current || !activePatient || sortedShap.length === 0) return;
    const svg = d3.select(waterfallRef.current);
    svg.selectAll('*').remove();

    const chartHeight = Math.max(dims.height * 0.55, 200);
    const innerW = dims.width - MARGIN.left - MARGIN.right;
    const innerH = chartHeight - MARGIN.top - MARGIN.bottom;
    if (innerW <= 0 || innerH <= 0) return;

    svg.attr('height', chartHeight);

    const g = svg.append('g')
      .attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

    const base = activePatient.baseValue;
    const items = sortedShap.map(([f, v]) => ({ feature: f, value: v }));

    // Build cumulative waterfall
    let cumulative = base;
    const bars = items.map(d => {
      const start = cumulative;
      cumulative += d.value;
      return { ...d, start, end: cumulative };
    });

    const allVals = [base, cumulative, ...bars.flatMap(b => [b.start, b.end])];
    const xExtent = d3.extent(allVals);
    const xScale = d3.scaleLinear()
      .domain([xExtent[0] - 0.05, xExtent[1] + 0.05])
      .range([0, innerW]);

    const yScale = d3.scaleBand()
      .domain(items.map(d => d.feature))
      .range([0, innerH])
      .padding(0.25);

    // Base value line
    g.append('line')
      .attr('x1', xScale(base)).attr('x2', xScale(base))
      .attr('y1', -5).attr('y2', innerH + 5)
      .attr('stroke', '#94a3b8')
      .attr('stroke-dasharray', '4 3')
      .attr('stroke-width', 1);

    g.append('text')
      .attr('x', xScale(base)).attr('y', -8)
      .attr('text-anchor', 'middle')
      .attr('fill', '#94a3b8').attr('font-size', '9px')
      .text(`Base: ${base.toFixed(2)}`);

    // Bars
    bars.forEach(bar => {
      const x = xScale(Math.min(bar.start, bar.end));
      const w = Math.abs(xScale(bar.end) - xScale(bar.start));
      const isPositive = bar.value >= 0;

      g.append('rect')
        .attr('x', x)
        .attr('y', yScale(bar.feature))
        .attr('width', Math.max(w, 1))
        .attr('height', yScale.bandwidth())
        .attr('rx', 3)
        .attr('fill', isPositive ? '#ef4444' : '#3b82f6')
        .attr('opacity', 0.8);

      // Value label
      g.append('text')
        .attr('x', xScale(bar.end) + (isPositive ? 4 : -4))
        .attr('y', yScale(bar.feature) + yScale.bandwidth() / 2 + 3)
        .attr('text-anchor', isPositive ? 'start' : 'end')
        .attr('fill', '#475569')
        .attr('font-size', '9px')
        .attr('font-weight', '500')
        .text((isPositive ? '+' : '') + bar.value.toFixed(3));

      // Connector lines
      if (bars.indexOf(bar) < bars.length - 1) {
        const nextBar = bars[bars.indexOf(bar) + 1];
        g.append('line')
          .attr('x1', xScale(bar.end))
          .attr('x2', xScale(bar.end))
          .attr('y1', yScale(bar.feature) + yScale.bandwidth())
          .attr('y2', yScale(nextBar.feature))
          .attr('stroke', '#e2e8f0')
          .attr('stroke-dasharray', '2 2');
      }
    });

    // Final prediction line
    g.append('line')
      .attr('x1', xScale(cumulative)).attr('x2', xScale(cumulative))
      .attr('y1', -5).attr('y2', innerH + 5)
      .attr('stroke', '#1e293b')
      .attr('stroke-width', 1.5);

    g.append('text')
      .attr('x', xScale(cumulative)).attr('y', innerH + 18)
      .attr('text-anchor', 'middle')
      .attr('fill', '#1e293b').attr('font-size', '10px').attr('font-weight', '600')
      .text(`Prediction: ${cumulative.toFixed(2)}`);

    // Y axis
    g.append('g')
      .call(d3.axisLeft(yScale).tickFormat(f => FEATURE_LABELS[f] || f))
      .call(g => g.select('.domain').remove())
      .call(g => g.selectAll('.tick line').remove())
      .call(g => g.selectAll('.tick text').attr('fill', '#475569').attr('font-size', '10px'));

    // X axis
    g.append('g')
      .attr('transform', `translate(0,${innerH})`)
      .call(d3.axisBottom(xScale).ticks(5).tickFormat(d3.format('.2f')))
      .call(g => g.select('.domain').attr('stroke', '#cbd5e1'))
      .call(g => g.selectAll('.tick line').attr('stroke', '#e2e8f0'))
      .call(g => g.selectAll('.tick text').attr('fill', '#94a3b8').attr('font-size', '9px'));

  }, [activePatient, sortedShap, dims]);

  if (!activePatient) {
    return (
      <div className="detail-empty">
        <div className="detail-empty__icon">
          <svg width="48" height="48" viewBox="0 0 48 48" fill="none">
            <circle cx="24" cy="24" r="22" stroke="#e2e8f0" strokeWidth="2" fill="#f8fafc" />
            <path d="M24 14v10M24 28v2" stroke="#94a3b8" strokeWidth="2" strokeLinecap="round" />
          </svg>
        </div>
        <p className="detail-empty__text">Select a patient from the scatter plot to view their detailed analysis</p>
        <p className="detail-empty__hint">Click on any point or brush to select a group</p>
      </div>
    );
  }

  const risk = activePatient.riskProb >= 0.7 ? 'High' : activePatient.riskProb >= 0.4 ? 'Moderate' : 'Low';
  const riskColor = activePatient.riskProb >= 0.7 ? '#ef4444' : activePatient.riskProb >= 0.4 ? '#f59e0b' : '#22c55e';

  return (
    <div className="detail-container" ref={containerRef}>
      {/* Patient header */}
      <div className="detail-patient-header">
        <div className="detail-patient-info">
          <span className="detail-patient-id">Patient #{activePatient.id}</span>
          <span className="detail-patient-site">{activePatient.site}</span>
        </div>
        <div className="detail-risk-badge" style={{ background: riskColor + '18', color: riskColor, borderColor: riskColor + '40' }}>
          {risk} Risk ({(activePatient.riskProb * 100).toFixed(0)}%)
        </div>
        <button
          className={`detail-whatif-btn ${showWhatIf ? 'active' : ''}`}
          onClick={() => setShowWhatIf(!showWhatIf)}
        >
          {showWhatIf ? 'Hide' : 'What-If Editor'}
        </button>
      </div>

      <div className="detail-content">
        {/* Waterfall chart */}
        <div className="detail-waterfall">
          <svg ref={waterfallRef} width={dims.width} />
        </div>

        {/* LLM Narrative */}
        <div className="detail-narrative">
          <div className="detail-narrative__label">
            AI Clinical Narrative
            {narrativeLoading && <span className="detail-narrative__loading"> Generating...</span>}
          </div>
          <div className="detail-narrative__text">{narrative || 'Loading narrative...'}</div>
          {whatIfResult && (
            <div className="detail-narrative__whatif-risk">
              What-If Prediction: <strong>{(whatIfResult.riskProb * 100).toFixed(0)}%</strong>
              {whatIfLoading && ' (updating...)'}
            </div>
          )}
        </div>

        {/* What-if editor */}
        {showWhatIf && whatIfValues && (
          <div className="detail-whatif">
            <div className="detail-whatif__title">What-If Feature Editor</div>
            <div className="detail-whatif__grid">
              {Object.entries(whatIfValues).map(([feature, value]) => (
                <div key={feature} className="detail-whatif__row">
                  <label className="detail-whatif__label">{FEATURE_LABELS[feature]}</label>
                  <input
                    className="detail-whatif__input"
                    type="number"
                    value={value}
                    onChange={e => handleWhatIfChange(feature, e.target.value)}
                  />
                  <span className="detail-whatif__orig">
                    orig: {activePatient.features[feature]}
                  </span>
                </div>
              ))}
            </div>
            <button
              className="detail-whatif__reset"
              onClick={() => {
                setWhatIfValues({ ...activePatient.features });
                setWhatIfResult(null);
              }}
            >
              Reset to Original
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
