import { useRef, useEffect, useState, useMemo } from 'react';
import * as d3 from 'd3';
import { usePatients } from '../../context/PatientContext';
import { SITES, FEATURE_LABELS } from '../../data/mockData';
import './PopulationScatter.css';

const RISK_COLORS = ['#22c55e', '#84cc16', '#eab308', '#f97316', '#ef4444'];
const SITE_COLORS = { Cleveland: '#3b82f6', Hungary: '#8b5cf6', Switzerland: '#f59e0b', 'VA Long Beach': '#10b981' };

const MARGIN = { top: 24, right: 16, bottom: 56, left: 40 };

export default function PopulationScatter() {
  const svgRef = useRef(null);
  const containerRef = useRef(null);
  const [dims, setDims] = useState({ width: 500, height: 350 });
  const {
    allPatients, selectedPatientIds, activePatient, colorMode,
    brushPatients, selectPatient, toggleColorMode, activeFeature
  } = usePatients();

  // Responsive sizing
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

  const riskColorScale = useMemo(() =>
    d3.scaleQuantize().domain([0, 1]).range(RISK_COLORS),
    []
  );

  const featureAbsShapExtent = useMemo(() => {
    if (!activeFeature) return null;
    return d3.extent(allPatients, p => Math.abs(p.shapValues[activeFeature]));
  }, [activeFeature, allPatients]);

  useEffect(() => {
    if (!svgRef.current) return;
    const { width, height } = dims;
    const innerW = width - MARGIN.left - MARGIN.right;
    const innerH = height - MARGIN.top - MARGIN.bottom;
    if (innerW <= 0 || innerH <= 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const xScale = d3.scaleLinear()
      .domain(d3.extent(allPatients, d => d.umapX)).nice()
      .range([0, innerW]);
    const yScale = d3.scaleLinear()
      .domain(d3.extent(allPatients, d => d.umapY)).nice()
      .range([innerH, 0]);

    const g = svg.append('g')
      .attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

    // Subtle background grid only — no axis numbers (UMAP coords are meaningless)
    const xTicks = xScale.ticks(5);
    const yTicks = yScale.ticks(5);
    xTicks.forEach(t => {
      g.append('line')
        .attr('x1', xScale(t)).attr('x2', xScale(t))
        .attr('y1', 0).attr('y2', innerH)
        .attr('stroke', '#f1f5f9').attr('stroke-width', 1);
    });
    yTicks.forEach(t => {
      g.append('line')
        .attr('x1', 0).attr('x2', innerW)
        .attr('y1', yScale(t)).attr('y2', yScale(t))
        .attr('stroke', '#f1f5f9').attr('stroke-width', 1);
    });

    // ── Cluster zone annotations (risk mode only) ─────────────────────────
    if (colorMode === 'risk') {
      const groups = {
        high:     { patients: allPatients.filter(p => p.riskProb >= 0.7),  color: '#ef4444', label: 'Higher Risk' },
        moderate: { patients: allPatients.filter(p => p.riskProb >= 0.4 && p.riskProb < 0.7), color: '#f59e0b', label: 'Moderate Risk' },
        low:      { patients: allPatients.filter(p => p.riskProb < 0.4),   color: '#22c55e', label: 'Lower Risk' },
      };
      Object.values(groups).forEach(({ patients, color, label }) => {
        if (patients.length < 5) return;
        const cx = d3.mean(patients, p => xScale(p.umapX));
        const cy = d3.mean(patients, p => yScale(p.umapY));
        // Soft pill background
        const tw = label.length * 6.4 + 16;
        g.append('rect')
          .attr('x', cx - tw / 2).attr('y', cy - 18)
          .attr('width', tw).attr('height', 18)
          .attr('rx', 9)
          .attr('fill', color).attr('opacity', 0.12)
          .attr('pointer-events', 'none');
        g.append('text')
          .attr('x', cx).attr('y', cy - 5)
          .attr('text-anchor', 'middle')
          .attr('fill', color).attr('opacity', 0.75)
          .attr('font-size', '10.5px').attr('font-weight', '700')
          .attr('pointer-events', 'none')
          .text(label);
      });
    }

    // ── Site cluster labels (site mode only) ──────────────────────────────
    if (colorMode === 'site') {
      SITES.forEach(site => {
        const pts = allPatients.filter(p => p.site === site);
        if (pts.length < 3) return;
        const cx = d3.mean(pts, p => xScale(p.umapX));
        const cy = d3.mean(pts, p => yScale(p.umapY));
        const color = SITE_COLORS[site] || '#94a3b8';
        const tw = site.length * 6.4 + 16;
        g.append('rect')
          .attr('x', cx - tw / 2).attr('y', cy - 18)
          .attr('width', tw).attr('height', 18)
          .attr('rx', 9)
          .attr('fill', color).attr('opacity', 0.12)
          .attr('pointer-events', 'none');
        g.append('text')
          .attr('x', cx).attr('y', cy - 5)
          .attr('text-anchor', 'middle')
          .attr('fill', color).attr('opacity', 0.8)
          .attr('font-size', '10px').attr('font-weight', '700')
          .attr('pointer-events', 'none')
          .text(site);
      });
    }

    // ── Axis labels ───────────────────────────────────────────────────────
    g.append('text')
      .attr('x', innerW / 2).attr('y', innerH + 20)
      .attr('text-anchor', 'middle')
      .attr('fill', '#94a3b8').attr('font-size', '10px').attr('font-weight', '500')
      .text('UMAP Dimension 1');

    g.append('text')
      .attr('transform', `translate(-30, ${innerH / 2}) rotate(-90)`)
      .attr('text-anchor', 'middle')
      .attr('fill', '#94a3b8').attr('font-size', '10px').attr('font-weight', '500')
      .text('UMAP Dimension 2');

    // ── Bottom caption ────────────────────────────────────────────────────
    const caption = activeFeature
      ? `Dot size = how strongly "${FEATURE_LABELS[activeFeature] || activeFeature}" affects each patient's risk`
      : 'Each dot is one patient — patients closer together have more similar health profiles';

    g.append('text')
      .attr('x', innerW / 2).attr('y', innerH + 40)
      .attr('text-anchor', 'middle')
      .attr('fill', '#94a3b8').attr('font-size', '9.5px')
      .text(caption);

    const hasBrush = selectedPatientIds.size > 0;

    // Brush FIRST (behind dots) so dots receive clicks
    const brush = d3.brush()
      .extent([[0, 0], [innerW, innerH]])
      .on('end', (event) => {
        if (!event.sourceEvent) return; // ignore programmatic clears
        if (!event.selection) {
          brushPatients([]);
          return;
        }
        const [[x0, y0], [x1, y1]] = event.selection;
        const ids = allPatients
          .filter(d => {
            const px = xScale(d.umapX);
            const py = yScale(d.umapY);
            return px >= x0 && px <= x1 && py >= y0 && py <= y1;
          })
          .map(d => d.id);
        brushPatients(ids);
        // Clear the brush rectangle after selection
        g.select('.brush').call(brush.move, null);
      });

    g.append('g')
      .attr('class', 'brush')
      .call(brush);

    // Points ON TOP of brush so they receive click events
    const tooltip = d3.select(containerRef.current).select('.scatter-tooltip');

    const points = g.selectAll('.dot')
      .data(allPatients)
      .enter()
      .append('circle')
      .attr('class', 'dot')
      .attr('cx', d => xScale(d.umapX))
      .attr('cy', d => yScale(d.umapY))
      .attr('r', d => {
        if (activePatient && d.id === activePatient.id) return 7;
        if (activeFeature && featureAbsShapExtent) {
          const norm = Math.abs(d.shapValues[activeFeature]) / (featureAbsShapExtent[1] || 1);
          return 2.5 + norm * 5;
        }
        return 3.5;
      })
      .attr('fill', d => {
        if (colorMode === 'site') return SITE_COLORS[d.site] || '#94a3b8';
        return riskColorScale(d.riskProb);
      })
      .attr('stroke', d => {
        if (activePatient && d.id === activePatient.id) return '#1e293b';
        return 'rgba(255,255,255,0.7)';
      })
      .attr('stroke-width', d => activePatient && d.id === activePatient.id ? 2.5 : 0.5)
      .attr('opacity', d => {
        if (hasBrush && !selectedPatientIds.has(d.id)) return 0.1;
        return 0.75;
      })
      .style('cursor', 'pointer')
      .style('pointer-events', 'all')
      .on('click', (event, d) => {
        event.stopPropagation();
        selectPatient(d);
      })
      .on('mouseenter', (event, d) => {
        const risk = d.riskProb >= 0.7 ? 'High' : d.riskProb >= 0.4 ? 'Moderate' : 'Low';
        tooltip
          .style('opacity', 1)
          .style('left', `${event.offsetX + 14}px`)
          .style('top', `${event.offsetY - 10}px`)
          .html(`
            <strong>Patient #${d.id}</strong><br/>
            <span style="color:#64748b">Site:</span> ${d.site}<br/>
            <span style="color:#64748b">Risk:</span> ${risk} (${(d.riskProb * 100).toFixed(0)}%)<br/>
            <span style="color:#64748b">Age:</span> ${d.features.age} | <span style="color:#64748b">Sex:</span> ${d.features.sex === 1 ? 'M' : 'F'}
          `);
      })
      .on('mouseleave', () => tooltip.style('opacity', 0));

  }, [dims, allPatients, selectedPatientIds, activePatient, colorMode, activeFeature, featureAbsShapExtent, riskColorScale, brushPatients, selectPatient]);

  return (
    <div className="scatter-container" ref={containerRef}>
      <div className="scatter-controls">
        <button
          className={`scatter-toggle ${colorMode === 'risk' ? 'active' : ''}`}
          onClick={toggleColorMode}
        >
          {colorMode === 'risk' ? 'Colour by Risk' : 'Colour by Hospital'}
        </button>

        {colorMode === 'risk' ? (
          <div className="scatter-legend scatter-legend--risk">
            {[
              { color: '#22c55e', label: 'Low Risk'      },
              { color: '#f59e0b', label: 'Moderate Risk' },
              { color: '#ef4444', label: 'High Risk'     },
            ].map(({ color, label }) => (
              <span key={label} className="scatter-legend__pill">
                <span className="scatter-legend__dot" style={{ background: color }} />
                {label}
              </span>
            ))}
          </div>
        ) : (
          <div className="scatter-legend scatter-legend--sites">
            {SITES.map(s => (
              <span key={s} className="scatter-legend__site">
                <span className="scatter-legend__dot" style={{ background: SITE_COLORS[s] }} />
                {s}
              </span>
            ))}
          </div>
        )}
      </div>
      <svg ref={svgRef} width={dims.width} height={dims.height} />
      <div className="scatter-tooltip" />
    </div>
  );
}
