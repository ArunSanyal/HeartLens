import { useRef, useEffect, useState, useMemo } from 'react';
import * as d3 from 'd3';
import { usePatients } from '../../context/PatientContext';
import { SITES } from '../../data/mockData';
import './PopulationScatter.css';

const RISK_COLORS = ['#22c55e', '#84cc16', '#eab308', '#f97316', '#ef4444'];
const SITE_COLORS = { Cleveland: '#3b82f6', Hungary: '#8b5cf6', Switzerland: '#f59e0b', 'VA Long Beach': '#10b981' };

const MARGIN = { top: 20, right: 20, bottom: 40, left: 50 };

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

    // Axes
    g.append('g')
      .attr('transform', `translate(0,${innerH})`)
      .call(d3.axisBottom(xScale).ticks(6))
      .call(g => g.select('.domain').attr('stroke', '#cbd5e1'))
      .call(g => g.selectAll('.tick line').attr('stroke', '#e2e8f0'))
      .call(g => g.selectAll('.tick text').attr('fill', '#94a3b8').attr('font-size', '10px'));

    g.append('g')
      .call(d3.axisLeft(yScale).ticks(6))
      .call(g => g.select('.domain').attr('stroke', '#cbd5e1'))
      .call(g => g.selectAll('.tick line').attr('stroke', '#e2e8f0'))
      .call(g => g.selectAll('.tick text').attr('fill', '#94a3b8').attr('font-size', '10px'));

    // Axis labels
    g.append('text')
      .attr('x', innerW / 2).attr('y', innerH + 32)
      .attr('text-anchor', 'middle')
      .attr('fill', '#94a3b8').attr('font-size', '11px')
      .text('UMAP-1');

    g.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -innerH / 2).attr('y', -36)
      .attr('text-anchor', 'middle')
      .attr('fill', '#94a3b8').attr('font-size', '11px')
      .text('UMAP-2');

    const hasBrush = selectedPatientIds.size > 0;

    // Points
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
      .on('click', (event, d) => {
        event.stopPropagation();
        selectPatient(d);
      });

    // Tooltip
    const tooltip = d3.select(containerRef.current).select('.scatter-tooltip');

    points
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

    // Brush for lasso selection
    const brush = d3.brush()
      .extent([[0, 0], [innerW, innerH]])
      .on('end', (event) => {
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
      });

    g.append('g')
      .attr('class', 'brush')
      .call(brush);

  }, [dims, allPatients, selectedPatientIds, activePatient, colorMode, activeFeature, featureAbsShapExtent, riskColorScale, brushPatients, selectPatient]);

  return (
    <div className="scatter-container" ref={containerRef}>
      <div className="scatter-controls">
        <button
          className={`scatter-toggle ${colorMode === 'risk' ? 'active' : ''}`}
          onClick={toggleColorMode}
        >
          {colorMode === 'risk' ? 'Color: Risk Level' : 'Color: Hospital Site'}
        </button>
        {colorMode === 'risk' ? (
          <div className="scatter-legend">
            <span className="scatter-legend__label">Low</span>
            <div className="scatter-legend__gradient" />
            <span className="scatter-legend__label">High</span>
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
