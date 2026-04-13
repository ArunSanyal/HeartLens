import { useRef, useEffect, useState, useMemo } from 'react';
import * as d3 from 'd3';
import { usePatients } from '../../context/PatientContext';
import { FEATURES, FEATURE_LABELS, getGlobalFeatureImportance } from '../../data/mockData';
import './FeatureImportance.css';

const MARGIN = { top: 16, right: 56, bottom: 40, left: 172 };

export default function FeatureImportance() {
  const svgRef        = useRef(null);
  const containerRef  = useRef(null);
  const tooltipRef    = useRef(null);
  const [dims, setDims] = useState({ width: 500, height: 350 });
  const { selectedPatients, activeFeature, setActiveFeature } = usePatients();

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

  // Sort features by |mean SHAP| descending, cap at 10 to avoid label crowding
  const sortedFeatures = useMemo(
    () => getGlobalFeatureImportance(selectedPatients).map(d => d.feature).slice(0, 10),
    [selectedPatients]
  );

  // Pre-compute mean SHAP per feature
  const meanShap = useMemo(() => {
    const out = {};
    FEATURES.forEach(f => {
      out[f] = d3.mean(selectedPatients, p => p.shapValues[f]) ?? 0;
    });
    return out;
  }, [selectedPatients]);

  useEffect(() => {
    if (!svgRef.current) return;
    const { width, height } = dims;
    const innerW = width  - MARGIN.left - MARGIN.right;
    const innerH = height - MARGIN.top  - MARGIN.bottom;
    if (innerW <= 0 || innerH <= 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const g = svg.append('g')
      .attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

    // ── Scales ────────────────────────────────────────────────────────────────
    const absMax = d3.max(sortedFeatures, f => Math.abs(meanShap[f])) || 1;

    const xScale = d3.scaleLinear()
      .domain([-absMax * 1.15, absMax * 1.15])
      .range([0, innerW]);

    const yScale = d3.scaleBand()
      .domain(sortedFeatures)
      .range([0, innerH])
      .padding(0.22);

    const barH = yScale.bandwidth();

    // ── Light grid lines ──────────────────────────────────────────────────────
    const ticks = xScale.ticks(6);
    ticks.forEach(t => {
      g.append('line')
        .attr('x1', xScale(t)).attr('x2', xScale(t))
        .attr('y1', 0).attr('y2', innerH)
        .attr('stroke', '#f1f5f9').attr('stroke-width', 1);
    });

    // ── Zero line ─────────────────────────────────────────────────────────────
    g.append('line')
      .attr('x1', xScale(0)).attr('x2', xScale(0))
      .attr('y1', -6).attr('y2', innerH)
      .attr('stroke', '#94a3b8').attr('stroke-width', 1.5);

    // ── X axis ────────────────────────────────────────────────────────────────
    g.append('g')
      .attr('transform', `translate(0,${innerH})`)
      .call(d3.axisBottom(xScale).ticks(6).tickFormat(d3.format('.2f')))
      .call(gg => gg.select('.domain').attr('stroke', '#e2e8f0'))
      .call(gg => gg.selectAll('.tick line').remove())
      .call(gg => gg.selectAll('.tick text')
        .attr('fill', '#94a3b8').attr('font-size', '10px'));

    g.append('text')
      .attr('x', innerW / 2).attr('y', innerH + 34)
      .attr('text-anchor', 'middle')
      .attr('fill', '#94a3b8').attr('font-size', '10px')
      .text('Average impact on heart disease risk  (← reduces risk   |   increases risk →)');

    // ── Y axis ────────────────────────────────────────────────────────────────
    g.append('g')
      .call(d3.axisLeft(yScale).tickFormat(f => FEATURE_LABELS[f] || f))
      .call(gg => gg.select('.domain').remove())
      .call(gg => gg.selectAll('.tick line').remove())
      .call(gg => gg.selectAll('.tick text')
        .attr('fill', d => d === activeFeature ? '#3b82f6' : '#475569')
        .attr('font-size', '10px')
        .attr('font-weight', d => d === activeFeature ? '700' : '400')
        .style('cursor', 'pointer')
        .on('click', (_, f) => setActiveFeature(activeFeature === f ? null : f))
      );

    // ── Bars ──────────────────────────────────────────────────────────────────
    sortedFeatures.forEach(feature => {
      const val      = meanShap[feature];
      const positive = val >= 0;
      const color    = positive ? '#ef4444' : '#3b82f6';
      const isActive = feature === activeFeature;

      const barX = positive ? xScale(0) : xScale(val);
      const barW = Math.max(2, Math.abs(xScale(val) - xScale(0)));
      const barY = yScale(feature);

      // Subtle active-row highlight
      if (isActive) {
        g.append('rect')
          .attr('x', 0).attr('y', barY - yScale.step() * 0.14)
          .attr('width', innerW)
          .attr('height', yScale.step() * 1.28)
          .attr('fill', 'rgba(59,130,246,0.06)')
          .attr('rx', 4);
      }

      // Bar
      g.append('rect')
        .attr('x', barX)
        .attr('y', barY)
        .attr('width', barW)
        .attr('height', barH)
        .attr('fill', color)
        .attr('rx', 3)
        .attr('opacity', isActive ? 1 : 0.72);

      // Value label at bar tip
      const labelX = positive
        ? xScale(val) + 5
        : xScale(val) - 5;
      const anchor = positive ? 'start' : 'end';

      g.append('text')
        .attr('x', labelX)
        .attr('y', barY + barH / 2 + 3.5)
        .attr('text-anchor', anchor)
        .attr('fill', '#64748b')
        .attr('font-size', '9px')
        .text((val >= 0 ? '+' : '') + val.toFixed(3));

      // Invisible wide hit-target for tooltip/click
      g.append('rect')
        .attr('x', 0).attr('y', barY)
        .attr('width', innerW).attr('height', barH)
        .attr('fill', 'transparent')
        .style('cursor', 'pointer')
        .on('click', () => setActiveFeature(activeFeature === feature ? null : feature))
        .on('mouseover', function (event) {
          const direction = val >= 0 ? 'increases risk' : 'reduces risk';
          d3.select(tooltipRef.current)
            .style('display', 'block')
            .html(
              `<strong>${FEATURE_LABELS[feature] || feature}</strong><br/>` +
              `Average impact: <span style="color:${color}">${val >= 0 ? '+' : ''}${val.toFixed(3)}</span><br/>` +
              `<span style="opacity:0.7">${direction} on average</span>`
            );
        })
        .on('mousemove', function (event) {
          const rect = containerRef.current.getBoundingClientRect();
          d3.select(tooltipRef.current)
            .style('left', (event.clientX - rect.left + 14) + 'px')
            .style('top',  (event.clientY - rect.top  - 42) + 'px');
        })
        .on('mouseout', () =>
          d3.select(tooltipRef.current).style('display', 'none')
        );
    });

    // ── Legend ────────────────────────────────────────────────────────────────
    const legendG = svg.append('g')
      .attr('transform', `translate(${MARGIN.left + innerW / 2 - 90}, 2)`);

    [
      { color: '#ef4444', label: 'Increases risk' },
      { color: '#3b82f6', label: 'Reduces risk',  dx: 110 },
    ].forEach(({ color, label, dx = 0 }) => {
      legendG.append('rect')
        .attr('x', dx).attr('y', 0)
        .attr('width', 10).attr('height', 10)
        .attr('fill', color).attr('rx', 2).attr('opacity', 0.75);
      legendG.append('text')
        .attr('x', dx + 14).attr('y', 8.5)
        .attr('fill', '#64748b').attr('font-size', '10px')
        .text(label);
    });

  }, [dims, selectedPatients, sortedFeatures, meanShap, activeFeature, setActiveFeature]);

  return (
    <div className="beeswarm-container" ref={containerRef}>
      <svg ref={svgRef} width={dims.width} height={dims.height} />
      <div ref={tooltipRef} className="fi-tooltip" />
    </div>
  );
}
