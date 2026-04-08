import { useRef, useEffect, useState, useMemo } from 'react';
import * as d3 from 'd3';
import { usePatients } from '../../context/PatientContext';
import { FEATURES, FEATURE_LABELS, getGlobalFeatureImportance } from '../../data/mockData';
import './FeatureImportance.css';

const MARGIN = { top: 10, right: 30, bottom: 30, left: 130 };

export default function FeatureImportance() {
  const svgRef = useRef(null);
  const containerRef = useRef(null);
  const [dims, setDims] = useState({ width: 500, height: 350 });
  const { selectedPatients, activeFeature, setActiveFeature, allPatients } = usePatients();

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

  // Sort features by global importance
  const sortedFeatures = useMemo(() => {
    return getGlobalFeatureImportance(selectedPatients).map(d => d.feature);
  }, [selectedPatients]);

  useEffect(() => {
    if (!svgRef.current) return;
    const { width, height } = dims;
    const innerW = width - MARGIN.left - MARGIN.right;
    const innerH = height - MARGIN.top - MARGIN.bottom;
    if (innerW <= 0 || innerH <= 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const g = svg.append('g')
      .attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

    // Collect all SHAP values for beeswarm
    const shapExtent = d3.extent(
      selectedPatients.flatMap(p => FEATURES.map(f => p.shapValues[f]))
    );
    const xMax = Math.max(Math.abs(shapExtent[0]), Math.abs(shapExtent[1]));

    const xScale = d3.scaleLinear()
      .domain([-xMax, xMax])
      .range([0, innerW])
      .nice();

    const yScale = d3.scaleBand()
      .domain(sortedFeatures)
      .range([0, innerH])
      .padding(0.15);

    // Feature value color scale (low=blue, high=red) for beeswarm
    const featureScales = {};
    sortedFeatures.forEach(f => {
      const extent = d3.extent(selectedPatients, p => p.features[f]);
      featureScales[f] = d3.scaleSequential(d3.interpolateRdBu)
        .domain([extent[1], extent[0]]); // reversed so high=red
    });

    // Zero line
    g.append('line')
      .attr('x1', xScale(0)).attr('x2', xScale(0))
      .attr('y1', 0).attr('y2', innerH)
      .attr('stroke', '#cbd5e1')
      .attr('stroke-dasharray', '3 3');

    // X axis
    g.append('g')
      .attr('transform', `translate(0,${innerH})`)
      .call(d3.axisBottom(xScale).ticks(6).tickFormat(d3.format('.2f')))
      .call(g => g.select('.domain').attr('stroke', '#cbd5e1'))
      .call(g => g.selectAll('.tick line').attr('stroke', '#e2e8f0'))
      .call(g => g.selectAll('.tick text').attr('fill', '#94a3b8').attr('font-size', '10px'));

    g.append('text')
      .attr('x', innerW / 2).attr('y', innerH + 26)
      .attr('text-anchor', 'middle')
      .attr('fill', '#94a3b8').attr('font-size', '10px')
      .text('SHAP value (impact on prediction)');

    // Y axis (feature labels)
    const yAxis = g.append('g')
      .call(d3.axisLeft(yScale).tickFormat(f => FEATURE_LABELS[f] || f))
      .call(g => g.select('.domain').remove())
      .call(g => g.selectAll('.tick line').remove())
      .call(g => g.selectAll('.tick text')
        .attr('fill', d => d === activeFeature ? '#3b82f6' : '#475569')
        .attr('font-size', '11px')
        .attr('font-weight', d => d === activeFeature ? '600' : '400')
        .style('cursor', 'pointer')
        .on('click', (event, f) => {
          setActiveFeature(activeFeature === f ? null : f);
        })
      );

    // Beeswarm dots - sample for performance
    const sampleSize = Math.min(selectedPatients.length, 150);
    const step = Math.max(1, Math.floor(selectedPatients.length / sampleSize));
    const sampled = selectedPatients.filter((_, i) => i % step === 0);

    const dotR = 2.2;

    // Clip group to prevent overflow between rows
    const clipId = 'beeswarm-clip';
    const defs0 = g.append('defs');
    defs0.append('clipPath').attr('id', clipId)
      .append('rect')
      .attr('x', 0).attr('y', 0)
      .attr('width', innerW).attr('height', innerH);

    const dotsGroup = g.append('g').attr('clip-path', `url(#${clipId})`);

    sortedFeatures.forEach(feature => {
      const bandY = yScale(feature);
      const bandH = yScale.bandwidth();
      const bandCenter = bandY + bandH / 2;
      const maxYOffset = bandH / 2 - dotR; // clamp within band

      const dots = sampled.map(p => ({
        x: xScale(p.shapValues[feature]),
        shap: p.shapValues[feature],
        featureVal: p.features[feature],
        patientId: p.id,
      }));

      // Sort by x for left-to-right placement
      dots.sort((a, b) => a.x - b.x);

      // Collision-aware placement, clamped to band
      const placed = [];
      dots.forEach(dot => {
        let bestY = bandCenter;
        let yOff = 0;
        let dir = 1;

        for (let attempt = 0; attempt < 15; attempt++) {
          const testY = bandCenter + yOff;
          const collision = placed.some(p =>
            Math.abs(p.x - dot.x) < dotR * 2.2 && Math.abs(p.y - testY) < dotR * 2.2
          );
          if (!collision) {
            bestY = testY;
            break;
          }
          yOff = dir * (Math.abs(yOff) + dotR * 2);
          dir *= -1;

          // Stop if we've hit band edges
          if (Math.abs(yOff) > maxYOffset) break;
        }

        // Clamp to band boundaries
        const clampedY = Math.max(bandY + dotR, Math.min(bandY + bandH - dotR, bestY));
        dot.y = clampedY;
        placed.push({ x: dot.x, y: clampedY });
      });

      // Draw dots within clipped group
      dotsGroup.selectAll(null)
        .data(dots)
        .enter()
        .append('circle')
        .attr('cx', d => d.x)
        .attr('cy', d => d.y)
        .attr('r', dotR)
        .attr('fill', d => featureScales[feature](d.featureVal))
        .attr('opacity', feature === activeFeature ? 0.9 : 0.6)
        .attr('stroke', feature === activeFeature ? '#1e293b' : 'none')
        .attr('stroke-width', feature === activeFeature ? 0.5 : 0);

      // Clickable row overlay
      g.append('rect')
        .attr('x', 0)
        .attr('y', bandY)
        .attr('width', innerW)
        .attr('height', bandH)
        .attr('fill', feature === activeFeature ? 'rgba(59,130,246,0.04)' : 'transparent')
        .style('cursor', 'pointer')
        .on('click', () => setActiveFeature(activeFeature === feature ? null : feature));
    });

    // Color legend
    const legendW = 100;
    const legendH = 8;
    const legendG = svg.append('g')
      .attr('transform', `translate(${width - legendW - 30},${4})`);

    const defs = svg.append('defs');
    const grad = defs.append('linearGradient').attr('id', 'bee-grad');
    grad.append('stop').attr('offset', '0%').attr('stop-color', '#2563eb');
    grad.append('stop').attr('offset', '100%').attr('stop-color', '#ef4444');

    legendG.append('rect')
      .attr('width', legendW).attr('height', legendH)
      .attr('rx', 4)
      .attr('fill', 'url(#bee-grad)');
    legendG.append('text').attr('y', -3).attr('fill', '#94a3b8').attr('font-size', '9px').text('Low');
    legendG.append('text').attr('x', legendW).attr('y', -3).attr('text-anchor', 'end').attr('fill', '#94a3b8').attr('font-size', '9px').text('High');
    legendG.append('text').attr('x', legendW / 2).attr('y', -3).attr('text-anchor', 'middle').attr('fill', '#94a3b8').attr('font-size', '9px').text('Feature value');

  }, [dims, selectedPatients, sortedFeatures, activeFeature, setActiveFeature]);

  return (
    <div className="beeswarm-container" ref={containerRef}>
      <svg ref={svgRef} width={dims.width} height={dims.height} />
    </div>
  );
}
