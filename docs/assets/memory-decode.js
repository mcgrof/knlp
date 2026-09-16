/* Empirical fit illustrations. No synthetic point is a measured observation. */
(() => {
  'use strict';
  const startInput = document.getElementById('capacityStart');
  const factorInput = document.getElementById('capacityFactor');
  const batchInput = document.getElementById('batchSlider');
  const normalizedThroughput = batch => batch / (1 + batch);
  const historicalRatio = batch => 3.75 * batch ** 1.32 / (5.1 ** 1.32 + batch ** 1.32);
  let capacityChart;
  let historicalChart;

  function updateCapacity() {
    const start = Number(startInput.value) / 100;
    const factor = Number(factorInput.value);
    const batch = start / (1 - start);
    const expandedBatch = factor * batch;
    const expanded = normalizedThroughput(expandedBatch);
    const throughputRatio = expanded / start;
    const tokenTimeRatio = factor / throughputRatio;
    document.getElementById('capacityStartValue').textContent = `${Math.round(start * 100)}%`;
    document.getElementById('capacityResult').textContent =
      `${((factor - 1) * 100).toFixed(1)}% more batch → ${((throughputRatio - 1) * 100).toFixed(1)}% more throughput`;
    document.getElementById('capacityDetail').textContent =
      `${(start * 100).toFixed(1)}% → ${(expanded * 100).toFixed(1)}% of the ceiling. Approximate time per output token rises ${((tokenTimeRatio - 1) * 100).toFixed(1)}%.`;
    if (capacityChart) {
      capacityChart.data.datasets[1].data = [{x: batch, y: start}];
      capacityChart.data.datasets[2].data = [{x: expandedBatch, y: expanded}];
      capacityChart.update('none');
    }
  }

  function updateHistorical() {
    const batch = Number(batchInput.value);
    document.getElementById('historicalBatch').textContent = String(batch);
    document.getElementById('speedupVal').textContent = `${historicalRatio(batch).toFixed(2)}×`;
    if (historicalChart) {
      historicalChart.data.datasets[1].data = [{x: batch, y: historicalRatio(batch)}];
      historicalChart.update('none');
    }
  }

  startInput.addEventListener('input', updateCapacity);
  factorInput.addEventListener('change', updateCapacity);
  batchInput.addEventListener('input', updateHistorical);
  updateCapacity();
  updateHistorical();
  // The explanation, tables, and calculator remain usable if the chart CDN fails.
  if (typeof Chart === 'undefined') return;

  const css = getComputedStyle(document.documentElement);
  const color = name => css.getPropertyValue(name).trim();
  const green = color('--green');
  const cyan = color('--cyan');
  const muted = color('--text2');
  const border = color('--border');
  Chart.defaults.color = muted;
  Chart.defaults.font.family = 'system-ui, sans-serif';
  Chart.defaults.font.size = 12;
  Chart.defaults.animation = false;

  function options(xTitle, yTitle) {
    return {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {mode: 'nearest', intersect: false},
      scales: {
        x: {
          type: 'linear', min: 0,
          grid: {color: border},
          ticks: {maxTicksLimit: 5},
          title: {display: true, text: xTitle, color: muted}
        },
        y: {
          min: 0, grid: {color: border},
          ticks: {maxTicksLimit: 5},
          title: {display: true, text: yTitle, color: muted}
        }
      },
      plugins: {
        legend: {display: false},
        tooltip: {
          backgroundColor: color('--bg3'), borderColor: border, borderWidth: 1,
          titleColor: color('--text'), bodyColor: muted
        }
      }
    };
  }

  const capacityOptions = options('Active batch / half-saturation batch', 'Aggregate throughput / ceiling');
  capacityOptions.scales.x.max = 20;
  capacityOptions.scales.y.max = 1;
  capacityOptions.scales.y.ticks.callback = value => `${Math.round(value * 100)}%`;
  capacityOptions.plugins.tooltip.callbacks = {
    title: items => `Batch / half-saturation batch: ${items[0].parsed.x.toFixed(2)}`,
    label: item => `${item.dataset.label}: ${(100 * item.parsed.y).toFixed(1)}% of ceiling`
  };
  capacityChart = new Chart(document.getElementById('capacityChart'), {
    type: 'line',
    data: {datasets: [
      {label: 'Illustrative curve', data: Array.from({length: 401}, (_, i) => ({x: i / 20, y: normalizedThroughput(i / 20)})), borderColor: green, borderWidth: 2, pointRadius: 0, parsing: false},
      {label: 'Starting batch', data: [], borderColor: color('--bg'), backgroundColor: green, borderWidth: 2, pointRadius: 6, pointHoverRadius: 8, showLine: false, parsing: false},
      {label: 'Expanded batch', data: [], borderColor: color('--bg'), backgroundColor: cyan, borderWidth: 2, pointStyle: 'rect', pointRadius: 7, pointHoverRadius: 9, showLine: false, parsing: false}
    ]},
    options: capacityOptions
  });

  const ratioOptions = options('Active batch B', 'INT4 / FP16 SDPA speedup (×)');
  ratioOptions.scales.x.type = 'logarithmic';
  ratioOptions.scales.x.min = 4;
  ratioOptions.scales.x.max = 128;
  ratioOptions.scales.x.afterBuildTicks = axis => { axis.ticks = [4, 8, 16, 32, 64, 128].map(value => ({value})); };
  ratioOptions.scales.x.ticks.callback = value => String(value);
  ratioOptions.scales.y.max = 4;
  ratioOptions.plugins.tooltip.callbacks = {
    title: items => `Batch: ${items[0].parsed.x}`,
    label: item => `Fitted INT4 speedup ratio: ${item.parsed.y.toFixed(2)}×`
  };
  historicalChart = new Chart(document.getElementById('satChart'), {
    type: 'line',
    data: {datasets: [
      {label: 'Historical fitted ratio', data: Array.from({length: 125}, (_, i) => ({x: i + 4, y: historicalRatio(i + 4)})), borderColor: green, borderWidth: 2, pointRadius: 0, parsing: false},
      {label: 'Selected batch', data: [], backgroundColor: cyan, borderColor: color('--bg'), pointRadius: 6, showLine: false, parsing: false}
    ]},
    options: ratioOptions
  });
  document.getElementById('historical-speedup').addEventListener('toggle', event => {
    if (event.target.open) historicalChart.resize();
  });
  updateCapacity();
  updateHistorical();
})();
