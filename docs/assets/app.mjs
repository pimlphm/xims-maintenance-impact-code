import { MAX_FILE_BYTES, analyseSeries, createDemo, extractSeries, numericColumns, parseDelimited } from './analysis.mjs';

const elements = Object.fromEntries([
  'csvFile', 'signalColumn', 'eventIndex', 'windowSize', 'higherHealthy', 'analyzeButton', 'demoButton', 'heroDemo',
  'errorMessage', 'dataStatus', 'verdictBox', 'verdictIcon', 'verdictTitle', 'verdictDetail', 'beforeMean', 'afterMean',
  'relativeChange', 'absoluteChange', 'effectSize', 'effectLabel', 'beforeRange', 'afterRange', 'chartSubtitle', 'signalChart',
].map((id) => [id, document.getElementById(id)]));

let dataset = createDemo();

function format(value, digits = 4) {
  if (value === null || Number.isNaN(value)) return 'n/a';
  if (!Number.isFinite(value)) return value > 0 ? '+∞' : '−∞';
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: digits });
}

function setOptions(headers, rows, preferred) {
  const columns = numericColumns(headers, rows);
  if (!columns.length) throw new Error('No mostly numeric column was found in the CSV.');
  elements.signalColumn.replaceChildren(...columns.map((column) => {
    const option = document.createElement('option');
    option.value = column;
    option.textContent = column;
    option.selected = column === preferred;
    return option;
  }));
}

function drawChart(values, result) {
  const canvas = elements.signalChart;
  const width = canvas.clientWidth || 900;
  const height = 330;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const context = canvas.getContext('2d');
  context.scale(ratio, ratio);
  context.clearRect(0, 0, width, height);

  const pad = { left: 54, right: 18, top: 24, bottom: 38 };
  const plotWidth = width - pad.left - pad.right;
  const plotHeight = height - pad.top - pad.bottom;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const x = (index) => pad.left + (index / Math.max(1, values.length - 1)) * plotWidth;
  const y = (value) => pad.top + ((max - value) / range) * plotHeight;

  context.fillStyle = 'rgba(241, 168, 76, .12)';
  context.fillRect(x(result.beforeStart), pad.top, x(result.event) - x(result.beforeStart), plotHeight);
  context.fillStyle = 'rgba(55, 199, 161, .11)';
  context.fillRect(x(result.event), pad.top, x(result.afterEnd - 1) - x(result.event), plotHeight);

  context.strokeStyle = 'rgba(142, 169, 191, .16)';
  context.fillStyle = '#7891a7';
  context.font = '12px ui-monospace, SFMono-Regular, Consolas, monospace';
  context.lineWidth = 1;
  for (let tick = 0; tick <= 4; tick += 1) {
    const yy = pad.top + (tick / 4) * plotHeight;
    context.beginPath(); context.moveTo(pad.left, yy); context.lineTo(width - pad.right, yy); context.stroke();
    const label = max - (tick / 4) * range;
    context.fillText(format(label, 3), 4, yy + 4);
  }

  context.beginPath();
  values.forEach((value, index) => index ? context.lineTo(x(index), y(value)) : context.moveTo(x(index), y(value)));
  context.strokeStyle = '#9ad8ff'; context.lineWidth = 2.2; context.stroke();

  context.beginPath(); context.moveTo(x(result.event), pad.top); context.lineTo(x(result.event), pad.top + plotHeight);
  context.setLineDash([6, 5]); context.strokeStyle = '#f1a84c'; context.lineWidth = 1.5; context.stroke(); context.setLineDash([]);
  context.fillStyle = '#f1a84c'; context.fillText('maintenance', Math.min(width - 100, x(result.event) + 8), pad.top + 14);

  context.fillStyle = '#7891a7';
  context.fillText('0', pad.left - 3, height - 12);
  context.fillText(String(values.length - 1), width - pad.right - 28, height - 12);
}

function render() {
  elements.errorMessage.textContent = '';
  try {
    const column = elements.signalColumn.value;
    const values = extractSeries(dataset.rows, column);
    const result = analyseSeries(values, Number(elements.eventIndex.value), Number(elements.windowSize.value), elements.higherHealthy.checked);
    elements.beforeMean.textContent = format(result.beforeMean);
    elements.afterMean.textContent = format(result.afterMean);
    elements.relativeChange.textContent = result.relative === null ? 'n/a' : `${result.relative >= 0 ? '+' : ''}${format(result.relative, 2)}%`;
    elements.absoluteChange.textContent = `absolute ${result.delta >= 0 ? '+' : ''}${format(result.delta)}`;
    elements.effectSize.textContent = `${result.effect >= 0 ? '+' : ''}${format(result.effect, 2)}`;
    elements.effectLabel.textContent = `${result.strength} Cohen's d`;
    elements.beforeRange.textContent = `rows ${result.beforeStart}–${result.event - 1}`;
    elements.afterRange.textContent = `rows ${result.event}–${result.afterEnd - 1}`;
    elements.verdictTitle.textContent = result.title;
    elements.verdictDetail.textContent = result.detail;
    elements.verdictIcon.textContent = result.tone === 'positive' ? '↗' : result.tone === 'negative' ? '↘' : '→';
    elements.verdictBox.dataset.tone = result.tone;
    elements.chartSubtitle.textContent = `${column} · ${values.length.toLocaleString()} numeric observations`;
    drawChart(values, result);
  } catch (error) {
    elements.errorMessage.textContent = error.message;
  }
}

function loadDemo(scroll = false) {
  dataset = createDemo();
  setOptions(dataset.headers, dataset.rows, dataset.column);
  elements.eventIndex.value = dataset.eventIndex;
  elements.eventIndex.max = dataset.rows.length - 2;
  elements.windowSize.value = dataset.windowSize;
  elements.dataStatus.textContent = 'Synthetic demo ready';
  elements.csvFile.value = '';
  render();
  if (scroll) document.getElementById('workspace').scrollIntoView({ behavior: 'smooth' });
}

elements.csvFile.addEventListener('change', async () => {
  const [file] = elements.csvFile.files;
  if (!file) return;
  elements.errorMessage.textContent = '';
  try {
    if (file.size > MAX_FILE_BYTES) throw new Error('The selected file is larger than 5 MB.');
    const parsed = parseDelimited(await file.text());
    dataset = parsed;
    setOptions(parsed.headers, parsed.rows, parsed.headers[0]);
    const suggested = Math.floor(parsed.rows.length / 2);
    elements.eventIndex.value = suggested;
    elements.eventIndex.max = parsed.rows.length - 2;
    elements.windowSize.value = Math.max(2, Math.min(30, Math.floor(parsed.rows.length / 4)));
    elements.dataStatus.textContent = `${file.name} · ${parsed.rows.length.toLocaleString()} rows`;
    render();
  } catch (error) {
    elements.errorMessage.textContent = error.message;
  }
});

elements.analyzeButton.addEventListener('click', render);
elements.demoButton.addEventListener('click', () => loadDemo(false));
elements.heroDemo.addEventListener('click', () => loadDemo(true));
window.addEventListener('resize', () => window.requestAnimationFrame(render));
loadDemo(false);
