export const MAX_FILE_BYTES = 5 * 1024 * 1024;
export const MAX_ROWS = 200_000;

function countDelimiter(line, delimiter) {
  let count = 0;
  let quoted = false;
  for (let index = 0; index < line.length; index += 1) {
    const char = line[index];
    if (char === '"') quoted = !quoted;
    if (!quoted && char === delimiter) count += 1;
  }
  return count;
}

function splitRow(line, delimiter) {
  const cells = [];
  let value = '';
  let quoted = false;
  for (let index = 0; index < line.length; index += 1) {
    const char = line[index];
    const next = line[index + 1];
    if (char === '"' && quoted && next === '"') {
      value += '"';
      index += 1;
    } else if (char === '"') {
      quoted = !quoted;
    } else if (char === delimiter && !quoted) {
      cells.push(value.trim());
      value = '';
    } else {
      value += char;
    }
  }
  cells.push(value.trim());
  return cells;
}

export function parseDelimited(text) {
  const lines = text.replace(/^\uFEFF/, '').split(/\r?\n/).filter((line) => line.trim().length > 0);
  if (lines.length < 3) throw new Error('The CSV needs a header and at least two data rows.');
  if (lines.length - 1 > MAX_ROWS) throw new Error(`The browser GUI accepts at most ${MAX_ROWS.toLocaleString()} rows.`);

  const candidates = [',', ';', '\t'];
  const delimiter = candidates.sort((a, b) => countDelimiter(lines[0], b) - countDelimiter(lines[0], a))[0];
  const headers = splitRow(lines[0], delimiter).map((header, index) => header || `column_${index + 1}`);
  const rows = lines.slice(1).map((line) => {
    const values = splitRow(line, delimiter);
    return Object.fromEntries(headers.map((header, index) => [header, values[index] ?? '']));
  });
  return { headers, rows };
}

export function numericColumns(headers, rows) {
  return headers.filter((header) => {
    const sampled = rows.slice(0, 500).map((row) => row[header]).filter((value) => value !== '');
    return sampled.length > 0 && sampled.filter((value) => Number.isFinite(Number(value))).length / sampled.length >= 0.9;
  });
}

export function extractSeries(rows, column) {
  const values = rows.map((row) => Number(row[column])).filter(Number.isFinite);
  if (values.length < 6) throw new Error(`Column “${column}” does not contain enough numeric values.`);
  return values;
}

function mean(values) {
  return values.reduce((total, value) => total + value, 0) / values.length;
}

function sampleVariance(values, average) {
  if (values.length < 2) return 0;
  return values.reduce((total, value) => total + (value - average) ** 2, 0) / (values.length - 1);
}

export function analyseSeries(values, eventIndex, windowSize, higherIsHealthier = true) {
  const event = Number(eventIndex);
  const window = Number(windowSize);
  if (!Number.isInteger(event) || event < 2 || event > values.length - 2) {
    throw new Error(`Maintenance index must be between 2 and ${values.length - 2}.`);
  }
  if (!Number.isInteger(window) || window < 2) throw new Error('Window size must be an integer of at least 2.');

  const beforeStart = Math.max(0, event - window);
  const afterEnd = Math.min(values.length, event + window);
  const before = values.slice(beforeStart, event);
  const after = values.slice(event, afterEnd);
  if (before.length < 2 || after.length < 2) throw new Error('Both analysis windows need at least two observations.');

  const beforeMean = mean(before);
  const afterMean = mean(after);
  const delta = afterMean - beforeMean;
  const relative = Math.abs(beforeMean) > Number.EPSILON ? (delta / Math.abs(beforeMean)) * 100 : null;
  const beforeVariance = sampleVariance(before, beforeMean);
  const afterVariance = sampleVariance(after, afterMean);
  const pooled = Math.sqrt(((before.length - 1) * beforeVariance + (after.length - 1) * afterVariance) / (before.length + after.length - 2));
  const effect = pooled > Number.EPSILON ? delta / pooled : delta === 0 ? 0 : Math.sign(delta) * Infinity;
  const healthDelta = higherIsHealthier ? delta : -delta;
  const magnitude = Math.abs(effect);
  const strength = magnitude >= 0.8 ? 'large' : magnitude >= 0.5 ? 'moderate' : magnitude >= 0.2 ? 'small' : 'limited';
  const direction = healthDelta > 0 ? 'improvement' : healthDelta < 0 ? 'deterioration' : 'no material change';

  let title = 'No clear short-term change';
  let detail = 'The local pre- and post-maintenance means are effectively unchanged.';
  let tone = 'neutral';
  if (direction === 'improvement') {
    title = `${strength[0].toUpperCase()}${strength.slice(1)} short-term improvement`;
    detail = `The post-maintenance window moves in the healthier direction with a ${strength} standardized effect.`;
    tone = 'positive';
  } else if (direction === 'deterioration') {
    title = `${strength[0].toUpperCase()}${strength.slice(1)} short-term deterioration`;
    detail = `The post-maintenance window moves in the less healthy direction with a ${strength} standardized effect.`;
    tone = 'negative';
  }

  return {
    beforeMean, afterMean, delta, relative, effect, strength, direction, title, detail, tone,
    beforeStart, event, afterEnd, beforeCount: before.length, afterCount: after.length,
  };
}

export function createDemo() {
  const rows = [];
  for (let index = 0; index < 240; index += 1) {
    const seasonal = Math.sin(index / 7) * 0.018 + Math.sin(index / 17) * 0.011;
    const baseline = index < 120 ? 0.78 - index * 0.00155 : 0.86 - (index - 120) * 0.00082;
    rows.push({ cycle: index, health_index: Number((baseline + seasonal).toFixed(5)) });
  }
  return { headers: ['cycle', 'health_index'], rows, column: 'health_index', eventIndex: 120, windowSize: 30 };
}
