import test from 'node:test';
import assert from 'node:assert/strict';
import { analyseSeries, createDemo, extractSeries, numericColumns, parseDelimited } from '../docs/assets/analysis.mjs';

test('parses quoted CSV values and identifies numeric columns', () => {
  const parsed = parseDelimited('cycle,"health,index",label\n0,0.4,"before, baseline"\n1,0.5,after\n2,0.6,after');
  assert.deepEqual(parsed.headers, ['cycle', 'health,index', 'label']);
  assert.deepEqual(numericColumns(parsed.headers, parsed.rows), ['cycle', 'health,index']);
});

test('detects improvement when higher values represent healthier operation', () => {
  const values = [1, 1.1, 1.2, 1.1, 2, 2.1, 2.2, 2.1];
  const result = analyseSeries(values, 4, 4, true);
  assert.equal(result.direction, 'improvement');
  assert.ok(result.delta > 0.9);
  assert.equal(result.beforeCount, 4);
  assert.equal(result.afterCount, 4);
});

test('reverses interpretation for risk or vibration signals', () => {
  const values = [8, 8.2, 7.8, 8.1, 3, 3.1, 2.9, 3.2];
  const result = analyseSeries(values, 4, 4, false);
  assert.equal(result.direction, 'improvement');
  assert.ok(result.delta < 0);
});

test('synthetic demo is analysable and has an interpretable improvement', () => {
  const demo = createDemo();
  const values = extractSeries(demo.rows, demo.column);
  const result = analyseSeries(values, demo.eventIndex, demo.windowSize, true);
  assert.equal(values.length, 240);
  assert.equal(result.direction, 'improvement');
  assert.ok(result.relative > 5);
});

test('rejects invalid event positions', () => {
  assert.throws(() => analyseSeries([1, 2, 3, 4, 5], 0, 2), /Maintenance index/);
});
