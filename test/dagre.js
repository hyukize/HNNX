import * as dagre from '../source/dagre.js';
import assert from 'node:assert/strict';

const nodes = [{ v: 'node', width: 80, height: 40, parent: null }];
const edges = [{
    v: 'node',
    w: 'node',
    name: 'named-self-edge',
    minlen: 1,
    weight: 1,
    width: 0,
    height: 0,
    labeloffset: 10,
    labelpos: 'r'
}];

dagre.layout(nodes, edges, { nodesep: 20, ranksep: 20 }, {});

assert.equal(edges[0].name, 'named-self-edge');
assert.ok(edges[0].points.length >= 5);
assert.ok(edges[0].points.every((point) => Number.isFinite(point.x) && Number.isFinite(point.y)));
