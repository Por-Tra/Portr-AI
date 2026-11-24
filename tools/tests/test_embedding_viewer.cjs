/* Test the actual dashboard script with a small DOM stub, without a browser. */
'use strict';
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');

class Element {
  constructor() { this.children = []; this.style = {}; this.listeners = {}; this.value = ''; this._text = ''; }
  set textContent(value) { this._text = value; this.children = []; }
  get textContent() { return this._text; }
  append(...elements) { this.children.push(...elements); }
  replaceChildren(...elements) { this.children = elements; this._text = ''; }
  addEventListener(event, handler) { this.listeners[event] = handler; }
  querySelector() { return this.children[0]; }
  click() { if (this.onclick) this.onclick(); }
  on(event, handler) { this.listeners[event] = handler; }
}

for (const path of process.argv.slice(2)) {
  const page = fs.readFileSync(path, 'utf8');
  const data = JSON.parse(page.match(/<script id="embedding-data" type="application\/json">([\s\S]*?)<\/script>/)[1]);
  const scripts = [...page.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)];
  for (const script of scripts) {
    if (!script[0].startsWith('<script id="embedding-data"')) new vm.Script(script[1]);
  }
  const elements = new Map();
  const get = id => { if (!elements.has(id)) elements.set(id, new Element()); return elements.get(id); };
  get('embedding-data').textContent = JSON.stringify(data);
  const restyles = [];
  const context = vm.createContext({
    document: { getElementById: get, createElement: () => new Element() },
    Plotly: { restyle: (...args) => restyles.push(args) },
  });
  new vm.Script(scripts.at(-1)[1]).runInContext(context);

  const i = Math.min(1, data.indices.length - 1);
  const row = data.rows[data.indices[i]];
  get('search').value = '#' + row.id;
  get('search').listeners.input();
  assert.equal(get('results').children.length, 1);
  get('search').listeners.keydown({key: 'Enter'});
  assert.equal(get('selected').textContent, row.token);
  assert.equal(get('neighbor-list').children.length, data.neighbors[i].length);
  const overlay = restyles.at(-1)[1];
  assert.equal(overlay.x[0][0], data.coordinates[i][0]);
  assert.equal(overlay.y[0][0], data.coordinates[i][1]);
  if (data.dims === 3) assert.equal(overlay.z[0][0], data.coordinates[i][2]);

  get('embedding-map').listeners.plotly_click({points: [{customdata: 0}]});
  assert.equal(get('selected').textContent, data.rows[data.indices[0]].token);
  const neighborButton = get('neighbor-list').children.find(button => !button.disabled);
  if (neighborButton) neighborButton.onclick();
  get('search').value = '___no_matching_token_496a___';
  get('search').listeners.input();
  assert.equal(get('results').children.length, 0);
  assert.match(get('live').textContent, /^0 resultat/);
  console.log('OK: recherche, selection, voisins et syntaxe JS — ' + path);
}
