// report.js - logica client del comparison.html della valutazione.
// Iniettato in pagina da evaluate.py insieme ai dati (var EVAL_ROWS, definita
// prima di questo script). Selettore A vs B: mostra solo i due candidati
// scelti (Simulation Info affiancate, tabella-duello con vincitore/perdente
// colorati per campo, Gantt impilati) e ridimensiona i Gantt appena mostrati
// (Plotly li disegna a larghezza di default dentro i div nascosti).

var EVAL_METRICS = [
  { key: 'pipe',                  label: 'Pipeline',                    better: 'max' },
  { key: 'constellation_time_ms', label: 'Tempo totale di completamento C_T', better: 'min', suffix: 'ms' },
  { key: 'energy_satsas_J',       label: 'Costo energetico C_E',         better: 'min', suffix: 'J' },
  { key: 'energy_total_J',        label: 'Energia totale',              better: 'min', suffix: 'J' }
];

function evalVal(m, r) { return m.key === 'pipe' ? r.n_completed : r[m.key]; }

function evalFmt(m, r) {
  if (m.key === 'pipe') { return r.n_completed + '/' + r.n_pipelines; }
  var v = r[m.key];
  return (v === null || v === undefined) ? 'N/A' : v.toFixed(4) + m.suffix;
}

// Classe della cella di A confrontata con B: win / lose / pari (nessuna).
function evalCls(m, va, vb) {
  if (va === null || va === undefined) { return (vb === null || vb === undefined) ? '' : ' lose'; }
  if (vb === null || vb === undefined) { return ' win'; }
  if (va === vb) { return ''; }
  var aWins = (m.better === 'min') ? (va < vb) : (va > vb);
  return aWins ? ' win' : ' lose';
}

function evalRenderDuel(a, b) {
  var head = '<tr><th>Candidato</th><th>Policy / modello</th>';
  EVAL_METRICS.forEach(function (m) { head += '<th>' + m.label + '</th>'; });
  head += '</tr>';
  function row(r, other) {
    var tds = '<td class="name">' + r.name + '</td><td>' + r.label + '</td>';
    EVAL_METRICS.forEach(function (m) {
      tds += '<td class="num' + evalCls(m, evalVal(m, r), evalVal(m, other)) + '">'
           + evalFmt(m, r) + '</td>';
    });
    return '<tr>' + tds + '</tr>';
  }
  document.getElementById('eval-duel').innerHTML =
    '<table class="cmp"><thead>' + head + '</thead><tbody>'
    + row(EVAL_ROWS[a], EVAL_ROWS[b]) + row(EVAL_ROWS[b], EVAL_ROWS[a])
    + '</tbody></table>';
}

function evalUpdateCompare() {
  var a = +document.getElementById('eval-sel-a').value;
  var b = +document.getElementById('eval-sel-b').value;
  // I contenitori sono flex: `order` fa seguire a info e Gantt l'ordine della
  // selezione (A prima/sopra, B dopo/sotto), qualunque sia l'ordine in pagina.
  document.querySelectorAll('.eval-cand-info').forEach(function (d, i) {
    d.style.display = (i === a || i === b) ? 'block' : 'none';
    d.style.order   = (i === a) ? 0 : 1;
  });
  evalRenderDuel(a, b);
  document.querySelectorAll('.eval-cand-gantt').forEach(function (d, i) {
    var show = (i === a || i === b);
    d.style.display = show ? 'block' : 'none';
    d.style.order   = (i === a) ? 0 : 1;
    if (show) {
      d.querySelectorAll('.plotly-graph-div').forEach(function (g) {
        if (window.Plotly) { Plotly.Plots.resize(g); }
      });
    }
  });
}

window.addEventListener('load', evalUpdateCompare);
