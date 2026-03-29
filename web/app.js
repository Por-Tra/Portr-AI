const DOCS = [
  {
    module: "model/attention.py",
    symbols: [
      {
        name: "Attention",
        kind: "class",
        signature: "class Attention(nn.Module)",
        description: "Implementation pure de l attention multi-tete causale.",
        highlights: ["Q/K/V projections", "Causal mask", "Softmax", "Merge heads"],
        methods: [
          { sig: "__init__(self, embedding_dim: int, num_heads: int = 8, dropout: float = 0.1)", doc: "Initialise les dimensions, les projections lineaires Q/K/V/out et les dropout." },
          { sig: "forward(self, x: torch.Tensor) -> torch.Tensor", doc: "Pipeline complete: projection QKV, split heads, attention, merge heads, out projection." },
          { sig: "_split_heads(self, Q, K, V, B, T)", doc: "Reformate [B,T,C] en [B,H,T,D] pour chaque tenseur Q, K, V." },
          { sig: "_attend(self, Q, K, V, T)", doc: "Calcule scores QK^T / sqrt(d), applique masque causal, softmax puis ponderation par V." },
          { sig: "_merge_heads(self, x, B, T)", doc: "Recompose les heads en tenseur [B,T,C]." }
        ]
      }
    ]
  },
  {
    module: "model/transformer_block.py",
    symbols: [
      {
        name: "MLP",
        kind: "class",
        signature: "class MLP(nn.Module)",
        description: "FeedForward token-wise: Linear -> GELU -> Linear avec dropout.",
        highlights: ["Expansion", "Activation", "Projection retour"],
        methods: [
          { sig: "__init__(self, embedding_dim: int, hidden_dim: int, dropout: float = 0.1)", doc: "Cree deux couches lineaires fc1/fc2, une GELU et un dropout." },
          { sig: "forward(self, x: torch.Tensor) -> torch.Tensor", doc: "Applique fc1, activation GELU, dropout, fc2, dropout." }
        ]
      },
      {
        name: "TransformerBlock",
        kind: "class",
        signature: "class TransformerBlock(nn.Module)",
        description: "Bloc pre-norm: Attention + MLP avec connexions residuelles.",
        highlights: ["ln1", "attention", "ln2", "mlp"],
        methods: [
          { sig: "__init__(self, embedding_dim: int, num_heads: int, dropout: float = 0.1)", doc: "Initialise ln1 + Attention puis ln2 + MLP (4x embedding_dim)." },
          { sig: "forward(self, x: torch.Tensor) -> torch.Tensor", doc: "Calcule x = x + attention(ln1(x)) puis x = x + mlp(ln2(x))." }
        ]
      }
    ]
  },
  {
    module: "model/embedding.py",
    symbols: [
      {
        name: "Embedding",
        kind: "class",
        signature: "class Embedding(nn.Module)",
        description: "Combine token embeddings et position embeddings puis normalise.",
        highlights: ["token_emb", "position_emb", "layer_norm", "dropout"],
        methods: [
          { sig: "__init__(self, vocab_size, seq_length, embedding_dim=256)", doc: "Initialise matrices token_emb et position_emb, puis LayerNorm et Dropout." },
          { sig: "forward(self, tokens)", doc: "Recupere vectors de tokens, ajoute vecteurs de positions, applique LayerNorm + Dropout." }
        ]
      }
    ]
  },
  {
    module: "model/gpt_model.py",
    symbols: [
      {
        name: "GPT",
        kind: "class",
        signature: "class GPT(nn.Module)",
        description: "Assemble Embedding + stack de TransformerBlock + output head.",
        highlights: ["blocks", "final_ln", "output_head", "generate"],
        methods: [
          { sig: "__init__(self, vocab_size, seq_length, embedding_dim=256, num_heads=8, num_blocks=4, dropout=0.1)", doc: "Construit le modele complet et ses hyperparametres." },
          { sig: "forward(self, tokens: torch.Tensor) -> torch.Tensor", doc: "Fait passer tokens par embedding, blocs, final_ln puis projection vocabulaire." },
          { sig: "generate(self, initial_tokens, max_length, temperature=1.0, top_k=None)", doc: "Boucle autoregressive: contexte, logits dernier token, temperature/top_k, sample, concat." }
        ]
      },
      {
        name: "GPTConfig",
        kind: "class",
        signature: "class GPTConfig",
        description: "Container de configuration avec valeurs par defaut.",
        highlights: ["defaults", "to_dict"],
        methods: [
          { sig: "__init__(self, **kwargs)", doc: "Lit les hyperparametres depuis kwargs et complete avec des defaults." },
          { sig: "to_dict(self)", doc: "Retourne un dictionnaire reutilisable pour instancier GPT." }
        ]
      }
    ]
  },
  {
    module: "data/DataBase/database.py",
    symbols: [
      {
        name: "Database",
        kind: "class",
        signature: "class Database",
        description: "Gestion SQLite + FTS5 pour stockage et recherche full-text.",
        highlights: ["pages", "pages_fts", "triggers", "export_jsonl"],
        methods: [
          { sig: "__init__(self, db_file: str = 'data/crawler.db')", doc: "Definit le chemin DB, initialise la connexion et cree schema/index." },
          { sig: "_initialize(self) -> bool", doc: "Cree tables, FTS5, triggers et index si necessaire." },
          { sig: "insert_page(self, url, title, content, content_hash, depth, status_code, word_count, link_count, commit=True) -> bool", doc: "Insert/update une page avec ON CONFLICT(url)." },
          { sig: "commit(self) -> bool", doc: "Valide explicitement la transaction en cours." },
          { sig: "hash_exists(self, content_hash: str) -> bool", doc: "Teste si un hash de contenu existe deja." },
          { sig: "get_page_count(self) -> int", doc: "Retourne le nombre total de pages." },
          { sig: "export_jsonl(self, output_path: str) -> int", doc: "Exporte url/title/content/word_count en JSONL." },
          { sig: "search_full_text(self, query: str, limit: int = 20) -> list", doc: "Recherche FTS5 avec snippet et limite." },
          { sig: "get_statistics(self) -> sqlite3.Row | None", doc: "Statistiques globales: total_pages, domains, avg/max words, links." },
          { sig: "close(self)", doc: "Ferme la connexion SQLite." },
          { sig: "__enter__(self)", doc: "Support context manager, retourne self." },
          { sig: "__exit__(self, exc_type, exc_val, exc_tb)", doc: "Ferme la connexion en sortie de contexte." }
        ]
      }
    ]
  },
  {
    module: "data/dataset.py",
    symbols: [
      {
        name: "TextDataset",
        kind: "class",
        signature: "class TextDataset(Dataset)",
        description: "Fenetre glissante pour creer paires (x,y) d entrainement autoregressif.",
        highlights: ["__len__", "__getitem__", "seq_length"],
        methods: [
          { sig: "__init__(self, tokens, seq_length)", doc: "Stocke le tableau de tokens et la longueur de fenetre." },
          { sig: "__len__(self)", doc: "Retourne len(tokens) - seq_length." },
          { sig: "__getitem__(self, idx)", doc: "Retourne x=tokens[idx:idx+seq] et y=tokens[idx+1:idx+seq+1]." }
        ]
      }
    ]
  },
  {
    module: "tokenizer/tokenizer.py",
    symbols: [
      {
        name: "Tokenizer",
        kind: "class",
        signature: "class Tokenizer",
        description: "Tokenizer BPE optimise avec training incremental et encodage en IDs.",
        highlights: ["train", "merge_ranks", "encode", "save/load"],
        methods: [
          { sig: "__init__(self, end_of_word: str = '</w>')", doc: "Initialise vocab, merges, mapping token<->id et special tokens." },
          { sig: "split_text(self, text: str) -> List[str]", doc: "Decoupe texte en mots/ponctuation/espaces." },
          { sig: "normalize_token(self, token: str) -> str", doc: "Normalise les espaces vers un token dedie." },
          { sig: "prepare_word(self, word: str) -> List[str]", doc: "Transforme un mot en liste de symboles + fin de mot." },
          { sig: "_build_word_freq(self, texts) -> Dict[Tuple[str, ...], int]", doc: "Construit frequence des mots symboliques du corpus." },
          { sig: "_pairs_in_word(word) -> List[Tuple[str, str]]", doc: "Liste toutes les paires adjacentes dans un mot symbole." },
          { sig: "_count_all_pairs(self, word_freq)", doc: "Compte frequences de paires et index pair->mots." },
          { sig: "train(self, texts, target_vocab_size=5000, min_pair_freq=2, verbose=False)", doc: "Apprend merges BPE avec approche incremental + heap." },
          { sig: "_apply_merges(self, symbols)", doc: "Applique les merges appris a une sequence de symboles." },
          { sig: "tokenize_word(self, word, strip_eow=True)", doc: "Tokenise un mot unique avec merges BPE." },
          { sig: "tokenize_text(self, text, strip_eow=True)", doc: "Tokenise un texte complet en sous-mots." },
          { sig: "save(self, path=DEFAULT_MODEL_PATH)", doc: "Sauvegarde le modele BPE et mappings en JSON." },
          { sig: "load(cls, path=DEFAULT_MODEL_PATH)", doc: "Charge un modele BPE depuis JSON." },
          { sig: "encode(self, text) -> List[int]", doc: "Tokenise puis convertit en IDs." },
          { sig: "encode_corpus(self, corpus_path) -> List[int]", doc: "Encode un corpus texte complet en IDs." },
          { sig: "vocab_size(self) -> int", doc: "Retourne la taille du vocabulaire." },
          { sig: "get_vocab(self) -> set[str]", doc: "Expose l ensemble des tokens du vocabulaire." },
          { sig: "get_token_to_id(self) -> Dict[str, int]", doc: "Retourne mapping token -> id." },
          { sig: "get_id_to_token(self) -> Dict[int, str]", doc: "Retourne mapping id -> token." }
        ]
      }
    ]
  }
];

const els = {
  tabDocs: document.getElementById("tab-docs"),
  tabPlayground: document.getElementById("tab-playground"),
  panelDocs: document.getElementById("panel-docs"),
  panelPlayground: document.getElementById("panel-playground"),
  searchDocs: document.getElementById("search-docs"),
  moduleList: document.getElementById("module-list"),
  symbolTitle: document.getElementById("symbol-title"),
  symbolKind: document.getElementById("symbol-kind"),
  symbolDesc: document.getElementById("symbol-description"),
  symbolSig: document.getElementById("symbol-signature"),
  symbolMeta: document.getElementById("symbol-meta"),
  symbolMethods: document.getElementById("symbol-methods"),

  algoSelect: document.getElementById("algo-select"),
  algoTitle: document.getElementById("algo-title"),
  algoCode: document.getElementById("algo-code"),
  stepLabel: document.getElementById("step-label"),
  visualArea: document.getElementById("visual-area"),
  timeline: document.getElementById("timeline"),
  traceSummary: document.getElementById("trace-summary"),

  buildTrace: document.getElementById("build-trace"),
  playerReset: document.getElementById("player-reset"),
  playerStep: document.getElementById("player-step"),
  playerPlay: document.getElementById("player-play"),
  playerPause: document.getElementById("player-pause"),
  playerSpeed: document.getElementById("player-speed"),
  playerSpeedOut: document.getElementById("player-speed-out"),

  paramsAttention: document.getElementById("params-attention"),
  paramsDataset: document.getElementById("params-dataset"),
  paramsMlp: document.getElementById("params-mlp"),
  paramsGptgen: document.getElementById("params-gptgen"),
  paramsBpe: document.getElementById("params-bpe"),

  attnTokens: document.getElementById("attn-tokens"),
  attnHeads: document.getElementById("attn-heads"),
  attnHeadsOut: document.getElementById("attn-heads-out"),
  attnTemp: document.getElementById("attn-temp"),
  attnTempOut: document.getElementById("attn-temp-out"),
  attnCausal: document.getElementById("attn-causal"),

  dsTokens: document.getElementById("ds-tokens"),
  dsSeq: document.getElementById("ds-seq"),
  dsSeqOut: document.getElementById("ds-seq-out"),

  mlpInput: document.getElementById("mlp-input"),
  mlpMult: document.getElementById("mlp-mult"),
  mlpMultOut: document.getElementById("mlp-mult-out"),
  mlpActivation: document.getElementById("mlp-activation"),

  gptSeed: document.getElementById("gpt-seed"),
  gptMaxLen: document.getElementById("gpt-max-len"),
  gptMaxLenOut: document.getElementById("gpt-max-len-out"),
  gptVocab: document.getElementById("gpt-vocab"),
  gptVocabOut: document.getElementById("gpt-vocab-out"),
  gptTopk: document.getElementById("gpt-topk"),
  gptTopkOut: document.getElementById("gpt-topk-out"),

  bpeText: document.getElementById("bpe-text"),
  bpeMerges: document.getElementById("bpe-merges"),
  bpeMergesOut: document.getElementById("bpe-merges-out"),

  jsonlFile: document.getElementById("jsonl-file"),
  jsonlText: document.getElementById("jsonl-text"),
  runJsonl: document.getElementById("run-jsonl"),
  demoJsonl: document.getElementById("demo-jsonl"),
  jsonlStats: document.getElementById("jsonl-stats"),
  jsonlBody: document.getElementById("jsonl-body")
};

const player = {
  trace: [],
  code: [],
  step: -1,
  timer: null,
  speed: 700,
  algo: "attention"
};

function switchTab(which) {
  const docs = which === "docs";
  els.tabDocs.classList.toggle("active", docs);
  els.tabPlayground.classList.toggle("active", !docs);
  els.panelDocs.classList.toggle("active", docs);
  els.panelPlayground.classList.toggle("active", !docs);
}

els.tabDocs.addEventListener("click", () => switchTab("docs"));
els.tabPlayground.addEventListener("click", () => switchTab("playground"));

function renderDocs(filter = "") {
  const query = filter.trim().toLowerCase();
  els.moduleList.innerHTML = "";

  DOCS.forEach((mod) => {
    const filteredSymbols = mod.symbols.filter((s) => {
      const methodText = s.methods.map((m) => `${m.sig} ${m.doc}`).join(" ");
      const hay = `${mod.module} ${s.name} ${s.description} ${methodText}`.toLowerCase();
      return !query || hay.includes(query);
    });

    if (!filteredSymbols.length) return;

    const wrap = document.createElement("div");
    wrap.className = "mod-card";
    wrap.innerHTML = `<div class=\"mod-title\">${mod.module}</div>`;

    filteredSymbols.forEach((sym) => {
      const item = document.createElement("div");
      item.className = "mod-symbol";
      item.textContent = `${sym.kind} ${sym.name}`;
      item.addEventListener("click", () => showSymbol(mod.module, sym));
      wrap.appendChild(item);
    });

    els.moduleList.appendChild(wrap);
  });
}

function showSymbol(moduleName, sym) {
  els.symbolTitle.textContent = sym.name;
  els.symbolKind.textContent = sym.kind;
  els.symbolDesc.textContent = sym.description;
  els.symbolSig.textContent = sym.signature;

  els.symbolMeta.innerHTML = "";
  const meta = [
    ["Module", moduleName],
    ["Methodes", String(sym.methods.length)],
    ["Highlights", sym.highlights.join(" | ")]
  ];

  meta.forEach(([k, v]) => {
    const d = document.createElement("div");
    d.className = "meta-item";
    d.innerHTML = `<strong>${k}</strong><br>${v}`;
    els.symbolMeta.appendChild(d);
  });

  els.symbolMethods.innerHTML = "";
  sym.methods.forEach((method) => {
    const block = document.createElement("div");
    block.className = "method";
    block.innerHTML = `<div class=\"sig\">${method.sig}</div><div class=\"doc\">${method.doc}</div>`;
    els.symbolMethods.appendChild(block);
  });
}

els.searchDocs.addEventListener("input", (e) => {
  renderDocs(e.target.value);
});

function seededValue(seed) {
  const x = Math.sin(seed) * 10000;
  return x - Math.floor(x);
}

function parseNumberList(text) {
  return text
    .split(/[\s,;]+/)
    .map((x) => Number(x.trim()))
    .filter((x) => Number.isFinite(x));
}

function softmax(row, temp = 1) {
  const vals = row.map((v) => (Number.isFinite(v) ? v / temp : -1e9));
  const max = Math.max(...vals);
  const exps = vals.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0) || 1;
  return exps.map((v) => v / sum);
}

function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i += 1) s += a[i] * b[i];
  return s;
}

function randomMatrix(rows, cols, seed) {
  const out = [];
  for (let i = 0; i < rows; i += 1) {
    const row = [];
    for (let j = 0; j < cols; j += 1) {
      row.push((seededValue(seed + i * 37.1 + j * 19.7) - 0.5) * 1.3);
    }
    out.push(row);
  }
  return out;
}

function matVecMul(mat, vec) {
  return mat.map((row) => row.reduce((acc, v, i) => acc + v * vec[i], 0));
}

function relu(x) {
  return Math.max(0, x);
}

function gelu(x) {
  return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x)));
}

function tokenToVector(token, dim, offset) {
  const base = token.split("").reduce((acc, ch) => acc + ch.charCodeAt(0), 0) + offset;
  const v = [];
  for (let i = 0; i < dim; i += 1) v.push((seededValue(base * (i + 1.11)) - 0.5) * 2);
  return v;
}

const ALGOS = {
  attention: {
    title: "Attention (QK^T + masque causal + softmax)",
    code: [
      "tokens -> Q, K, V projections",
      "scores = Q @ K^T / sqrt(d)",
      "if causal: mask upper triangle",
      "weights = softmax(scores / temperature)",
      "output = weights @ V"
    ],
    buildTrace() {
      const tokens = els.attnTokens.value.trim().split(/\s+/).filter(Boolean).slice(0, 10);
      const heads = Number(els.attnHeads.value);
      const temp = Number(els.attnTemp.value);
      const causal = els.attnCausal.checked;
      const dim = Math.max(8, heads * 4);

      if (tokens.length < 2) {
        return { trace: [{ line: 0, label: "Ajoute au moins 2 tokens", state: { msg: "Minimum 2 tokens." } }] };
      }

      const Q = tokens.map((t) => tokenToVector(t, dim, 13));
      const K = tokens.map((t) => tokenToVector(t, dim, 29));
      const V = tokens.map((t) => tokenToVector(t, dim, 43));
      const scale = 1 / Math.sqrt(dim);

      const trace = [];
      trace.push({
        line: 0,
        label: "Projection des tokens en Q, K, V",
        state: { type: "chips", chips: tokens, extra: `dim=${dim}, heads=${heads}` }
      });

      const scoreMatrix = [];
      for (let i = 0; i < tokens.length; i += 1) {
        const row = [];
        for (let j = 0; j < tokens.length; j += 1) {
          row.push(dot(Q[i], K[j]) * scale);
        }
        scoreMatrix.push(row);
      }

      trace.push({
        line: 1,
        label: "Calcul de la matrice des scores bruts",
        state: { type: "heatmap", tokens, matrix: scoreMatrix, caption: "Scores QK^T / sqrt(d)" }
      });

      const masked = scoreMatrix.map((r) => r.slice());
      if (causal) {
        for (let i = 0; i < tokens.length; i += 1) {
          for (let j = i + 1; j < tokens.length; j += 1) masked[i][j] = -Infinity;
        }
      }

      trace.push({
        line: 2,
        label: causal ? "Application du masque causal" : "Masque causal desactive",
        state: { type: "heatmap", tokens, matrix: masked, caption: causal ? "Upper triangle masquee" : "Aucun masque" }
      });

      const weights = masked.map((row) => softmax(row, temp));
      trace.push({
        line: 3,
        label: "Softmax ligne par ligne pour obtenir les poids",
        state: { type: "heatmap", tokens, matrix: weights, caption: `Poids d attention (temperature=${temp.toFixed(1)})` }
      });

      const out = [];
      for (let i = 0; i < tokens.length; i += 1) {
        const v = new Array(dim).fill(0);
        for (let j = 0; j < tokens.length; j += 1) {
          const w = weights[i][j];
          for (let d = 0; d < dim; d += 1) v[d] += w * V[j][d];
        }
        out.push(v.slice(0, 6).map((x) => Number(x.toFixed(2))));
      }

      trace.push({
        line: 4,
        label: "Sortie = poids @ V",
        state: { type: "table", headers: ["token", "output(dim0..5)"], rows: tokens.map((t, i) => [t, `[${out[i].join(", ")}]`]) }
      });

      return { trace };
    }
  },

  dataset: {
    title: "TextDataset (fenetre glissante)",
    code: [
      "for idx in range(len(tokens) - seq_length):",
      "  x = tokens[idx: idx + seq_length]",
      "  y = tokens[idx + 1: idx + seq_length + 1]",
      "  yield (x, y)"
    ],
    buildTrace() {
      const tokens = parseNumberList(els.dsTokens.value).map((n) => Math.trunc(n));
      const seq = Number(els.dsSeq.value);
      const trace = [];

      trace.push({ line: 0, label: `Boucle sur ${Math.max(0, tokens.length - seq)} positions`, state: { type: "chips", chips: tokens.map(String), extra: `seq_length=${seq}` } });

      if (tokens.length <= seq) {
        trace.push({ line: 0, label: "Pas assez de tokens", state: { msg: "len(tokens) doit etre > seq_length" } });
        return { trace };
      }

      for (let idx = 0; idx < tokens.length - seq; idx += 1) {
        const x = tokens.slice(idx, idx + seq);
        const y = tokens.slice(idx + 1, idx + seq + 1);
        trace.push({
          line: 3,
          label: `idx=${idx} -> paire (x,y)`,
          state: {
            type: "table",
            headers: ["idx", "x", "y"],
            rows: [[String(idx), `[${x.join(", ")}]`, `[${y.join(", ")}]`]]
          }
        });
      }

      return { trace };
    }
  },

  mlp: {
    title: "MLP (Linear -> Activation -> Linear)",
    code: [
      "h1 = W1 @ x + b1",
      "h2 = activation(h1)",
      "y  = W2 @ h2 + b2"
    ],
    buildTrace() {
      const x = parseNumberList(els.mlpInput.value);
      const mult = Number(els.mlpMult.value);
      const actName = els.mlpActivation.value;
      if (x.length < 2 || x.length > 20) {
        return { trace: [{ line: 0, label: "Vecteur invalide", state: { msg: "Entre 2 et 20 valeurs." } }] };
      }

      const hidden = x.length * mult;
      const W1 = randomMatrix(hidden, x.length, 101);
      const W2 = randomMatrix(x.length, hidden, 303);
      const h1 = matVecMul(W1, x);
      const h2 = h1.map((v) => (actName === "relu" ? relu(v) : gelu(v)));
      const y = matVecMul(W2, h2);

      return {
        trace: [
          { line: 0, label: "Projection lineaire W1 @ x", state: { type: "bars", left: x, right: h1.slice(0, 16), leftLabel: "x", rightLabel: "h1" } },
          { line: 1, label: `Activation ${actName.toUpperCase()}`, state: { type: "bars", left: h1.slice(0, 16), right: h2.slice(0, 16), leftLabel: "h1", rightLabel: "h2" } },
          { line: 2, label: "Projection retour W2 @ h2", state: { type: "bars", left: h2.slice(0, 16), right: y, leftLabel: "h2", rightLabel: "y" } }
        ]
      };
    }
  },

  gptgen: {
    title: "GPT Generate (boucle auto-regressive)",
    code: [
      "current = seed_tokens",
      "while len(current) < max_length:",
      "  context = current[-seq_length:]",
      "  logits = model(context)",
      "  next = sample(top_k(logits[-1]))",
      "  current.append(next)"
    ],
    buildTrace() {
      const seed = parseNumberList(els.gptSeed.value).map((n) => Math.trunc(Math.abs(n)));
      const maxLen = Number(els.gptMaxLen.value);
      const vocab = Number(els.gptVocab.value);
      const topk = Number(els.gptTopk.value);
      const current = (seed.length ? seed : [1, 2, 3]).map((x) => x % vocab);
      const trace = [];

      trace.push({ line: 0, label: "Initialisation avec seed_tokens", state: { type: "chips", chips: current.map(String), extra: `max_len=${maxLen}, vocab=${vocab}, top_k=${topk}` } });

      while (current.length < maxLen) {
        const context = current.slice(-6);
        const ctxHash = context.reduce((a, b, i) => a + b * (i + 3), 0);
        const logits = [];
        for (let i = 0; i < vocab; i += 1) logits.push((seededValue(ctxHash + i * 0.37) - 0.5) * 8);

        const sorted = logits
          .map((v, i) => ({ i, v }))
          .sort((a, b) => b.v - a.v)
          .slice(0, topk);

        const nextToken = sorted[0].i;
        trace.push({
          line: 4,
          label: `Iteration ${current.length - seed.length + 1}: selection next token ${nextToken}`,
          state: {
            type: "table",
            headers: ["token_id", "logit"],
            rows: sorted.slice(0, 10).map((r) => [String(r.i), r.v.toFixed(3)]),
            chips: context.map(String),
            extra: `context=[${context.join(", ")}]`
          }
        });

        current.push(nextToken);
        trace.push({ line: 5, label: "Append du token choisi", state: { type: "chips", chips: current.map(String) } });
      }

      return { trace };
    }
  },

  bpe: {
    title: "Tokenizer BPE (merges successifs)",
    code: [
      "word_freq = count(words)",
      "repeat n merges:",
      "  pair = most_frequent_adjacent_pair(word_freq)",
      "  merge(pair)",
      "  update word_freq"
    ],
    buildTrace() {
      const text = els.bpeText.value.trim().toLowerCase();
      const merges = Number(els.bpeMerges.value);
      const words = text.split(/\s+/).filter(Boolean);
      const trace = [];

      if (!words.length) {
        return { trace: [{ line: 0, label: "Texte vide", state: { msg: "Ajoute des mots pour lancer BPE." } }] };
      }

      const wordFreq = new Map();
      words.forEach((w) => {
        const chars = [...w, "</w>"];
        const key = chars.join(" ");
        wordFreq.set(key, (wordFreq.get(key) || 0) + 1);
      });

      trace.push({
        line: 0,
        label: "Frequences initiales des mots symboliques",
        state: {
          type: "table",
          headers: ["word symbols", "freq"],
          rows: [...wordFreq.entries()].map(([k, v]) => [k, String(v)]).slice(0, 12)
        }
      });

      const keys = () => [...wordFreq.keys()];

      for (let step = 0; step < merges; step += 1) {
        const pairFreq = new Map();
        keys().forEach((k) => {
          const parts = k.split(" ");
          const freq = wordFreq.get(k) || 0;
          for (let i = 0; i < parts.length - 1; i += 1) {
            const pair = `${parts[i]} ${parts[i + 1]}`;
            pairFreq.set(pair, (pairFreq.get(pair) || 0) + freq);
          }
        });

        const best = [...pairFreq.entries()].sort((a, b) => b[1] - a[1])[0];
        if (!best || best[1] <= 1) break;

        const [bestPair, bestFreq] = best;
        trace.push({
          line: 2,
          label: `Merge ${step + 1}: paire la plus frequente = (${bestPair}) freq=${bestFreq}`,
          state: {
            type: "table",
            headers: ["pair", "freq"],
            rows: [...pairFreq.entries()].sort((a, b) => b[1] - a[1]).slice(0, 8).map(([p, f]) => [p, String(f)])
          }
        });

        const mergedToken = bestPair.replace(" ", "");
        const next = new Map();

        keys().forEach((k) => {
          const freq = wordFreq.get(k) || 0;
          const parts = k.split(" ");
          const out = [];
          for (let i = 0; i < parts.length; i += 1) {
            if (i < parts.length - 1 && `${parts[i]} ${parts[i + 1]}` === bestPair) {
              out.push(mergedToken);
              i += 1;
            } else {
              out.push(parts[i]);
            }
          }
          const nk = out.join(" ");
          next.set(nk, (next.get(nk) || 0) + freq);
        });

        wordFreq.clear();
        [...next.entries()].forEach(([k, v]) => wordFreq.set(k, v));

        trace.push({
          line: 4,
          label: `Mise a jour word_freq apres merge ${step + 1}`,
          state: {
            type: "table",
            headers: ["word symbols", "freq"],
            rows: [...wordFreq.entries()].map(([k, v]) => [k, String(v)]).slice(0, 12)
          }
        });
      }

      return { trace };
    }
  }
};

function showParamsForAlgo(algo) {
  [els.paramsAttention, els.paramsDataset, els.paramsMlp, els.paramsGptgen, els.paramsBpe].forEach((el) => el.classList.add("hidden"));
  if (algo === "attention") els.paramsAttention.classList.remove("hidden");
  if (algo === "dataset") els.paramsDataset.classList.remove("hidden");
  if (algo === "mlp") els.paramsMlp.classList.remove("hidden");
  if (algo === "gptgen") els.paramsGptgen.classList.remove("hidden");
  if (algo === "bpe") els.paramsBpe.classList.remove("hidden");
}

function buildTrace() {
  stopPlayback();
  const algo = els.algoSelect.value;
  player.algo = algo;
  const spec = ALGOS[algo];
  const result = spec.buildTrace();

  player.trace = result.trace || [];
  player.code = spec.code;
  player.step = -1;

  els.algoTitle.textContent = spec.title;
  renderCode(-1);
  renderTimeline();
  els.visualArea.innerHTML = "";
  els.stepLabel.textContent = "Trace construite. Clique Step +1 ou Play.";
  els.traceSummary.innerHTML = `<strong>Trace</strong><br>${player.trace.length} etapes`;
}

function renderCode(activeLine) {
  const rows = player.code.map((line, idx) => {
    const mark = idx === activeLine ? ">" : " ";
    return `${mark} ${String(idx + 1).padStart(2, "0")} | ${line}`;
  });
  els.algoCode.textContent = rows.join("\n");
}

function renderTimeline() {
  els.timeline.innerHTML = "";
  player.trace.forEach((step, i) => {
    const d = document.createElement("button");
    d.type = "button";
    d.className = `timeline-step ${i === player.step ? "active" : ""}`;
    d.textContent = String(i + 1);
    d.addEventListener("click", () => gotoStep(i));
    els.timeline.appendChild(d);
  });
}

function renderHeatmap(tokens, matrix, caption = "") {
  const n = tokens.length;
  const wrap = document.createElement("div");
  if (caption) {
    const p = document.createElement("p");
    p.className = "muted";
    p.textContent = caption;
    wrap.appendChild(p);
  }

  const grid = document.createElement("div");
  grid.className = "heatmap";
  grid.style.gridTemplateColumns = `repeat(${n + 1}, minmax(36px, 1fr))`;

  const tl = document.createElement("div");
  tl.className = "heat-cell";
  tl.textContent = "q\\k";
  tl.style.background = "#f6f2e8";
  grid.appendChild(tl);

  tokens.forEach((t) => {
    const h = document.createElement("div");
    h.className = "heat-cell";
    h.style.background = "#f6f2e8";
    h.textContent = String(t).slice(0, 6);
    grid.appendChild(h);
  });

  for (let i = 0; i < n; i += 1) {
    const rh = document.createElement("div");
    rh.className = "heat-cell";
    rh.style.background = "#f6f2e8";
    rh.textContent = String(tokens[i]).slice(0, 6);
    grid.appendChild(rh);

    for (let j = 0; j < n; j += 1) {
      const v = matrix[i][j];
      const alpha = Number.isFinite(v) ? Math.max(0, Math.min(1, Math.abs(v) <= 1 ? v : (v + 10) / 20)) : 0;
      const c = document.createElement("div");
      c.className = "heat-cell";
      c.style.background = Number.isFinite(v) ? `rgba(13,148,136,${0.1 + alpha * 0.9})` : "#f0ede4";
      c.style.color = alpha > 0.55 ? "#fff" : "#0b2522";
      c.textContent = Number.isFinite(v) ? v.toFixed(2) : "-inf";
      grid.appendChild(c);
    }
  }

  wrap.appendChild(grid);
  return wrap;
}

function renderTable(headers, rows) {
  const wrap = document.createElement("div");
  wrap.className = "table-wrap";
  const table = document.createElement("table");
  const thead = document.createElement("thead");
  const trh = document.createElement("tr");
  headers.forEach((h) => {
    const th = document.createElement("th");
    th.textContent = h;
    trh.appendChild(th);
  });
  thead.appendChild(trh);

  const tbody = document.createElement("tbody");
  rows.forEach((r) => {
    const tr = document.createElement("tr");
    r.forEach((cell) => {
      const td = document.createElement("td");
      td.textContent = cell;
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });

  table.appendChild(thead);
  table.appendChild(tbody);
  wrap.appendChild(table);
  return wrap;
}

function renderBarsPair(left, right, leftLabel, rightLabel) {
  const wrap = document.createElement("div");
  wrap.className = "row-2";

  const mk = (arr, label) => {
    const sec = document.createElement("div");
    const h = document.createElement("h3");
    h.textContent = label;
    const bars = document.createElement("div");
    bars.className = "bars";
    const absMax = Math.max(...arr.map((v) => Math.abs(v)), 1e-6);
    arr.forEach((v, i) => {
      const row = document.createElement("div");
      row.className = "bar";
      const fill = document.createElement("span");
      fill.style.width = `${Math.min(100, (Math.abs(v) / absMax) * 100)}%`;
      fill.style.background = v >= 0 ? "linear-gradient(90deg,#0f766e,#14b8a6)" : "linear-gradient(90deg,#9f1239,#f43f5e)";
      const em = document.createElement("em");
      em.textContent = `${i}: ${Number(v).toFixed(3)}`;
      row.appendChild(fill);
      row.appendChild(em);
      bars.appendChild(row);
    });
    sec.appendChild(h);
    sec.appendChild(bars);
    return sec;
  };

  wrap.appendChild(mk(left, leftLabel));
  wrap.appendChild(mk(right, rightLabel));
  return wrap;
}

function renderChips(chips, extra = "") {
  const wrap = document.createElement("div");
  if (extra) {
    const p = document.createElement("p");
    p.className = "muted";
    p.textContent = extra;
    wrap.appendChild(p);
  }
  const row = document.createElement("div");
  chips.forEach((c) => {
    const d = document.createElement("span");
    d.className = "token-chip";
    d.textContent = c;
    row.appendChild(d);
  });
  wrap.appendChild(row);
  return wrap;
}

function renderState(stepObj) {
  els.visualArea.innerHTML = "";
  const st = stepObj.state || {};

  if (st.msg) {
    const p = document.createElement("p");
    p.className = "muted";
    p.textContent = st.msg;
    els.visualArea.appendChild(p);
    return;
  }

  if (st.chips) {
    els.visualArea.appendChild(renderChips(st.chips, st.extra || ""));
  }

  if (st.type === "heatmap") {
    els.visualArea.appendChild(renderHeatmap(st.tokens || [], st.matrix || [], st.caption || ""));
    return;
  }

  if (st.type === "table") {
    if (st.extra && !st.chips) {
      const p = document.createElement("p");
      p.className = "muted";
      p.textContent = st.extra;
      els.visualArea.appendChild(p);
    }
    els.visualArea.appendChild(renderTable(st.headers || [], st.rows || []));
    return;
  }

  if (st.type === "bars") {
    els.visualArea.appendChild(renderBarsPair(st.left || [], st.right || [], st.leftLabel || "left", st.rightLabel || "right"));
  }
}

function gotoStep(idx) {
  if (!player.trace.length) return;
  player.step = Math.max(0, Math.min(player.trace.length - 1, idx));
  const s = player.trace[player.step];
  renderCode(s.line ?? -1);
  renderTimeline();
  els.stepLabel.textContent = `Etape ${player.step + 1}/${player.trace.length} - ${s.label}`;
  renderState(s);
}

function stepNext() {
  if (!player.trace.length) return;
  if (player.step + 1 >= player.trace.length) {
    stopPlayback();
    return;
  }
  gotoStep(player.step + 1);
}

function resetPlayer() {
  stopPlayback();
  player.step = -1;
  renderCode(-1);
  renderTimeline();
  els.visualArea.innerHTML = "";
  els.stepLabel.textContent = "Player reset. Clique Step +1 ou Play.";
}

function play() {
  stopPlayback();
  if (!player.trace.length) return;
  if (player.step >= player.trace.length - 1) player.step = -1;
  player.timer = setInterval(() => {
    if (player.step + 1 >= player.trace.length) {
      stopPlayback();
      return;
    }
    stepNext();
  }, player.speed);
}

function stopPlayback() {
  if (player.timer) {
    clearInterval(player.timer);
    player.timer = null;
  }
}

function updateAlgoUI() {
  const algo = els.algoSelect.value;
  showParamsForAlgo(algo);
  const spec = ALGOS[algo];
  els.algoTitle.textContent = spec.title;
  player.code = spec.code;
  renderCode(-1);
  els.traceSummary.innerHTML = "<strong>Trace</strong><br>pas encore construite";
}

els.algoSelect.addEventListener("change", () => {
  stopPlayback();
  updateAlgoUI();
});

els.buildTrace.addEventListener("click", buildTrace);
els.playerReset.addEventListener("click", resetPlayer);
els.playerStep.addEventListener("click", stepNext);
els.playerPlay.addEventListener("click", play);
els.playerPause.addEventListener("click", stopPlayback);

els.playerSpeed.addEventListener("input", () => {
  player.speed = Number(els.playerSpeed.value);
  els.playerSpeedOut.value = `${player.speed} ms`;
  if (player.timer) {
    play();
  }
});

els.attnHeads.addEventListener("input", () => {
  els.attnHeadsOut.value = els.attnHeads.value;
});
els.attnTemp.addEventListener("input", () => {
  els.attnTempOut.value = els.attnTemp.value;
});
els.dsSeq.addEventListener("input", () => {
  els.dsSeqOut.value = els.dsSeq.value;
});
els.mlpMult.addEventListener("input", () => {
  els.mlpMultOut.value = els.mlpMult.value;
});
els.gptMaxLen.addEventListener("input", () => {
  els.gptMaxLenOut.value = els.gptMaxLen.value;
});
els.gptVocab.addEventListener("input", () => {
  els.gptVocabOut.value = els.gptVocab.value;
});
els.gptTopk.addEventListener("input", () => {
  els.gptTopkOut.value = els.gptTopk.value;
});
els.bpeMerges.addEventListener("input", () => {
  els.bpeMergesOut.value = els.bpeMerges.value;
});

function parseJsonl(text) {
  const rows = [];
  const lines = text.split(/\r?\n/).filter((l) => l.trim());
  for (const line of lines.slice(0, 3000)) {
    try {
      rows.push(JSON.parse(line));
    } catch {
      // ignore broken lines
    }
  }
  return rows;
}

function domainFromUrl(url) {
  try {
    return new URL(url).hostname;
  } catch {
    return "invalid";
  }
}

function renderJsonl(rows) {
  const total = rows.length;
  const avgWords = total ? Math.round(rows.reduce((acc, r) => acc + Number(r.word_count || 0), 0) / total) : 0;

  const domainCounts = new Map();
  rows.forEach((r) => {
    const d = domainFromUrl(String(r.url || ""));
    domainCounts.set(d, (domainCounts.get(d) || 0) + 1);
  });

  const topDomains = [...domainCounts.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5)
    .map(([d, c]) => `${d} (${c})`)
    .join(" | ");

  els.jsonlStats.innerHTML = "";
  [
    ["Records", String(total)],
    ["Avg words", String(avgWords)],
    ["Top domains", topDomains || "n/a"]
  ].forEach(([k, v]) => {
    const d = document.createElement("div");
    d.className = "meta-item";
    d.innerHTML = `<strong>${k}</strong><br>${v}`;
    els.jsonlStats.appendChild(d);
  });

  els.jsonlBody.innerHTML = "";
  rows.slice(0, 25).forEach((r) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${String(r.url || "").slice(0, 90)}</td><td>${String(r.title || "").slice(0, 70)}</td><td>${Number(r.word_count || 0)}</td>`;
    els.jsonlBody.appendChild(tr);
  });
}

async function loadJsonlFromFile(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ""));
    reader.onerror = () => reject(reader.error);
    reader.readAsText(file);
  });
}

async function runJsonlInspector() {
  let text = els.jsonlText.value.trim();
  const file = els.jsonlFile.files && els.jsonlFile.files[0];
  if (!text && file) {
    text = await loadJsonlFromFile(file);
  }
  if (!text) {
    alert("Colle du JSONL ou charge un fichier.");
    return;
  }
  renderJsonl(parseJsonl(text));
}

function loadDemoJsonl() {
  const demo = [
    { text: "Transformers are sequence models.", url: "https://example.org/transformers", title: "Transformers", word_count: 320 },
    { text: "BPE builds a subword vocabulary.", url: "https://example.org/tokenizer", title: "Tokenizer", word_count: 210 },
    { text: "Attention allows global context.", url: "https://docs.python.org/", title: "Attention", word_count: 180 },
    { text: "SQLite FTS5 enables full text search.", url: "https://sqlite.org/fts5.html", title: "FTS5", word_count: 265 }
  ];
  els.jsonlText.value = demo.map((r) => JSON.stringify(r)).join("\n");
}

els.runJsonl.addEventListener("click", () => runJsonlInspector().catch((e) => alert(`Erreur: ${e.message}`)));
els.demoJsonl.addEventListener("click", loadDemoJsonl);

renderDocs();
showSymbol(DOCS[0].module, DOCS[0].symbols[0]);
updateAlgoUI();
buildTrace();
