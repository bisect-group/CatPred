"""Fast trainer for one CatPred run (10 ensemble members) with the maths of catpred.train.run_training.

What is the same as the original: model, loss, Noam schedule, Adam, batch order, padding of every batch,
initial weights of every member, validation rule, best-epoch checkpoint and ensemble averaging.
What is different: how batches reach the model.
- Graph tensors and ESM rows are built from caches by background threads and copied to the GPU in the background.
- The Python loop of the original forward (per-record tensors, pad_sequence on the CPU) is replaced by a few tensor ops.
- Members can train in separate processes, because member inits and the batch order are computed up front.
Optional deviations (all off by default) are listed in `Options`.
"""

import functools
import os
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from random import Random

import numpy as np
import torch
import torch.nn.functional as F

from emulator_bench.common import SMILES_COL, TARGET_COL, ensure_repo_on_path
from emulator_bench.esm_cache import ESM_DIM, sequence_key
from emulator_bench.fused_attention import PRECISIONS, fused_mha

ensure_repo_on_path()

AA_TO_ID = {"A": 0, "R": 1, "N": 2, "D": 3, "C": 4, "Q": 5, "E": 6, "G": 7, "H": 8, "I": 9,
            "L": 10, "K": 11, "M": 12, "F": 13, "P": 14, "S": 15, "T": 16, "W": 17, "Y": 18, "V": 19}  # same as seq_to_tensor
PAD_ID = 20
RECORDS_PLACEHOLDER = "esm2_feats_from_cache"


@dataclass(frozen=True)
class Options:
    """Deviations from the original maths. All off by default."""

    tf32: bool = False  # TF32 matmuls
    amp: bool = False  # bf16 autocast in forward
    fused_adam: bool = False  # torch.optim.Adam(fused=True)
    skip_pad: bool = False  # first pooling layer: ESM block of linear1 only on real residues (same maths, other rounding)
    fused_attn: bool = False  # Triton kernel for the attention core of the Seq-Attn block (same maths, other rounding, GPU only)
    compile: bool = False  # torch.compile(dynamic=True) of the attentive pooling (same maths, other rounding)
    attn_precision: str = "ieee"  # dot products of the kernel: ieee (fp32), tf32 or bf16 (fp32 accumulation). Not exact unless ieee


def build_train_args(train_csv, val_csv, test_csv, save_dir, seed, epochs=30, ensemble_size=10, batch_size=16, cpu=False, egnn_path=None):
    """TrainArgs for the released kcat and Km settings (scripts/reproduce_fig5_catpred.sh).
    With egnn_path, the ProGRES vectors of that dict are added as in the original `--add_pretrained_egnn_feats`."""
    from catpred.args import TrainArgs

    argv = [
        "--protein_records_path", RECORDS_PLACEHOLDER,
        "--data_path", str(train_csv), "--separate_val_path", str(val_csv), "--separate_test_path", str(test_csv),
        "--dataset_type", "regression", "--smiles_columns", SMILES_COL, "--target_columns", TARGET_COL,
        "--extra_metrics", "mae", "mse", "r2",
        "--ensemble_size", str(ensemble_size), "--seq_embed_dim", "36", "--seq_self_attn_nheads", "6",
        "--loss_function", "mve", "--batch_size", str(batch_size), "--save_dir", str(save_dir), "--epochs", str(epochs),
        "--add_esm_feats", "--seed", str(seed), "--pytorch_seed", str(seed),
    ]
    if egnn_path:
        argv += ["--add_pretrained_egnn_feats", "--pretrained_egnn_feats_path", str(egnn_path)]
    args = TrainArgs().parse_args(argv + (["--no_cuda"] if cpu else ["--gpu", "0"]))
    args.num_folds = 1
    return args


# ----------------------------------------------------------------------------- data


@dataclass
class GraphArrays:
    """One molecule as arrays. Index arrays hold the 1-based local index and 0 for padding."""

    n_atoms: int
    n_bonds: int
    f_atoms: np.ndarray
    f_bonds: np.ndarray
    a2b: np.ndarray
    b2a: np.ndarray
    b2revb: np.ndarray


def graph_arrays(mol_graph) -> GraphArrays:
    from catpred.features import get_atom_fdim, get_bond_fdim

    width = max(1, max((len(x) for x in mol_graph.a2b), default=1))
    a2b = np.zeros((mol_graph.n_atoms, width), dtype=np.int64)
    for a, bonds in enumerate(mol_graph.a2b):
        a2b[a, : len(bonds)] = np.asarray(bonds, dtype=np.int64) + 1
    return GraphArrays(
        mol_graph.n_atoms,
        mol_graph.n_bonds,
        np.asarray(mol_graph.f_atoms, dtype=np.float32).reshape(mol_graph.n_atoms, get_atom_fdim()),
        np.asarray(mol_graph.f_bonds, dtype=np.float32).reshape(mol_graph.n_bonds, get_bond_fdim()),
        a2b,
        np.asarray(mol_graph.b2a, dtype=np.int64) + 1,
        np.asarray(mol_graph.b2revb, dtype=np.int64) + 1,
    )


class SplitData:
    """Rows of one split: graph arrays, token ids, ESM rows and optional ProGRES vectors, shared between rows with the same SMILES or sequence.

    egnn is (dict of sequence key to vector, mean vector). A sequence missing from the dict gets the mean, as in the original model.
    """

    def __init__(self, dataset, sequences, esm_cache, graph_by_smiles=None, egnn=None):
        from catpred.features import MolGraph

        assert len(dataset) == len(sequences)
        self.dataset = dataset
        self.sequences = list(sequences)
        self.n = len(dataset)
        graph_by_smiles = {} if graph_by_smiles is None else graph_by_smiles
        self.graphs, self.tokens, self.esm, self.egnn = [], [], [], []
        hits = 0
        token_by_seq = {}
        for d, seq in zip(dataset._data, sequences):
            smiles = d.smiles[0]
            if smiles not in graph_by_smiles:
                graph_by_smiles[smiles] = graph_arrays(
                    MolGraph(d.mol[0], d.atom_features, d.bond_features,
                             overwrite_default_atom_features=d.overwrite_default_atom_features,
                             overwrite_default_bond_features=d.overwrite_default_bond_features)
                )
            self.graphs.append(graph_by_smiles[smiles])
            if seq not in token_by_seq:
                token_by_seq[seq] = np.fromiter((AA_TO_ID[a] for a in seq), dtype=np.int64, count=len(seq))
            self.tokens.append(token_by_seq[seq])
            self.esm.append(esm_cache.get(seq))
            if egnn is not None:
                key = sequence_key(seq)
                hits += key in egnn[0]
                self.egnn.append(egnn[0].get(key, egnn[1]))
        self.egnn_hit_rate = hits / max(self.n, 1) if egnn is not None else float("nan")  # share of rows with a real vector
        self.targets = [d.targets for d in dataset._data]  # scaled for the train split, raw for the others


@dataclass
class HostBatch:
    floats: torch.Tensor  # f_atoms and f_bonds, flat
    ints: torch.Tensor  # a2b, b2a, b2revb, index of each ESM row in the padded tensor
    tokens: torch.Tensor
    targets: torch.Tensor
    esm_rows: torch.Tensor
    egnn: torch.Tensor  # (batch, 128), or (batch, 0) without ProGRES vectors
    dims: tuple  # n_atoms+1, n_bonds+1, width, atom_fdim, bond_fdim, batch size, padded length
    a_scope: list
    b_scope: list

    def tensors(self) -> list:
        return [self.floats, self.ints, self.tokens, self.targets, self.esm_rows, self.egnn]


def _pinned(shape, dtype):
    return torch.empty(shape, dtype=dtype, pin_memory=torch.cuda.is_available())


def build_host_batch(split: SplitData, rows) -> HostBatch:
    """Same tensors as BatchMolGraph plus padded tokens and ESM rows for the rows of one batch."""
    graphs = [split.graphs[r] for r in rows]
    n_atoms = 1 + sum(g.n_atoms for g in graphs)
    n_bonds = 1 + sum(g.n_bonds for g in graphs)
    width = max(1, max(g.a2b.shape[1] for g in graphs))
    atom_fdim, bond_fdim = graphs[0].f_atoms.shape[1], graphs[0].f_bonds.shape[1]
    esm = [split.esm[r] for r in rows]
    length = max(e.shape[0] for e in esm)
    n_rows = sum(e.shape[0] for e in esm)

    floats = _pinned(n_atoms * atom_fdim + n_bonds * bond_fdim, torch.float32)
    fa = floats[: n_atoms * atom_fdim].view(n_atoms, atom_fdim).numpy()
    fb = floats[n_atoms * atom_fdim :].view(n_bonds, bond_fdim).numpy()
    fa[0] = 0
    fb[0] = 0
    ints = _pinned(n_atoms * width + 2 * n_bonds + n_rows, torch.int64)
    a2b = ints[: n_atoms * width].view(n_atoms, width).numpy()
    b2a = ints[n_atoms * width : n_atoms * width + n_bonds].numpy()
    b2revb = ints[n_atoms * width + n_bonds : n_atoms * width + 2 * n_bonds].numpy()
    esm_index = ints[n_atoms * width + 2 * n_bonds :].numpy()
    a2b[:] = 0
    b2a[0] = b2revb[0] = 0
    esm_rows = _pinned((n_rows, ESM_DIM), torch.float32)
    esm_np = esm_rows.numpy()
    tokens = torch.full((len(rows), length), PAD_ID, dtype=torch.int64)

    a0, b0, r0 = 1, 1, 0
    a_scope, b_scope = [], []
    for i, (g, e) in enumerate(zip(graphs, esm)):
        fa[a0 : a0 + g.n_atoms] = g.f_atoms
        fb[b0 : b0 + g.n_bonds] = g.f_bonds
        block = a2b[a0 : a0 + g.n_atoms, : g.a2b.shape[1]]
        block[:] = g.a2b
        block[g.a2b > 0] += b0 - 1
        b2a[b0 : b0 + g.n_bonds] = g.b2a + (a0 - 1)
        b2revb[b0 : b0 + g.n_bonds] = g.b2revb + (b0 - 1)
        a_scope.append((a0, g.n_atoms))
        b_scope.append((b0, g.n_bonds))
        esm_np[r0 : r0 + e.shape[0]] = e
        esm_index[r0 : r0 + e.shape[0]] = i * length + np.arange(e.shape[0])
        tok = split.tokens[rows[i]][:length]
        tokens[i, : len(tok)] = torch.from_numpy(tok)
        a0 += g.n_atoms
        b0 += g.n_bonds
        r0 += e.shape[0]
    targets = torch.tensor([split.targets[r] for r in rows])
    egnn = torch.stack([split.egnn[r] for r in rows]) if split.egnn else torch.empty(len(rows), 0)
    return HostBatch(floats, ints, tokens, targets, esm_rows, egnn,
                     (n_atoms, n_bonds, width, atom_fdim, bond_fdim, len(rows), length), a_scope, b_scope)


class FastGraph:
    """Stands in for BatchMolGraph inside MPNEncoder.forward."""

    def __init__(self, components):
        self.components = components

    def get_components(self, atom_messages: bool = False):
        return self.components


@dataclass
class DeviceBatch:
    graph: FastGraph
    tokens: torch.Tensor
    esm: torch.Tensor
    esm_rows: torch.Tensor  # the real residues only, in row order
    esm_index: torch.Tensor  # position of each esm_rows row in esm viewed as (batch * length, 1280)
    egnn: torch.Tensor
    targets: torch.Tensor
    n: int


def to_device(hb: HostBatch, device) -> list:
    """Start the copies of a host batch. Returns the device tensors in a list."""
    return [t.to(device, non_blocking=True) for t in hb.tensors()]


def assemble(hb: HostBatch, tensors) -> DeviceBatch:
    """Cut views out of the copied buffers and scatter the ESM rows into the zero-padded tensor."""
    floats, ints, tokens, targets, esm_rows, egnn = tensors
    n_atoms, n_bonds, width, atom_fdim, bond_fdim, batch, length = hb.dims
    f_atoms = floats[: n_atoms * atom_fdim].view(n_atoms, atom_fdim)
    f_bonds = floats[n_atoms * atom_fdim :].view(n_bonds, bond_fdim)
    a2b = ints[: n_atoms * width].view(n_atoms, width)
    b2a = ints[n_atoms * width : n_atoms * width + n_bonds]
    b2revb = ints[n_atoms * width + n_bonds : n_atoms * width + 2 * n_bonds]
    esm_index = ints[n_atoms * width + 2 * n_bonds :]
    esm = torch.zeros(batch * length, ESM_DIM, device=esm_rows.device)
    esm.index_copy_(0, esm_index, esm_rows)
    graph = FastGraph((f_atoms, f_bonds, a2b, b2a, b2revb, hb.a_scope, hb.b_scope))
    return DeviceBatch(graph, tokens, esm.view(batch, length, ESM_DIM), esm_rows, esm_index, egnn, targets, batch)


class Prefetcher:
    """Builds host batches in threads and copies them to the GPU on a side stream, in order."""

    def __init__(self, split: SplitData, batches, device, threads=2, depth=4):
        self.split, self.batches, self.device = split, batches, device
        self.threads, self.depth = threads, depth
        self.local = threading.local()

    def _produce(self, rows):
        hb = build_host_batch(self.split, rows)
        if self.device.type != "cuda":
            return hb, hb.tensors(), None, None
        if not hasattr(self.local, "stream"):
            self.local.stream = torch.cuda.Stream(self.device)
        with torch.cuda.stream(self.local.stream):
            tensors = to_device(hb, self.device)
            event = torch.cuda.Event()
            event.record(self.local.stream)
        return hb, tensors, event, hb

    def __iter__(self):
        pending = deque()
        items = iter(self.batches)
        with ThreadPoolExecutor(self.threads) as pool:
            for rows in items:
                pending.append(pool.submit(self._produce, rows))
                if len(pending) >= self.depth:
                    break
            while pending:
                hb, tensors, event, keep = pending.popleft().result()
                for rows in items:
                    pending.append(pool.submit(self._produce, rows))
                    break
                if event is None:
                    yield assemble(hb, tensors)
                    continue
                stream = torch.cuda.current_stream(self.device)
                stream.wait_event(event)
                for t in tensors:
                    t.record_stream(stream)
                yield assemble(hb, tensors)


# ----------------------------------------------------------------------------- model


@functools.cache
def compiled(fn):
    return torch.compile(fn, dynamic=True)


def _pool_math(w1, b1, w2, b2, seq_outs, esm_rows, esm_index, esm):
    batch, length, _ = seq_outs.shape
    hidden = F.linear(seq_outs, w1[:, ESM_DIM:], b1).view(batch * length, -1)
    hidden = hidden.index_add(0, esm_index, F.linear(esm_rows, w1[:, :ESM_DIM])).view(batch, length, -1)
    weights = torch.softmax(F.linear(torch.tanh(hidden), w2, b2), dim=1).permute(0, 2, 1)
    return torch.cat([torch.bmm(weights, esm), torch.bmm(weights, seq_outs)], dim=-1).squeeze(1)


def pool_skip_pad(pooler, db: DeviceBatch, seq_outs, use_compile: bool = False):
    """AttentivePooling.forward on cat([esm, seq_outs]) without the ESM columns at padded positions.

    The ESM input is zero at padded positions, so the ESM block of linear1 contributes zero there and is applied to the real residues
    only. The attention weights still cover every position, as in the original. The maths is equal, the rounding differs.
    """
    math = compiled(_pool_math) if use_compile else _pool_math
    return math(pooler.linear1.weight, pooler.linear1.bias, pooler.linear2.weight, pooler.linear2.bias, seq_outs, db.esm_rows, db.esm_index, db.esm)


def forward(model, db: DeviceBatch, opts: Options = Options()):
    """MoleculeModel.forward for Seq-Attn + ESM2 features (and ProGRES vectors if given) and the mve loss, from prepared tensors."""
    encodings = model.encoder.encoder[0](db.graph)
    seq_outs = model.seq_embedder(db.tokens)
    rotate = model.rotary_embedder.rotate_queries_or_keys  # not compiled: its cached frequencies recompile at every new longest sequence
    q = rotate(seq_outs, seq_dim=1)
    if opts.fused_attn:  # q and k are equal tensors in the original, so one rotary call serves both
        seq_outs = fused_mha(model.multihead_attn, q, q, seq_outs, opts.attn_precision)
    else:
        k = rotate(seq_outs, seq_dim=1)
        seq_outs, _ = model.multihead_attn(q, k, seq_outs)
    if opts.skip_pad:
        pooled = pool_skip_pad(model.attentive_pooler, db, seq_outs, opts.compile)
    else:
        pooler = compiled(model.attentive_pooler.forward) if opts.compile else model.attentive_pooler
        pooled, _ = pooler(torch.cat([db.esm, seq_outs], dim=-1))
    if db.egnn.shape[1]:
        pooled = torch.cat([pooled, db.egnn], dim=-1)
    output = model.readout(torch.cat([pooled, encodings], dim=-1))
    means, variances = torch.split(output, output.shape[1] // 2, dim=1)
    return torch.cat([means, model.softplus(variances)], dim=1)


def batches_of(order, batch_size):
    return [order[i : i + batch_size] for i in range(0, len(order), batch_size)]


@torch.no_grad()
def predict_split(model, split: SplitData, scaler, batch_size, device, opts=Options()) -> np.ndarray:
    """Predicted means in the original target scale, shape (n, 1), for sequential batches as in catpred.train.predict."""
    model.eval()
    outputs = []
    for db in Prefetcher(split, batches_of(list(range(split.n)), batch_size), device):
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=opts.amp):
            out = forward(model, db, opts)
        outputs.append(out.float()[:, :1])
    means = torch.cat(outputs).cpu().numpy()
    return scaler.inverse_transform(means).astype(np.float64)


def member_inits(args, n_members, epochs):
    """State dict of each member's fresh model, with the torch RNG stream of run_training.

    Between two members the original run draws one value from the global RNG for each data loader iteration
    (train and validation per epoch, then test), and builds a second model when it reloads the best checkpoint.
    """
    from catpred.models import MoleculeModel

    torch.manual_seed(args.pytorch_seed)
    states = []
    for _ in range(n_members):
        states.append({k: v.clone() for k, v in MoleculeModel(args).state_dict().items()})
        for _ in range(2 * epochs):
            torch.empty((), dtype=torch.int64).random_()
        MoleculeModel(args)
        torch.empty((), dtype=torch.int64).random_()
    return states


def epoch_orders(seed, n_rows, n_epochs_total):
    """Row order of every epoch of the whole run. The original sampler keeps one Random(seed) over all members."""
    rnd = Random(seed)
    for _ in range(n_epochs_total):
        order = list(range(n_rows))
        rnd.shuffle(order)
        yield order


def train_step(model, db: DeviceBatch, loss_func, optimizer, scheduler, count_cache: dict, opts: Options):
    """One step of catpred.train.train.train. The original masks and data weights are all ones, so they are left out."""
    if db.n not in count_cache:
        count_cache[db.n] = torch.tensor(db.n, device=db.targets.device)
    optimizer.zero_grad()
    with torch.autocast(db.targets.device.type, dtype=torch.bfloat16, enabled=opts.amp):
        preds = forward(model, db, opts)
    loss = loss_func(preds.float(), db.targets).sum() / count_cache[db.n]
    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss.detach()


def train_member(args, member, init_state, orders, train: SplitData, val: SplitData, scaler, device, opts=Options(), log=print):
    """Train one member for args.epochs epochs. Returns the best state dict, its epoch and validation score, and the history."""
    from catpred.models import MoleculeModel
    from catpred.train.evaluate import evaluate_predictions
    from catpred.train.loss_functions import get_loss_func
    from catpred.utils import build_lr_scheduler, multitask_mean

    model = MoleculeModel(args)
    model.load_state_dict(init_state)
    model = model.to(device)
    optimizer = torch.optim.Adam([{"params": model.parameters(), "lr": args.init_lr, "weight_decay": 0}], fused=opts.fused_adam)
    scheduler = build_lr_scheduler(optimizer, args)
    loss_func = get_loss_func(args)
    best_score, best_epoch, best_state, history = float("inf"), 0, None, []
    count_cache = {}
    for epoch in range(args.epochs):
        model.train()
        loss_sum = torch.zeros((), device=device)
        for db in Prefetcher(train, batches_of(next(orders), args.batch_size), device):
            loss = train_step(model, db, loss_func, optimizer, scheduler, count_cache, opts)
            loss_sum += loss
        preds = predict_split(model, val, scaler, args.batch_size, device, opts)
        scores = evaluate_predictions(preds=preds.tolist(), targets=val.targets, num_tasks=1, metrics=args.metrics,
                                      dataset_type=args.dataset_type)
        score = multitask_mean(scores[args.metric], metric=args.metric)
        history.append({"epoch": epoch, "train_loss_sum": loss_sum.item(), "val_" + args.metric: score})
        log(f"member {member} epoch {epoch} val_{args.metric} {score:.6f}")
        if score < best_score:
            best_score, best_epoch = score, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best_epoch, best_score, history


def settings_options(opts: Options) -> dict:
    """Options for run_settings.json. The keys of the newer options appear only when used, so older run folders keep their settings."""
    absent = ({"fused_attn", "attn_precision"} if not opts.fused_attn else set()) | ({"compile"} if not opts.compile else set())
    return {k: v for k, v in vars(opts).items() if k not in absent}


def configure_runtime(opts: Options, deterministic: bool = False) -> None:
    assert opts.attn_precision in PRECISIONS
    torch.backends.cuda.matmul.allow_tf32 = opts.tf32
    torch.backends.cudnn.allow_tf32 = opts.tf32
    if deterministic:
        torch.use_deterministic_algorithms(True)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

