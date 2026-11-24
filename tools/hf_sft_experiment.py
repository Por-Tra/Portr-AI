"""
████████████   ██ ██████              ██                            ████      ██
██        ██ ████     ████            ███                          ██ ███     ██
██        ██████        ███ ██████████████████ ████████           ███  ███    ██
███████████████         ███ ██        ███      ██                ██     ███   ██
██          ███         ███ ██        ███      ██     █████████ ████████████  ██
██           █████   █████  ██        ████     ██              ███        ███ ██
██             █████████    ██         ███████ ██             ███          ██ ██

auteur : @ChatGPT(LLM)

File name : hf_sft_experiment.py

Pinned Luth data, grouped holdouts and isolated 256/512-context SFT trials.

"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import main
import prepare_sft_data as prep
from data.artifacts import atomic_json,sha256_file
from data.dataset import ResumableRandomSampler
from data.sft_dataset import SFTDataset
from inference.generate import build_sft_prompt_ids,load_model_from_final_state
from tokenizer.tokenizer import Tokenizer
from tools.audit_hf_sft import pair
from tools.repair_sft import curriculum,dump_lines,read_lines,score
from training.sft_train import SFTConfig,train_sft_model

OUT=ROOT/'artifacts/sft/hf_training/20261117'
REV='07229214c6ceba25575775efeab12675d7722aaf'
SEED=20261117
QUOTAS={'luth_aya_dataset':(1000,400),'luth_croissantllm':(3000,3000),
        'luth_smoltalk2':(3000,4000),'luth_tulu3_persona_instruct':(3000,3000),
        'luth_openhermes':(20000,7000)}


def protect():
    OUT.mkdir(parents=True,exist_ok=True)
    file=OUT/'protected.json'
    if not file.exists():
        paths=[main._base_model_path(),Path(main.CHECKPOINT_DIR)/main.CHECKPOINT_NAME,
               Path(main.TOKENIZER_MODEL_PATH),main._sft_model_path(),main.SFT_ACTIVE_RUN_PATH,
               main.SFT_TRAIN_PATH,main.SFT_VAL_PATH,main.SFT_META_PATH]
        atomic_json(file,{'files':{str(p.resolve()):sha256_file(p) for p in paths},
                         'initial_sft':str(main._sft_model_path().resolve()),
                         'active_selection':main.SFT_ACTIVE_RUN_PATH.read_text(encoding='utf-8')})
    data=json.loads(file.read_text(encoding='utf-8'))
    for p,digest in data['files'].items():
        if sha256_file(p)==digest:continue
        if Path(p).resolve()==main.SFT_ACTIVE_RUN_PATH.resolve() and (OUT/'promotion.json').exists():
            promotion=json.loads((OUT/'promotion.json').read_text(encoding='utf-8'))
            selected=Path(promotion['run_dir']).resolve()
            pointer=json.loads(Path(p).read_text(encoding='utf-8'))
            if (promotion['previous_pointer_sha256']==digest and Path(pointer['run_dir']).resolve()==selected
                and sha256_file(selected/'models'/main.SFT_FINAL_MODEL_NAME)==promotion['model_sha256']):continue
        raise RuntimeError('Protected input changed: '+p)
    return data


def choose(name):
    saved=protect()
    results={n:json.loads((OUT/'evaluation'/n/'dev_metrics.json').read_text(encoding='utf-8'))
             for n in ('baseline','luth256','luth512')}
    if name!='luth512':raise ValueError('This documented selection is for luth512')
    reference=results['baseline'];candidate=results[name]
    if not (candidate['hf_short']['ce']<reference['hf_short']['ce']
            and candidate['original_corpus']['ce']<reference['original_corpus']['ce']
            and candidate['rules']['accuracy']>=reference['rules']['accuracy']
            and candidate['long']['accuracy']>=.95):raise ValueError('Development selection failed')
    path=OUT/'selection.json'
    decision={'name':name,'model_sha256':candidate['sha256'],'context':candidate['context'],
        'initial_sft':saved['initial_sft'],'initial_sft_sha256':reference['sha256'],
        'reason':'Best novel Luth short CE and corpus retention; long retrieval 100%; simple rules above baseline. Arithmetic still poor.',
        'dev_metrics_sha256':{n:sha256_file(OUT/'evaluation'/n/'dev_metrics.json') for n in results},
        'test_files_sha256':{str(p.relative_to(OUT)):sha256_file(p) for p in
            [OUT/'rules_test.jsonl',OUT/'long_test.jsonl',OUT/'ctx256/sft_test.npz',OUT/'ctx512/sft_test.npz']}}
    if path.exists():raise RuntimeError('Selection already frozen')
    atomic_json(path,decision)
    print('[selection]',name,'frozen before final test')


def check_selection(name):
    decision=json.loads((OUT/'selection.json').read_text(encoding='utf-8'))
    if name not in ('baseline',decision['name']):raise ValueError('Only frozen candidate and baseline may use final test')
    for p,digest in decision['test_files_sha256'].items():
        if sha256_file(OUT/p)!=digest:raise ValueError('Frozen test changed: '+p)
    for n,digest in decision['dev_metrics_sha256'].items():
        if sha256_file(OUT/'evaluation'/n/'dev_metrics.json')!=digest:raise ValueError('Development metrics changed after selection')
    return decision


def promote(name):
    saved=protect();decision=check_selection(name)
    if name!=decision['name']:raise ValueError('Can only promote selected candidate')
    metrics=json.loads((OUT/'evaluation'/name/'test_metrics.json').read_text(encoding='utf-8'))
    baseline=json.loads((OUT/'evaluation/baseline/test_metrics.json').read_text(encoding='utf-8'))
    if metrics['sha256']!=decision['model_sha256']:raise ValueError('Selected model changed')
    if metrics['hf_short']['ce']>=baseline['hf_short']['ce'] or metrics['long']['accuracy']<.95:
        raise ValueError('Final quality gate failed; retain previous active SFT')
    run=main.SFT_RUNS_DIR/('hf-20261117-'+name)
    atomic_json(OUT/'promotion.json',{'run_dir':str(run.resolve()),'model_sha256':metrics['sha256'],
        'previous_pointer_sha256':saved['files'][str(main.SFT_ACTIVE_RUN_PATH.resolve())],
        'selection_sha256':sha256_file(OUT/'selection.json')})
    main._activate_sft_run(run)
    protect()
    print('[active]',run)


def verify():
    saved=protect();decision=check_selection('luth512')
    tok=Tokenizer.load(main.TOKENIZER_MODEL_PATH)
    if not main._sft_data_ready():raise ValueError('Manager rejects selected dataset')
    if sha256_file(main._sft_model_path())!=decision['model_sha256']:raise ValueError('Wrong active SFT')
    directory=OUT/'ctx512';meta=json.loads((directory/'sft_meta.json').read_text(encoding='utf-8'))
    rows=read_lines(directory/'source.jsonl')
    if any(r['split'] not in ('train','dev') for r in rows):raise ValueError('Test in preparation source')
    assignments={}
    for r in rows:
        if r['group'] in assignments and assignments[r['group']]!=r['split']:raise ValueError('Group leakage')
        assignments[r['group']]=r['split']
    previous=SFTDataset(next(Path(p) for p in saved['files'] if p.endswith('sft_train.npz')))
    seen={tuple(prep.prompt_key((x.tolist(),y.tolist()))[1:-1])
          for x,y in zip(previous.input_ids,previous.labels)}
    data={s:SFTDataset(directory/f'sft_{s}.npz') for s in ('train','val','test')}
    keys={s:{tuple(prep.prompt_key((x.tolist(),y.tolist()))[1:-1])
             for x,y in zip(ds.input_ids,ds.labels)} for s,ds in data.items()}
    if keys['train']&keys['val'] or keys['train']&keys['test'] or keys['val']&keys['test']:
        raise ValueError('Tokenized prompt leakage')
    report={'protected_original_files':len(saved['files']),'manager_ready':True,'active_context':512,
            'base_context':main.SEQ_LENGTH,'source_counts':dict(Counter(r['split'] for r in rows)),
            'model_sha256':sha256_file(main._sft_model_path()),'tokenized_prompt_overlap':0}
    for split,ds in data.items():
        if sha256_file(directory/f'sft_{split}.npz')!=meta[split+'_sha256']:raise ValueError('Dataset changed')
        if len(ds)!=meta[split]['examples']:raise ValueError('Wrong dataset count')
        if ds.input_ids.min()<0 or ds.input_ids.max()>=len(tok.token_to_id):raise ValueError('Token range')
        if (ds.labels!=-100).sum()!=meta[split]['supervised_tokens']:raise ValueError('Label count')
    for split in ('dev','test'):
        for family in ('rules','long'):
            cases=read_lines(OUT/f'{family}_{split}.jsonl')
            case_keys={tuple(tok.encode(r['prompt'])) for r in cases}
            if case_keys&(seen|keys['train']):raise ValueError('Fresh generation case in training')
            report[f'{family}_{split}_cases']=len(cases)
    rng=random.Random(SEED);special=tuple(tok.token_to_id[s] for s in prep.SPECIAL_TOKENS)
    for split,label in (('train','train'),('dev','val')):
        candidates=[r for r in rows if r['split']==split]
        for i in rng.sample(range(len(candidates)),16):
            row=candidates[i];result,reason=prep.build_example(row['prompt'],row['answer'],tok.encode,
                special,512,1,320,None)
            if reason or not np.array_equal(result[0],data[label].input_ids[i]) or not np.array_equal(result[1],data[label].labels[i]):
                raise ValueError('Native tokenizer / labels differ')
    report['native_encoding_and_labels_checked']=32
    atomic_json(OUT/'final_verification.json',report)
    print(json.dumps(report,indent=2))


class CachedEncoder:
    """Exact native BPE with a bounded word cache and early token-limit rejection."""
    def __init__(self,tok):
        self.tok=tok
        self.word=lru_cache(maxsize=120000)(self._word)
    def _word(self,raw):
        return tuple(self.tok.token_to_id.get(t,self.tok.token_to_id['<UNK>'])
                     for t in self.tok.tokenize_word(self.tok.normalize_token(raw)))
    def encode(self,text,limit=512):
        result=[]
        for word in self.tok.split_text(text):
            result.extend(self.word(word))
            if len(result)>limit:return None
        return result


def prompt_hash(ids):return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def encode_row(row,enc):
    p=enc.encode(row['prompt'],509)
    if p is None:return None,'prompt_length'
    a=enc.encode(row['answer'],min(320,509-len(p)))
    if not a:return None,'answer_length'
    unknown=enc.tok.token_to_id['<UNK>']
    if (p+a).count(unknown)>max(1,int(.01*(len(p)+len(a)))):return None,'unknown_tokens'
    return {**row,'_p':p,'_a':a,'tokens':len(p)+len(a)+3},None


def partition(group):
    bucket=int(hashlib.sha256(('split:'+group).encode()).hexdigest()[:8],16)%1000
    return 'dev' if bucket<20 else 'test' if bucket<40 else 'train'


def native_example(row,tok,context):
    p,a=row['_p'],row['_a'];ids=tok.token_to_id
    x=[ids['<|user|>'],*p,ids['<|assistant|>'],*a,ids['<|end|>']]
    y=[-100]*(len(p)+1)+[*a,ids['<|end|>'],-100]
    if len(x)>context:raise ValueError('Context overflow')
    pad=context-len(x)
    return x+[ids['<PAD>']]*pad,y+[-100]*pad


def long_examples(tok):
    rng=random.Random(SEED)
    cities='Paris Lyon Lille Nice Nantes Rennes Rouen Tours Brest Metz Dijon Caen Rome Madrid Berlin Lisbonne Londres Dublin Vienne Oslo'.split()
    filler=['Le jardin est calme ce matin.','Les oiseaux se posent dans les arbres.','Une personne lit un livre près de la fenêtre.',
            'Le vent fait bouger les feuilles.','La table est placée au milieu de la pièce.','Les nuages cachent un peu le soleil.',
            'Un train passe au loin.','La journée se termine tranquillement.','Une lampe éclaire le bureau.',
            'Le petit chat dort sur une chaise.','La porte donne sur une cour.','Une musique douce vient de la pièce voisine.']
    rows=[]
    for i in range(1200):
        city=rng.choice(cities)
        text=f'Information importante : la destination du voyage est {city}.\n'
        goal=rng.randint(300,450)
        while len(tok.encode(text))<goal:text+=rng.choice(filler)+' '
        prompt=text+'\nQuelle est la destination du voyage indiquée au début ? Réponds uniquement par le nom de la ville.'
        split='dev' if i<100 else 'test' if i<200 else 'train'
        rows.append({'prompt':prompt,'answer':city,'family':'memoire_longue','source':'verified_long_context',
                     'group':f'verified-long:{i}','forced_split':split})
    return rows


def build():
    tok=Tokenizer.load(main.TOKENIZER_MODEL_PATH);enc=CachedEncoder(tok)
    # Existing test/dev cases are regression-only. Do not train on their prompts.
    old=ROOT/'artifacts/sft/repairs/20261116'
    blocked={prompt_hash(tok.encode(r['prompt'])) for file in ('test_locked.jsonl','dev_generation.jsonl')
             for r in read_lines(old/file)}
    foundation=curriculum()[0]
    foundation_groups={prompt_hash(tok.encode(r['prompt'])):r['group'] for r in foundation}
    old_train=SFTDataset(old/'sft_train.npz')
    valid_foundation={tuple(prep.prompt_key((x.tolist(),y.tolist()))[1:-1]) for x,y in zip(old_train.input_ids,old_train.labels)}
    verified=[]
    for r in foundation:
        if tuple(tok.encode(r['prompt'])) in valid_foundation:
            e,_=encode_row({**r,'source':'verified_curriculum'},enc)
            if e and prompt_hash(e['_p']) not in blocked:verified.append(e)
    rng=random.Random(SEED);rng.shuffle(verified);records=verified[:6000]
    seen={(prompt_hash(r['_p']),tuple(r['_a'])) for r in records}
    stats={}
    for source,quotas in QUOTAS.items():
        files=sorted((OUT/'downloads/data').glob(source+'-*.parquet'))
        if not files:raise RuntimeError('Missing download: '+source)
        groups=[]
        for file in files:
            parquet=pq.ParquetFile(file)
            groups.extend((file,g) for g in range(parquet.num_row_groups))
        rng.shuffle(groups);kept=[0,0];rejected=Counter();read=0
        for file,g in groups:
            table=pq.ParquetFile(file).read_row_group(g)
            raw=table.to_pylist();rng.shuffle(raw)
            for row in raw:
                read+=1
                result,reason=pair(row)
                if reason:rejected[reason]+=1;continue
                p,a=result
                if any(t in p or t in a for t in ('```','<think>','<tool_call>','\\begin{')):
                    rejected['code_reasoning']+=1;continue
                if any(t in a.casefold() for t in prep.BANNED_IN_ANSWER):
                    rejected['boilerplate_identity']+=1;continue
                e,reason=encode_row({'prompt':p,'answer':a,'source':source,'revision':REV,
                                    'license':'odc-by','source_file':file.name,'row_group':g},enc)
                if reason:rejected[reason]+=1;continue
                ph=prompt_hash(e['_p'])
                if ph in blocked:rejected['reserved_prompt']+=1;continue
                key=(ph,tuple(e['_a']))
                if key in seen:rejected['duplicate_pair']+=1;continue
                grams=[tuple(e['_a'][i:i+4]) for i in range(len(e['_a'])-3)]
                if len(grams)>20 and (len(grams)-len(set(grams)))/len(grams)>.15:
                    rejected['repetitive_answer']+=1;continue
                bucket=int(e['tokens']>256)
                if kept[bucket]>=quotas[bucket]:continue
                e['group']=foundation_groups.get(ph,'prompt:'+ph)
                kept[bucket]+=1;seen.add(key);records.append(e)
                if all(k>=q for k,q in zip(kept,quotas)):break
            if all(k>=q for k,q in zip(kept,quotas)):break
        stats[source]={'read':read,'short':kept[0],'long':kept[1],'rejected':dict(rejected)}
        atomic_json(OUT/'ingest_progress.json',stats)
        print('[data]',source,stats[source],flush=True)
    for r in long_examples(tok):
        e,reason=encode_row(r,enc)
        if reason:raise RuntimeError(reason)
        records.append(e)
    for r in records:r['split']=r.get('forced_split') or partition(r['group'])
    # Fresh, unused rule instances for development and final regression testing.
    old_prompts={r['prompt'] for name in ('test_locked.jsonl','dev_generation.jsonl') for r in read_lines(old/name)}
    residual=[r for r in curriculum()[1] if r['prompt'] not in old_prompts and r['family']!='faits_appris']
    grouped={}
    for r in residual:grouped.setdefault(r['group'],[]).append(r)
    dev,test=[],[]
    for key,values in grouped.items():
        (dev if int(hashlib.sha256(key.encode()).hexdigest()[:8],16)%2 else test).extend(values)
    from tools.repair_sft import balanced
    rule_dev,rule_test=balanced(dev,12),balanced(test,15)
    dump_lines(OUT/'rules_dev.jsonl',rule_dev);dump_lines(OUT/'rules_test.jsonl',rule_test)
    rule_keys={prompt_hash(tok.encode(r['prompt'])) for r in rule_dev+rule_test}
    records=[r for r in records if prompt_hash(r['_p']) not in rule_keys]
    dump_lines(OUT/'long_dev.jsonl',[r for r in records if r['source']=='verified_long_context' and r['split']=='dev'])
    dump_lines(OUT/'long_test.jsonl',[{k:v for k,v in r.items() if not k.startswith('_')}
                                    for r in records if r['source']=='verified_long_context' and r['split']=='test'])
    for context in (256,512):
        directory=OUT/f'ctx{context}';directory.mkdir(parents=True,exist_ok=True)
        eligible=[r for r in records if r['tokens']<=context]
        source_rows=[]
        groups_by_split={}
        metadata={'dataset':'pinned Luth plus verified exercises','revision':REV,'seq_length':context,
            'ignore_index':-100,'label_alignment_version':3,'split_groups':{'strategy':prep.SPLIT_STRATEGY,'prompt_overlap':0},
            'tokenizer_vocab_size':len(tok.token_to_id),'tokenizer_fingerprint':tok.fingerprint(),
            'special_tokens':{s:tok.token_to_id[s] for s in prep.SPECIAL_TOKENS}}
        for split in ('train','dev','test'):
            subset=[r for r in eligible if r['split']==split]
            rng.shuffle(subset)
            target='val' if split=='dev' else split
            examples=[native_example(r,tok,context) for r in subset]
            metadata[target]=prep.save_split(directory/f'sft_{target}.npz',examples,tok.token_to_id['<PAD>'])
            metadata[target+'_sha256']=sha256_file(directory/f'sft_{target}.npz')
            groups_by_split[split]={tuple(r['_p']) for r in subset}
            for r in subset:source_rows.append({k:v for k,v in r.items() if not k.startswith('_')})
            dump_lines(directory/(split+'_examples.jsonl'),[{k:v for k,v in r.items() if not k.startswith('_')} for r in subset])
        assert not (groups_by_split['train']&groups_by_split['dev'] or groups_by_split['train']&groups_by_split['test'] or groups_by_split['test']&groups_by_split['dev'])
        dump_lines(directory/'source.jsonl',[r for r in source_rows if r['split']!='test'])
        atomic_json(directory/'sft_meta.json',metadata)
        print('[data] context',context,'train',metadata['train']['examples'],'val',metadata['val']['examples'],'test',metadata['test']['examples'],flush=True)
    atomic_json(OUT/'data_manifest.json',{'revision':REV,'source_stats':stats,'total':len(records),
        'rules_test_sha256':sha256_file(OUT/'rules_test.jsonl'),'long_test_sha256':sha256_file(OUT/'long_test.jsonl'),
        'files':{str(p.relative_to(OUT)):sha256_file(p) for p in (OUT/'downloads/data').glob('*.parquet')}})


def train(name,context,epochs,lr,normalization):
    saved=protect();tok=Tokenizer.load(main.TOKENIZER_MODEL_PATH)
    device=main._setup_device();torch.set_num_threads(12);torch.set_num_interop_threads(6)
    if device.type!='cuda':raise RuntimeError('CUDA required')
    directory=OUT/f'ctx{context}';run=main.SFT_RUNS_DIR/('hf-20261117-'+name)
    if run.exists():raise RuntimeError('Run already exists; choose a fresh name')
    torch.manual_seed(main.SEED)
    model,_=load_model_from_final_state(saved['initial_sft'],tok,str(device))
    model.resize_context(context)
    bs=32 if context==256 else 16
    ds=SFTDataset(directory/'sft_train.npz');val=SFTDataset(directory/'sft_val.npz')
    loader=DataLoader(ds,batch_size=bs,sampler=ResumableRandomSampler(len(ds),seed=main.SEED),pin_memory=True)
    vl=DataLoader(val,batch_size=bs,shuffle=False,pin_memory=True)
    config=SFTConfig(epochs=epochs,learning_rate=lr,device=str(device),grad_accum_steps=64//bs,
        loss_normalization=normalization,warmup_ratio=.02,seed=main.SEED,checkpoint_every_steps=0,
        resume_from_checkpoint=False,eval_max_batches=None,log_interval=100,max_cpu_threads=12,max_cpu_interop_threads=6,
        checkpoint_dir=str(run/'checkpoints'),checkpoint_name=main.SFT_CHECKPOINT_NAME,
        final_model_dir=str(run/'models'),final_model_name=main.SFT_FINAL_MODEL_NAME,
        benchmark_path=str(run/'training_benchmark.csv'),tokenizer_fingerprint=tok.fingerprint(),
        unlikelihood_exclude_token_ids=tuple(tok.token_to_id[t] for t in tok.special_tokens),
        data_identity={'initial_sft_sha256':sha256_file(saved['initial_sft']),
                       'train_sha256':sha256_file(directory/'sft_train.npz'),'val_sha256':sha256_file(directory/'sft_val.npz')})
    atomic_json(run/'config.json',asdict(config))
    history=train_sft_model(model,loader,config,val_dataloader=vl)
    atomic_json(run/'history.json',history)
    del model;torch.cuda.empty_cache()
    score(Path(history['final_model_path']),read_lines(OUT/'rules_dev.jsonl'),tok,device,OUT/(name+'_rules_dev.json'))
    score(Path(history['final_model_path']),read_lines(OUT/'long_dev.jsonl'),tok,device,OUT/(name+'_long_dev.json'))
    protect()


def reference_subset(context, split, saved, tok, long_only=False):
    """Only Luth questions absent from the initial SFT training set count as novel."""
    from torch.utils.data import Subset
    initial_train=next(Path(p) for p in saved['files'] if p.endswith('sft_train.npz'))
    previous=SFTDataset(initial_train)
    blocked={tuple(prep.prompt_key((x.tolist(),y.tolist()))[1:-1])
             for x,y in zip(previous.input_ids,previous.labels)}
    directory=OUT/f'ctx{context}'
    rows=read_lines(directory/(split+'_examples.jsonl'))
    target='val' if split=='dev' else 'test'
    dataset=SFTDataset(directory/f'sft_{target}.npz')
    indices=[i for i,r in enumerate(rows) if r['source'].startswith('luth_')
             and (not long_only or r['tokens']>256) and tuple(tok.encode(r['prompt'])) not in blocked]
    if not indices:raise ValueError('No novel Luth references')
    atomic_json(directory/(split+('_long' if long_only else '_short')+'_evaluation_filter.json'),
        {'initial_train':str(initial_train),'total':len(rows),'retained':len(indices),
         'excluded':len(rows)-len(indices),'indices':indices,
         'policy':'Luth only; no exact tokenized prompt in initial SFT training; long >256 if requested'})
    return Subset(dataset,indices)


def evaluate(name,split,references_only=False):
    """Common short references plus fresh rule/long tests; no unequal-context CE comparison."""
    from tools.diagnose_sft import measure
    from data.artifacts import load_token_cache
    from data.dataset import TextDataset
    if split=='test':check_selection(name)
    saved=protect();tok=Tokenizer.load(main.TOKENIZER_MODEL_PATH)
    device=main._setup_device();torch.set_num_threads(12);torch.set_num_interop_threads(6)
    path=Path(saved['initial_sft']) if name=='baseline' else main.SFT_RUNS_DIR/('hf-20261117-'+name)/'models'/main.SFT_FINAL_MODEL_NAME
    model,payload=load_model_from_final_state(path,tok,str(device))
    end=tok.token_to_id['<|end|>'];destination=OUT/'evaluation'/name;destination.mkdir(parents=True,exist_ok=True)
    target='val' if split=='dev' else 'test'
    metric_path=destination/(split+'_metrics.json')
    result=json.loads(metric_path.read_text(encoding='utf-8')) if references_only else {}
    result.update(model=str(path),sha256=sha256_file(path),context=model.seq_length,selection=payload.get('selection'))
    short=reference_subset(256,split,saved,tok)
    m,sums,counts=measure(model,short,device,end)
    result['hf_short']=m
    np.savez_compressed(destination/(split+'_short_per_example.npz'),sums=sums,counts=counts)
    print('[eval]',name,'common HF short CE',m['ce'],flush=True)
    # Corpus retention always uses original 256-token windows, even for a 512 model.
    if split=='dev' and not references_only:
        corpus=TextDataset(load_token_cache(main.PREPARED_CACHE_DIR/'val_tokens.json',tok),256)
        result['original_corpus']=measure(model,corpus,device,end)[0]
        result['original_sft']=measure(model,SFTDataset(ROOT/'artifacts/sft/sft_val.npz'),device,end)[0]
        print('[eval]',name,'corpus CE',result['original_corpus']['ce'],flush=True)
    if model.seq_length>=512:
        ds=reference_subset(512,split,saved,tok,long_only=True)
        result['hf_long']=measure(model,ds,device,end)[0]
        print('[eval]',name,'HF long CE',result['hf_long']['ce'],flush=True)
    atomic_json(destination/(split+'_metrics.json'),result)
    del model,payload;torch.cuda.empty_cache()
    if references_only:
        protect()
        return
    result['rules']=score(path,read_lines(OUT/f'rules_{split}.jsonl'),tok,device,destination/(split+'_rules.json'))
    result['long']=score(path,read_lines(OUT/f'long_{split}.jsonl'),tok,device,destination/(split+'_long.json'))
    if split=='dev':
        old=ROOT/'artifacts/sft/repairs/20261116/dev_generation.jsonl'
        result['old_rules_regression']=score(path,read_lines(old),tok,device,destination/'old_dev_rules.json')
        # Same independent generative samples as previous measurements, for a qualitative comparison.
        from tools.diagnose_sft import generations
        model,_=load_model_from_final_state(path,tok,str(device))
        atomic_json(destination/'general_generations.json',generations(model,tok,SFTDataset(ROOT/'artifacts/sft/sft_val.npz'),device))
        del model;torch.cuda.empty_cache()
    atomic_json(destination/(split+'_metrics.json'),result)
    protect()


def cli():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('build','train','evaluate','references','choose','promote','verify'))
    parser.add_argument('--name',default='candidate256')
    parser.add_argument('--context',type=int,choices=(256,512),default=256)
    parser.add_argument('--epochs',type=int,default=2)
    parser.add_argument('--lr',type=float,default=1e-5)
    parser.add_argument('--normalization',choices=('sequences','supervised_tokens'),default='sequences')
    parser.add_argument('--split',choices=('dev','test'),default='dev')
    args=parser.parse_args()
    import re
    if not re.fullmatch('[a-z0-9_-]+',args.name):parser.error('Invalid run name')
    protect()
    if args.stage=='build':build()
    elif args.stage=='choose':choose(args.name)
    elif args.stage=='promote':promote(args.name)
    elif args.stage=='verify':verify()
    elif args.stage in ('evaluate','references'):evaluate(args.name,args.split,references_only=args.stage=='references')
    else:train(args.name,args.context,args.epochs,args.lr,args.normalization)
    protect()


if __name__=='__main__':cli()
