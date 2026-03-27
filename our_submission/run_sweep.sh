#!/bin/bash
# one-shot sweep: train once, eval many configs
# designed to run on vast.ai 8xH100 SXM in under 55 minutes
# leaves 5 min buffer before the 1hr mark
set -e

DEADLINE=$((SECONDS + 3300))  # 55 min hard cap

echo "=== PARAMETER GOLF SWEEP ==="
echo "start: $(date)"
echo "deadline: 55 min from now"

cd /workspace

# --- SETUP (2 min) ---
echo "--- SETUP ---"
git clone https://github.com/openai/parameter-golf.git pg 2>&1 | tail -1
git clone -b ppmd-submission https://github.com/pablinga19/parameter-golf.git ours 2>&1 | tail -1
cd pg
pip install sentencepiece flash-attn numba zstandard huggingface_hub -q 2>&1 | tail -1
python3 data/cached_challenge_fineweb.py --variant sp1024 2>&1 | tail -3

# copy our modules
cp /workspace/ours/our_submission/mixer.py .
cp /workspace/ours/our_submission/ppmd_numba.py .
cp /workspace/ours/our_submission/bayesian_mixer.py .
cp /workspace/ours/our_submission/rans_numba.py .
cp /workspace/ours/our_submission/nonuniform_quant.py .
echo "setup done: $(date)"

# --- FETCH SOTA TRAIN SCRIPT (PR #761 architecture) ---
echo "--- FETCHING SOTA ARCHITECTURE ---"
# try to get the actual competitive train_gpt.py from PR #761
# if gh not available, use our baseline
if command -v gh &> /dev/null; then
    gh pr checkout 761 -- records/track_10min_16mb/2026-03-26_ScoreFirst_TTT_Ngram_Backoff/train_gpt.py 2>/dev/null && \
    cp records/track_10min_16mb/2026-03-26_ScoreFirst_TTT_Ngram_Backoff/train_gpt.py train_gpt_sota.py 2>/dev/null || true
    gh pr checkout 727 -- records/track_10min_16mb/2026-03-26_Backoff_Entropy_Adaptive/train_gpt.py 2>/dev/null && \
    cp records/track_10min_16mb/2026-03-26_Backoff_Entropy_Adaptive/train_gpt.py train_gpt_727.py 2>/dev/null || true
fi

# --- TRAIN (10 min) ---
echo "--- TRAINING ---"
echo "train start: $(date)"
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tail -5
echo "train done: $(date)"

# save checkpoint location
MODEL_PT="$(pwd)/final_model.pt"
MODEL_PTZ="$(pwd)/final_model.int8.ptz"
echo "model: $MODEL_PT"
echo "compressed: $MODEL_PTZ"

# --- EVAL SWEEP ---
# write the eval script inline — no external deps except our modules
python3 << 'EVALSCRIPT'
import torch, sys, time, math, types, os
import numpy as np
import torch.nn.functional as F
import sentencepiece as spm

sys.path.insert(0, ".")
from train_gpt import Hyperparameters, GPT, build_sentencepiece_luts, load_validation_tokens

args = Hyperparameters()
device = torch.device("cuda:0")

model = GPT(
    vocab_size=args.vocab_size, num_layers=args.num_layers,
    model_dim=args.model_dim, num_heads=args.num_heads,
    num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
    tie_embeddings=args.tie_embeddings,
    tied_embed_init_std=args.tied_embed_init_std,
    logit_softcap=args.logit_softcap, rope_base=args.rope_base,
    qk_gain_init=args.qk_gain_init,
).to(device)
sd = torch.load("final_model.pt", map_location=device, weights_only=True)
model.load_state_dict(sd, strict=True)
model.eval()

def fwd_logits(self, ids):
    ids = ids.long()
    x = self.tok_emb(ids)
    x = F.rms_norm(x, (x.size(-1),))
    x0 = x; skips = []
    for i in range(self.num_encoder_layers):
        x = self.blocks[i](x, x0); skips.append(x)
    for i in range(self.num_decoder_layers):
        if skips: x = x + self.skip_weights[i].to(dtype=x.dtype)[None,None,:] * skips.pop()
        x = self.blocks[self.num_encoder_layers+i](x, x0)
    x = self.final_norm(x)
    lp = F.linear(x, self.tok_emb.weight) if self.tie_embeddings else self.lm_head(x)
    return self.logit_softcap * torch.tanh(lp / self.logit_softcap)
model.forward_logits = types.MethodType(fwd_logits, model)

sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
bb, hs, ib = build_sentencepiece_luts(sp, args.vocab_size, device)
val = load_validation_tokens(args.val_files, args.train_seq_len).to(device)
val_np = val.cpu().numpy().astype(np.int32)
print(f"val: {val.numel():,} tokens", flush=True)

from ppmd_numba import PPMDNumba
from mixer import logistic_mix, entropy_adaptive_alpha

def run_eval(label, max_win, use_ppmd=False, ppmd_boost=1.2, ppmd_tau=30.0):
    stride=64; sl=args.train_seq_len; total=val.numel()-1
    wins=list(range(0,total,stride))[:max_win]
    scored=np.zeros(total+1,dtype=bool)
    bk=4194304; mk=np.uint64(bk-1); no=6; mo=2
    pr=np.array([36313,27191,51647,81929,131071,175447,209591],dtype=np.uint64)

    if use_ppmd:
        pp=PPMDNumba(count_gate_tau=ppmd_tau,use_singleton_escape=True,depth_boost_base=ppmd_boost)
        # warmup
        pp.update_tables(val_np,0,100)
        gj=np.arange(10,100,dtype=np.int64);pp.predict_blended(val_np,gj,len(gj))
        pp=PPMDNumba(count_gate_tau=ppmd_tau,use_singleton_escape=True,depth_boost_base=ppmd_boost)
    else:
        ct=[np.zeros(bk,dtype=np.uint32) for _ in range(no)]
        ft=[np.zeros(bk,dtype=np.uint32) for _ in range(no)]

    ls=0.0;tc=0;bc=0.0;t0=time.time()
    with torch.inference_mode():
        for wi,ws in enumerate(wins):
            wl=min(ws+sl,total)-ws
            if wl<2:continue
            x=val[ws:ws+wl].unsqueeze(0).to(device)
            with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
                lo=model.forward_logits(x)
            lo=lo.float()
            y=val[ws+1:ws+wl+1].long().to(device)
            nll=F.cross_entropy(lo[0,:wl-1],y[:wl-1],reduction="none")
            s=0
            for j in range(len(nll)):
                if not scored[ws+1+j]:s=j;break
            else:continue
            sn=nll[s:].cpu().numpy().astype(np.float64);ns=len(sn)
            if ns==0:continue
            gj=np.arange(ws+s+1,ws+s+1+ns,dtype=np.int64)
            sp_=np.exp(-sn)
            with torch.no_grad():
                lp=F.log_softmax(lo[0,s:s+ns],dim=-1)
                en=-(lp.exp()*lp).sum(-1).cpu().numpy()

            if use_ppmd:
                pg,hm=pp.predict_blended(val_np,gj,ns)
                if hm.any():
                    a=np.clip(entropy_adaptive_alpha(en[hm]),0,0.95)
                    sp_[hm]=(1-a)*sp_[hm]+a*pg[hm]
                sn=-np.log(np.clip(sp_,1e-12,1.0))
                pp.update_tables(val_np,int(gj[0]),int(gj[-1])+1)
            else:
                best=np.full(ns,-1.0)
                for oi in range(no):
                    cl=mo+oi;v=gj>=cl
                    if not v.any():continue
                    vi=np.nonzero(v)[0]
                    ck=np.zeros(len(vi),dtype=np.uint64)
                    for k in range(cl):
                        ti=np.clip(gj[vi]-cl+k,0,len(val_np)-1)
                        ck^=pr[k]*val_np[ti].astype(np.uint64)
                    ck&=mk;ti=np.clip(gj[vi],0,len(val_np)-1)
                    pi=min(cl,len(pr)-1)
                    fk=ck^(pr[pi]*val_np[ti].astype(np.uint64));fk&=mk
                    cc=ct[oi][ck].astype(np.float64);fc=ft[oi][fk].astype(np.float64)
                    g_=cc>=2.0;n_=g_&(best[vi]<0)
                    if n_.any():
                        fi=vi[n_];best[fi]=np.clip(np.minimum(fc[n_],cc[n_])/np.maximum(cc[n_],1.0),0,1)
                hm=best>=0
                if hm.any():
                    a=np.clip(entropy_adaptive_alpha(en[hm]),0,0.95)
                    sp_[hm]=(1-a)*sp_[hm]+a*best[hm]
                sn=-np.log(np.clip(sp_,1e-12,1.0))
                for oi in range(no):
                    cl=mo+oi
                    for jl in range(ns):
                        jg=int(gj[jl])
                        if jg<cl:continue
                        ck=np.uint64(0)
                        for k in range(cl):ck^=pr[k]*np.uint64(val_np[jg-cl+k])
                        ck&=mk;pi=min(cl,len(pr)-1)
                        fk=ck^(pr[pi]*np.uint64(val_np[jg]));fk&=mk
                        ct[oi][ck]+=1;ft[oi][fk]+=1

            ls+=sn.sum();tc+=ns
            tgt=val_np[gj].astype(np.int64);prv=val_np[np.clip(gj-1,0,len(val_np)-1)].astype(np.int64)
            b_=bb[torch.from_numpy(tgt).to(device)].cpu().numpy().astype(np.float64)
            h_=hs[torch.from_numpy(tgt).to(device)].cpu().numpy()
            i_=ib[torch.from_numpy(prv).to(device)].cpu().numpy()
            b_+=(h_&~i_).astype(np.float64);bc+=b_.sum()
            for jl in range(ns):scored[int(gj[jl])]=True

            if(wi+1)%2000==0:
                bpb=(ls/tc/math.log(2))*(tc/bc) if bc>0 else 0
                print(f"  [{label}] {wi+1}/{len(wins)} bpb={bpb:.4f} {time.time()-t0:.0f}s",flush=True)

    bpb=(ls/tc/math.log(2))*(tc/bc) if bc>0 else 0
    elapsed=time.time()-t0
    print(f"  [{label}] DONE bpb={bpb:.4f} tok={tc:,} {elapsed:.0f}s",flush=True)
    return bpb, tc, elapsed

# scenario 1: backoff baseline (10K windows = ~640K tokens)
print("\n=== S1: BACKOFF BASELINE (10K windows) ===",flush=True)
b1,t1,d1 = run_eval("backoff", 10000, use_ppmd=False)

# scenario 2: PPM-D fixed, boost=1.2, tau=30
print("\n=== S2: PPMD fc>0, boost=1.2, tau=30 ===",flush=True)
b2,t2,d2 = run_eval("ppmd-1.2", 10000, use_ppmd=True, ppmd_boost=1.2, ppmd_tau=30.0)

# scenario 3: PPM-D flat, boost=1.0 (no depth preference)
print("\n=== S3: PPMD fc>0, boost=1.0, tau=30 ===",flush=True)
b3,t3,d3 = run_eval("ppmd-flat", 10000, use_ppmd=True, ppmd_boost=1.0, ppmd_tau=30.0)

# scenario 4: PPM-D boost=1.5, tau=50 (moderate)
print("\n=== S4: PPMD fc>0, boost=1.5, tau=50 ===",flush=True)
b4,t4,d4 = run_eval("ppmd-1.5", 10000, use_ppmd=True, ppmd_boost=1.5, ppmd_tau=50.0)

# scenario 5: backoff, larger eval (20K windows) for statistical power
print("\n=== S5: BACKOFF EXTENDED (20K windows) ===",flush=True)
b5,t5,d5 = run_eval("backoff-20k", 20000, use_ppmd=False)

# scenario 6: best PPM-D config from above, also 20K windows
best_boost = 1.0
best_label = "flat"
if b2 < b3 and b2 < b4:
    best_boost = 1.2; best_label = "1.2"
elif b4 < b3:
    best_boost = 1.5; best_label = "1.5"
print(f"\n=== S6: BEST PPMD ({best_label}), 20K windows ===",flush=True)
b6,t6,d6 = run_eval(f"ppmd-best-{best_label}", 20000, use_ppmd=True, ppmd_boost=best_boost, ppmd_tau=30.0 if best_boost != 1.5 else 50.0)

print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)
print(f"{'Config':<30} {'BPB':>8} {'Tokens':>10} {'Time':>6}")
print("-"*60)
print(f"{'S1: Backoff 10K':<30} {b1:>8.4f} {t1:>10,} {d1:>5.0f}s")
print(f"{'S2: PPMD boost=1.2 tau=30':<30} {b2:>8.4f} {t2:>10,} {d2:>5.0f}s")
print(f"{'S3: PPMD boost=1.0 tau=30':<30} {b3:>8.4f} {t3:>10,} {d3:>5.0f}s")
print(f"{'S4: PPMD boost=1.5 tau=50':<30} {b4:>8.4f} {t4:>10,} {d4:>5.0f}s")
print(f"{'S5: Backoff 20K':<30} {b5:>8.4f} {t5:>10,} {d5:>5.0f}s")
print(f"{'S6: Best PPMD 20K':<30} {b6:>8.4f} {t6:>10,} {d6:>5.0f}s")
print()
print(f"PPMD 1.2 vs backoff (10K): {b1-b2:+.4f} ({(b1-b2)/b1*100:+.2f}%)")
print(f"PPMD flat vs backoff (10K): {b1-b3:+.4f} ({(b1-b3)/b1*100:+.2f}%)")
print(f"PPMD 1.5 vs backoff (10K): {b1-b4:+.4f} ({(b1-b4)/b1*100:+.2f}%)")
print(f"Best PPMD vs backoff (20K): {b5-b6:+.4f} ({(b5-b6)/b5*100:+.2f}%)")

winner = "BACKOFF" if min(b2,b3,b4) >= b1 else f"PPMD (boost={best_boost})"
print(f"\nWINNER: {winner}")
if winner == "BACKOFF":
    print("VERDICT: Abandon PPM-D. Ship with SOTA backoff + our compression improvements.")
else:
    print("VERDICT: PPM-D fix works! Integrate into submission.")
EVALSCRIPT

echo "sweep done: $(date)"
echo "=== ALL DONE ==="
