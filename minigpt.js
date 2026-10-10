// MiniGPT JS推理引擎——从《从零构建大模型》PyTorch版移植
// int8权重 + 纯Float32Array前向，无任何依赖
(function(global){
  'use strict';

  let W = null; // 权重
  let b64cache = {};

  function b64ToInt8(b64){
    const bin = atob(b64);
    const n = bin.length;
    const a = new Int8Array(n);
    for (let i=0;i<n;i++) a[i] = bin.charCodeAt(i) - 128 + 128; // charCode→int8 (0..255 → -128..127)
    // 修正：charCodeAt 给 0..255，int8 语义 = (b ^ 128) - 128 即 b<128? b : b-256
    for (let i=0;i<n;i++){ const b=bin.charCodeAt(i); a[i] = b < 128 ? b : b - 256; }
    return a;
  }
  function deq(arr, scale){ // int8数组 → Float32Array
    const f = new Float32Array(arr.length);
    for (let i=0;i<arr.length;i++) f[i] = arr[i]*scale;
    return f;
  }

  function loadWeights(json){
    W = { vocab: json.vocab, V: json.vocab.length, D: 96, BLOCK: 64, H: 3, L: json.blocks.length };
    W.emb = deq(b64ToInt8(json.emb), json.emb_scale);       // V×D
    W.pos = deq(b64ToInt8(json.pos), json.pos_scale);       // 64×D
    W.lnfg = json.lnfg; W.lnfb = json.lnfb;
    W.blocks = json.blocks.map(b=>({
      ln1g:b.ln1g, ln1b:b.ln1b, ln2g:b.ln2g, ln2b:b.ln2b,
      qkvw:deq(b64ToInt8(b.qkvw), b.qkvw_s), qkvb:deq(b64ToInt8(b.qkvb), b.qkvb_s),
      projw:deq(b64ToInt8(b.projw), b.projw_s), projb:deq(b64ToInt8(b.projb), b.projb_s),
      fcw:deq(b64ToInt8(b.fcw), b.fcw_s), fcb:deq(b64ToInt8(b.fcb), b.fcb_s),
      fc2w:deq(b64ToInt8(b.fc2w), b.fc2w_s), fc2b:deq(b64ToInt8(b.fc2b), b.fc2b_s)
    }));
    W.stoi = {};
    for (let i=0;i<W.vocab.length;i++) W.stoi[W.vocab[i]] = i;
  }

  const D = 96, HD = 32; // d_model, head_dim (3头×32)

  function layernorm(x, g, b, off, n){
    let m=0; for(let i=0;i<n;i++) m += x[off+i]; m/=n;
    let v=0; for(let i=0;i<n;i++){ const d=x[off+i]-m; v+=d*d; } v/=n;
    const s = 1/Math.sqrt(v+1e-5);
    for(let i=0;i<n;i++) x[off+i] = (x[off+i]-m)*s*g[i]+b[i];
  }

  function matvec(x, w, bias, out, inN, outN, xOff, outOff){
    outOff = outOff || 0;
    for(let o=0;o<outN;o++){
      let s = bias? bias[o] : 0;
      const wo = o*inN;
      for(let i=0;i<inN;i++) s += x[xOff+i]*w[wo+i];
      out[outOff+o] = s;
    }
  }

  // 前向：ids(Int32Array 长度T) → logits(Float32Array 长度V)，只算最后一个位置的logits
  // 完整实现：T个位置的hidden全部算出（attention需要全序列），返回末位logits
  function forward(ids){
    const T = ids.length, V = W.V;
    // token+pos嵌入
    const x = new Float32Array(T*D);
    for(let t=0;t<T;t++){
      const e = ids[t]*D, p = t*D, o = t*D;
      for(let i=0;i<D;i++) x[o+i] = W.emb[e+i] + W.pos[p+i];
    }
    // 逐层
    const buf1 = new Float32Array(T*D);       // 残差中间
    const qkv = new Float32Array(T*3*D);
    const attv = new Float32Array(T*D);
    const mlp1 = new Float32Array(T*4*D);
    const head_in = new Float32Array(D);
    const logits = new Float32Array(V);
    // head权重 = emb (tied)，需要反量化一次（V×96 float）
    if (!W._headf) W._headf = Float32Array.from(W.emb);

    for(let l=0;l<W.L;l++){
      const B = W.blocks[l];
      // ---- x = x + attn(ln1(x)) ----
      buf1.set(x);
      for(let t=0;t<T;t++) layernorm(x, B.ln1g, B.ln1b, t*D, D);
      // qkv
      for(let t=0;t<T;t++) matvec(x, B.qkvw, B.qkvb, qkv, D, 3*D, t*D, t*3*D);
      // causal attention（每头）
      for(let h=0;h<W.H;h++){
        const qo = h*HD, ko = D + h*HD, vo = 2*D + h*HD;
        for(let t=0;t<T;t++){
          // q·k for all s<=t
          let maxs = -1e30;
          const scores = new Float32Array(t+1);
          for(let s=0;s<=t;s++){
            let d = 0;
            for(let i=0;i<HD;i++) d += qkv[t*3*D+qo+i]*qkv[s*3*D+ko+i];
            d /= Math.sqrt(HD);
            scores[s]=d; if(d>maxs) maxs=d;
          }
          let sum=0;
          for(let s=0;s<=t;s++){ scores[s]=Math.exp(scores[s]-maxs); sum+=scores[s]; }
          for(let i=0;i<HD;i++){
            let a=0;
            for(let s=0;s<=t;s++) a += scores[s]/sum * qkv[s*3*D+vo+i];
            attv[t*D+qo+i]=a;
          }
        }
      }
      // proj + 残差
      const projout = new Float32Array(D);
      for(let t=0;t<T;t++){
        matvec(attv, B.projw, B.projb, projout, D, D, t*D);
        for(let i=0;i<D;i++) x[t*D+i] = buf1[t*D+i] + projout[i];
      }
      // ---- x = x + mlp(ln2(x)) ----
      buf1.set(x);
      for(let t=0;t<T;t++) layernorm(x, B.ln2g, B.ln2b, t*D, D);
      for(let t=0;t<T;t++){
        matvec(x, B.fcw, B.fcb, mlp1, D, 4*D, t*D, t*4*D);
        for(let i=0;i<4*D;i++){
          const v = mlp1[t*4*D+i];
          mlp1[t*4*D+i] = 0.5*v*(1+Math.tanh(Math.sqrt(2/Math.PI)*(v+0.044715*v*v*v))); // GELU(tanh近似)
        }
        matvec(mlp1, B.fc2w, B.fc2b, projout, 4*D, D, t*4*D);
        for(let i=0;i<D;i++) x[t*D+i] = buf1[t*D+i] + projout[i];
      }
    }
    // 末位ln_f + head
    layernorm(x, W.lnfg, W.lnfb, (T-1)*D, D);
    const last = (T-1)*D;
    for(let i=0;i<D;i++) head_in[i] = x[last+i];
    matvec(head_in, W._headf, null, logits, D, V, 0);
    return logits;
  }

  // 生成：prompt字符串 → 逐字生成（每字回调onChar，结束回调onDone）
  // 异步分片：每字让出主线程，DOM流式渲染
  async function generate(prompt, maxNew, temperature, topK, onChar, onDone){
    const stoi = W.stoi;
    let ids = [];
    for (const c of prompt) if (stoi[c] !== undefined) ids.push(stoi[c]);
    if (!ids.length) ids = [stoi['兮']] || [0];
    let out = prompt;
    if (onChar) onChar('', out);
    for (let n=0; n<maxNew; n++){
      const ctx = ids.slice(-W.BLOCK);
      const logits = forward(Int32Array.from(ctx));
      // 温度
      const lg = new Float32Array(logits.length);
      let max = -1e30;
      for(let i=0;i<logits.length;i++){ lg[i]=logits[i]/temperature; if(lg[i]>max) max=lg[i]; }
      // top-k + softmax
      if (topK && topK < lg.length){
        const idx = Array.from(lg.keys()).sort((a,b)=>lg[b]-lg[a]).slice(0, topK);
        const set = new Set(idx);
        let sum=0;
        for(let i=0;i<lg.length;i++){
          if(!set.has(i)) lg[i]=-1e30;
        }
      }
      let sum=0;
      for(let i=0;i<lg.length;i++){ lg[i]=Math.exp(lg[i]-max); sum+=lg[i]; }
      // 采样
      let r = Math.random()*sum, pick = lg.length-1;
      for(let i=0;i<lg.length;i++){ r-=lg[i]; if(r<=0){ pick=i; break; } }
      const ch = W.vocab[pick];
      out += ch;
      ids.push(pick);
      if (onChar) onChar(ch, out);
      if (n % 3 === 2) await new Promise(function(r){ setTimeout(r, 0); }); // 每3字让出一次
    }
    if (onDone) onDone(out);
    return out;
  }

  global.MiniGPT = { loadWeights, generate };
})(window);
