import {Comparison, HoldoutTable} from "./dashboard";

export default function Results({data}) {
  return (
    <section id="results" data-view>
      <div className="page-heading"><div><h1>实验对照</h1><p>查看同样本基线、成本压力与后续检验。</p></div></div>
      <article className="panel section-panel"><div className="panel-head"><div><h2>周频 XGBoost 配对</h2><p>定向反馈产生的低波动平滑因子作为第五项输入</p></div></div><div id="comparison"><Comparison pair={data.paired} /></div></article>
      <article className="panel section-panel"><div className="panel-head"><div><h2>冻结四因子留出回放</h2><p>独立组合实验 · 485 个交易日</p></div></div><div id="holdout-table"><HoldoutTable holdout={data.holdout} /></div></article>
    </section>
  );
}
