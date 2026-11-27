import {SearchRounds} from "./dashboard";

export default function Workflow({data}) {
  return (
    <section id="workflow" data-view>
      <div className="page-heading"><div><h1>Agent loop</h1><p>从失败指标生成下一轮可验证的因子假设。</p></div></div>
      <article className="panel section-panel">
        <div className="panel-head"><div><h2>训练期反馈搜索</h2><p>DeepSeek 提案、程序评估、滚动反馈与独立准入</p></div></div>
        <div className="steps detailed"><div><b>提案</b><span>LangGraph 将预算分配给研究角色，收集结构化表达式。</span></div><div><b>评估</b><span>校验 AST，跨轮去重，在已兑现的标签上做滚动折评估。</span></div><div><b>修订</b><span>根据失败原因修改一个参数，或提出具备独立机制的新因子。</span></div><div><b>准入</b><span>达到预算后冻结候选，并在隔离窗口按严格规则复核。</span></div></div>
        <div id="rounds"><SearchRounds search={data.search} mechanism={data.mechanism} /></div>
      </article>
      <article className="panel section-panel"><h2>交易层反馈</h2><p>在相同样本上比较四因子 XGBoost 和新增因子，复用相同的成交规则。将两档成本下的净收益、回撤、换手和逐年差异送回 Miner。</p><a className="text-link" href="#results">查看配对结果</a></article>
    </section>
  );
}
