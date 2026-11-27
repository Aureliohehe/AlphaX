import {useDeferredValue, useEffect, useMemo, useRef, useState} from "react";
import {format, grades, integer, LibraryOptions, Table} from "./shared";

const pageSize = 30;
export const gradeOptions = Object.entries(grades).map(([id, label]) => <option key={id} value={id}>{label}</option>);

function matchFactors(catalog, source, grade, search) {
  const matches = [];
  for (const library of catalog) {
    if (source && library.id !== source) continue;
    for (const factor of library.factors) {
      if (grade && factor.grade !== grade) continue;
      if (![factor.id, factor.expression, factor.hypothesis].some(
        (value) => value.toLocaleLowerCase().includes(search))) continue;
      matches.push([factor.id, grades[factor.grade], factor.expression, factor.hypothesis || "—",
        format(factor.rank_ic, 4), library.name]);
    }
  }
  return matches;
}

export default function Factors({catalog, loading, error, reload, requestedSource}) {
  const [search, setSearch] = useState("");
  const [source, setSource] = useState("");
  const [grade, setGrade] = useState("");
  const [page, setPage] = useState(0);
  const appliedSource = useRef(null);
  const query = useDeferredValue(search.trim().toLocaleLowerCase());

  useEffect(() => {
    if (requestedSource?.id && requestedSource !== appliedSource.current &&
        !loading && catalog.some(({id}) => id === requestedSource.id)) {
      appliedSource.current = requestedSource;
      setSource(requestedSource.id);
      setSearch("");
      setGrade("");
      setPage(0);
    } else {
      setSource((previous) => catalog.some(({id}) => id === previous) ? previous : "");
    }
  }, [catalog, loading, requestedSource]);

  const matches = useMemo(() => matchFactors(catalog, source, grade, query), [catalog, source, grade, query]);
  const pageCount = Math.max(1, Math.ceil(matches.length / pageSize));
  const currentPage = Math.min(page, pageCount - 1);
  useEffect(() => {
    if (page !== currentPage) setPage(currentPage);
  }, [page, currentPage]);

  function filter(setter) {
    return (event) => {
      setter(event.target.value);
      setPage(0);
    };
  }

  return (
    <section id="factors" data-view>
      <div className="page-heading"><div><h1>因子库</h1><p>按来源、质量和表达式浏览本机 Miner 产物。</p></div><button className="button secondary" type="button" disabled={loading} onClick={reload}>刷新因子库</button></div>
      <article className="panel section-panel">
        <div className="form-grid">
          <label>搜索因子<input type="search" value={search} onChange={filter(setSearch)} placeholder="名称、表达式或研究假设" /></label>
          <label>实验来源<select value={source} onChange={filter(setSource)}><option value="">全部来源</option><LibraryOptions catalog={catalog} /></select></label>
          <label>质量分级<select value={grade} onChange={filter(setGrade)}><option value="">全部等级</option>{gradeOptions}</select></label>
        </div>
        <p className="form-hint">分级沿用来源实验的准入结果。Rank IC 显示来源实验的 1 日验证期指标。</p>
        <p id="factor-state" role="status" data-invalid={Boolean(error)}>{error || (loading ? "正在读取因子产物…" : "找到 " + integer(matches.length) + " 条因子记录")}</p>
        <div id="factor-table">{matches.length ? <Table headers={["因子", "质量", "表达式", "研究假设", "Rank IC", "来源"]}
          rows={matches.slice(currentPage * pageSize, (currentPage + 1) * pageSize)} /> :
          !loading && !error && <p className="empty">没有匹配的因子。检查扫描目录或调整筛选条件。</p>}</div>
        {pageCount > 1 && <div className="form-actions">
          <button className="button secondary" type="button" disabled={currentPage === 0} onClick={() => setPage(currentPage - 1)}>上一页</button>
          <span>{currentPage + 1} / {pageCount}</span>
          <button className="button secondary" type="button" disabled={currentPage === pageCount - 1} onClick={() => setPage(currentPage + 1)}>下一页</button>
        </div>}
      </article>
    </section>
  );
}
