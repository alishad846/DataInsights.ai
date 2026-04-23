import { useEffect, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { MessageSquare, Sparkles, Download, FileText, Database } from 'lucide-react';
import EmployeeLayout from '../../layout/EmployeeLayout';
import { getAnalysis, getCleanedData, getDatasets } from '../../services/api';

const TOC_SECTIONS = [
  { id: 'overview', label: 'Overview' },
  { id: 'schema', label: 'Schema / Columns' },
  { id: 'numeric', label: 'Numeric Stats' },
  { id: 'categorical', label: 'Categorical Profiles' },
  { id: 'actions', label: 'Actions' },
];

const typeColors = {
  num: { bg: 'rgba(63,185,80,0.1)', color: 'var(--success)' },
  cat: { bg: 'rgba(188,140,255,0.1)', color: 'var(--accent)' },
  date: { bg: 'rgba(210,153,34,0.1)', color: 'var(--warning)' },
};

const normalizeType = (value) => {
  if (!value) return 'string';
  const lower = String(value).toLowerCase();
  if (['numeric', 'number', 'float64', 'int64'].includes(lower)) return 'numeric';
  if (['datetime', 'date', 'timestamp'].includes(lower)) return 'datetime';
  if (['categorical', 'category'].includes(lower)) return 'categorical';
  return 'string';
};

const EmployeeSummaryPage = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const datasetId = searchParams.get('ds');
  const datasetName = searchParams.get('name') || datasetId || 'Dataset';

  const [activeSection, setActiveSection] = useState('overview');
  const [availableDatasets, setAvailableDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState(null);
  const [summaryData, setSummaryData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [summaryError, setSummaryError] = useState('');

  useEffect(() => {
    const loadDatasets = async () => {
      setLoading(true);
      setSummaryError('');

      try {
        const res = await getDatasets();
        if (!res.success || !Array.isArray(res.data)) {
          setSummaryError('Could not load datasets.');
          setLoading(false);
          return;
        }

        const allDatasets = res.data;
        const readyDatasets = allDatasets.filter((d) =>
          ['completed', 'ready', 'cleaned'].includes(d.status)
        );

        setAvailableDatasets(readyDatasets);

        const selected =
          (datasetId && allDatasets.find((d) => (d.dataset_id || d.id) === datasetId)) ||
          readyDatasets[0] ||
          allDatasets[0] ||
          null;

        if (selected) {
          setSelectedDataset(selected);
        } else {
          setSummaryError('No dataset is available for summary yet.');
          setLoading(false);
        }
      } catch (err) {
        console.warn('Could not load datasets:', err.message);
        setSummaryError('Could not load datasets.');
        setLoading(false);
      }
    };

    loadDatasets();
  }, [datasetId]);

  useEffect(() => {
    const loadSummary = async () => {
      if (!selectedDataset) return;

      const dsId = selectedDataset.dataset_id || selectedDataset.id;
      if (!dsId) {
        setSummaryError('This dataset does not have a valid identifier.');
        setLoading(false);
        return;
      }

      setLoading(true);
      setSummaryError('');

      try {
        const [analysisRes, cleanedRes] = await Promise.allSettled([
          getAnalysis(dsId),
          getCleanedData(dsId, { limit: 1, page: 1 }),
        ]);

        const analysis = analysisRes.status === 'fulfilled' && analysisRes.value?.success
          ? analysisRes.value
          : null;
        const cleaned = cleanedRes.status === 'fulfilled' && cleanedRes.value?.success
          ? cleanedRes.value
          : null;

        if (!analysis && !cleaned) {
          setSummaryData(null);
          setSummaryError('No dataset summary data is available yet.');
          setLoading(false);
          return;
        }

        const analysisColumns = analysis?.columns || [];
        const headers = cleaned?.headers?.length
          ? cleaned.headers
          : analysisColumns.map((col) => col.name);
        const columnTypes = cleaned?.columnTypes || Object.fromEntries(
          analysisColumns.map((col) => [col.name, normalizeType(col.inferred_type || col.type)])
        );
        const columnStats = cleaned?.columnStats || {};

        setSummaryData({
          name: analysis?.dataset_name || selectedDataset.name || datasetName,
          rows: analysis?.row_count ?? cleaned?.totalRows ?? 0,
          columns: analysis?.column_count ?? headers.length,
          headers,
          columnTypes,
          columnStats,
          qualityScore: analysis?.quality_score ?? null,
          totalNulls: analysis?.total_nulls ?? 0,
          duplicateRows: analysis?.duplicate_rows ?? 0,
          cleaningReport: analysis?.cleaning_report || [],
          analysisColumns,
        });
      } catch (err) {
        console.warn('Could not load dataset summary:', err.message);
        setSummaryError('Could not load summary data for this dataset.');
      }

      setLoading(false);
    };

    loadSummary();
  }, [selectedDataset, datasetName]);

  useEffect(() => {
    const handleScroll = () => {
      for (const { id } of TOC_SECTIONS) {
        const el = document.getElementById(id);
        if (el && el.getBoundingClientRect().top < 120) {
          setActiveSection(id);
        }
      }
    };

    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const scrollTo = (id) => {
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' });
    setActiveSection(id);
  };

  const analysisColumnMap = Object.fromEntries(
    (summaryData?.analysisColumns || []).map((col) => [col.name, col])
  );

  const schema = (summaryData?.headers || summaryData?.analysisColumns?.map((col) => col.name) || []).map((colName) => {
    const source = analysisColumnMap[colName];
    const type = normalizeType(summaryData?.columnTypes?.[colName] || source?.inferred_type || source?.type);
    const uniqueCount = summaryData?.columnStats?.[colName]?.uniqueCount ?? source?.nunique ?? 0;
    const nullCount = source?.null_count ?? 0;

    return {
      name: colName,
      type,
      nullable: nullCount > 0 ? 'Yes' : 'No',
      unique: uniqueCount ? uniqueCount.toLocaleString() : '0',
      typeClass: type === 'numeric' ? 'num' : type === 'datetime' ? 'date' : 'cat',
    };
  });

  const numericStats = (summaryData?.headers || [])
    .filter((header) => normalizeType(summaryData?.columnTypes?.[header]) === 'numeric')
    .map((col) => {
      const stats = summaryData?.columnStats?.[col];
      return {
        name: col,
        stats: stats ? {
          min: typeof stats.min === 'number' ? stats.min.toFixed(2) : '—',
          max: typeof stats.max === 'number' ? stats.max.toFixed(2) : '—',
          mean: typeof stats.mean === 'number' ? stats.mean.toFixed(2) : '—',
          count: typeof stats.count === 'number' ? stats.count.toLocaleString() : '0',
        } : { min: '—', max: '—', mean: '—', count: '0' },
      };
    });

  const categoricalData = (summaryData?.headers || [])
    .filter((header) => normalizeType(summaryData?.columnTypes?.[header]) === 'categorical')
    .map((col) => {
      const stats = summaryData?.columnStats?.[col];
      const topValues = stats?.topValues || [];
      return {
        name: col,
        values: topValues.slice(0, 5).map((item) => ({
          label: item.value,
          pct: typeof item.pct === 'number' ? Math.round(item.pct) : 0,
          rows: typeof item.count === 'number' ? item.count : 0,
        })),
        color: 'var(--accent)',
      };
    });

  const nullAnalysis = (summaryData?.analysisColumns || [])
    .map((col) => ({
      col: col.name,
      pct: Number(col.null_pct || 0),
      label: col.null_count ? `${Number(col.null_pct || 0).toFixed(1)}% nulls` : 'No nulls',
    }))
    .sort((a, b) => b.pct - a.pct)
    .slice(0, 3);

  const cleaningCards = summaryData ? [
    { val: (summaryData.totalNulls ?? 0).toLocaleString(), lbl: 'Nulls Found', color: 'var(--warning)' },
    { val: (summaryData.duplicateRows ?? 0).toLocaleString(), lbl: 'Duplicate Rows', color: 'var(--accent)' },
    { val: summaryData.qualityScore ?? 'N/A', lbl: 'Quality Score', color: 'var(--success)' },
    { val: summaryData.cleaningReport?.length || 0, lbl: 'Cleaning Steps', color: 'var(--primary)' },
  ] : [];

  const cleaningSteps = summaryData?.cleaningReport?.length
    ? summaryData.cleaningReport.map((item, idx) => ({
        num: String(idx + 1),
        name: item.category || item.title || `Step ${idx + 1}`,
        detail: item.action || item.reason || 'Completed',
        result: typeof item.count === 'number' ? `${item.count.toLocaleString()} affected` : 'Done',
        skipped: !item.count,
      }))
    : [];

  const currentDataset = selectedDataset || {
    name: datasetName,
    rows_count: summaryData?.rows,
    columns_count: summaryData?.columns,
  };

  const summaryMessage = loading
    ? 'Loading dataset summary...'
    : summaryError
      ? summaryError
      : summaryData
        ? `This dataset contains ${summaryData.rows?.toLocaleString() || '—'} records with ${summaryData.columns || '—'} attributes.`
        : 'No dataset summary data is available yet.';

  return (
    <EmployeeLayout>
      <div className="emp-topbar">
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <button className="emp-btn emp-btn-ghost emp-btn-sm" onClick={() => navigate('/employee/datasets')}>
            Back
          </button>
          {availableDatasets.length > 1 && (
            <select
              className="emp-filter-select"
              value={selectedDataset?.dataset_id || selectedDataset?.id || ''}
              onChange={(e) => {
                const ds = availableDatasets.find((d) => (d.dataset_id || d.id) === e.target.value);
                if (ds) {
                  setSelectedDataset(ds);
                  navigate(`/employee/summary?ds=${ds.dataset_id || ds.id}&name=${encodeURIComponent(ds.name || '')}`);
                }
              }}
              style={{ minWidth: 180, fontSize: 11 }}
            >
              {availableDatasets.map((ds) => (
                <option key={ds.dataset_id || ds.id} value={ds.dataset_id || ds.id}>
                  {ds.name}
                </option>
              ))}
            </select>
          )}
          <div>
            <div className="emp-topbar-title">Dataset Summary</div>
            <div className="emp-topbar-sub">
              {currentDataset.name} · {summaryData?.rows?.toLocaleString() || '—'} rows · {summaryData?.columns || '—'} columns
            </div>
          </div>
        </div>
        <div className="emp-topbar-actions">
          <button className="emp-btn emp-btn-ghost emp-btn-sm">
            <Download size={12} /> Export
          </button>
          <button
            className="emp-btn emp-btn-primary emp-btn-sm"
            onClick={() => navigate(`/employee/chat?ds=${selectedDataset?.dataset_id || selectedDataset?.id}`)}
          >
            <MessageSquare size={12} /> Ask Chatbot
          </button>
        </div>
      </div>

      <div style={{ display: 'flex', flex: 1 }}>
        <div
          style={{
            width: 220,
            flexShrink: 0,
            padding: '24px 16px',
            borderRight: '1px solid var(--border-color)',
            position: 'sticky',
            top: 58,
            height: 'calc(100vh - 58px)',
            overflowY: 'auto',
          }}
        >
          <div style={{ fontFamily: "'DM Mono', monospace", fontSize: 9, color: 'var(--text-muted)', letterSpacing: 2, textTransform: 'uppercase', marginBottom: 10 }}>
            Contents
          </div>
          {TOC_SECTIONS.map((section) => (
            <div
              key={section.id}
              onClick={() => scrollTo(section.id)}
              style={{
                fontFamily: "'DM Mono', monospace",
                fontSize: 10,
                color: activeSection === section.id ? 'var(--primary)' : 'var(--text-muted)',
                padding: '5px 8px',
                borderRadius: 6,
                cursor: 'pointer',
                marginBottom: 2,
                display: 'flex',
                alignItems: 'center',
                gap: 7,
                transition: 'all 0.15s',
                background: activeSection === section.id ? 'rgba(88,166,255,0.08)' : 'transparent',
              }}
              onMouseEnter={(e) => {
                if (activeSection !== section.id) e.currentTarget.style.background = 'rgba(255,255,255,0.04)';
              }}
              onMouseLeave={(e) => {
                if (activeSection !== section.id) e.currentTarget.style.background = 'transparent';
              }}
            >
              <div style={{ width: 5, height: 5, borderRadius: '50%', background: 'currentColor', flexShrink: 0 }} />
              {section.label}
            </div>
          ))}
        </div>

        <div style={{ flex: 1, padding: '28px 32px', maxWidth: 900 }}>
          <div
            className="glass-panel"
            id="overview"
            style={{ padding: '24px 28px', marginBottom: 28, position: 'relative', overflow: 'hidden', scrollMarginTop: 80 }}
          >
            <div
              style={{
                position: 'absolute',
                top: -40,
                right: -40,
                width: 200,
                height: 200,
                background: 'radial-gradient(circle, rgba(88,166,255,0.08), transparent 70%)',
                borderRadius: '50%',
              }}
            />
            <div style={{ display: 'flex', alignItems: 'flex-start', gap: 16, marginBottom: 16 }}>
              <div
                style={{
                  width: 48,
                  height: 48,
                  borderRadius: 12,
                  background: 'rgba(88,166,255,0.1)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: 22,
                  flexShrink: 0,
                }}
              >
                📊
              </div>
              <div>
                <div style={{ fontSize: 22, fontWeight: 600, color: '#fff', lineHeight: 1.2 }}>
                  {currentDataset.name}
                </div>
                <div style={{ fontFamily: "'DM Mono', monospace", fontSize: 10, color: 'var(--text-muted)', marginTop: 4 }}>
                  Processed · Cleaned · Analysis Ready
                </div>
              </div>
              <div style={{ display: 'flex', gap: 8, marginLeft: 'auto' }}>
                <span
                  style={{
                    fontFamily: "'DM Mono', monospace",
                    fontSize: 10,
                    padding: '4px 10px',
                    borderRadius: 20,
                    background: 'rgba(63,185,80,0.08)',
                    color: 'var(--success)',
                  }}
                >
                  ● Cleaned
                </span>
                <span
                  style={{
                    fontFamily: "'DM Mono', monospace",
                    fontSize: 10,
                    padding: '4px 10px',
                    borderRadius: 20,
                    background: 'rgba(88,166,255,0.08)',
                    color: 'var(--primary)',
                  }}
                >
                  Chatbot Unlocked
                </span>
              </div>
            </div>

            <div
              style={{
                fontSize: 13.5,
                color: 'var(--text-main)',
                lineHeight: 1.75,
                padding: '14px 16px',
                background: 'rgba(13,17,23,0.95)',
                borderRadius: 10,
                borderLeft: '3px solid var(--primary)',
                fontStyle: 'italic',
              }}
            >
              {summaryMessage}
              {summaryData && (
                <>
                  {' '}
                  The data includes {Object.values(summaryData.columnTypes || {}).filter((t) => normalizeType(t) === 'numeric').length} numeric columns and{' '}
                  {Object.values(summaryData.columnTypes || {}).filter((t) => normalizeType(t) === 'categorical').length} categorical columns.
                </>
              )}
            </div>

            <div style={{ display: 'flex', gap: 24, marginTop: 16, flexWrap: 'wrap' }}>
              {[
                { val: summaryData?.rows?.toLocaleString() || '—', lbl: 'Total Rows' },
                { val: summaryData?.columns || '—', lbl: 'Columns' },
                { val: schema.length, lbl: 'Attributes' },
                {
                  val: selectedDataset?.status === 'completed' || selectedDataset?.status === 'ready' || selectedDataset?.status === 'cleaned' ? 'Cleaned' : 'Processing',
                  lbl: 'Status',
                  color: 'var(--success)',
                },
              ].map((metric, index) => (
                <div key={index} style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: 20, fontWeight: 600, color: metric.color || '#fff' }}>{metric.val}</div>
                  <div style={{ fontFamily: "'DM Mono', monospace", fontSize: 9, color: 'var(--text-muted)', marginTop: 2, textTransform: 'uppercase', letterSpacing: 0.5 }}>
                    {metric.lbl}
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div id="cleaning" style={{ marginBottom: 32, scrollMarginTop: 80 }}>
            <div style={sectionTitleStyle}>
              <Sparkles size={18} /> Cleaning Summary
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 10, marginBottom: 16 }}>
              {cleaningCards.map((card, index) => (
                <div key={index} className="glass-panel" style={{ padding: 14, textAlign: 'center' }}>
                  <div style={{ fontSize: 20, fontWeight: 600, color: card.color }}>{card.val}</div>
                  <div style={{ fontFamily: "'DM Mono', monospace", fontSize: 9, color: 'var(--text-muted)', marginTop: 3, textTransform: 'uppercase' }}>
                    {card.lbl}
                  </div>
                </div>
              ))}
            </div>

            <div className="glass-panel" style={{ overflow: 'hidden' }}>
              {cleaningSteps.length > 0 ? cleaningSteps.map((step, index) => (
                <div
                  key={index}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 12,
                    padding: '11px 16px',
                    borderBottom: index < cleaningSteps.length - 1 ? '1px solid rgba(255,255,255,0.025)' : 'none',
                  }}
                >
                  <div
                    style={{
                      width: 22,
                      height: 22,
                      borderRadius: '50%',
                      background: step.skipped ? 'rgba(255,255,255,0.04)' : 'rgba(63,185,80,0.08)',
                      color: step.skipped ? 'var(--text-muted)' : 'var(--success)',
                      fontFamily: "'DM Mono', monospace",
                      fontSize: 10,
                      fontWeight: 600,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      flexShrink: 0,
                    }}
                  >
                    {step.num}
                  </div>
                  <div style={{ flex: 1, fontSize: 13, fontWeight: 500, color: step.skipped ? 'var(--text-muted)' : '#fff' }}>{step.name}</div>
                  <div style={{ fontFamily: "'DM Mono', monospace", fontSize: 10, color: 'var(--text-muted)' }}>{step.detail}</div>
                  <div style={{ fontFamily: "'DM Mono', monospace", fontSize: 10, color: step.skipped ? 'var(--text-muted)' : 'var(--success)' }}>{step.result}</div>
                </div>
              )) : (
                <div style={{ padding: 16, color: 'var(--text-muted)', fontSize: 12 }}>
                  No cleaning report is available for this dataset yet.
                </div>
              )}
            </div>
          </div>

          <div id="schema" style={{ marginBottom: 32, scrollMarginTop: 80 }}>
            <div style={sectionTitleStyle}>⊞ Schema · {schema.length} Columns</div>
            <div className="glass-panel" style={{ overflow: 'hidden' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr>
                    {['#', 'Column Name', 'Type', 'Nullable', 'Unique Values'].map((col) => (
                      <th
                        key={col}
                        style={{
                          background: 'rgba(13,17,23,0.95)',
                          padding: '9px 12px',
                          textAlign: 'left',
                          fontFamily: "'DM Mono', monospace",
                          fontSize: 9,
                          color: 'var(--text-muted)',
                          letterSpacing: 1,
                          textTransform: 'uppercase',
                          borderBottom: '1px solid var(--border-color)',
                        }}
                      >
                        {col}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {schema.slice(0, 15).map((col, index) => (
                    <tr
                      key={index}
                      onMouseEnter={(e) => { e.currentTarget.style.background = 'rgba(22,27,34,0.7)'; }}
                      onMouseLeave={(e) => { e.currentTarget.style.background = ''; }}
                      style={{ transition: 'background 0.15s' }}
                    >
                      <td style={{ padding: '10px 12px', borderBottom: '1px solid rgba(255,255,255,0.025)', color: 'var(--text-muted)', fontFamily: "'DM Mono', monospace", fontSize: 10 }}>
                        {index + 1}
                      </td>
                      <td style={{ padding: '10px 12px', borderBottom: '1px solid rgba(255,255,255,0.025)' }}>
                        <code style={{ fontFamily: "'DM Mono', monospace", fontSize: 11, color: 'var(--primary)' }}>{col.name}</code>
                      </td>
                      <td style={{ padding: '10px 12px', borderBottom: '1px solid rgba(255,255,255,0.025)' }}>
                        <span
                          style={{
                            fontFamily: "'DM Mono', monospace",
                            fontSize: 9,
                            padding: '2px 8px',
                            borderRadius: 5,
                            background: typeColors[col.typeClass]?.bg,
                            color: typeColors[col.typeClass]?.color,
                          }}
                        >
                          {summaryData?.columnTypes?.[col.name] || col.type}
                        </span>
                      </td>
                      <td style={{ padding: '10px 12px', borderBottom: '1px solid rgba(255,255,255,0.025)', color: col.nullable === 'Yes' ? 'var(--warning)' : 'var(--success)', fontSize: 11 }}>
                        {col.nullable}
                      </td>
                      <td style={{ padding: '10px 12px', borderBottom: '1px solid rgba(255,255,255,0.025)', fontFamily: "'DM Mono', monospace", fontSize: 10, color: 'var(--text-main)' }}>
                        {col.unique}
                      </td>
                    </tr>
                  ))}
                  {schema.length > 15 && (
                    <tr>
                      <td colSpan={5} style={{ padding: '10px 12px', fontFamily: "'DM Mono', monospace", fontSize: 10, color: 'var(--text-muted)', textAlign: 'center' }}>
                        + {schema.length - 15} more columns
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>

          <div id="nulls" style={{ marginBottom: 32, scrollMarginTop: 80 }}>
            <div style={sectionTitleStyle}>○ Null Analysis (Post-Cleaning)</div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 8 }}>
              {nullAnalysis.length > 0 ? nullAnalysis.map((item, index) => (
                <div key={index} className="glass-panel" style={{ padding: '10px 12px' }}>
                  <div style={{ fontFamily: "'DM Mono', monospace", fontSize: 10, color: 'var(--text-main)', marginBottom: 6 }}>{item.col}</div>
                  <div style={{ height: 5, background: 'rgba(255,255,255,0.06)', borderRadius: 5, overflow: 'hidden', marginBottom: 4 }}>
                    <div style={{ height: '100%', width: `${item.pct}%`, background: item.pct > 0 ? 'var(--warning)' : 'var(--success)', borderRadius: 5 }} />
                  </div>
                  <div style={{ fontFamily: "'DM Mono', monospace", fontSize: 9, color: item.pct === 0 ? 'var(--success)' : 'var(--text-muted)' }}>
                    {item.label}
                  </div>
                </div>
              )) : (
                <div className="glass-panel" style={{ padding: 12, color: 'var(--text-muted)', fontSize: 12, gridColumn: '1 / -1' }}>
                  Null statistics are not available yet.
                </div>
              )}
            </div>
          </div>

          <div id="numeric" style={{ marginBottom: 32, scrollMarginTop: 80 }}>
            <div style={sectionTitleStyle}>∑ Numeric Column Statistics</div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
              {(numericStats.length > 0 ? numericStats : [{ name: 'No numeric columns', stats: { min: '—', max: '—', mean: '—', count: '0' } }]).map((col, index) => (
                <div key={index} className="glass-panel" style={{ padding: '14px 16px' }}>
                  <div style={{ fontFamily: "'DM Mono', monospace", fontSize: 11, color: 'var(--primary)', marginBottom: 8, fontWeight: 600 }}>
                    {col.name}
                  </div>
                  {Object.entries(col.stats).map(([key, value]) => (
                    <div key={key} style={{ display: 'flex', justifyContent: 'space-between', padding: '4px 0', borderBottom: '1px solid rgba(255,255,255,0.025)' }}>
                      <span style={{ fontFamily: "'DM Mono', monospace", fontSize: 10, color: 'var(--text-muted)' }}>{key}</span>
                      <span style={{ fontFamily: "'DM Mono', monospace", fontSize: 10, color: '#fff', fontWeight: 500 }}>{value}</span>
                    </div>
                  ))}
                </div>
              ))}
            </div>
          </div>

          <div id="categorical" style={{ marginBottom: 32, scrollMarginTop: 80 }}>
            <div style={sectionTitleStyle}>◈ Categorical Profiles</div>
            {(categoricalData.length > 0 ? categoricalData : [{ name: 'No categorical columns', values: [] }]).map((cat, catIndex) => (
              <div key={catIndex} className="glass-panel" style={{ padding: '14px 16px', marginBottom: 10 }}>
                <div style={{ fontFamily: "'DM Mono', monospace", fontSize: 11, color: 'var(--accent)', marginBottom: 10 }}>
                  {cat.name} ({cat.values.length} unique values)
                </div>
                {cat.values.length > 0 ? cat.values.map((value, valueIndex) => (
                  <div key={valueIndex} style={{ marginBottom: 6 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontFamily: "'DM Mono', monospace", fontSize: 9, color: 'var(--text-muted)', marginBottom: 3 }}>
                      <span>{value.label}</span>
                      <span>{value.pct}% · {value.rows} rows</span>
                    </div>
                    <div style={{ height: 4, background: 'rgba(255,255,255,0.06)', borderRadius: 4, overflow: 'hidden' }}>
                      <div
                        style={{
                          height: '100%',
                          width: `${value.pct}%`,
                          borderRadius: 4,
                          background: value.color ? `linear-gradient(90deg, ${value.color}, ${value.color})` : 'linear-gradient(90deg, var(--accent), #c4b5fd)',
                        }}
                      />
                    </div>
                  </div>
                )) : (
                  <div style={{ color: 'var(--text-muted)', fontSize: 12 }}>No categorical values available yet.</div>
                )}
              </div>
            ))}
          </div>

          <div id="actions" style={{ marginBottom: 32, scrollMarginTop: 80 }}>
            <div style={sectionTitleStyle}>→ Next Steps</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              {[
                {
                  icon: '◎',
                  title: 'Ask the Chatbot',
                  sub: 'Query this dataset in natural language · Chatbot has full context',
                  action: () => navigate(`/employee/chat?ds=${selectedDataset?.dataset_id || selectedDataset?.id}`),
                  label: 'Open Chatbot →',
                  primary: true,
                },
                {
                  icon: '▦',
                  title: 'View Dashboard',
                  sub: 'See auto-generated charts and AI insights for this dataset',
                  action: () => navigate(`/employee/dashboard?ds=${selectedDataset?.dataset_id || selectedDataset?.id}`),
                  label: 'Open Dashboard →',
                },
                {
                  icon: '✦',
                  title: 'Re-clean Dataset',
                  sub: 'Go back to cleaning wizard with saved selections',
                  action: () => navigate(`/employee/cleaning?ds=${selectedDataset?.dataset_id || selectedDataset?.id}`),
                  label: 'Open Cleaning →',
                },
              ].map((item, index) => (
                <div key={index} className="glass-panel" style={{ padding: '18px 22px', display: 'flex', alignItems: 'center', gap: 16 }}>
                  <div style={{ fontSize: 24 }}>{item.icon}</div>
                  <div style={{ flex: 1 }}>
                    <div style={{ fontSize: 14, fontWeight: 600, color: '#fff', marginBottom: 2 }}>{item.title}</div>
                    <div style={{ fontFamily: "'DM Mono', monospace", fontSize: 9, color: 'var(--text-muted)' }}>{item.sub}</div>
                  </div>
                  <button className={`emp-btn ${item.primary ? 'emp-btn-primary' : 'emp-btn-ghost'}`} onClick={item.action}>
                    {item.label}
                  </button>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </EmployeeLayout>
  );
};

const sectionTitleStyle = {
  fontSize: 17,
  fontWeight: 600,
  color: '#fff',
  marginBottom: 14,
  display: 'flex',
  alignItems: 'center',
  gap: 10,
  paddingBottom: 10,
  borderBottom: '1px solid var(--border-color)',
};

export default EmployeeSummaryPage;
