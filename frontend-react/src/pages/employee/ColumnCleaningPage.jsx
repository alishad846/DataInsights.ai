import { useState, useEffect } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { ArrowLeft, CheckCircle2, Sparkles, Zap, AlertTriangle, ArrowRight, Settings2, RefreshCw, FileCheck, Activity, BarChart3, UploadCloud } from 'lucide-react';
import EmployeeLayout from '../../layout/EmployeeLayout';

const PIPELINE_STEPS = [
  { id: 1, label: 'Uploading', icon: UploadCloud },
  { id: 2, label: 'Validating', icon: FileCheck },
  { id: 3, label: 'Cleaning', icon: Sparkles },
  { id: 4, label: 'Analyzing', icon: Activity },
  { id: 5, label: 'Dashboard', icon: BarChart3 },
];

const CLEANING_STRATEGIES = {
  numeric: [
    { value: 'mean', label: 'Fill with Mean', desc: 'Replace nulls with column average', ai: true },
    { value: 'median', label: 'Fill with Median', desc: 'Replace nulls with column median' },
    { value: 'zero', label: 'Fill with Zero', desc: 'Replace nulls with 0' },
    { value: 'ffill', label: 'Forward Fill', desc: 'Use previous row value' },
    { value: 'drop', label: 'Drop Rows', desc: 'Remove rows with nulls in this column' },
  ],
  categorical: [
    { value: 'mode', label: 'Fill with Mode', desc: 'Replace nulls with most frequent value', ai: true },
    { value: 'unknown', label: 'Fill with "Unknown"', desc: 'Replace nulls with placeholder' },
    { value: 'drop', label: 'Drop Rows', desc: 'Remove rows with nulls in this column' },
  ],
  datetime: [
    { value: 'ffill', label: 'Forward Fill', desc: 'Use previous row date', ai: true },
    { value: 'bfill', label: 'Backward Fill', desc: 'Use next row date' },
    { value: 'drop', label: 'Drop Rows', desc: 'Remove rows with nulls' },
  ],
  text: [
    { value: 'empty', label: 'Fill with Empty', desc: 'Replace nulls with empty string', ai: true },
    { value: 'unknown', label: 'Fill with "N/A"', desc: 'Replace nulls with placeholder' },
    { value: 'drop', label: 'Drop Rows', desc: 'Remove rows with nulls' },
  ],
};

const ADDITIONAL_OPS = {
  numeric: [
    { id: 'trim_whitespace', label: 'Trim Whitespace', desc: 'Remove leading/trailing spaces', default: false },
    { id: 'remove_outliers', label: 'Remove Outliers (IQR)', desc: 'Cap values outside 1.5×IQR', default: false },
    { id: 'round_values', label: 'Round to 2 Decimals', desc: 'Round float values', default: false },
  ],
  categorical: [
    { id: 'trim_whitespace', label: 'Trim Whitespace', desc: 'Remove leading/trailing spaces', default: true },
    { id: 'fix_case', label: 'Fix Casing (Title)', desc: 'Convert to Title Case', default: false },
    { id: 'remove_special', label: 'Remove Special Chars', desc: 'Strip non-alphanumeric characters', default: false },
  ],
  datetime: [
    { id: 'standardize', label: 'Standardize Format', desc: 'Convert to YYYY-MM-DD', default: true },
  ],
  text: [
    { id: 'trim_whitespace', label: 'Trim Whitespace', desc: 'Remove leading/trailing spaces', default: true },
    { id: 'fix_case', label: 'Fix Casing (Title)', desc: 'Convert to Title Case', default: false },
  ],
};

const ColumnCleaningPage = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const datasetId = searchParams.get('ds') || 'dataset_001';
  const datasetName = searchParams.get('name') || 'Customer_Data.xlsx';
  const colNames = searchParams.get('cols')?.split(',') || [];

  const [columns, setColumns] = useState([]);
  const [cleaning, setCleaning] = useState(false);
  const [cleaningStep, setCleaningStep] = useState(0);
  const [completed, setCompleted] = useState(false);
  const [beforeAfter, setBeforeAfter] = useState(null);

  // Build column configs from the column names passed via URL
  useEffect(() => {
    const configs = colNames.map((name, i) => {
      const isNumeric = ['revenue', 'orders', 'retention_score', 'discount', 'profit', 'unit_price', 'cost', 'score', 'quantity', 'age'].includes(name);
      const isDate = name.includes('date');
      const isCategorical = ['segment', 'region', 'channel', 'product', 'status', 'city', 'gender'].includes(name);
      const isText = ['email', 'name', 'rep_name', 'notes'].includes(name);

      let type = 'text';
      let nulls = 0;
      let nullPct = 0;
      let strategy = 'empty';
      let additionalOps = {};

      if (isNumeric) {
        type = 'numeric';
        nulls = Math.floor(Math.random() * 50);
        nullPct = parseFloat((Math.random() * 5).toFixed(2));
        strategy = 'mean';
        ADDITIONAL_OPS.numeric.forEach(op => { additionalOps[op.id] = op.default; });
      } else if (isDate) {
        type = 'datetime';
        nulls = Math.floor(Math.random() * 30);
        nullPct = parseFloat((Math.random() * 3).toFixed(2));
        strategy = 'ffill';
        ADDITIONAL_OPS.datetime.forEach(op => { additionalOps[op.id] = op.default; });
      } else if (isCategorical) {
        type = 'categorical';
        nulls = Math.floor(Math.random() * 15);
        nullPct = parseFloat((Math.random() * 2).toFixed(2));
        strategy = 'mode';
        ADDITIONAL_OPS.categorical.forEach(op => { additionalOps[op.id] = op.default; });
      } else {
        ADDITIONAL_OPS.text.forEach(op => { additionalOps[op.id] = op.default; });
        nulls = Math.floor(Math.random() * 100);
        nullPct = parseFloat((Math.random() * 10).toFixed(2));
        strategy = 'empty';
      }

      return { name, type, nulls, nullPct, strategy, additionalOps, id: i };
    });
    setColumns(configs);
  }, [colNames.join(',')]);

  const updateStrategy = (colName, strategy) => {
    setColumns(prev => prev.map(c => c.name === colName ? { ...c, strategy } : c));
  };

  const toggleOp = (colName, opId) => {
    setColumns(prev => prev.map(c => c.name === colName ? {
      ...c, additionalOps: { ...c.additionalOps, [opId]: !c.additionalOps[opId] }
    } : c));
  };

  const handleClean = async () => {
    setCleaning(true);
    // Simulate cleaning pipeline steps
    const stepOrder = [0, 1, 2, 3, 4]; // Uploading -> Validating -> Cleaning -> Analyzing -> Dashboard
    for (let i = 0; i < stepOrder.length; i++) {
      setCleaningStep(stepOrder[i]);
      await new Promise(r => setTimeout(r, 800 + Math.random() * 500));
    }
    setCompleted(true);
    setCleaning(false);
  };

  const handleViewDashboard = () => {
    navigate(`/employee/dashboard?ds=${datasetId}&name=${encodeURIComponent(datasetName)}`);
  };

  // Get step status
  const getStepStatus = (stepIndex) => {
    if (completed) return 'completed';
    if (cleaningStep >= stepIndex) return 'active';
    return 'pending';
  };

  // Cleaning progress
  if (cleaning || completed) {
    return (
      <EmployeeLayout>
        <div className="emp-topbar">
          <div>
            <div className="emp-topbar-title">{completed ? 'Pipeline Complete' : 'Processing Dataset'}</div>
            <div className="emp-topbar-sub">{datasetName}</div>
          </div>
        </div>
        <div className="emp-content" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '60vh' }}>
          <div className="glass-panel" style={{ width: '100%', maxWidth: 700, padding: '2.5rem 2rem', textAlign: 'center' }}>
            
            {/* Pipeline Steps */}
            <div style={{ display: 'flex', justifyContent: 'center', gap: 8, marginBottom: 32 }}>
              {PIPELINE_STEPS.map((step, i) => {
                const status = getStepStatus(i);
                const Icon = step.icon;
                return (
                  <div key={step.id} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6, flex: 1 }}>
                    <div style={{
                      width: 44, height: 44, borderRadius: '50%',
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      border: `2px solid ${
                        status === 'completed' ? 'var(--success)' : 
                        status === 'active' ? 'var(--primary)' : 'var(--border-color)'
                      }`,
                      background: status === 'completed' ? 'var(--success)' : status === 'active' ? 'var(--primary)' : 'transparent',
                      color: (status === 'completed' || status === 'active') ? '#fff' : 'var(--text-muted)',
                      transition: 'all 0.3s',
                    }}>
                      {status === 'completed' ? <CheckCircle2 size={20} /> : status === 'active' ? <Icon size={18} className="spin" /> : <Icon size={18} />}
                    </div>
                    <div style={{ fontFamily: "'DM Mono', monospace", fontSize: 9, textTransform: 'uppercase', letterSpacing: 0.5,
                      color: status === 'active' ? 'var(--primary)' : status === 'completed' ? 'var(--success)' : 'var(--text-muted)',
                    }}>{step.label}</div>
                  </div>
                );
              })}
            </div>

            {completed ? (
              <>
                <CheckCircle2 size={56} color="var(--success)" style={{ marginBottom: 16 }} />
                <h2 style={{ color: 'var(--success)', marginBottom: 8, fontSize: '1.5rem' }}>Dataset Processed Successfully!</h2>
                <p style={{ color: 'var(--text-muted)', marginBottom: 24, fontSize: 13 }}>
                  {columns.length} columns cleaned · All pipeline stages completed
                </p>

                {/* Summary */}
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12, margin: '20px 0' }}>
                  <div style={{ background: 'rgba(88,166,255,0.08)', border: '1px solid rgba(88,166,255,0.2)', borderRadius: 8, padding: 12 }}>
                    <div style={{ fontFamily: "'DM Mono', monospace", fontSize: 20, fontWeight: 600, color: 'var(--primary)' }}>✓</div>
                    <div style={{ fontFamily: "'DM Mono', monospace", fontSize: 9, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Uploaded</div>
                  </div>
                  <div style={{ background: 'rgba(88,166,255,0.08)', border: '1px solid rgba(88,166,255,0.2)', borderRadius: 8, padding: 12 }}>
                    <div style={{ fontFamily: "'DM Mono', monospace", fontSize: 20, fontWeight: 600, color: 'var(--primary)' }}>✓</div>
                    <div style={{ fontFamily: "'DM Mono', monospace", fontSize: 9, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Validated</div>
                  </div>
                  <div style={{ background: 'rgba(63,185,80,0.08)', border: '1px solid rgba(63,185,80,0.2)', borderRadius: 8, padding: 12 }}>
                    <div style={{ fontFamily: "'DM Mono', monospace", fontSize: 20, fontWeight: 600, color: 'var(--success)' }}>{columns.length}</div>
                    <div style={{ fontFamily: "'DM Mono', monospace", fontSize: 9, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Cleaned</div>
                  </div>
                  <div style={{ background: 'rgba(188,140,255,0.08)', border: '1px solid rgba(188,140,255,0.2)', borderRadius: 8, padding: 12 }}>
                    <div style={{ fontFamily: "'DM Mono', monospace", fontSize: 20, fontWeight: 600, color: 'var(--accent)' }}>✓</div>
                    <div style={{ fontFamily: "'DM Mono', monospace", fontSize: 9, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Analyzed</div>
                  </div>
                </div>

                <div style={{ display: 'flex', gap: 12, justifyContent: 'center', marginTop: 24 }}>
                  <button className="emp-btn emp-btn-primary" onClick={handleViewDashboard} style={{ padding: '12px 28px', fontSize: 14 }}>
                    <BarChart3 size={16} /> View Dashboard →
                  </button>
                  <button className="emp-btn emp-btn-ghost" onClick={() => navigate('/employee/analysis')}>
                    Back to Analysis
                  </button>
                </div>
              </>
            ) : (
              <>
                <div style={{ marginBottom: 16 }}>
                  <RefreshCw size={48} color="var(--primary)" className="spin" />
                </div>
                <h2 style={{ color: '#fff', marginBottom: 8 }}>Processing Your Dataset</h2>
                <p style={{ color: 'var(--primary)', fontFamily: "'DM Mono', monospace", fontSize: 14, marginBottom: 16 }}>
                  {PIPELINE_STEPS[cleaningStep].label}...
                </p>
                <div style={{ height: 6, background: 'rgba(255,255,255,0.06)', borderRadius: 6, overflow: 'hidden', marginBottom: 12 }}>
                  <div style={{
                    height: '100%', width: `${((cleaningStep + 1) / PIPELINE_STEPS.length) * 100}%`,
                    background: 'linear-gradient(90deg, var(--primary), var(--accent))',
                    borderRadius: 6, transition: 'width 0.5s ease',
                  }} />
                </div>
                <p style={{ fontFamily: "'DM Mono', monospace", fontSize: 10, color: 'var(--text-muted)' }}>
                  Step {cleaningStep + 1} of {PIPELINE_STEPS.length}
                </p>
              </>
            )}
          </div>
        </div>
        <style>{`
          @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
          .spin { animation: spin 1s linear infinite; }
        `}</style>
      </EmployeeLayout>
    );
  }

  return (
    <EmployeeLayout>
      <div className="emp-topbar">
        <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
          <button className="emp-btn emp-btn-ghost emp-btn-sm" onClick={() => navigate(-1)}>
            <ArrowLeft size={14} /> Back
          </button>
          <div>
            <div className="emp-topbar-title">Column Cleaning</div>
            <div className="emp-topbar-sub">{datasetName} · {columns.length} selected columns</div>
          </div>
        </div>
        <div className="emp-topbar-actions">
          <button className="emp-btn emp-btn-primary" onClick={handleClean} disabled={columns.length === 0}
            style={{ padding: '8px 20px' }}>
            <Sparkles size={14} /> Run Cleaning →
          </button>
        </div>
      </div>

      <div className="emp-content">
        {/* Column Cards */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
          {columns.map((col, i) => {
            const strategies = CLEANING_STRATEGIES[col.type] || CLEANING_STRATEGIES.text;
            const additionalOps = ADDITIONAL_OPS[col.type] || ADDITIONAL_OPS.text;
            const hasNulls = col.nulls > 0;

            return (
              <div key={col.name} className="glass-panel" style={{
                padding: '18px 22px',
                animation: 'adminFadeUp 0.4s ease both',
                animationDelay: `${i * 0.04}s`,
                borderColor: hasNulls ? 'rgba(210,153,34,0.15)' : undefined,
              }}>
                <div style={{ display: 'flex', alignItems: 'flex-start', gap: 16 }}>
                  {/* Column Info */}
                  <div style={{ flex: '0 0 200px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
                      <code style={{ fontFamily: "'DM Mono', monospace", fontSize: 14, color: '#fff', fontWeight: 600 }}>{col.name}</code>
                      {hasNulls && <AlertTriangle size={14} color="var(--warning)" />}
                    </div>
                    <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                      <span style={{
                        fontFamily: "'DM Mono', monospace", fontSize: 9, padding: '2px 7px', borderRadius: 5,
                        background: col.type === 'numeric' ? 'rgba(63,185,80,0.1)' : col.type === 'categorical' ? 'rgba(188,140,255,0.1)' : col.type === 'datetime' ? 'rgba(210,153,34,0.1)' : 'rgba(88,166,255,0.1)',
                        color: col.type === 'numeric' ? 'var(--success)' : col.type === 'categorical' ? 'var(--accent)' : col.type === 'datetime' ? 'var(--warning)' : 'var(--primary)',
                      }}>{col.type}</span>
                      {hasNulls && (
                        <span style={{ fontFamily: "'DM Mono', monospace", fontSize: 9, padding: '2px 7px', borderRadius: 5, background: 'rgba(210,153,34,0.1)', color: 'var(--warning)' }}>
                          {col.nulls} nulls ({col.nullPct}%)
                        </span>
                      )}
                    </div>
                  </div>

                  {/* Null Strategy */}
                  <div style={{ flex: 1, minWidth: 200 }}>
                    <div style={{ fontFamily: "'DM Mono', monospace", fontSize: 9, color: 'var(--text-muted)', letterSpacing: 1, textTransform: 'uppercase', marginBottom: 8 }}>
                      Null Handling Strategy
                    </div>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                      {strategies.map(s => (
                        <label key={s.value} style={{
                          display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer',
                          padding: '6px 10px', borderRadius: 6,
                          border: `1px solid ${col.strategy === s.value ? 'var(--primary)' : 'var(--border-color)'}`,
                          background: col.strategy === s.value ? 'rgba(88,166,255,0.06)' : 'transparent',
                          transition: 'all 0.15s',
                        }}>
                          <input type="radio" name={`strategy-${col.name}`} checked={col.strategy === s.value}
                            onChange={() => updateStrategy(col.name, s.value)}
                            style={{ accentColor: 'var(--primary)' }} />
                          <div>
                            <div style={{ fontSize: 12, fontWeight: 500, color: '#fff' }}>{s.label}</div>
                            <div style={{ fontFamily: "'DM Mono', monospace", fontSize: 9, color: 'var(--text-muted)' }}>{s.desc}</div>
                          </div>
                          {s.ai && (
                            <span style={{ marginLeft: 'auto', fontFamily: "'DM Mono', monospace", fontSize: 9, background: 'rgba(188,140,255,0.1)', color: 'var(--accent)', padding: '2px 6px', borderRadius: 5 }}>
                              ✦ AI pick
                            </span>
                          )}
                        </label>
                      ))}
                    </div>
                  </div>

                  {/* Additional Operations */}
                  <div style={{ flex: '0 0 220px' }}>
                    <div style={{ fontFamily: "'DM Mono', monospace", fontSize: 9, color: 'var(--text-muted)', letterSpacing: 1, textTransform: 'uppercase', marginBottom: 8 }}>
                      Additional Operations
                    </div>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                      {additionalOps.map(op => (
                        <label key={op.id} style={{
                          display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer',
                          padding: '5px 8px', borderRadius: 6,
                          border: '1px solid var(--border-color)',
                          background: col.additionalOps[op.id] ? 'rgba(63,185,80,0.06)' : 'transparent',
                          transition: 'all 0.15s',
                        }}>
                          <input type="checkbox" checked={col.additionalOps[op.id] || false}
                            onChange={() => toggleOp(col.name, op.id)}
                            style={{ accentColor: 'var(--success)' }} />
                          <div>
                            <div style={{ fontSize: 11, fontWeight: 500, color: '#fff' }}>{op.label}</div>
                            <div style={{ fontFamily: "'DM Mono', monospace", fontSize: 8, color: 'var(--text-muted)' }}>{op.desc}</div>
                          </div>
                        </label>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        {/* Bottom Action */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 20, padding: '16px 0', borderTop: '1px solid var(--border-color)' }}>
          <div style={{ fontFamily: "'DM Mono', monospace", fontSize: 11, color: 'var(--text-muted)' }}>
            {columns.length} columns will be cleaned · {columns.filter(c => c.nulls > 0).length} have null values
          </div>
          <div style={{ display: 'flex', gap: 10 }}>
            <button className="emp-btn emp-btn-ghost" onClick={() => navigate(-1)}>
              <ArrowLeft size={14} /> Back to Analysis
            </button>
            <button className="emp-btn emp-btn-primary" onClick={handleClean} style={{ padding: '10px 24px', fontSize: 13 }}>
              <Sparkles size={14} /> Run Cleaning →
            </button>
          </div>
        </div>
      </div>
    </EmployeeLayout>
  );
};

export default ColumnCleaningPage;
