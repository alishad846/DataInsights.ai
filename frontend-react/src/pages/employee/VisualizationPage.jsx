import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { useParams, useSearchParams, useNavigate } from 'react-router-dom';
import {
  PieChart, Pie, Cell, BarChart, Bar, LineChart, Line, AreaChart, Area,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts';
import {
  Filter, BarChart3, TrendingUp, ChevronDown, ChevronUp,
  Search, RefreshCw, ArrowLeft, Loader, PieChart as PieIcon,
  Database,
  Activity
} from 'lucide-react';
import EmployeeLayout from '../../layout/EmployeeLayout';
import { getDashboardConfig, getDatasets } from '../../services/api';

const api = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5000/api';

const fetchCleanedData = async (datasetId, params = {}) => {
  const token = sessionStorage.getItem('token');
  const queryStr = new URLSearchParams();
  if (params.filters) queryStr.set('filters', JSON.stringify(params.filters));
  if (params.search) queryStr.set('search', params.search);
  if (params.page) queryStr.set('page', params.page);
  if (params.limit) queryStr.set('limit', params.limit || 500);
  const res = await fetch(`${api}/cleaned-data/${datasetId}?${queryStr}`, {
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  });
  return res.json();
};

const COLORS = ['#58a6ff', '#3fb950', '#bc8cff', '#d29922', '#f85149', '#79c0ff', '#d2a8ff', '#ffa657'];

const TooltipBox = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: 'rgba(22,27,34,0.96)', border: '1px solid var(--border-color)',
      borderRadius: 8, padding: '8px 12px', fontSize: 12, maxWidth: 220,
    }}>
      {label && <div style={{ color: 'var(--text-muted)', marginBottom: 4, fontSize: 11 }}>{label}</div>}
      {payload.map((p, i) => (
        <div key={i} style={{ color: p.color || '#58a6ff', fontWeight: 600 }}>
          {p.name}: {typeof p.value === 'number' ? p.value.toLocaleString() : p.value}
        </div>
      ))}
    </div>
  );
};

// Rich Pie/Donut hover tooltip — mirrors matplotlib hover behaviour
const PieTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null;
  const d = payload[0];
  const total = payload[0]?.payload?.total || 1;
  const pct = ((d.value / total) * 100).toFixed(1);
  return (
    <div style={{
      background: 'rgba(22,27,34,0.97)',
      border: `1px solid ${d.payload?.fill || d.color || '#58a6ff'}`,
      borderRadius: 10, padding: '10px 14px', minWidth: 160, boxShadow: '0 8px 32px rgba(0,0,0,0.4)',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
        <span style={{ width: 10, height: 10, borderRadius: '50%', background: d.payload?.fill || d.color, display: 'inline-block', flexShrink: 0 }} />
        <span style={{ color: '#fff', fontWeight: 700, fontSize: 13, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: 130 }}>{d.name}</span>
      </div>
      <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 11, display: 'flex', flexDirection: 'column', gap: 3 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', gap: 16 }}>
          <span style={{ color: '#8b949e' }}>Value</span>
          <span style={{ color: d.payload?.fill || '#58a6ff', fontWeight: 600 }}>{typeof d.value === 'number' ? d.value.toLocaleString(undefined, { maximumFractionDigits: 2 }) : d.value}</span>
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', gap: 16 }}>
          <span style={{ color: '#8b949e' }}>Share</span>
          <span style={{ color: '#3fb950', fontWeight: 600 }}>{pct}%</span>
        </div>
      </div>
    </div>
  );
};

const VisualizationPage = () => {
  const { datasetId: paramDatasetId } = useParams();
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const datasetId = paramDatasetId || searchParams.get('ds');
  const datasetName = searchParams.get('name') || datasetId;

  const [data, setData] = useState(null);
  const [dashboardConfig, setDashboardConfig] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [search, setSearch] = useState('');
  const [filters, setFilters] = useState({});
  const [appliedFilters, setAppliedFilters] = useState({});
  const [expandedFilter, setExpandedFilter] = useState(null);
  const [page, setPage] = useState(1);
  const [tablePage, setTablePage] = useState(1);
  const TABLE_PAGE_SIZE = 50;
  const [chartType, setChartType] = useState('bar');
  const [aggregation, setAggregation] = useState('sum');
  
  const [chartXAxis, setChartXAxis] = useState('');
  const [chartYAxis, setChartYAxis] = useState('');
  
  const [availableDatasets, setAvailableDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState(null);
  const [emptyStateMessage, setEmptyStateMessage] = useState('');

  const isInitialized = useRef(false);
  const NO_DATA_MESSAGE = 'You need data first. Upload a dataset and complete cleaning before using Visualize.';

  const loadData = useCallback(async (currentFilters = {}, currentSearch = '', currentPage = 1) => {
    if (!datasetId) {
      setLoading(false);
      return;
    }
    setLoading(true);
    setError('');
    setEmptyStateMessage('');
    try {
      const [cleanedRes, dashRes] = await Promise.all([
        fetchCleanedData(datasetId, { filters: currentFilters, search: currentSearch, page: currentPage, limit: 500 }),
        getDashboardConfig(datasetId).catch(() => null),
      ]);
      
      if (cleanedRes.success) {
        const hasRows = Array.isArray(cleanedRes.rows) && cleanedRes.rows.length > 0;
        const hasHeaders = Array.isArray(cleanedRes.headers) && cleanedRes.headers.length > 0;

        if (!hasRows || !hasHeaders || (cleanedRes.totalRows ?? 0) === 0) {
          setData(null);
          setEmptyStateMessage(NO_DATA_MESSAGE);
          setLoading(false);
          return;
        }

        setData(cleanedRes);
        setEmptyStateMessage('');
        
        if (!isInitialized.current && cleanedRes.headers?.length > 0) {
          const catCol = cleanedRes.headers.find(h => cleanedRes.columnTypes?.[h] === 'categorical');
          const numCol = cleanedRes.headers.find(h => cleanedRes.columnTypes?.[h] === 'numeric');
          setChartXAxis(catCol || cleanedRes.headers[0]);
          setChartYAxis(numCol || cleanedRes.headers[1] || '');
          isInitialized.current = true;
        }
      } else {
        setData(null);
        setEmptyStateMessage(
          cleanedRes.message?.toLowerCase().includes('cleaned data not found') ||
          cleanedRes.message?.toLowerCase().includes('complete the cleaning')
            ? NO_DATA_MESSAGE
            : cleanedRes.message || NO_DATA_MESSAGE
        );
      }
      
      if (dashRes) setDashboardConfig(dashRes);
    } catch {
      setData(null);
      setEmptyStateMessage(NO_DATA_MESSAGE);
    } finally {
      setLoading(false);
    }
  }, [datasetId]);

  useEffect(() => { 
    if (datasetId) {
      loadData(appliedFilters, search, page); 
    }
  }, [datasetId]);

  useEffect(() => {
    const loadDatasets = async () => {
      try {
        const res = await getDatasets();
        if (res.success && res.data) {
          const readyDatasets = res.data.filter(d => d.status === 'completed' || d.status === 'ready' || d.status === 'cleaned');
          setAvailableDatasets(readyDatasets);
          
          if (!datasetId) {
            if (readyDatasets.length > 0) {
              const firstReady = readyDatasets[0];
              setSelectedDataset(firstReady);
              navigate(`/employee/visualization?ds=${firstReady.dataset_id || firstReady.id}&name=${encodeURIComponent(firstReady.name || '')}`, { replace: true });
            } else {
              setSelectedDataset(null);
              setEmptyStateMessage(NO_DATA_MESSAGE);
              setLoading(false);
            }
          } else {
            const selected = res.data.find(d => (d.dataset_id || d.id) === datasetId);
            if (selected) {
              setSelectedDataset(selected);
            } else {
              setSelectedDataset(null);
              setEmptyStateMessage(NO_DATA_MESSAGE);
              setLoading(false);
            }
          }
        }
        if (!res.success || !res.data || res.data.length === 0) {
          setEmptyStateMessage(NO_DATA_MESSAGE);
          setLoading(false);
        }
      } catch (err) {
        console.warn('Could not load datasets:', err.message);
        setEmptyStateMessage(NO_DATA_MESSAGE);
        setLoading(false);
      }
    };
    loadDatasets();
  }, [datasetId]);

  const applyFilters = useCallback(() => { 
    setAppliedFilters({ ...filters }); 
    setPage(1); 
    loadData({ ...filters }, search, 1); 
  }, [filters, search, loadData]);

  const clearFilters = useCallback(() => { 
    setFilters({}); 
    setAppliedFilters({}); 
    setSearch(''); 
    setPage(1); 
    loadData({}, '', 1); 
  }, [loadData]);

  const toggleFilterValue = useCallback((col, val) => {
    setFilters(prev => {
      const cur = prev[col] || [];
      if (cur.includes(val)) return { ...prev, [col]: cur.filter(v => v !== val) };
      return { ...prev, [col]: [...cur, val] };
    });
  }, []);

  const setNumericFilter = useCallback((col, min, max) => {
    setFilters(prev => ({
      ...prev,
      [col]: { min: min !== '' ? parseFloat(min) : undefined, max: max !== '' ? parseFloat(max) : undefined },
    }));
  }, []);

  const appliedFilterCount = useMemo(() => {
    return Object.keys(appliedFilters).filter(k => {
      const v = appliedFilters[k];
      if (Array.isArray(v)) return v.length > 0;
      return v?.min !== undefined || v?.max !== undefined;
    }).length;
  }, [appliedFilters]);

  const headers = useMemo(() => {
    if (!data?.headers) return [];
    return data.headers.filter(h => h !== 'Unnamed: 0.1' && h !== 'Unnamed: 0');
  }, [data?.headers]);

  const chartData = useMemo(() => {
    if (!data?.rows || !chartXAxis || !chartYAxis || !headers.length) return [];
    
    const isNumericY = data.columnTypes?.[chartYAxis] === 'numeric';
    const grouped = {};
    const counts = {};
    const maxs = {};
    const mins = {};
    const sums = {};
    
    for (let i = 0; i < data.rows.length; i++) {
      const row = data.rows[i];
      const key = row[chartXAxis] || 'Unknown';
      const val = row[chartYAxis];
      
      if (isNumericY) {
        const numVal = parseFloat(val);
        if (!isNaN(numVal)) {
          grouped[key] = (grouped[key] || 0) + numVal;
          sums[key] = (sums[key] || 0) + numVal;
          counts[key] = (counts[key] || 0) + 1;
          maxs[key] = Math.max(maxs[key] || -Infinity, numVal);
          mins[key] = mins[key] === undefined ? numVal : Math.min(mins[key], numVal);
        }
      } else {
        grouped[key] = (grouped[key] || 0) + 1;
        counts[key] = (counts[key] || 0) + 1;
      }
    }
    
    const entries = Object.entries(grouped);
    const result = [];
    
    for (let i = 0; i < entries.length; i++) {
      const [name, value] = entries[i];
      result.push({
        name: String(name).substring(0, 18),
        value: isNumericY 
          ? (aggregation === 'sum' ? Math.round((sums[name] || 0) * 100) / 100 : 
             aggregation === 'count' ? counts[name] || 0 :
             aggregation === 'avg' ? Math.round(((sums[name] || 0) / (counts[name] || 1)) * 100) / 100 :
             aggregation === 'max' ? Math.round((maxs[name] || 0) * 100) / 100 :
             aggregation === 'min' ? Math.round((mins[name] || 0) * 100) / 100 : value)
          : value,
        rawValue: value,
        count: counts[name] || 0,
        max: maxs[name],
        min: mins[name]
      });
    }
    
    result.sort((a, b) => b.value - a.value);
    return result.slice(0, 10);
  }, [data, chartXAxis, chartYAxis, aggregation, headers]);

  const chartStats = useMemo(() => {
    if (!chartData.length) return null;
    const isNumericY = data?.columnTypes?.[chartYAxis] === 'numeric';
    if (!isNumericY) {
      return { totalSum: null, totalCount: chartData.reduce((acc, d) => acc + d.rawValue, 0), avg: null, max: null, min: null };
    }
    const totalSum = chartData.reduce((acc, d) => acc + d.rawValue, 0);
    const totalCount = chartData.reduce((acc, d) => acc + d.count, 0);
    const avg = totalCount > 0 ? totalSum / totalCount : 0;
    const max = Math.max(...chartData.map(d => d.rawValue));
    const min = Math.min(...chartData.filter(d => d.rawValue > 0).map(d => d.rawValue), 0);
    return { totalSum, totalCount, avg, max, min };
  }, [chartData, chartYAxis, data, aggregation]);

  const renderMainChart = () => {
    if (!chartData.length) return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: 'var(--text-muted)', fontSize: 13 }}>
        Select X and Y axes to generate a chart
      </div>
    );
    
    const chartProps = {
      data: chartData,
      margin: { top: 10, right: 10, left: 0, bottom: 0 }
    };
    
    switch (chartType) {
      case 'bar':
        return (
          <ResponsiveContainer width="100%" height="100%">
            <BarChart {...chartProps}>
              <defs>
                <linearGradient id="barFillGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#58a6ff" stopOpacity={1} />
                  <stop offset="100%" stopColor="#58a6ff" stopOpacity={0.5} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
              <XAxis dataKey="name" tick={{ fill: '#3d4f6e', fontSize: 9 }} />
              <YAxis tick={{ fill: '#3d4f6e', fontSize: 9 }} />
              <Tooltip content={<TooltipBox />} />
              <Bar dataKey="value" name={chartYAxis} fill="url(#barFillGrad)" radius={[6, 6, 0, 0]}>
                {chartData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        );
      case 'line':
        return (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart {...chartProps}>
              <defs>
                <linearGradient id="lineGrad" x1="0" y1="0" x2="1" y2="0">
                  <stop offset="0%" stopColor="#58a6ff" />
                  <stop offset="100%" stopColor="#bc8cff" />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
              <XAxis dataKey="name" tick={{ fill: '#3d4f6e', fontSize: 9 }} />
              <YAxis tick={{ fill: '#3d4f6e', fontSize: 9 }} />
              <Tooltip content={<TooltipBox />} />
              <Line type="monotone" dataKey="value" name={chartYAxis} stroke="url(#lineGrad)" strokeWidth={3} dot={{ fill: '#58a6ff', strokeWidth: 2, stroke: '#fff', r: 4 }} />
            </LineChart>
          </ResponsiveContainer>
        );
      case 'pie':
        return (
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <defs>
                {COLORS.map((color, i) => (
                  <linearGradient key={i} id={`pieGrad${i}`} x1="0" y1="0" x2="1" y2="1">
                    <stop offset="0%" stopColor={color} stopOpacity={1} />
                    <stop offset="100%" stopColor={color} stopOpacity={0.6} />
                  </linearGradient>
                ))}
              </defs>
              <Pie data={chartData} dataKey="value" nameKey="name" outerRadius={85} innerRadius={40} paddingAngle={3}
                label={({ name, percent }) => `${name} (${(percent * 100).toFixed(0)}%)`}>
                {chartData.map((_, i) => <Cell key={i} fill={`url(#pieGrad${i % COLORS.length})`} stroke="rgba(22,27,34,0.5)" strokeWidth={2} />)}
              </Pie>
              <Tooltip content={<TooltipBox />} />
              <Legend formatter={v => <span style={{ color: '#8b949e', fontSize: 9 }}>{v}</span>} />
            </PieChart>
          </ResponsiveContainer>
        );
      case 'area':
        return (
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart {...chartProps}>
              <defs>
                <linearGradient id="areaGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#58a6ff" stopOpacity={0.5} />
                  <stop offset="50%" stopColor="#58a6ff" stopOpacity={0.25} />
                  <stop offset="100%" stopColor="#58a6ff" stopOpacity={0.02} />
                </linearGradient>
                <linearGradient id="areaStroke" x1="0" y1="0" x2="1" y2="0">
                  <stop offset="0%" stopColor="#58a6ff" />
                  <stop offset="100%" stopColor="#3fb950" />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
              <XAxis dataKey="name" tick={{ fill: '#3d4f6e', fontSize: 9 }} />
              <YAxis tick={{ fill: '#3d4f6e', fontSize: 9 }} />
              <Tooltip content={<TooltipBox />} />
              <Area type="monotone" dataKey="value" name={chartYAxis} stroke="url(#areaStroke)" fill="url(#areaGrad)" strokeWidth={3} />
            </AreaChart>
          </ResponsiveContainer>
        );
      default:
        return null;
    }
  };

  if (emptyStateMessage) {
    return (
      <EmployeeLayout>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: 'calc(100vh - 80px)', padding: '24px' }}>
          <div className="glass-panel" style={{ padding: '2.5rem', textAlign: 'center', maxWidth: 560, width: '100%' }}>
            <div style={{ width: 60, height: 60, borderRadius: 18, background: 'rgba(88,166,255,0.12)', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 1rem' }}>
              <Database size={28} color="var(--primary)" />
            </div>
            <h2 style={{ color: '#fff', marginBottom: '0.75rem' }}>You need data first</h2>
            <p style={{ color: 'var(--text-muted)', marginBottom: '1.5rem', lineHeight: 1.6 }}>{emptyStateMessage}</p>
            <div style={{ display: 'flex', gap: 10, justifyContent: 'center', flexWrap: 'wrap' }}>
              <button className="emp-btn emp-btn-primary" onClick={() => navigate('/employee/datasets')}>
                Back to Datasets
              </button>
              <button className="emp-btn emp-btn-ghost" onClick={() => navigate('/employee/cleaning')}>
                Open Cleaning
              </button>
            </div>
          </div>
        </div>
      </EmployeeLayout>
    );
  }

  if (loading) {
    return (
      <EmployeeLayout>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: 'calc(100vh - 80px)' }}>
          <div style={{ textAlign: 'center' }}>
            <Loader size={40} color="var(--primary)" style={{ animation: 'spin 1s linear infinite' }} />
            <p style={{ marginTop: '1rem', color: 'var(--text-muted)' }}>Loading visualization data...</p>
          </div>
        </div>
      </EmployeeLayout>
    );
  }

  if (error) {
    return (
      <EmployeeLayout>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: 'calc(100vh - 80px)' }}>
          <div className="glass-panel" style={{ padding: '3rem', textAlign: 'center', maxWidth: 500 }}>
            <h2 style={{ color: 'var(--warning)', marginBottom: '1rem' }}>Data Not Available</h2>
            <p style={{ color: 'var(--text-muted)', marginBottom: '2rem' }}>{error}</p>
            <button className="emp-btn emp-btn-primary" onClick={() => navigate('/employee/datasets')}>Back to Datasets</button>
          </div>
        </div>
      </EmployeeLayout>
    );
  }

  return (
    <EmployeeLayout>
      <div className="emp-topbar" style={{ padding: '8px 16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <button className="emp-btn emp-btn-ghost emp-btn-sm" onClick={() => navigate('/employee/datasets')}>
            <ArrowLeft size={14} /> Back
          </button>
          {availableDatasets.length > 1 && (
            <select className="emp-filter-select" value={selectedDataset?.dataset_id || selectedDataset?.id || ''}
              onChange={(e) => {
                const ds = availableDatasets.find(d => (d.dataset_id || d.id) === e.target.value);
                if (ds) {
                  isInitialized.current = false;
                  setSelectedDataset(ds);
                  setData(null);
                  setDashboardConfig(null);
                  setChartXAxis('');
                  setChartYAxis('');
                  navigate(`/employee/visualization?ds=${ds.dataset_id || ds.id}&name=${encodeURIComponent(ds.name || '')}`);
                }
              }} style={{ minWidth: 180, fontSize: 11 }}>
              {availableDatasets.map(ds => <option key={ds.dataset_id || ds.id} value={ds.dataset_id || ds.id}>{ds.name}</option>)}
            </select>
          )}
          <div>
            <div className="emp-topbar-title">Data Visualization</div>
            <div className="emp-topbar-sub">
              {datasetName} · {data?.totalRows?.toLocaleString() || '0'} rows · {data?.headers?.length || 0} columns · Cleaned
            </div>
          </div>
        </div>
        <div className="emp-topbar-actions">
          <button className="emp-btn emp-btn-ghost emp-btn-sm" onClick={() => loadData(appliedFilters, search, page)}>
            <RefreshCw size={12} /> Refresh
          </button>
        </div>
      </div>

      <div style={{ display: 'flex', height: 'calc(100vh - 60px)', overflow: 'hidden' }}>
        {/* Filters Sidebar */}
        <div style={{
          width: 220, flexShrink: 0, overflowY: 'auto',
          borderRight: '1px solid var(--border-color)', background: 'rgba(22,27,34,0.5)',
        }}>
          <div style={{
            padding: '14px 16px', borderBottom: '1px solid var(--border-color)',
            display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          }}>
            <div style={{ fontSize: 13, fontWeight: 600, color: '#fff', display: 'flex', alignItems: 'center', gap: 6 }}>
              <Filter size={14} /> Filters
            </div>
            {appliedFilterCount > 0 && (
              <button className="emp-btn emp-btn-ghost emp-btn-sm" onClick={clearFilters} style={{ fontSize: 9, padding: '2px 8px' }}>
                Clear
              </button>
            )}
          </div>

          <div style={{ padding: '10px 12px' }}>
            <div className="emp-search-bar" style={{ width: '100%' }}>
              <Search size={12} />
              <input type="text" placeholder="Search..." value={search}
                onChange={e => setSearch(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && loadData(appliedFilters, search, 1)}
                style={{ width: '100%', fontSize: 11, background: 'transparent', border: 'none', outline: 'none', color: '#fff' }}
              />
            </div>
          </div>

          {headers.map(col => {
            const type = data?.columnTypes?.[col];
            const stats = data?.columnStats?.[col];
            const isExpanded = expandedFilter === col;
            const filterVal = filters[col];
            const isNum = type === 'numeric';

            return (
              <div key={col} style={{ borderTop: '1px solid rgba(255,255,255,0.03)' }}>
                <div onClick={() => setExpandedFilter(isExpanded ? null : col)} style={{
                  padding: '8px 12px', cursor: 'pointer', display: 'flex',
                  alignItems: 'center', justifyContent: 'space-between',
                  background: isExpanded ? 'rgba(88,166,255,0.05)' : 'transparent',
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6, overflow: 'hidden' }}>
                    <span style={{
                      fontFamily: "'DM Mono', monospace", fontSize: 8, padding: '1px 4px', borderRadius: 3,
                      background: isNum ? 'rgba(63,185,80,0.1)' : 'rgba(188,140,255,0.1)',
                      color: isNum ? '#3fb950' : '#bc8cff', flexShrink: 0,
                    }}>{isNum ? 'NUM' : 'CAT'}</span>
                    <span style={{ fontSize: 11, color: 'var(--text-main)', fontWeight: 500, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{col}</span>
                  </div>
                  {isExpanded ? <ChevronUp size={12} color="var(--text-muted)" /> : <ChevronDown size={12} color="var(--text-muted)" />}
                </div>

                {isExpanded && (
                  <div style={{ padding: '4px 12px 12px' }}>
                    {isNum ? (
                      <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                        <input type="number" placeholder={stats?.min?.toFixed(0)}
                          value={filterVal?.min ?? ''}
                          onChange={e => setNumericFilter(col, e.target.value, filterVal?.max ?? '')}
                          style={numInputStyle} />
                        <span style={{ color: 'var(--text-muted)', fontSize: 10 }}>to</span>
                        <input type="number" placeholder={stats?.max?.toFixed(0)}
                          value={filterVal?.max ?? ''}
                          onChange={e => setNumericFilter(col, filterVal?.min ?? '', e.target.value)}
                          style={numInputStyle} />
                      </div>
                    ) : (
                      <div style={{ maxHeight: 150, overflow: 'auto' }}>
                        {stats?.values?.slice(0, 25).map(val => {
                          const isSelected = Array.isArray(filterVal) && filterVal.includes(val);
                          return (
                            <label key={val} style={{
                              display: 'flex', alignItems: 'center', gap: 6, padding: '2px 0',
                              cursor: 'pointer', fontSize: 11,
                              color: isSelected ? 'var(--primary)' : 'var(--text-muted)',
                            }}>
                              <input type="checkbox" checked={isSelected}
                                onChange={() => toggleFilterValue(col, val)}
                                style={{ accentColor: 'var(--primary)' }} />
                              <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                {String(val).substring(0, 20)}
                              </span>
                            </label>
                          );
                        })}
                        {stats?.uniqueCount > 25 && (
                          <div style={{ fontFamily: "'DM Mono', monospace", fontSize: 9, color: 'var(--text-muted)', marginTop: 4 }}>
                            + {stats.uniqueCount - 25} more
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                )}
              </div>
            );
          })}

          <div style={{ padding: '12px', borderTop: '1px solid var(--border-color)' }}>
            <button className="emp-btn emp-btn-primary emp-btn-sm" onClick={applyFilters}
              style={{ width: '100%', justifyContent: 'center' }}>
              Apply Filters {appliedFilterCount > 0 && `(${appliedFilterCount})`}
            </button>
          </div>
        </div>

        {/* Main Content Area */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '12px 14px' }}>
          {/* Chart Controls */}
          <div className="glass-panel" style={{ padding: '8px 12px', marginBottom: 10, display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span style={{ fontFamily: "'DM Mono', monospace", fontSize: 9, color: 'var(--text-muted)' }}>TYPE</span>
              <div style={{ display: 'flex', gap: 3 }}>
                {[
                  { type: 'bar', icon: BarChart3 },
                  { type: 'line', icon: TrendingUp },
                  { type: 'pie', icon: PieIcon },
                  { type: 'area', icon: Activity },
                ].map(({ type, icon: Icon }) => (
                  <button key={type} className="emp-btn emp-btn-sm" onClick={() => setChartType(type)} style={{
                    background: chartType === type ? 'rgba(88,166,255,0.15)' : 'transparent',
                    color: chartType === type ? 'var(--primary)' : 'var(--text-muted)',
                    border: `1px solid ${chartType === type ? 'var(--primary)' : 'var(--border-color)'}`,
                    padding: '3px 8px',
                  }}><Icon size={11} /></button>
                ))}
              </div>
            </div>
            <div style={{ width: 1, height: 16, background: 'var(--border-color)' }} />
            <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <span style={{ fontFamily: "'DM Mono', monospace", fontSize: 9, color: 'var(--text-muted)' }}>X</span>
              <select 
                className="admin-filter-select" 
                value={chartXAxis} 
                onChange={(e) => setChartXAxis(e.target.value)} 
                style={{ fontSize: 9 }}
              >
                <option value="">Select</option>
                {headers.map(h => (
                  <option key={h} value={h}>{h}</option>
                ))}
              </select>
            </div>
            <div style={{ width: 1, height: 16, background: 'var(--border-color)' }} />
            <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <span style={{ fontFamily: "'DM Mono', monospace", fontSize: 9, color: 'var(--text-muted)' }}>Y</span>
              <select 
                className="admin-filter-select" 
                value={chartYAxis} 
                onChange={(e) => setChartYAxis(e.target.value)} 
                style={{ fontSize: 9 }}
              >
                <option value="">Select</option>
                {headers.map(h => (
                  <option key={h} value={h}>{h}</option>
                ))}
              </select>
            </div>
            <div style={{ width: 1, height: 16, background: 'var(--border-color)' }} />
            <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <span style={{ fontFamily: "'DM Mono', monospace", fontSize: 9, color: 'var(--text-muted)' }}>AGG</span>
              <select 
                className="admin-filter-select" 
                value={aggregation} 
                onChange={(e) => setAggregation(e.target.value)} 
                style={{ fontSize: 9, minWidth: 70 }}
              >
                <option value="sum">Sum</option>
                <option value="count">Count</option>
                <option value="avg">Avg</option>
                <option value="max">Max</option>
                <option value="min">Min</option>
              </select>
            </div>
            {chartStats && (
              <>
                <div style={{ width: 1, height: 16, background: 'var(--border-color)' }} />
                <div style={{ display: 'flex', alignItems: 'center', gap: 12, fontFamily: "'DM Mono', monospace", fontSize: 9 }}>
                  <span style={{ color: 'var(--primary)' }}>Σ: <strong>{chartStats.totalSum?.toLocaleString(undefined, { maximumFractionDigits: 2 })}</strong></span>
                  <span style={{ color: 'var(--accent)' }}>Cnt: <strong>{chartStats.totalCount?.toLocaleString()}</strong></span>
                  <span style={{ color: 'var(--success)' }}>Avg: <strong>{chartStats.avg?.toLocaleString(undefined, { maximumFractionDigits: 2 })}</strong></span>
                </div>
              </>
            )}
            <div style={{ marginLeft: 'auto', fontFamily: "'DM Mono', monospace", fontSize: 9, color: 'var(--text-muted)' }}>
              {data?.totalRows?.toLocaleString() || '0'} rows · {appliedFilterCount} filter{appliedFilterCount !== 1 ? 's' : ''}
            </div>
          </div>

          {/* Main Chart + Pie Side by Side */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10, marginBottom: 10 }}>
            <div className="glass-panel" style={{ padding: '12px 14px' }}>
              <div style={{ marginBottom: 6 }}>
                <div style={{ fontSize: 12, fontWeight: 600, color: '#fff' }}>
                  {chartYAxis || 'Value'} by {chartXAxis || 'Category'}
                </div>
                <div style={{ fontFamily: "'DM Mono', monospace", fontSize: 8, color: 'var(--text-muted)' }}>
                  {chartType.charAt(0).toUpperCase() + chartType.slice(1)} · Top 10 groups
                </div>
              </div>
              <div style={{ height: 190 }}>{renderMainChart()}</div>
            </div>

            {/* Companion Donut Chart - hover-only details, no clutter labels */}
            {data && (() => {
              const catCol = data.headers?.find(h =>
                data.columnTypes?.[h] === 'categorical' &&
                data.columnStats?.[h]?.uniqueCount >= 2 &&
                data.columnStats?.[h]?.uniqueCount <= 12
              );
              const numCol = data.headers?.find(h => data.columnTypes?.[h] === 'numeric');
              if (!catCol || !numCol) return null;
              const grouped = {};
              data.rows.forEach(row => {
                const key = row[catCol] || 'Unknown';
                const val = parseFloat(row[numCol]);
                if (!isNaN(val)) grouped[key] = (grouped[key] || 0) + val;
              });
              const total = Object.values(grouped).reduce((a, b) => a + b, 0);
              const pieData = Object.entries(grouped)
                .map(([name, value]) => ({ 
                  name, 
                  value: Math.round(value * 100) / 100,
                  total, // passed into tooltip for % calculation
                  fill: COLORS[Object.keys(grouped).indexOf(name) % COLORS.length]
                }))
                .filter(d => d.value > 0)
                .sort((a, b) => b.value - a.value)
                .slice(0, 8);
              if (pieData.length < 2) return null;
              return (
                <div className="glass-panel" style={{ padding: '12px 14px' }}>
                  <div style={{ marginBottom: 8 }}>
                    <div style={{ fontSize: 12, fontWeight: 600, color: '#fff' }}>{numCol} by {catCol}</div>
                    <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 8, color: 'var(--text-muted)', marginTop: 2 }}>
                      Hover a slice to see details
                    </div>
                  </div>
                  <div style={{ height: 190, position: 'relative' }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <defs>
                          {COLORS.map((color, i) => (
                            <radialGradient key={i} id={`rg${i}`} cx="50%" cy="50%" r="50%">
                              <stop offset="0%" stopColor={color} stopOpacity={1} />
                              <stop offset="100%" stopColor={color} stopOpacity={0.65} />
                            </radialGradient>
                          ))}
                        </defs>
                        <Pie
                          data={pieData}
                          dataKey="value"
                          nameKey="name"
                          innerRadius={45}
                          outerRadius={80}
                          paddingAngle={2}
                          stroke="none"
                        >
                          {pieData.map((entry, i) => (
                            <Cell 
                              key={i} 
                              fill={`url(#rg${i % COLORS.length})`}
                              style={{ cursor: 'pointer', outline: 'none' }}
                            />
                          ))}
                        </Pie>
                        <Tooltip 
                          content={<PieTooltip />}
                          wrapperStyle={{ zIndex: 100 }}
                        />
                        <Legend
                          iconType="circle"
                          iconSize={7}
                          formatter={v => <span style={{ color: '#6e7681', fontSize: 9, fontFamily: "'IBM Plex Mono',monospace" }}>{String(v).substring(0, 16)}</span>}
                        />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              );
            })()}
          </div>

          {/* Pipeline Charts Row */}
          {dashboardConfig?.charts?.length > 0 && (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 10, marginBottom: 10 }}>
              {dashboardConfig.charts.slice(0, 4).map(chart => {
                const cData = chart.data || [];
                return (
                  <div key={chart.id} className="glass-panel" style={{ padding: '10px 12px' }}>
                    <div style={{ marginBottom: 4 }}>
                      <div style={{ fontSize: 11, fontWeight: 600, color: '#fff', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{chart.title}</div>
                      <div style={{ fontFamily: "'DM Mono', monospace", fontSize: 8, color: 'var(--text-muted)' }}>
                        {chart.x} vs {chart.y}
                      </div>
                    </div>
                    <div style={{ height: 140 }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={cData}>
                          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                          <XAxis dataKey={chart.x} tick={{ fill: '#3d4f6e', fontSize: 7 }} />
                          <YAxis tick={{ fill: '#3d4f6e', fontSize: 7 }} width={30} />
                          <Tooltip content={<TooltipBox />} />
                          <Bar dataKey={chart.y} name={chart.y} radius={[2, 2, 0, 0]}>
                            {cData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} opacity={0.85} />)}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                );
              })}
            </div>
          )}

          {/* Data Table - 50 rows per page with proper 1-50 of N pagination */}
          {data?.rows?.length > 0 && (() => {
            const tableRows = data.rows;
            const tableTotalRows = data.totalRows;
            const totalTablePages = Math.ceil(tableRows.length / TABLE_PAGE_SIZE);
            const tableStart = (tablePage - 1) * TABLE_PAGE_SIZE;
            const tableEnd = Math.min(tablePage * TABLE_PAGE_SIZE, tableRows.length);
            const visibleRows = tableRows.slice(tableStart, tableEnd);
            // Global row offset: page is the backend page (each page = 500 rows)
            const globalOffset = (page - 1) * 500;
            return (
              <div className="glass-panel" style={{ overflow: 'hidden' }}>
                <div style={{
                  padding: '12px 16px', borderBottom: '1px solid var(--border-color)',
                  display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                }}>
                  <div style={{ fontSize: 13, fontWeight: 600, color: '#fff' }}>Cleaned Data</div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                    <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 10, color: 'var(--text-muted)', background: 'rgba(255,255,255,0.05)', padding: '3px 10px', borderRadius: 20 }}>
                      {globalOffset + tableStart + 1}–{globalOffset + tableEnd} of {tableTotalRows?.toLocaleString()}
                    </div>
                    <div style={{ display: 'flex', gap: 4 }}>
                      <button
                        onClick={() => {
                          if (tablePage > 1) { setTablePage(p => p - 1); }
                          else if (page > 1) { setPage(p => p - 1); setTablePage(totalTablePages); loadData(appliedFilters, search, page - 1); }
                        }}
                        disabled={tablePage <= 1 && page <= 1}
                        style={{ background: 'none', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 6, color: (tablePage <= 1 && page <= 1) ? '#444' : '#aaa', cursor: 'pointer', padding: '2px 8px', fontSize: 13 }}
                      >←</button>
                      <button
                        onClick={() => {
                          if (tablePage < totalTablePages) { setTablePage(p => p + 1); }
                          else if (tableEnd + globalOffset < tableTotalRows) { setPage(p => p + 1); setTablePage(1); loadData(appliedFilters, search, page + 1); }
                        }}
                        disabled={tableEnd + globalOffset >= tableTotalRows}
                        style={{ background: 'none', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 6, color: tableEnd + globalOffset >= tableTotalRows ? '#444' : '#aaa', cursor: 'pointer', padding: '2px 8px', fontSize: 13 }}
                      >→</button>
                    </div>
                  </div>
                </div>
                <div style={{ overflowX: 'auto', maxHeight: 320, overflowY: 'auto' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontFamily: "'IBM Plex Mono', monospace", fontSize: 10 }}>
                    <thead>
                      <tr>
                        <th style={thStyle}>#</th>
                        {headers.map(h => (<th key={h} style={thStyle}>{h}</th>))}
                      </tr>
                    </thead>
                    <tbody>
                      {visibleRows.map((row, ri) => (
                        <tr key={ri}
                          onMouseEnter={e => e.currentTarget.style.background = 'rgba(255,255,255,0.03)'}
                          onMouseLeave={e => e.currentTarget.style.background = ''}>
                          <td style={tdStyle}>{globalOffset + tableStart + ri + 1}</td>
                          {headers.map(h => (
                            <td key={h} style={{
                              ...tdStyle,
                              color: data.columnTypes?.[h] === 'numeric' ? '#3fb950' : '#8b949e',
                              whiteSpace: 'nowrap', maxWidth: 140, overflow: 'hidden', textOverflow: 'ellipsis',
                            }} title={row[h]}>
                              {row[h] || <span style={{ color: '#f85149', fontStyle: 'italic' }}>—</span>}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            );
          })()}
        </div>
      </div>

      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </EmployeeLayout>
  );
};

const thStyle = {
  background: 'rgba(13,17,23,0.95)', padding: '8px 10px', textAlign: 'left',
  fontFamily: "'DM Mono', monospace", fontSize: 9, color: '#6b7280',
  letterSpacing: 1, textTransform: 'uppercase', borderBottom: '1px solid var(--border-color)',
  position: 'sticky', top: 0, whiteSpace: 'nowrap', zIndex: 1,
};

const tdStyle = {
  padding: '6px 10px', borderBottom: '1px solid rgba(255,255,255,0.025)',
};

const numInputStyle = {
  width: '50%', padding: '4px 6px', borderRadius: 6,
  border: '1px solid var(--border-color)', background: 'rgba(255,255,255,0.05)',
  color: '#fff', fontSize: 10, fontFamily: "'DM Mono', monospace", outline: 'none',
};

export default VisualizationPage;
