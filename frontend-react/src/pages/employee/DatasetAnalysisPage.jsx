import { useState, useEffect, useMemo } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { ArrowRight, Filter, BarChart3, AlertTriangle, CheckCircle2, Eye, ChevronDown, Search, Sparkles, Upload, FileSpreadsheet, AlertCircle, Database, TrendingUp, Activity, PieChart as PieChartIcon, RefreshCw, Download, FileText, Info, Clock, HardDrive, Layers, Zap } from 'lucide-react';
import { PieChart, Pie, Cell, BarChart as RechartsBarChart, LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, Legend, Bar } from 'recharts';
import axios from 'axios';
import EmployeeLayout from '../../layout/EmployeeLayout';
import { getDatasets, downloadDataset } from '../../services/api';

const CHART_COLORS = ['#58a6ff', '#3fb950', '#bc8cff', '#d29922', '#f85149', '#79c0ff', '#d2a8ff', '#ffa657'];

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

const typeColors = {
  string: { bg: 'rgba(188,140,255,0.1)', color: '#bc8cff' },
  float64: { bg: 'rgba(63,185,80,0.1)', color: '#3fb950' },
  int64: { bg: 'rgba(63,185,80,0.1)', color: '#3fb950' },
  datetime: { bg: 'rgba(210,153,34,0.1)', color: '#d29922' },
};

const inferredColors = {
  numeric: { bg: 'rgba(63,185,80,0.08)', color: '#3fb950', label: 'NUM' },
  categorical: { bg: 'rgba(188,140,255,0.08)', color: '#bc8cff', label: 'CAT' },
  datetime: { bg: 'rgba(210,153,34,0.08)', color: '#d29922', label: 'DATE' },
  text: { bg: 'rgba(88,166,255,0.08)', color: '#58a6ff', label: 'TEXT' },
  identifier: { bg: 'rgba(139,148,158,0.08)', color: '#8b949e', label: 'ID' },
};

const DatasetAnalysisPage = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const datasetId = searchParams.get('ds') || null;
  const datasetName = searchParams.get('name') || '';

  // State
  const [columns, setColumns] = useState([]);
  const [cleanedData, setCleanedData] = useState([]);
  const [search, setSearch] = useState('');
  const [typeFilter, setTypeFilter] = useState('all');
  const [nullFilter, setNullFilter] = useState('all');
  const [viewMode, setViewMode] = useState('analysis'); // analysis | data | dashboard | summary | original
  const [showPreview, setShowPreview] = useState(false);
  const [summaryDataMode, setSummaryDataMode] = useState('cleaned'); // cleaned | original
  const [loading, setLoading] = useState(true);
  const [analysisLoading, setAnalysisLoading] = useState(false);
  const [availableDatasets, setAvailableDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState(null);
  const [analysisData, setAnalysisData] = useState(null);
  const [dashboardData, setDashboardData] = useState(null);
  const [analysisError, setAnalysisError] = useState(null);
  const [dataPage, setDataPage] = useState(1);
  const [dataTotalRows, setDataTotalRows] = useState(0);
  const [loadingAllData, setLoadingAllData] = useState(false);
  const [originalData, setOriginalData] = useState([]);
  const [originalTotalRows, setOriginalTotalRows] = useState(0);
  const [loadingOriginal, setLoadingOriginal] = useState(false);
  const [originalAnalysis, setOriginalAnalysis] = useState(null);
  const [vizData, setVizData] = useState(null);
  const [vizLoading, setVizLoading] = useState(false);

  const API_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5000/api';

  const getAuthHeaders = () => {
    const token = sessionStorage.getItem('token');
    return token ? { Authorization: `Bearer ${token}` } : {};
  };

  // Load available datasets
  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      try {
        const res = await getDatasets();
        if (res.success && res.data && res.data.length > 0) {
          const readyDatasets = res.data.filter(d => 
            d.status === 'completed' || d.status === 'ready' || d.status === 'cleaned'
          );
          setAvailableDatasets(readyDatasets);

          if (!datasetId && readyDatasets.length > 0) {
            const firstReady = readyDatasets[0];
            setSelectedDataset(firstReady);
            navigate(`/employee/analysis?ds=${firstReady.dataset_id || firstReady.id}&name=${encodeURIComponent(firstReady.name || '')}`, { replace: true });
          } else if (datasetId) {
            const selected = readyDatasets.find(d => (d.dataset_id || d.id) === datasetId);
            if (selected) setSelectedDataset(selected);
          }
        }
      } catch (err) {
        console.warn('Could not load datasets:', err.message);
      }
      setLoading(false);
    };

    loadData();
  }, [datasetId]);

  // Load analysis data when dataset is selected
  useEffect(() => {
    const loadAnalysis = async () => {
      if (!selectedDataset) return;
      
      const dsId = selectedDataset.dataset_id || selectedDataset.id;
      if (!dsId) return;

      setAnalysisLoading(true);
      setAnalysisError(null);
      setCleanedData([]);
      
      try {
        // Load analysis
        const response = await axios.get(`${API_URL}/datasets/${dsId}/analysis`, { headers: getAuthHeaders() });
        if (response.data && response.data.columns) {
          const cols = response.data.columns.map((col, idx) => ({
            name: col.name,
            type: col.dtype || 'string',
            nulls: col.null_count || 0,
            nullPct: col.null_pct || 0,
            unique: col.nunique || 0,
            sample: col.sample?.[0] || '-',
            inferred: col.inferred_type || 'text',
            selected: (col.null_count || 0) > 0 || col.inferred_type === 'numeric',
          }));
          setColumns(cols);
          setAnalysisData(response.data);
        } else {
          setAnalysisError('No analysis data available');
        }

        // Load dashboard config
        try {
          const dashRes = await axios.get(`${API_URL}/dashboard/${dsId}`, { headers: getAuthHeaders() });
          if (dashRes.data && dashRes.data.charts) {
            setDashboardData(dashRes.data);
          }
        } catch (dashErr) {
          console.warn('Dashboard not available:', dashErr.message);
        }

        // Load cleaned data
        try {
          const dataRes = await axios.get(`${API_URL}/cleaned-data/${dsId}?limit=100&page=1`, { headers: getAuthHeaders() });
          if (dataRes.data && dataRes.data.rows) {
            setCleanedData(dataRes.data.rows);
            setDataTotalRows(dataRes.data.totalRows || 0);
          }
        } catch (dataErr) {
          console.warn('Cleaned data not available:', dataErr.message);
        }

        // Load original dataset stats for summary view
        try {
          const origRes = await axios.get(`${API_URL}/original-data/${dsId}?limit=1&page=1`, { headers: getAuthHeaders() });
          if (origRes.data && origRes.data.headers) {
            setOriginalAnalysis({
              headers: origRes.data.headers,
              totalRows: origRes.data.totalRows,
              columnCount: origRes.data.headers.length,
            });
          }
        } catch (origErr) {
          console.warn('Original data not available:', origErr.message);
        }

      } catch (err) {
        console.warn('Could not load analysis:', err.message);
        setAnalysisError(err.response?.data?.message || 'Failed to load analysis');
        setColumns([]);
      }
      setAnalysisLoading(false);
    };

    loadAnalysis();
  }, [selectedDataset, API_URL]);

  // Load visualization data when dashboard view is selected
  useEffect(() => {
    const loadVizData = async () => {
      if (viewMode !== 'dashboard' || !selectedDataset) return;
      
      const dsId = selectedDataset.dataset_id || selectedDataset.id;
      if (!dsId) return;

      setVizLoading(true);
      try {
        // First try to use existing cleaned data if available
        let rowsToUse = cleanedData;
        let headersToUse = [];
        let columnTypesToUse = {};
        let totalRows = dataTotalRows;
        
        if (!rowsToUse || rowsToUse.length === 0) {
          // Fetch cleaned data if not already loaded
          const dataRes = await axios.get(`${API_URL}/cleaned-data/${dsId}?limit=500&page=1`, { headers: getAuthHeaders() });
          if (dataRes.data && dataRes.data.rows) {
            rowsToUse = dataRes.data.rows;
            headersToUse = dataRes.data.headers || [];
            columnTypesToUse = dataRes.data.columnTypes || {};
            totalRows = dataRes.data.totalRows || 0;
            
            // Store in cleanedData state
            setCleanedData(rowsToUse);
            setDataTotalRows(totalRows);
          }
        } else {
          // Use existing cleaned data
          headersToUse = Object.keys(rowsToUse[0] || {});
          // Detect column types from sample data
          if (rowsToUse.length > 0) {
            headersToUse.forEach(col => {
              const val = rowsToUse[0][col];
              if (!isNaN(parseFloat(val)) && isFinite(val)) {
                columnTypesToUse[col] = 'numeric';
              } else {
                columnTypesToUse[col] = 'categorical';
              }
            });
          }
          totalRows = dataTotalRows;
        }

        if (!rowsToUse || rowsToUse.length === 0) {
          setVizData(null);
          setVizLoading(false);
          return;
        }

        const numericCols = [];
        const categoricalCols = [];
        
        headersToUse.forEach(col => {
          if (col === 'Unnamed: 0.1' || col === 'Unnamed: 0') return;
          const type = columnTypesToUse[col];
          if (type === 'numeric') numericCols.push(col);
          else categoricalCols.push(col);
        });

        // Auto-select first categorical for X and first numeric for Y
        const defaultX = categoricalCols[0] || headersToUse[0];
        const defaultY = numericCols[0] || headersToUse.find(h => {
          const val = rowsToUse[0]?.[h];
          return val !== undefined && !isNaN(parseFloat(val));
        });

        // Generate chart data
        const generateChartData = () => {
          if (!defaultX || !defaultY) return [];
          const grouped = {};
          rowsToUse.forEach(row => {
            const key = row[defaultX] || 'Unknown';
            const val = parseFloat(row[defaultY]);
            if (!isNaN(val)) grouped[key] = (grouped[key] || 0) + val;
          });
          return Object.entries(grouped)
            .map(([name, value]) => ({ name: String(name).substring(0, 18), value: Math.round(value * 100) / 100 }))
            .sort((a, b) => b.value - a.value).slice(0, 10);
        };

        const generateCountData = () => {
          if (!defaultX) return [];
          const counts = {};
          rowsToUse.forEach(row => {
            const key = row[defaultX] || 'Unknown';
            counts[key] = (counts[key] || 0) + 1;
          });
          return Object.entries(counts)
            .map(([name, count]) => ({ name: String(name).substring(0, 18), count }))
            .sort((a, b) => b.count - a.count).slice(0, 10);
        };

        setVizData({
          headers: headersToUse,
          rows: rowsToUse,
          totalRows: totalRows,
          columnTypes: columnTypesToUse,
          numericColumns: numericCols,
          categoricalColumns: categoricalCols,
          chartX: defaultX,
          chartY: defaultY,
          chartType: 'bar',
          chartData: generateChartData(),
          countData: generateCountData(),
        });
      } catch (err) {
        console.warn('Could not load visualization data:', err.message);
        setVizData(null);
      }
      setVizLoading(false);
    };

    loadVizData();
  }, [viewMode, selectedDataset, API_URL, cleanedData, dataTotalRows]);

  // Generate chart data when vizData selections change
  const chartInfo = useMemo(() => {
    if (!vizData?.rows || !vizData.chartX) return null;

    // Sum/aggregate data for selected columns
    const grouped = {};
    vizData.rows.forEach(row => {
      const key = row[vizData.chartX] || 'Unknown';
      if (vizData.chartY) {
        const val = parseFloat(row[vizData.chartY]);
        if (!isNaN(val)) grouped[key] = (grouped[key] || 0) + val;
      } else {
        grouped[key] = (grouped[key] || 0) + 1;
      }
    });
    const chartData = Object.entries(grouped)
      .map(([name, value]) => ({ 
        name: String(name).substring(0, 18), 
        value: vizData.chartY ? Math.round(value * 100) / 100 : value 
      }))
      .sort((a, b) => b.value - a.value).slice(0, 10);

    // Count data
    const countGrouped = {};
    vizData.rows.forEach(row => {
      const key = row[vizData.chartX] || 'Unknown';
      countGrouped[key] = (countGrouped[key] || 0) + 1;
    });
    const countData = Object.entries(countGrouped)
      .map(([name, count]) => ({ name: String(name).substring(0, 18), count }))
      .sort((a, b) => b.count - a.count).slice(0, 10);

    return { chartData, countData };
  }, [vizData]);

  // Update vizData when chart selections change
  useEffect(() => {
    if (!vizData || !chartInfo) return;
    setVizData(prev => prev ? { ...prev, chartData: chartInfo.chartData, countData: chartInfo.countData } : null);
  }, [chartInfo]);

  // Load more cleaned data when page changes
  useEffect(() => {
    const loadMoreData = async () => {
      if (!selectedDataset || dataPage === 1) return;
      
      const dsId = selectedDataset.dataset_id || selectedDataset.id;
      if (!dsId) return;

      try {
        const dataRes = await axios.get(`${API_URL}/cleaned-data/${dsId}?limit=100&page=${dataPage}`, { headers: getAuthHeaders() });
        if (dataRes.data && dataRes.data.rows) {
          setCleanedData(prev => [...prev, ...dataRes.data.rows]);
        }
      } catch (err) {
        console.warn('Could not load more data:', err.message);
      }
    };

    loadMoreData();
  }, [dataPage, selectedDataset]);

  // Load entire cleaned dataset
  const loadAllData = async () => {
    if (!selectedDataset) return;
    
    const dsId = selectedDataset.dataset_id || selectedDataset.id;
    if (!dsId) return;

    setLoadingAllData(true);
    try {
      const dataRes = await axios.get(`${API_URL}/cleaned-data/${dsId}?limit=1000000&page=1`, { headers: getAuthHeaders() });
      if (dataRes.data && dataRes.data.rows) {
        setCleanedData(dataRes.data.rows);
        setDataTotalRows(dataRes.data.totalRows || dataRes.data.rows.length);
      }
    } catch (err) {
      console.warn('Could not load all data:', err.message);
    }
    setLoadingAllData(false);
  };

  // Reset to paginated view
  const resetDataView = async () => {
    if (!selectedDataset) return;
    
    const dsId = selectedDataset.dataset_id || selectedDataset.id;
    if (!dsId) return;

    try {
      const dataRes = await axios.get(`${API_URL}/cleaned-data/${dsId}?limit=100&page=1`, { headers: getAuthHeaders() });
      if (dataRes.data && dataRes.data.rows) {
        setCleanedData(dataRes.data.rows);
        setDataTotalRows(dataRes.data.totalRows || 0);
        setDataPage(1);
      }
    } catch (err) {
      console.warn('Could not reset data:', err.message);
    }
  };

  // Load original dataset
  const loadOriginalData = async (page = 1) => {
    if (!selectedDataset) return;
    
    const dsId = selectedDataset.dataset_id || selectedDataset.id;
    if (!dsId) return;

    setLoadingOriginal(true);
    try {
      const dataRes = await axios.get(`${API_URL}/original-data/${dsId}?limit=100&page=${page}`, { headers: getAuthHeaders() });
      if (dataRes.data && dataRes.data.rows) {
        if (page === 1) {
          setOriginalData(dataRes.data.rows);
        } else {
          setOriginalData(prev => [...prev, ...dataRes.data.rows]);
        }
        setOriginalTotalRows(dataRes.data.totalRows || 0);
      }
    } catch (err) {
      console.warn('Could not load original data:', err.message);
    }
    setLoadingOriginal(false);
  };

  // Load entire original dataset
  const loadAllOriginalData = async () => {
    if (!selectedDataset) return;
    
    const dsId = selectedDataset.dataset_id || selectedDataset.id;
    if (!dsId) return;

    setLoadingOriginal(true);
    try {
      const dataRes = await axios.get(`${API_URL}/original-data/${dsId}?limit=1000000&page=1`, { headers: getAuthHeaders() });
      if (dataRes.data && dataRes.data.rows) {
        setOriginalData(dataRes.data.rows);
        setOriginalTotalRows(dataRes.data.totalRows || dataRes.data.rows.length);
      }
    } catch (err) {
      console.warn('Could not load all original data:', err.message);
    }
    setLoadingOriginal(false);
  };

  // Download original data as CSV
  const downloadOriginalData = () => {
    if (originalData.length === 0) return;
    
    const headers = Object.keys(originalData[0]);
    const csvContent = [
      headers.join(','),
      ...originalData.map(row => 
        headers.map(h => {
          const val = row[h];
          if (val === null || val === undefined) return '';
          const str = String(val);
          return str.includes(',') || str.includes('"') || str.includes('\n') 
            ? `"${str.replace(/"/g, '""')}"` 
            : str;
        }).join(',')
      )
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${selectedDataset?.name || 'original_data'}_original.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  // Download cleaned data as CSV
  const downloadCleanedData = async () => {
    if (!selectedDataset) return;

    const dsId = selectedDataset.dataset_id || selectedDataset.id;
    if (!dsId) return;

    try {
      await downloadDataset(dsId, `${selectedDataset?.name || 'cleaned_data'}_cleaned.csv`);
    } catch (err) {
      console.warn('Could not download cleaned data:', err.message);
    }
  };

  // If no datasets available
  if (!loading && availableDatasets.length === 0) {
    return (
      <EmployeeLayout>
        <div className="emp-topbar">
          <div>
            <div className="emp-topbar-title">Dataset Analysis</div>
            <div className="emp-topbar-sub">No processed datasets available</div>
          </div>
        </div>
        <div className="emp-content">
          <div className="glass-panel" style={{ padding: '4rem 2rem', textAlign: 'center', maxWidth: 500, margin: '2rem auto' }}>
            <Database size={64} color="var(--text-muted)" style={{ marginBottom: '1.5rem', opacity: 0.5 }} />
            <h2 style={{ color: '#fff', marginBottom: '1rem' }}>No Datasets Available</h2>
            <p style={{ color: 'var(--text-muted)', marginBottom: '2rem', fontSize: '0.95rem' }}>
              You have no datasets available for analysis.
            </p>
          </div>
        </div>
      </EmployeeLayout>
    );
  }

  // If dataset is still processing
  if (selectedDataset && (selectedDataset.status === 'processing' || selectedDataset.status === 'cleaning' || selectedDataset.status === 'not_cleaned')) {
    return (
      <EmployeeLayout>
        <div className="emp-topbar">
          <div>
            <div className="emp-topbar-title">Dataset Analysis</div>
            <div className="emp-topbar-sub">Dataset is being processed</div>
          </div>
        </div>
        <div className="emp-content">
          <div className="glass-panel" style={{ padding: '4rem 2rem', textAlign: 'center', maxWidth: 500, margin: '2rem auto' }}>
            <div className="spin" style={{ marginBottom: '1.5rem' }}>
              <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="var(--primary)" strokeWidth="2">
                <path d="M21 12a9 9 0 1 1-6.219-8.56" />
              </svg>
            </div>
            <h2 style={{ color: 'var(--primary)', marginBottom: '1rem' }}>Processing Dataset</h2>
            <p style={{ color: 'var(--text-muted)', marginBottom: '1rem' }}>{selectedDataset.name} is still being processed.</p>
            <button className="btn-primary" onClick={() => window.location.reload()} style={{ padding: '0.75rem 1.5rem' }}>Refresh</button>
          </div>
        </div>
        <style>{`@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } } .spin { animation: spin 1s linear infinite; }`}</style>
      </EmployeeLayout>
    );
  }

  const toggleColumn = (name) => setColumns(prev => prev.map(c => c.name === name ? { ...c, selected: !c.selected } : c));
  const selectAll = () => setColumns(prev => prev.map(c => ({ ...c, selected: true })));
  const selectNone = () => setColumns(prev => prev.map(c => ({ ...c, selected: false })));
  const selectNulls = () => setColumns(prev => prev.map(c => ({ ...c, selected: c.nulls > 0 })));
  const selectNumeric = () => setColumns(prev => prev.map(c => ({ ...c, selected: c.inferred === 'numeric' })));

  const filtered = columns.filter(c => {
    if (search && !c.name.toLowerCase().includes(search.toLowerCase())) return false;
    if (typeFilter !== 'all' && c.inferred !== typeFilter) return false;
    if (nullFilter === 'has-nulls' && c.nulls === 0) return false;
    if (nullFilter === 'no-nulls' && c.nulls > 0) return false;
    if (nullFilter === 'high-nulls' && c.nullPct < 5) return false;
    return true;
  });

  const selectedCount = columns.filter(c => c.selected).length;
  const selectedNulls = columns.filter(c => c.selected && c.nulls > 0).length;
  const totalNullCells = columns.filter(c => c.selected).reduce((sum, c) => sum + c.nulls, 0);

  const handleProceed = () => {
    const selectedCols = columns.filter(c => c.selected).map(c => c.name);
    const dsId = selectedDataset?.dataset_id || selectedDataset?.id || datasetId;
    const dsName = selectedDataset?.name || datasetName;
    navigate(`/employee/column-cleaning?ds=${dsId}&name=${encodeURIComponent(dsName)}&cols=${selectedCols.join(',')}`);
  };

  const NullBar = ({ pct }) => (
    <div style={{ width: 60, height: 5, background: 'rgba(255,255,255,0.06)', borderRadius: 5, overflow: 'hidden' }}>
      <div style={{ height: '100%', width: `${Math.min(pct, 100)}%`, borderRadius: 5, background: pct > 10 ? 'var(--danger)' : pct > 1 ? 'var(--warning)' : 'var(--success)' }} />
    </div>
  );

  return (
    <EmployeeLayout>
      <div className="emp-topbar">
        <div>
          <div className="emp-topbar-title">Dataset Analysis</div>
          <div className="emp-topbar-sub">{selectedDataset?.name || datasetName || 'Select a dataset'}</div>
        </div>
        <div className="emp-topbar-actions">
          {availableDatasets.length > 1 && (
            <select className="emp-filter-select" value={selectedDataset?.dataset_id || selectedDataset?.id || ''}
              onChange={(e) => {
                const ds = availableDatasets.find(d => (d.dataset_id || d.id) === e.target.value);
                if (ds) {
                  setSelectedDataset(ds);
                  setColumns([]);
                  setAnalysisData(null);
                  setDashboardData(null);
                  setCleanedData([]);
                  setDataPage(1);
                  navigate(`/employee/analysis?ds=${ds.dataset_id || ds.id}&name=${encodeURIComponent(ds.name || '')}`, { replace: true });
                }
              }} style={{ minWidth: 200 }}>
              {availableDatasets.map(ds => <option key={ds.dataset_id || ds.id} value={ds.dataset_id || ds.id}>{ds.name}</option>)}
            </select>
          )}
          {selectedDataset && (
            <button className="emp-btn emp-btn-primary" onClick={() => {
              if (selectedDataset) {
                navigate(`/employee/visualization?ds=${selectedDataset.dataset_id || selectedDataset.id}&name=${encodeURIComponent(selectedDataset.name || '')}`);
              }
            }} style={{ padding: '6px 14px', fontSize: 11 }}>
              <BarChart3 size={14} /> Visualize
            </button>
          )}
        </div>
      </div>

      <div className="emp-content">
        {/* View Mode Tabs */}
        <div style={{ display: 'flex', gap: 8, marginBottom: 20, borderBottom: '1px solid var(--border-color)', paddingBottom: 12 }}>
          <button className={`emp-btn ${viewMode === 'analysis' ? 'emp-btn-primary' : 'emp-btn-ghost'}`} onClick={() => setViewMode('analysis')}>
            <Activity size={14} /> Analysis
          </button>
          <button className={`emp-btn ${viewMode === 'data' ? 'emp-btn-primary' : 'emp-btn-ghost'}`} onClick={() => setViewMode('data')}>
            <Database size={14} /> Cleaned Data
          </button>
          <button className={`emp-btn ${viewMode === 'summary' ? 'emp-btn-primary' : 'emp-btn-ghost'}`} onClick={() => setViewMode('summary')}>
            <FileText size={14} /> Summary
          </button>
        </div>

        {/* ==================== ANALYSIS VIEW ==================== */}
        {viewMode === 'analysis' && (
          <>
            {analysisData && (
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 12, marginBottom: 20 }}>
                <div className="glass-panel" style={{ padding: '14px 16px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                    <FileSpreadsheet size={16} color="var(--primary)" />
                    <span style={{ fontSize: 12, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Dataset</span>
                  </div>
                  <div style={{ fontSize: 16, fontWeight: 600, color: '#fff' }}>{analysisData.dataset_name}</div>
                  <div style={{ fontFamily: "'DM Mono', monospace", fontSize: 10, color: 'var(--text-muted)', marginTop: 4 }}>{analysisData.row_count?.toLocaleString()} rows · {analysisData.column_count} columns</div>
                </div>
                <div className="glass-panel" style={{ padding: '14px 16px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                    <CheckCircle2 size={16} color="var(--success)" />
                    <span style={{ fontSize: 12, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Data Quality</span>
                  </div>
                  <div style={{ fontSize: 24, fontWeight: 600, color: 'var(--success)' }}>{analysisData.quality_score || 'N/A'}</div>
                  <div style={{ fontFamily: "'DM Mono', monospace", fontSize: 10, color: 'var(--text-muted)', marginTop: 4 }}>Quality Score (0-100)</div>
                </div>
                <div className="glass-panel" style={{ padding: '14px 16px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                    <AlertTriangle size={16} color="var(--warning)" />
                    <span style={{ fontSize: 12, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Missing Values</span>
                  </div>
                  <div style={{ fontSize: 24, fontWeight: 600, color: 'var(--warning)' }}>{analysisData.total_nulls?.toLocaleString() || 0}</div>
                  <div style={{ fontFamily: "'DM Mono', monospace", fontSize: 10, color: 'var(--text-muted)', marginTop: 4 }}>Total null cells</div>
                </div>
                <div className="glass-panel" style={{ padding: '14px 16px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                    <TrendingUp size={16} color="var(--accent)" />
                    <span style={{ fontSize: 12, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Columns</span>
                  </div>
                  <div style={{ fontSize: 24, fontWeight: 600, color: 'var(--accent)' }}>{columns.length}</div>
                  <div style={{ fontFamily: "'DM Mono', monospace", fontSize: 10, color: 'var(--text-muted)', marginTop: 4 }}>{columns.filter(c => c.inferred === 'numeric').length} numeric · {columns.filter(c => c.inferred === 'categorical').length} categorical</div>
                </div>
              </div>
            )}

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 12, marginBottom: 20 }}>
              {[{ val: columns.length, lbl: 'Total Columns', color: 'var(--primary)' }, { val: selectedCount, lbl: 'Selected for Cleaning', color: 'var(--accent)' }, { val: columns.filter(c => c.nulls > 0).length, lbl: 'Columns with Nulls', color: 'var(--warning)' }, { val: totalNullCells.toLocaleString(), lbl: 'Total Null Cells', color: 'var(--danger)' }].map((s, i) => (
                <div key={i} className="glass-panel" style={{ padding: '14px 16px', animation: 'adminFadeUp 0.4s ease both', animationDelay: `${i * 0.04}s` }}>
                  <div style={{ fontSize: 22, fontWeight: 600, color: s.color }}>{s.val}</div>
                  <div style={{ fontFamily: "'DM Mono', monospace", fontSize: 9, color: 'var(--text-muted)', marginTop: 2, textTransform: 'uppercase' }}>{s.lbl}</div>
                </div>
              ))}
            </div>

            {analysisLoading && <div className="glass-panel" style={{ padding: '3rem', textAlign: 'center' }}><div className="spin"><svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="var(--primary)" strokeWidth="2"><path d="M21 12a9 9 0 1 1-6.219-8.56" /></svg></div><p style={{ color: 'var(--text-muted)' }}>Loading analysis data...</p></div>}

            {analysisError && !analysisLoading && <div className="glass-panel" style={{ padding: '2rem', textAlign: 'center', background: 'rgba(248,81,73,0.05)', border: '1px solid rgba(248,81,73,0.2)' }}><AlertCircle size={40} color="var(--danger)" style={{ marginBottom: '1rem' }} /><p style={{ color: 'var(--danger)' }}>{analysisError}</p><button className="emp-btn emp-btn-ghost" onClick={() => window.location.reload()}><RefreshCw size={14} /> Retry</button></div>}

            {!analysisLoading && !analysisError && columns.length > 0 && (
              <>
                <div style={{ display: 'flex', gap: 8, marginBottom: 16, flexWrap: 'wrap', alignItems: 'center' }}>
                  <div className="emp-search-bar" style={{ flex: 1, maxWidth: 260 }}><Search size={14} /><input type="text" placeholder="Search columns…" value={search} onChange={e => setSearch(e.target.value)} /></div>
                  <select className="admin-filter-select" value={typeFilter} onChange={e => setTypeFilter(e.target.value)} style={{ fontSize: 11 }}><option value="all">All Types</option><option value="numeric">Numeric</option><option value="categorical">Categorical</option><option value="datetime">Datetime</option><option value="text">Text</option></select>
                  <select className="admin-filter-select" value={nullFilter} onChange={e => setNullFilter(e.target.value)} style={{ fontSize: 11 }}><option value="all">All Null Status</option><option value="has-nulls">Has Nulls</option><option value="no-nulls">No Nulls</option><option value="high-nulls">High Nulls (&gt;5%)</option></select>
                  <div style={{ width: 1, height: 20, background: 'var(--border-color)', margin: '0 4px' }} />
                  <button className="emp-btn emp-btn-ghost emp-btn-sm" onClick={selectAll}>Select All</button>
                  <button className="emp-btn emp-btn-ghost emp-btn-sm" onClick={selectNone}>Select None</button>
                  <button className="emp-btn emp-btn-ghost emp-btn-sm" onClick={selectNulls}>Select Nulls</button>
                </div>

                <div className="glass-panel" style={{ overflow: 'hidden' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                    <thead><tr><th style={thStyle}></th><th style={thStyle}>#</th><th style={thStyle}>Column Name</th><th style={thStyle}>Type</th><th style={thStyle}>Inferred</th><th style={thStyle}>Nulls</th><th style={thStyle}>Null %</th><th style={thStyle}>Unique</th><th style={thStyle}>Sample</th></tr></thead>
                    <tbody>
                      {filtered.map((col, i) => (
                        <tr key={col.name} onClick={() => toggleColumn(col.name)} style={{ cursor: 'pointer', background: col.selected ? 'rgba(88,166,255,0.04)' : '' }}>
                          <td style={tdStyle}><input type="checkbox" checked={col.selected} onChange={() => toggleColumn(col.name)} style={{ accentColor: 'var(--primary)', cursor: 'pointer' }} /></td>
                          <td style={{ ...tdStyle, color: 'var(--text-muted)', fontSize: 10 }}>{i + 1}</td>
                          <td style={tdStyle}><code style={{ fontFamily: "'DM Mono', monospace", fontSize: 11, color: col.selected ? 'var(--primary)' : '#fff' }}>{col.name}</code></td>
                          <td style={tdStyle}><span style={{ fontFamily: "'DM Mono', monospace", fontSize: 9, padding: '2px 7px', borderRadius: 5, background: typeColors[col.type]?.bg, color: typeColors[col.type]?.color }}>{col.type}</span></td>
                          <td style={tdStyle}><span style={{ fontFamily: "'DM Mono', monospace", fontSize: 9, padding: '2px 7px', borderRadius: 5, background: inferredColors[col.inferred]?.bg, color: inferredColors[col.inferred]?.color }}>{inferredColors[col.inferred]?.label}</span></td>
                          <td style={tdStyle}><div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>{col.nulls > 0 && <AlertTriangle size={12} color={col.nullPct > 5 ? 'var(--danger)' : 'var(--warning)'} />}<span style={{ fontFamily: "'DM Mono', monospace", fontSize: 11, color: col.nulls > 0 ? 'var(--warning)' : 'var(--success)' }}>{col.nulls}</span></div></td>
                          <td style={tdStyle}><div style={{ display: 'flex', alignItems: 'center', gap: 6 }}><NullBar pct={col.nullPct} /><span style={{ fontFamily: "'DM Mono', monospace", fontSize: 10, color: col.nullPct > 5 ? 'var(--danger)' : col.nullPct > 0 ? 'var(--warning)' : 'var(--text-muted)' }}>{col.nullPct.toFixed(2)}%</span></div></td>
                          <td style={{ ...tdStyle, fontFamily: "'DM Mono', monospace", fontSize: 11 }}>{col.unique?.toLocaleString()}</td>
                          <td style={{ ...tdStyle, fontFamily: "'DM Mono', monospace", fontSize: 10, color: 'var(--text-muted)', maxWidth: 120, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{col.sample}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 20, padding: '16px 0', borderTop: '1px solid var(--border-color)' }}>
                  <div style={{ fontFamily: "'DM Mono', monospace", fontSize: 11, color: 'var(--text-muted)' }}>{selectedCount} columns selected · {selectedNulls} have nulls · {totalNullCells.toLocaleString()} null cells</div>
                  <button className="emp-btn emp-btn-primary" onClick={handleProceed} disabled={selectedCount === 0} style={{ padding: '10px 24px', fontSize: 13 }}><Sparkles size={14} /> Clean Selected →</button>
                </div>
              </>
            )}
          </>
        )}

        {/* ==================== CLEANED DATA VIEW ==================== */}
        {viewMode === 'data' && (
          <div className="glass-panel" style={{ overflow: 'hidden' }}>
            <div style={{ padding: '12px 16px', borderBottom: '1px solid var(--border-color)', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div style={{ fontSize: 14, fontWeight: 600, color: '#fff' }}>Cleaned Dataset (After ML Pipeline Processing)</div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                <div style={{ fontFamily: "'DM Mono', monospace", fontSize: 10, color: 'var(--text-muted)' }}>{dataTotalRows > 0 ? `${dataTotalRows.toLocaleString()} total rows` : `${cleanedData.length} rows shown`}</div>
                {cleanedData.length > 0 && (
                  <button className="emp-btn emp-btn-ghost" onClick={downloadCleanedData} style={{ padding: '6px 14px', fontSize: 11 }}>
                    <Download size={14} /> Download CSV
                  </button>
                )}
                {cleanedData.length === dataTotalRows && cleanedData.length > 100 && (
                  <button className="emp-btn emp-btn-ghost" onClick={resetDataView} style={{ padding: '6px 14px', fontSize: 11 }}>
                    Show Less
                  </button>
                )}
                {!loadingAllData && cleanedData.length < dataTotalRows && cleanedData.length <= 100 && (
                  <button className="emp-btn emp-btn-primary" onClick={loadAllData} style={{ padding: '6px 14px', fontSize: 11 }}>
                    Load Entire Dataset
                  </button>
                )}
                {loadingAllData && (
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: 'var(--primary)', fontSize: 12 }}>
                    <div className="spin"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 12a9 9 0 1 1-6.219-8.56" /></svg></div>
                    Loading...
                  </div>
                )}
              </div>
            </div>
            {cleanedData.length > 0 ? (
              <div style={{ overflowX: 'auto', maxHeight: '70vh' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                  <thead>
                    <tr>
                      <th style={{ ...thStyle, width: 50 }}>#</th>
                      {Object.keys(cleanedData[0] || {}).map(col => (
                        <th key={col} style={{ ...thStyle, fontSize: 10, minWidth: 120 }}>{col}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {cleanedData.map((row, ri) => (
                      <tr key={ri}>
                        <td style={{ ...tdStyle, color: 'var(--text-muted)', fontSize: 10 }}>{ri + 1}</td>
                        {Object.values(row).map((val, ci) => (
                          <td key={ci} style={{ ...tdStyle, fontFamily: "'DM Mono', monospace", whiteSpace: 'nowrap', maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis' }}>
                            {val === null || val === undefined || val === '' ? <span style={{ color: 'var(--text-muted)', fontStyle: 'italic', opacity: 0.5 }}>null</span> : String(val)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div style={{ padding: '3rem', textAlign: 'center', color: 'var(--text-muted)' }}>
                <Database size={40} style={{ marginBottom: '1rem', opacity: 0.5 }} />
                <p>Cleaned data not available yet. The dataset needs to be processed by the ML pipeline first.</p>
              </div>
            )}
            {cleanedData.length > 0 && dataTotalRows > cleanedData.length && (
              <div style={{ padding: '12px 16px', borderTop: '1px solid var(--border-color)', textAlign: 'center' }}>
                <button className="emp-btn emp-btn-ghost" onClick={() => setDataPage(prev => prev + 1)}>
                  Load More Rows
                </button>
              </div>
            )}
          </div>
        )}

        {/* ==================== ORIGINAL DATA VIEW ==================== */}
        {viewMode === 'original' && (
          <div className="glass-panel" style={{ overflow: 'hidden' }}>
            <div style={{ padding: '12px 16px', borderBottom: '1px solid var(--border-color)', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div style={{ fontSize: 14, fontWeight: 600, color: '#fff' }}>Raw Uploaded Data (Before Cleaning Pipeline)</div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                <div style={{ fontFamily: "'DM Mono', monospace", fontSize: 10, color: 'var(--text-muted)' }}>{originalTotalRows > 0 ? `${originalTotalRows.toLocaleString()} total rows` : `${originalData.length} rows shown`}</div>
                {originalData.length > 0 && (
                  <button className="emp-btn emp-btn-ghost" onClick={downloadOriginalData} style={{ padding: '6px 14px', fontSize: 11 }}>
                    <Download size={14} /> Download CSV
                  </button>
                )}
                {originalData.length === originalTotalRows && originalData.length > 100 && (
                  <button className="emp-btn emp-btn-ghost" onClick={() => { setOriginalData([]); loadOriginalData(1); }} style={{ padding: '6px 14px', fontSize: 11 }}>
                    Show Less
                  </button>
                )}
                {!loadingOriginal && originalData.length < originalTotalRows && originalData.length <= 100 && (
                  <button className="emp-btn emp-btn-primary" onClick={loadAllOriginalData} style={{ padding: '6px 14px', fontSize: 11 }}>
                    Load Entire Dataset
                  </button>
                )}
                {loadingOriginal && (
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: 'var(--primary)', fontSize: 12 }}>
                    <div className="spin"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 12a9 9 0 1 1-6.219-8.56" /></svg></div>
                    Loading...
                  </div>
                )}
              </div>
            </div>
            {originalData.length > 0 ? (
              <div style={{ overflowX: 'auto', maxHeight: '70vh' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                  <thead>
                    <tr>
                      <th style={{ ...thStyle, width: 50 }}>#</th>
                      {Object.keys(originalData[0] || {}).map(col => (
                        <th key={col} style={{ ...thStyle, fontSize: 10, minWidth: 120 }}>{col}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {originalData.map((row, ri) => (
                      <tr key={ri}>
                        <td style={{ ...tdStyle, color: 'var(--text-muted)', fontSize: 10 }}>{ri + 1}</td>
                        {Object.values(row).map((val, ci) => (
                          <td key={ci} style={{ ...tdStyle, fontFamily: "'DM Mono', monospace", whiteSpace: 'nowrap', maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis' }}>
                            {val === null || val === undefined || val === '' ? <span style={{ color: 'var(--text-muted)', fontStyle: 'italic', opacity: 0.5 }}>null</span> : String(val)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div style={{ padding: '3rem', textAlign: 'center', color: 'var(--text-muted)' }}>
                <Eye size={40} style={{ marginBottom: '1rem', opacity: 0.5 }} />
                <p>No raw uploaded data found. This dataset may have been uploaded directly to the cleaned pipeline, or the raw data file is not available.</p>
              </div>
            )}
            {originalData.length > 0 && originalTotalRows > originalData.length && (
              <div style={{ padding: '12px 16px', borderTop: '1px solid var(--border-color)', textAlign: 'center' }}>
                <button className="emp-btn emp-btn-ghost" onClick={() => loadOriginalData(Math.ceil(originalData.length / 100) + 1)}>
                  Load More Rows
                </button>
              </div>
            )}
          </div>
        )}

        {/* ==================== SUMMARY VIEW ==================== */}
        {viewMode === 'summary' && (
          <div>
            {/* Data Source Toggle */}
            <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
              <button className={`emp-btn ${summaryDataMode === 'cleaned' ? 'emp-btn-primary' : 'emp-btn-ghost'}`} onClick={() => setSummaryDataMode('cleaned')}>
                <Database size={14} /> Cleaned Data
              </button>
              <button className={`emp-btn ${summaryDataMode === 'original' ? 'emp-btn-primary' : 'emp-btn-ghost'}`} onClick={() => setSummaryDataMode('original')} disabled={!originalAnalysis}>
                <Eye size={14} /> Original Data
              </button>
              {!originalAnalysis && <span style={{ fontSize: 11, color: 'var(--text-muted)', alignSelf: 'center' }}>Original data not available</span>}
            </div>

            {/* Cleaned Data Summary */}
            {summaryDataMode === 'cleaned' && analysisData && (
              <div style={{ display: 'grid', gap: 16 }}>
                {/* Dataset Overview */}
                <div className="glass-panel" style={{ padding: 20 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 16 }}>
                    <Info size={18} color="var(--primary)" />
                    <h3 style={{ color: '#fff', margin: 0, fontSize: 16 }}>Dataset Overview</h3>
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 16 }}>
                    <div style={{ background: 'rgba(88,166,255,0.08)', padding: 14, borderRadius: 8 }}>
                      <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 6, textTransform: 'uppercase' }}>Dataset Name</div>
                      <div style={{ fontSize: 14, color: '#fff', fontWeight: 500 }}>{analysisData.dataset_name || selectedDataset?.name}</div>
                    </div>
                    <div style={{ background: 'rgba(63,185,80,0.08)', padding: 14, borderRadius: 8 }}>
                      <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 6, textTransform: 'uppercase' }}>Total Rows</div>
                      <div style={{ fontSize: 20, color: 'var(--success)', fontWeight: 600 }}>{analysisData.row_count?.toLocaleString()}</div>
                    </div>
                    <div style={{ background: 'rgba(188,140,255,0.08)', padding: 14, borderRadius: 8 }}>
                      <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 6, textTransform: 'uppercase' }}>Total Columns</div>
                      <div style={{ fontSize: 20, color: '#bc8cff', fontWeight: 600 }}>{analysisData.column_count}</div>
                    </div>
                    <div style={{ background: 'rgba(210,153,34,0.08)', padding: 14, borderRadius: 8 }}>
                      <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 6, textTransform: 'uppercase' }}>Quality Score</div>
                      <div style={{ fontSize: 20, color: 'var(--warning)', fontWeight: 600 }}>{analysisData.quality_score || 'N/A'}</div>
                    </div>
                  </div>
                </div>

                {/* Data Quality Summary */}
                <div className="glass-panel" style={{ padding: 20 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 16 }}>
                    <Zap size={18} color="var(--warning)" />
                    <h3 style={{ color: '#fff', margin: 0, fontSize: 16 }}>Data Quality Summary</h3>
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 16 }}>
                    <div style={{ padding: 12, background: 'rgba(248,81,73,0.06)', borderRadius: 8, border: '1px solid rgba(248,81,73,0.15)' }}>
                      <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 4, textTransform: 'uppercase' }}>Missing Values</div>
                      <div style={{ fontSize: 24, fontWeight: 600, color: 'var(--danger)' }}>{analysisData.total_nulls?.toLocaleString() || 0}</div>
                      <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 4 }}>cells with null values</div>
                    </div>
                    <div style={{ padding: 12, background: 'rgba(63,185,80,0.06)', borderRadius: 8, border: '1px solid rgba(63,185,80,0.15)' }}>
                      <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 4, textTransform: 'uppercase' }}>Complete Rows</div>
                      <div style={{ fontSize: 24, fontWeight: 600, color: 'var(--success)' }}>{(analysisData.row_count - (analysisData.total_nulls || 0)).toLocaleString()}</div>
                      <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 4 }}>rows without nulls</div>
                    </div>
                    <div style={{ padding: 12, background: 'rgba(88,166,255,0.06)', borderRadius: 8, border: '1px solid rgba(88,166,255,0.15)' }}>
                      <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 4, textTransform: 'uppercase' }}>Complete Cells</div>
                      <div style={{ fontSize: 24, fontWeight: 600, color: 'var(--primary)' }}>{((analysisData.row_count * analysisData.column_count) - (analysisData.total_nulls || 0)).toLocaleString()}</div>
                      <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 4 }}>non-null cells</div>
                    </div>
                    <div style={{ padding: 12, background: 'rgba(188,140,255,0.06)', borderRadius: 8, border: '1px solid rgba(188,140,255,0.15)' }}>
                      <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 4, textTransform: 'uppercase' }}>Completeness</div>
                      <div style={{ fontSize: 24, fontWeight: 600, color: '#bc8cff' }}>{analysisData.total_nulls ? (100 - (analysisData.total_nulls / (analysisData.row_count * analysisData.column_count) * 100)).toFixed(1) : 100}%</div>
                      <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 4 }}>data completeness rate</div>
                    </div>
                  </div>
                </div>

                {/* Column Type Distribution */}
                <div className="glass-panel" style={{ padding: 20 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 16 }}>
                    <Layers size={18} color="var(--accent)" />
                    <h3 style={{ color: '#fff', margin: 0, fontSize: 16 }}>Column Type Distribution</h3>
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 12 }}>
                    {['numeric', 'categorical', 'datetime', 'text', 'identifier'].map(type => {
                      const count = columns.filter(c => c.inferred === type).length;
                      const pct = columns.length > 0 ? (count / columns.length * 100).toFixed(1) : 0;
                      if (count === 0) return null;
                      return (
                        <div key={type} style={{ padding: 12, background: 'rgba(255,255,255,0.03)', borderRadius: 8, border: '1px solid var(--border-color)' }}>
                          <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 6, textTransform: 'capitalize' }}>{type}</div>
                          <div style={{ fontSize: 20, fontWeight: 600, color: inferredColors[type]?.color || '#fff' }}>{count}</div>
                          <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>{pct}% of columns</div>
                          <div style={{ height: 4, background: 'rgba(255,255,255,0.1)', borderRadius: 2, marginTop: 8, overflow: 'hidden' }}>
                            <div style={{ height: '100%', width: `${pct}%`, background: inferredColors[type]?.color, borderRadius: 2 }} />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* Columns with Issues */}
                <div className="glass-panel" style={{ padding: 20 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 16 }}>
                    <AlertTriangle size={18} color="var(--danger)" />
                    <h3 style={{ color: '#fff', margin: 0, fontSize: 16 }}>Columns Requiring Attention</h3>
                  </div>
                  {columns.filter(c => c.nulls > 0).length > 0 ? (
                    <div style={{ display: 'grid', gap: 8 }}>
                      {columns.filter(c => c.nulls > 0).sort((a, b) => b.nulls - a.nulls).slice(0, 10).map(col => (
                        <div key={col.name} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '10px 14px', background: 'rgba(248,81,73,0.05)', borderRadius: 6, border: '1px solid rgba(248,81,73,0.1)' }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                            <code style={{ fontFamily: "'DM Mono', monospace", fontSize: 12, color: '#fff' }}>{col.name}</code>
                            <span style={{ fontSize: 10, padding: '2px 6px', borderRadius: 4, background: inferredColors[col.inferred]?.bg, color: inferredColors[col.inferred]?.color }}>{col.inferred}</span>
                          </div>
                          <div style={{ textAlign: 'right' }}>
                            <div style={{ fontSize: 14, fontWeight: 600, color: 'var(--danger)' }}>{col.nulls.toLocaleString()}</div>
                            <div style={{ fontSize: 9, color: 'var(--text-muted)' }}>nulls ({col.nullPct.toFixed(1)}%)</div>
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div style={{ padding: 20, textAlign: 'center', color: 'var(--success)' }}>
                      <CheckCircle2 size={24} style={{ marginBottom: 8 }} />
                      <p>All columns are clean - no null values found!</p>
                    </div>
                  )}
                </div>

                {/* Processing Info */}
                <div className="glass-panel" style={{ padding: 20 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 16 }}>
                    <Clock size={18} color="var(--text-muted)" />
                    <h3 style={{ color: '#fff', margin: 0, fontSize: 16 }}>Processing Information</h3>
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 12 }}>
                    <div style={{ padding: 12, background: 'rgba(255,255,255,0.03)', borderRadius: 8 }}>
                      <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 4, textTransform: 'uppercase' }}>Dataset ID</div>
                      <div style={{ fontSize: 12, color: '#fff', fontFamily: "'DM Mono', monospace" }}>{selectedDataset?.dataset_id || selectedDataset?.id || 'N/A'}</div>
                    </div>
                    <div style={{ padding: 12, background: 'rgba(255,255,255,0.03)', borderRadius: 8 }}>
                      <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 4, textTransform: 'uppercase' }}>Status</div>
                      <div style={{ fontSize: 12, color: 'var(--success)', fontWeight: 500, textTransform: 'capitalize' }}>{selectedDataset?.status || 'completed'}</div>
                    </div>
                    <div style={{ padding: 12, background: 'rgba(255,255,255,0.03)', borderRadius: 8 }}>
                      <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 4, textTransform: 'uppercase' }}>Unique Values (Total)</div>
                      <div style={{ fontSize: 12, color: '#fff' }}>{columns.reduce((sum, c) => sum + c.unique, 0).toLocaleString()}</div>
                    </div>
                    <div style={{ padding: 12, background: 'rgba(255,255,255,0.03)', borderRadius: 8 }}>
                      <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 4, textTransform: 'uppercase' }}>Columns Analyzed</div>
                      <div style={{ fontSize: 12, color: '#fff' }}>{columns.length}</div>
                    </div>
                  </div>
                </div>

                {/* Overall Summary */}
                <div className="glass-panel" style={{ padding: 20, background: 'linear-gradient(135deg, rgba(88,166,255,0.08) 0%, rgba(188,140,255,0.08) 100%)' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 16 }}>
                    <FileText size={18} color="var(--primary)" />
                    <h3 style={{ color: '#fff', margin: 0, fontSize: 16 }}>Overall Summary</h3>
                  </div>
                  <div style={{ fontSize: 14, color: 'var(--text-main)', lineHeight: 1.7 }}>
                    <p>This dataset contains <strong style={{ color: '#fff' }}>{analysisData.row_count?.toLocaleString()}</strong> rows and <strong style={{ color: '#fff' }}>{analysisData.column_count}</strong> columns. 
                    The overall data quality score is <strong style={{ color: analysisData.quality_score >= 70 ? 'var(--success)' : 'var(--warning)' }}>{analysisData.quality_score || 'N/A'}</strong>/100. 
                    {analysisData.total_nulls > 0 ? (
                      <>There are <strong style={{ color: 'var(--danger)' }}>{analysisData.total_nulls?.toLocaleString()}</strong> missing values across the dataset 
                      (representing {((analysisData.total_nulls / (analysisData.row_count * analysisData.column_count)) * 100).toFixed(1)}% of total cells). 
                      {columns.filter(c => c.nulls > 0).length} column{columns.filter(c => c.nulls > 0).length !== 1 ? 's' : ''} contain{columns.filter(c => c.nulls > 0).length === 1 ? 's' : ''} null values and may require cleaning.</>
                    ) : (
                      <> The dataset has no missing values and appears to be clean.</>
                    )}</p>
                    <p style={{ marginTop: 12 }}>The dataset includes <strong style={{ color: '#fff' }}>{columns.filter(c => c.inferred === 'numeric').length}</strong> numeric, 
                    <strong style={{ color: '#fff' }}> {columns.filter(c => c.inferred === 'categorical').length}</strong> categorical, 
                    and <strong style={{ color: '#fff' }}> {columns.filter(c => c.inferred === 'datetime').length}</strong> datetime columns.</p>
                  </div>
                </div>

                {/* Cleaning Report */}
                {analysisData?.cleaning_report && analysisData.cleaning_report.length > 0 && (
                  <div className="glass-panel" style={{ padding: 20, background: 'linear-gradient(135deg, rgba(63,185,80,0.08) 0%, rgba(88,166,255,0.08) 100%)' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 16 }}>
                      <CheckCircle2 size={18} color="var(--success)" />
                      <h3 style={{ color: '#fff', margin: 0, fontSize: 16 }}>Cleaning Report</h3>
                    </div>
                    <div style={{ display: 'grid', gap: 12 }}>
                      {analysisData.cleaning_report.map((item, idx) => (
                        <div key={idx} style={{ 
                          padding: '12px 16px', 
                          background: 'rgba(255,255,255,0.03)', 
                          borderRadius: 8,
                          borderLeft: '3px solid var(--success)'
                        }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                            <span style={{ color: '#fff', fontWeight: 600 }}>{item.category}</span>
                            <span style={{ color: 'var(--success)', fontWeight: 600 }}>{item.count.toLocaleString()}</span>
                          </div>
                          <div style={{ fontSize: 13, color: 'var(--text-muted)', marginBottom: 4 }}>
                            <strong style={{ color: 'var(--primary)' }}>Action:</strong> {item.action}
                          </div>
                          <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                            <strong style={{ color: 'var(--accent)' }}>Reason:</strong> {item.reason}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Original Data Summary */}
            {summaryDataMode === 'original' && originalAnalysis && (
              <div style={{ display: 'grid', gap: 16 }}>
                <div className="glass-panel" style={{ padding: 20 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 16 }}>
                    <Eye size={18} color="var(--primary)" />
                    <h3 style={{ color: '#fff', margin: 0, fontSize: 16 }}>Original Dataset Overview</h3>
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 16 }}>
                    <div style={{ background: 'rgba(88,166,255,0.08)', padding: 14, borderRadius: 8 }}>
                      <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 6, textTransform: 'uppercase' }}>Dataset Name</div>
                      <div style={{ fontSize: 14, color: '#fff', fontWeight: 500 }}>{selectedDataset?.name}</div>
                    </div>
                    <div style={{ background: 'rgba(63,185,80,0.08)', padding: 14, borderRadius: 8 }}>
                      <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 6, textTransform: 'uppercase' }}>Total Rows</div>
                      <div style={{ fontSize: 20, color: 'var(--success)', fontWeight: 600 }}>{originalAnalysis.totalRows?.toLocaleString()}</div>
                    </div>
                    <div style={{ background: 'rgba(188,140,255,0.08)', padding: 14, borderRadius: 8 }}>
                      <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 6, textTransform: 'uppercase' }}>Total Columns</div>
                      <div style={{ fontSize: 20, color: '#bc8cff', fontWeight: 600 }}>{originalAnalysis.columnCount}</div>
                    </div>
                  </div>
                </div>

                <div className="glass-panel" style={{ padding: 20 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 16 }}>
                    <Layers size={18} color="var(--accent)" />
                    <h3 style={{ color: '#fff', margin: 0, fontSize: 16 }}>Column List (Original)</h3>
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 8 }}>
                    {originalAnalysis.headers?.map((col, i) => (
                      <div key={i} style={{ padding: '8px 12px', background: 'rgba(255,255,255,0.03)', borderRadius: 6, border: '1px solid var(--border-color)' }}>
                        <div style={{ fontSize: 12, color: '#fff', fontFamily: "'DM Mono', monospace" }}>{col}</div>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="glass-panel" style={{ padding: 20, background: 'linear-gradient(135deg, rgba(88,166,255,0.08) 0%, rgba(188,140,255,0.08) 100%)' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 16 }}>
                    <FileText size={18} color="var(--primary)" />
                    <h3 style={{ color: '#fff', margin: 0, fontSize: 16 }}>Original Data Summary</h3>
                  </div>
                  <div style={{ fontSize: 14, color: 'var(--text-main)', lineHeight: 1.7 }}>
                    <p>This is the <strong style={{ color: '#fff' }}>original dataset</strong> before ML pipeline processing. 
                    It contains <strong style={{ color: '#fff' }}>{originalAnalysis.totalRows?.toLocaleString()}</strong> rows and 
                    <strong style={{ color: '#fff' }}> {originalAnalysis.columnCount}</strong> columns.</p>
                    <p style={{ marginTop: 12 }}>The data was uploaded and stored as-is before any cleaning or transformation.
                    The cleaned version of this dataset is created after the ML pipeline processing completes.</p>
                  </div>
                </div>
              </div>
            )}

            {/* No Data Available */}
            {summaryDataMode === 'cleaned' && !analysisData && (
              <div className="glass-panel" style={{ padding: '4rem 2rem', textAlign: 'center' }}>
                <FileText size={48} color="var(--text-muted)" style={{ marginBottom: '1rem', opacity: 0.5 }} />
                <h3 style={{ color: '#fff', marginBottom: '0.5rem' }}>Summary Not Available</h3>
                <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>Complete the dataset analysis to view the summary.</p>
              </div>
            )}
          </div>
        )}
      </div>

      <style>{`@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } } .spin { animation: spin 1s linear infinite; }`}</style>
    </EmployeeLayout>
  );
};

const thStyle = {
  background: 'rgba(13,17,23,0.95)', padding: '9px 12px', textAlign: 'left',
  fontFamily: "'DM Mono', monospace", fontSize: 9, color: 'var(--text-muted)',
  letterSpacing: 1, textTransform: 'uppercase', borderBottom: '1px solid var(--border-color)',
  position: 'sticky', top: 0, whiteSpace: 'nowrap',
};

const tdStyle = {
  padding: '9px 12px', borderBottom: '1px solid rgba(255,255,255,0.025)', color: 'var(--text-muted)',
};

export default DatasetAnalysisPage;
