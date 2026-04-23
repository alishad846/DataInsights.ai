import React, { useState, useEffect } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import './EmployeeCleaningPage.css';
import { getDatasetPreview, transformDataset, finalizeDataset, pauseCleaning, downloadDataset } from '../../services/api';


const STEPS = [
  { id: 1, name: 'Null Values', shortName: 'Null Values' },
  { id: 2, name: 'Duplicates', shortName: 'Duplicates' },
  { id: 3, name: 'Data Types', shortName: 'Data Types' },
  { id: 4, name: 'Outliers', shortName: 'Outliers' },
  { id: 5, name: 'Feature Eng.', shortName: 'Feature Eng.' },
];

const NULL_STRATEGIES = ['Keep as-is', 'Fill with 0', 'Fill with mean', 'Fill with median', 'Fill with mode', 'Drop rows'];
const DUPE_STRATEGIES = [
  { id: 'Keep first', label: 'Keep First Occurrence', desc: 'Remove all but first duplicate row' },
  { id: 'Keep last', label: 'Keep Last Occurrence', desc: 'Remove all but last duplicate row' },
  { id: 'Keep as-is', label: 'Ignore', desc: 'Keep all rows as-is' }
];

const TYPE_STRATEGIES = ['Auto-detect', 'String', 'Integer', 'Float', 'Date', 'Boolean'];
const OUTLIER_STRATEGIES = ['Keep as-is', 'Remove rows', 'IQR capping', 'Z-score capping'];

const EmployeeCleaningPage = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const dsId = searchParams.get('ds');
  const dsName = searchParams.get('name') || 'Dataset';

  const [currentStep, setCurrentStep] = useState(1);
  const [verifyOpen, setVerifyOpen] = useState(false);
  const [showFullCleaned, setShowFullCleaned] = useState(false);
  const [loading, setLoading] = useState(true);
  
  const [tableRows, setTableRows] = useState([]);
  const [cleanedRows, setCleanedRows] = useState([]);
  const [tableHeaders, setTableHeaders] = useState([]);
  
  const [settings, setSettings] = useState({
    1: {}, 2: { strategy: 'Keep first' }, 3: {}, 4: {}, 5: {}
  });

  const [leftWidth, setLeftWidth] = useState(68);
  const [dragging, setDragging] = useState(false);

  const [featStreaming, setFeatStreaming] = useState(false);
  const [featDone, setFeatDone] = useState(false);
  const [featStreamText, setFeatStreamText] = useState('');
  const STREAM_LINES = [
    '> Connecting to Ollama (llama3.2)...',
    '> Analyzing schema...',
    '> Detecting column relationships...',
    '> Computing value distributions...',
    '> Generating feature suggestions...'
  ];
  const [aiSuggestions, setAiSuggestions] = useState([]);
  const [featStatuses, setFeatStatuses] = useState({});
  const [error, setError] = useState(null);
  const [page, setPage] = useState(1);
  const [totalRows, setTotalRows] = useState(0);
  const [rawStats, setRawStats] = useState({
    totalRows: 0,
    totalNulls: 0,
    totalDuplicates: 0,
    columnNulls: {}
  });
  const [colFilter, setColFilter] = useState('all');

  const normalizeFeatureName = (value = '') =>
    String(value).toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '');

  const isNumericLikeColumn = (rows, col, minCount = 5) => {
    const sampleVals = rows.slice(0, 20).map(r => parseFloat(r[col])).filter(v => !isNaN(v));
    return sampleVals.length > minCount;
  };

  const getNumericColumns = (headers, rows) => headers.filter(col => isNumericLikeColumn(rows, col));

  const buildFeatureSuggestions = (headers, rows) => {
    const suggestions = [];
    const numericCols = getNumericColumns(headers, rows);
    const addSuggestion = (feature) => {
      if (!feature?.col || suggestions.some(existing => existing.col === feature.col)) return;
      suggestions.push(feature);
    };

    const findColumn = (tokens) => headers.find((col) => {
      const normalized = normalizeFeatureName(col);
      return tokens.some((token) => normalized.includes(token));
    });

    const getColumnMax = (col) => {
      const values = rows.map(r => parseFloat(r[col])).filter(v => !isNaN(v));
      if (values.length === 0) return 0;
      return Math.max(...values);
    };

    const slug = (value) => normalizeFeatureName(value || 'feature');

    const lengthCol = findColumn(['length', 'len']);
    const widthCol = findColumn(['width', 'breadth', 'wid']);
    const heightCol = findColumn(['height', 'depth', 'ht']);
    const salesCol = findColumn(['sales']);
    const quantityCol = findColumn(['quantity']);
    const profitCol = findColumn(['profit']);
    const discountCol = findColumn(['discount']);

    if (lengthCol && widthCol) {
      addSuggestion({
        id: suggestions.length,
        originalCol: lengthCol,
        inputs: [lengthCol, widthCol],
        col: `area_${slug(lengthCol)}_${slug(widthCol)}`,
        type: 'derived',
        operation: 'product',
        formula: `${lengthCol} x ${widthCol}`,
        desc: 'Creates an area-style feature from the two dimension columns.',
      });
    }

    if (lengthCol && widthCol && heightCol) {
      addSuggestion({
        id: suggestions.length,
        originalCol: lengthCol,
        inputs: [lengthCol, widthCol, heightCol],
        col: `volume_${slug(lengthCol)}_${slug(widthCol)}_${slug(heightCol)}`,
        type: 'derived',
        operation: 'product',
        formula: `${lengthCol} x ${widthCol} x ${heightCol}`,
        desc: 'Creates a volume-style feature from the three dimension columns.',
      });
    }

    if (salesCol && quantityCol) {
      addSuggestion({
        id: suggestions.length,
        originalCol: salesCol,
        inputs: [salesCol, quantityCol],
        col: `sales_per_${slug(quantityCol)}`,
        type: 'derived',
        operation: 'ratio',
        formula: `${salesCol} / ${quantityCol}`,
        desc: 'Creates a per-unit sales feature from sales and quantity.',
      });
    }

    if (profitCol && salesCol) {
      addSuggestion({
        id: suggestions.length,
        originalCol: profitCol,
        inputs: [profitCol, salesCol],
        col: `profit_margin`,
        type: 'derived',
        operation: 'ratio',
        formula: `${profitCol} / ${salesCol}`,
        desc: 'Creates a profit margin style feature.',
      });
    }

    if (discountCol && salesCol) {
      addSuggestion({
        id: suggestions.length,
        originalCol: discountCol,
        inputs: [discountCol, salesCol],
        col: `discount_rate`,
        type: 'derived',
        operation: 'ratio',
        formula: `${discountCol} / ${salesCol}`,
        desc: 'Creates a discount rate feature relative to sales.',
      });
    }

    const skewedNumericCols = numericCols
      .map((col) => {
        const values = rows.map(r => parseFloat(r[col])).filter(v => !isNaN(v) && v >= 0);
        if (values.length < 8) return null;
        const sorted = [...values].sort((a, b) => a - b);
        const median = sorted[Math.floor(sorted.length / 2)] || 0;
        const max = Math.max(...values);
        return { col, median, max };
      })
      .filter(Boolean)
      .filter(({ max, median }) => median > 0 && max > median * 5)
      .sort((a, b) => (b.max / (b.median || 1)) - (a.max / (a.median || 1)));

    skewedNumericCols.slice(0, 2).forEach(({ col }) => {
      addSuggestion({
        id: suggestions.length,
        originalCol: col,
        inputs: [col],
        col: `${slug(col)}_log`,
        type: 'derived',
        operation: 'log',
        formula: `log1p(${col})`,
        desc: 'Applies a log transform to stabilize a skewed numeric column.',
      });
    });

    if (suggestions.length < 4) {
      const fallbackNumeric = numericCols.find(col => !suggestions.some(s => s.inputs?.includes(col)));
      if (fallbackNumeric) {
        addSuggestion({
          id: suggestions.length,
          originalCol: fallbackNumeric,
          inputs: [fallbackNumeric],
          col: `${slug(fallbackNumeric)}_normalized`,
          type: 'numeric',
          operation: 'normalize',
          formula: `${fallbackNumeric} / max`,
          desc: `Scales ${fallbackNumeric} to a 0-1 range based on its max value.`,
        });
      }
    }

    return suggestions.slice(0, 6);
  };

  const buildFeatureMeta = (rows, features) => {
    const meta = {};
    features.forEach((feat) => {
      const operation = feat.operation || feat.kind || feat.type;
      if (operation !== 'normalize' && operation !== 'numeric') return;
      const sourceCol = feat.inputs?.[0] || feat.originalCol;
      if (!sourceCol) return;
      const values = rows.map(r => parseFloat(r[sourceCol])).filter(v => !isNaN(v));
      const max = values.length > 0 ? Math.max(...values) : 1;
      meta[feat.col || sourceCol] = max > 0 ? max : 1;
    });
    return meta;
  };

  const applyFeatureSuggestion = (row, feat, meta) => {
    const operation = feat.operation || feat.kind || (feat.type === 'boolean' ? 'boolean_flag' : feat.type === 'numeric' ? 'normalize' : feat.type);
    const inputs = Array.isArray(feat.inputs) && feat.inputs.length > 0
      ? feat.inputs
      : (feat.originalCol ? [feat.originalCol] : []);
    const values = inputs.map((col) => {
      const parsed = parseFloat(row[col]);
      return Number.isNaN(parsed) ? 0 : parsed;
    });

    switch (operation) {
      case 'normalize':
      case 'numeric': {
        const sourceCol = inputs[0];
        const max = meta[feat.col || sourceCol] || 1;
        const parsed = parseFloat(row[sourceCol]);
        return Number.isNaN(parsed) ? 0 : Number((parsed / max).toFixed(4));
      }
      case 'boolean':
      case 'boolean_flag':
        return inputs.some((col) => row[col] != null && String(row[col]).trim() !== '') ? 1 : 0;
      case 'product':
        return values.reduce((acc, value) => acc * (Number.isNaN(value) ? 1 : value), 1);
      case 'sum':
        return values.reduce((acc, value) => acc + (Number.isNaN(value) ? 0 : value), 0);
      case 'difference':
        return (values[0] || 0) - (values[1] || 0);
      case 'ratio': {
        const numerator = values[0] || 0;
        const denominator = values[1] || 0;
        return denominator === 0 ? 0 : Number((numerator / denominator).toFixed(4));
      }
      case 'log': {
        const parsed = values[0];
        return Number.isNaN(parsed) ? 0 : Number(Math.log1p(Math.max(parsed, 0)).toFixed(4));
      }
      default: {
        const sourceCol = inputs[0];
        const val = row[sourceCol];
        return val != null && String(val).trim() !== '' ? 1 : 0;
      }
    }
  };

  useEffect(() => {
    const fetchData = async () => {
      if (!dsId) { setLoading(false); return; }
      setLoading(true);
      try {
        const data = await getDatasetPreview(dsId, page);
        if (data.success) {
          setTableRows(data.data || []);
          setCleanedRows(data.data || []);
          setTotalRows(data.totalRows || 0);
          if (data.currentStats || data.rawStats) setRawStats(data.currentStats || data.rawStats);
          if (data.data?.length > 0) {
            setTableHeaders(Object.keys(data.data[0]));
          }
        }
      } catch (err) {
        console.warn('Fetch error:', err);
        setError("Failed to load dataset data.");
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, [dsId, page]);

  // Track page exit (Pause Logging)
  useEffect(() => {
    return () => {
      if (dsId) {
        pauseCleaning(dsId).catch(err => console.error("Pause logging failed:", err));
      }
    };
  }, [dsId]);

  // Handle Drag Resizing

  useEffect(() => {
    const handleMouseMove = (e) => {
      if (!dragging) return;
      const pct = Math.min(85, Math.max(25, (e.clientX / window.innerWidth) * 100));
      setLeftWidth(pct);
    };
    const handleMouseUp = () => setDragging(false);
    if (dragging) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
    }
    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [dragging]);

  // Process data based on settings
  useEffect(() => {
    let data = [...tableRows];

    // STEP 1: Null Values
    const s1 = settings[1] || {};
    const colStats = {};
    tableHeaders.forEach(col => {
      const numericVals = tableRows.map(r => parseFloat(r[col])).filter(v => !isNaN(v));
      if (numericVals.length > 0) {
        const sorted = [...numericVals].sort((a, b) => a - b);
        const sum = numericVals.reduce((a, b) => a + b, 0);
        colStats[col] = {
          mean: sum / numericVals.length,
          median: numericVals.length % 2 === 0 
            ? (sorted[numericVals.length / 2 - 1] + sorted[numericVals.length / 2]) / 2 
            : sorted[Math.floor(numericVals.length / 2)],
        };
      }
      const textVals = tableRows.map(r => r[col])?.filter(v => v != null && v !== '');
      if (textVals?.length > 0) {
        const freq = {};
        textVals.forEach(v => { freq[v] = (freq[v] || 0) + 1; });
        const mode = Object.entries(freq).sort((a, b) => b[1] - a[1])[0]?.[0];
        colStats[col] = { ...colStats[col], mode };
      }
    });

    data = data.map(row => {
      let newRow = { ...row };
      
      // Remove white spaces automatically regardless of strategy
      Object.keys(newRow).forEach(col => {
        if (typeof newRow[col] === 'string') {
          newRow[col] = newRow[col].trim();
        }
      });

      let drop = false;
      Object.entries(s1).forEach(([col, strategy]) => {
        if (!strategy || strategy === 'Keep as-is') return;
        const val = newRow[col];
        if (val == null || val === '') {
          if (strategy === 'Fill with 0') newRow[col] = 0;
          else if (strategy === 'Fill with mean') newRow[col] = colStats[col]?.mean != null ? Math.round(colStats[col].mean * 100) / 100 : 0;
          else if (strategy === 'Fill with median') newRow[col] = colStats[col]?.median != null ? Math.round(colStats[col].median * 100) / 100 : 0;
          else if (strategy === 'Fill with mode') newRow[col] = colStats[col]?.mode || 'Unknown';
          else if (strategy === 'Drop rows') drop = true;
        }
      });
      return drop ? null : newRow;
    }).filter(Boolean);

    // STEP 2: Duplicates
    const s2 = settings[2]?.strategy || 'Keep first';
    if (s2 !== 'Keep as-is') {
      const seen = new Set();
      if (s2 === 'Keep first') {
        data = data.filter(row => {
          const key = JSON.stringify(row);
          if (seen.has(key)) return false;
          seen.add(key);
          return true;
        });
      } else if (s2 === 'Keep last') {
        const newCleaned = [];
        for (let i = data.length - 1; i >= 0; i--) {
          const row = data[i];
          const key = JSON.stringify(row);
          if (!seen.has(key)) {
            seen.add(key);
            newCleaned.unshift(row);
          }
        }
        data = newCleaned;
      }
    }

    // STEP 3: Data Types
    const s3 = settings[3] || {};
    data = data.map(row => {
      const newRow = { ...row };
      Object.entries(s3).forEach(([col, targetType]) => {
        if (!targetType || targetType === 'Auto-detect') return;
        const val = newRow[col];
        if (targetType === 'Integer') {
          const parsed = parseInt(val, 10);
          newRow[col] = !isNaN(parsed) ? parsed : 0;
        } else if (targetType === 'Float') {
          const parsed = parseFloat(val);
          newRow[col] = !isNaN(parsed) ? parsed : 0;
        } else if (targetType === 'String') {
          newRow[col] = String(val ?? '');
        } else if (targetType === 'Boolean') {
          newRow[col] = val && val !== '0' && String(val).toLowerCase() !== 'false' ? true : false;
        } else if (targetType === 'Date') {
          const date = new Date(val);
          newRow[col] = !isNaN(date.getTime()) ? val : '';
        }
      });
      return newRow;
    });

    // STEP 4: Outliers
    const s4 = settings[4] || {};
    const numericCols = tableHeaders.filter(col => {
      const sampleVals = tableRows.slice(0, 20).map(r => parseFloat(r[col])).filter(v => !isNaN(v));
      return sampleVals.length > 5;
    });

    numericCols.forEach(col => {
      const strategy = s4[col];
      if (!strategy || strategy === 'Keep as-is') return;

      const vals = data.map(r => parseFloat(r[col])).filter(v => !isNaN(v));
      if (vals.length === 0) return;

      const sorted = [...vals].sort((a, b) => a - b);
      const q1 = sorted[Math.floor(vals.length * 0.25)] || 0;
      const q3 = sorted[Math.floor(vals.length * 0.75)] || 0;
      const iqr = q3 - q1;
      const lowerBound = q1 - 1.5 * iqr;
      const upperBound = q3 + 1.5 * iqr;

      if (strategy === 'Remove rows') {
        data = data.filter(row => {
          const val = parseFloat(row[col]);
          if (!isNaN(val) && (val < lowerBound || val > upperBound)) return false;
          return true;
        });
      } else if (strategy === 'IQR capping') {
        data = data.map(row => {
          const newRow = { ...row };
          const val = parseFloat(row[col]);
          if (!isNaN(val)) {
            if (val < lowerBound) newRow[col] = lowerBound;
            else if (val > upperBound) newRow[col] = upperBound;
          }
          return newRow;
        });
      }
    });

    // STEP 5: Feature Extraction
    const acceptedFeatures = aiSuggestions.filter(s => featStatuses[s.id] === 'accept');
    if (acceptedFeatures.length > 0) {
      const meta = buildFeatureMeta(data, acceptedFeatures);

      data = data.map(row => {
        const newRow = { ...row };
        acceptedFeatures.forEach(feat => {
          const outputCol = feat.col || feat.newCol || feat.output;
          if (!outputCol) return;
          newRow[outputCol] = applyFeatureSuggestion(newRow, feat, meta);
        });
        return newRow;
      });
    }

    setCleanedRows(data);
  }, [settings, tableRows, tableHeaders, featStatuses, aiSuggestions]);

  const handleTransformation = async (stepId, manualSettings = null) => {
    setLoading(true);
    setError(null);

    const refreshPreview = async () => {
      const previewRes = await getDatasetPreview(dsId, 1);
      if (!previewRes.success) {
        return false;
      }

      setPage(1);
      setTableRows(previewRes.data || []);
      setCleanedRows(previewRes.data || []);
      setTableHeaders(previewRes.data?.length > 0 ? Object.keys(previewRes.data[0]) : []);
      setTotalRows(previewRes.totalRows || 0);
      if (previewRes.currentStats || previewRes.rawStats) {
        setRawStats(previewRes.currentStats || previewRes.rawStats);
      }
      return true;
    };

    try {
      let transformConfig = {};
      const activeSettings = manualSettings || settings[stepId];

      if (stepId === 1) transformConfig = { type: 'null_fill', params: activeSettings };
      else if (stepId === 2) transformConfig = { type: 'drop_duplicates', params: activeSettings };
      else if (stepId === 3) transformConfig = { type: 'type_conversion', params: activeSettings };
      else if (stepId === 4) transformConfig = { type: 'outlier_handling', params: activeSettings };
      else if (stepId === 5) {
        const acceptedFeatures = aiSuggestions.filter(s => featStatuses[s.id] === 'accept');
        transformConfig = { type: 'feature_eng', params: { features: acceptedFeatures } };
      }

      const res = await transformDataset(dsId, transformConfig.type, transformConfig.params);
      if (res.success) {
        const previewLoaded = await refreshPreview();
        if (previewLoaded) {
          // Update current step to next after the file has been refreshed.
          if (currentStep < 5) {
            const nextS = currentStep + 1;
            setCurrentStep(nextS);
            if (nextS === 5 && !featDone && !featStreaming) startFeatStream();
          } else {
            setVerifyOpen(true);
          }
        }
      } else {
        throw new Error(res.message || "Backend transformation error");
      }
    } catch (err) {
      setError(err.message || "Transformation failed. Please try a different strategy.");
      // DO NOT reset Step to 1 here. Just refresh to show data hasn't changed.
      await refreshPreview();
    } finally {
      setLoading(false);
    }
  };

  const handleSkip = () => {
    if (currentStep < 5) {
      const nextS = currentStep + 1;
      setCurrentStep(nextS);
      if (nextS === 5 && !featDone && !featStreaming) startFeatStream();
    } else {
      setVerifyOpen(true);
    }
  };

  const handleLetAiDecide = async () => {
    // 1. Generate AI settings for current step
    const newSettings = handleAiDecide(); // This currently just updates state, let's make it return settings
    // 2. Apply them
    await handleTransformation(currentStep, newSettings);
  };

  const handleApplyManual = async () => {
    await handleTransformation(currentStep);
  };

  const startFeatStream = () => {
    setFeatStreaming(true);
    setFeatDone(false);
    setFeatStreamText('');
    let line = 0, ch = 0, text = '';
    
    // Determine dynamic suggestions based on data
    const suggestions = buildFeatureSuggestions(tableHeaders, tableRows);
    setFeatStatuses({});
    setAiSuggestions(suggestions);
    
    const tick = () => {
      if (line >= STREAM_LINES.length) {
        setTimeout(() => {
          setFeatStreaming(false);
          setFeatDone(true);
        }, 500);
        return;
      }
      const L = STREAM_LINES[line];
      if (ch < L.length) {
        text += L[ch++];
        setFeatStreamText(text);
        setTimeout(tick, 20);
      } else {
        text += '\n'; line++; ch = 0;
        setFeatStreamText(text);
        setTimeout(tick, 200);
      }
    };
    tick();
  };

  const getNumCols = () => {
    return tableHeaders.filter(col => {
      const vals = tableRows.slice(0, 20).map(r => parseFloat(r[col])).filter(v => !isNaN(v));
      return vals.length > 5;
    });
  };

  const cNulls = rawStats.columnNulls || {};
  const totNulls = rawStats.totalNulls || 0;
  const nullCols = Object.keys(cNulls).filter(c => cNulls[c] > 0);
  const totDupes = rawStats.totalDuplicates || 0;
  const numCols = getNumCols();

  const handleAiDecide = () => {
    const newSettings = { ...settings };
    
    if (currentStep === 1) {
      const s1 = {};
      nullCols.forEach(col => {
        // We still use a preview check for numeric vs string because raw_stats doesn't hold types yet
        const isNumeric = tableRows.slice(0, 20).map(r => parseFloat(r[col])).filter(v => !isNaN(v)).length > 5;
        s1[col] = isNumeric ? 'Fill with median' : 'Fill with mode';
      });
      newSettings[1] = {...newSettings[1], ...s1};
    } else if (currentStep === 2) {
      newSettings[2] = { strategy: 'Keep first' };
    } else if (currentStep === 3) {
      const s3 = {};
      tableHeaders.forEach(col => {
        const sampleVals = tableRows.slice(0, 20).map(r => r[col]).filter(v => v != null && String(v).trim() !== '');
        if (sampleVals.length === 0) {
          s3[col] = 'String';
        } else {
          const isBool = sampleVals.every(v => ['true', 'false', '0', '1'].includes(String(v).toLowerCase()));
          if (isBool) { s3[col] = 'Boolean'; }
          else {
            const numVals = sampleVals.map(v => parseFloat(v)).filter(v => !isNaN(v));
            if (numVals.length === sampleVals.length) {
              const isInt = sampleVals.every(v => Number.isInteger(parseFloat(v)));
              s3[col] = isInt ? 'Integer' : 'Float';
            } else {
              const dateVals = sampleVals.filter(v => !isNaN(new Date(v).getTime()) && isNaN(v));
              if (dateVals.length === sampleVals.length) s3[col] = 'Date';
              else s3[col] = 'String';
            }
          }
        }
      });
      newSettings[3] = {...newSettings[3], ...s3};
    } else if (currentStep === 4) {
      const s4 = {};
      numCols.forEach(col => s4[col] = 'IQR capping');
      newSettings[4] = {...newSettings[4], ...s4};
    }
    
    setSettings(newSettings);
    return newSettings[currentStep];
  };

  const syncPreview = async () => {
    if (!dsId) return false;

    const previewRes = await getDatasetPreview(dsId, 1);
    if (!previewRes.success) {
      return false;
    }

    setPage(1);
    setTableRows(previewRes.data || []);
    setCleanedRows(previewRes.data || []);
    setTableHeaders(previewRes.data?.length > 0 ? Object.keys(previewRes.data[0]) : []);
    setTotalRows(previewRes.totalRows || 0);
    if (previewRes.currentStats || previewRes.rawStats) {
      setRawStats(previewRes.currentStats || previewRes.rawStats);
    }
    return true;
  };

  const activeData = cleanedRows.length > 0 ? cleanedRows : tableRows;
  
  const acceptedFeatObj = aiSuggestions.filter(s => featStatuses[s.id] === 'accept');
  const acceptedFeaturesPersisted =
    acceptedFeatObj.length > 0 && acceptedFeatObj.every((feat) => tableHeaders.includes(feat.col));
  
  // Logical Filtering for headers
  let showHeaders = [...tableHeaders];
  
  if (colFilter === 'highlighted') {
    if (currentStep === 1) {
      showHeaders = tableHeaders.filter(c => (rawStats.columnNulls?.[c] || 0) > 0);
    } else if (currentStep === 4) {
      // Outliers: use numeric columns that were flagged (this is a simplified check)
      showHeaders = getNumCols();
    }
    // For other steps, we show all as they affect rows or are global
  }
  
  // Limit initial view to prevent lag, append new features
  const limitedHeaders = [...(showHeaders.length > 0 ? showHeaders.slice(0, 15) : []), ...acceptedFeatObj.map(f => f.col)];
  const finalHeaders = Array.from(
    new Set([
      ...showHeaders,
      ...acceptedFeatObj.map(f => f.col)
    ])
  );

  const getBadge = () => {
    switch (currentStep) {
      case 1: return { cls: 'clean-null-badge', txt: '● Nulls highlighted' };
      case 2: return { cls: 'clean-dupe-badge', txt: '● Dupes highlighted' };
      case 3: return { cls: 'clean-type-badge', txt: '● Type fixes highlighted' };
      case 4: return { cls: 'clean-outlier-badge', txt: '● Outliers highlighted' };
      case 5: return { cls: 'clean-feat-badge', txt: '✦ New features highlighted' };
      default: return { cls: '', txt: '' };
    }
  };

  const handleDownload = async () => {
    if (!dsId) return;
    try {
      if (acceptedFeatObj.length > 0 && !acceptedFeaturesPersisted) {
        setLoading(true);
        const transformRes = await transformDataset(dsId, 'feature_eng', { features: acceptedFeatObj });
        if (!transformRes.success) {
          throw new Error(transformRes.message || 'Failed to generate features before download.');
        }
        await syncPreview();
      }
      await downloadDataset(dsId, `${dsName}_cleaned.csv`);
    } catch (err) {
      setError(err.message || 'Failed to download cleaned CSV.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="clean-root">
      {/* Top Nav */}
      <div className="clean-topnav">
        <button className="clean-back-btn" onClick={() => navigate('/employee/datasets')}>← Back</button>
        <div>
          <div className="clean-ds-label">{dsName}</div>
          <div className="clean-ds-sublabel">
            v1 · {tableRows.length.toLocaleString()} rows · {tableHeaders.length} cols · Cleaning in progress
          </div>
        </div>
        <div className="clean-topnav-right">
          <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 10, color: 'var(--amber)', display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--amber)', display: 'inline-block', animation: 'cleanBlink 1s infinite' }}></span>
            Cleaning in progress
          </div>
          <button className="clean-btn clean-btn-ghost clean-btn-sm" onClick={() => setVerifyOpen(true)}>Verify Dataset</button>
        </div>
      </div>

      {/* Timeline */}
      <div className="clean-timeline">
        {STEPS.map((s, idx) => {
          let cls = 'clean-step';
          if (s.id < currentStep) cls += ' done';
          else if (s.id === currentStep) cls += ' active';
          if (s.id === 5) cls += ' feat';

          let status = 'Pending';
          if (s.id < currentStep) status = 'Done';
          else if (s.id === currentStep) {
            if (s.id === 1) status = `${totNulls} found`;
            if (s.id === 2) status = `${totDupes} dupes`;
            if (s.id === 3) status = 'Checking';
            if (s.id === 4) status = 'Scanning';
            if (s.id === 5) status = 'Ready';
          }

          return (
            <div key={s.id} className={cls} style={{ cursor: 'default' }}>
              <div className="clean-step-circle">{s.id === 5 ? '✦' : s.id}</div>
              <div className="clean-step-name">{s.shortName}</div>
              <div className="clean-step-status">{status}</div>
            </div>
          );
        })}
      </div>

      {/* Split View */}
      <div className="clean-split-wrap">
        
        {/* LEFT PANEL - Table */}
        <div className="clean-panel-left" style={{ width: `${leftWidth}%` }}>
          <div className="clean-panel-toolbar" style={{ justifyContent: 'space-between' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
              <div className="clean-toolbar-label">Showing <strong>all {showHeaders.length} columns</strong></div>
              <div className={`clean-step-badge ${getBadge().cls}`} style={{ marginLeft: 0 }}>{getBadge().txt}</div>
            </div>
            
            {/* Pagination Controls */}
            {totalRows > 0 && (
              <div style={{ display: 'flex', alignItems: 'center', gap: 10, background: 'rgba(255,255,255,0.05)', padding: '4px 12px', borderRadius: 20, border: '1px solid rgba(255,255,255,0.1)' }}>
                <span style={{ fontSize: 12, color: 'var(--ink3)', fontFamily: "'IBM Plex Mono', monospace" }}>
                  {(page - 1) * 50 + 1}-{Math.min(page * 50, totalRows)} of {totalRows.toLocaleString()}
                </span>
                <div style={{ display: 'flex', gap: 4 }}>
                  <button 
                    disabled={page === 1 || loading} 
                    onClick={() => setPage(p => p - 1)}
                    style={{ background: 'none', border: 'none', color: page === 1 ? '#555' : '#aaa', cursor: page === 1 ? 'default' : 'pointer', fontSize: 16, padding: '0 4px' }}
                  >
                    ←
                  </button>
                  <button 
                    disabled={page * 50 >= totalRows || loading} 
                    onClick={() => setPage(p => p + 1)}
                    style={{ background: 'none', border: 'none', color: page * 50 >= totalRows ? '#555' : '#aaa', cursor: page * 50 >= totalRows ? 'default' : 'pointer', fontSize: 16, padding: '0 4px' }}
                  >
                    →
                  </button>
                </div>
              </div>
            )}

            <select 
              className="clean-col-select" 
              value={colFilter} 
              onChange={(e) => setColFilter(e.target.value)}
            >
              <option value="all">All Columns</option>
              <option value="highlighted">Affected Columns Only</option>
            </select>
          </div>
          <div className="clean-data-scroll">
            {loading ? (
              <div style={{ padding: 40, textAlign: 'center', color: 'var(--ink3)' }}>Loading data...</div>
            ) : tableRows.length === 0 ? (
              <div style={{ padding: 60, textAlign: 'center', color: 'var(--ink3)', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 16, marginTop: 40 }}>
                <div style={{ fontSize: 18, color: 'var(--ink)' }}>No Dataset Selected</div>
                <div style={{ fontSize: 12 }}>You haven't selected a dataset to clean yet. Please choose one to get started.</div>
                <button 
                  className="clean-btn clean-btn-primary" 
                  onClick={() => navigate('/employee/datasets')}
                  style={{ marginTop: 8 }}
                >
                  Go to Datasets Page →
                </button>
              </div>
            ) : (
              <table className="clean-data-table">
                <thead>
                  <tr>
                    <th>#</th>
                    {limitedHeaders.map(col => {
                      const isNew = acceptedFeatObj.some(f => f.col === col);
                      const isNullCol = (rawStats.columnNulls?.[col] || 0) > 0;
                      const thCls = `${isNew ? 'col-new' : ''} ${isNullCol && currentStep === 1 ? 'col-problem' : ''}`;
                      return <th key={col} className={thCls}>{col}</th>;
                    })}
                  </tr>
                </thead>
                <tbody>
                  {activeData.map((row, ri) => (
                    <tr key={ri}>
                      <td className="row-num">{(page - 1) * 50 + ri + 1}</td>
                      {limitedHeaders.map(col => {
                        const val = row[col];
                        const isNull = val == null || String(val).trim() === '';
                        let cls = '';
                        if (currentStep === 1) {
                          if (isNull) cls = 'cell-null';
                          else if (!isNull && settings[1][col] && settings[1][col] !== 'Keep as-is') cls = 'cell-filled';
                        }
                        const isNew = acceptedFeatObj.some(f => f.col === col);
                        if (isNew) cls = 'cell-new';
                        return <td key={col} className={cls}>{isNull && currentStep === 1 ? (colFilter === 'highlighted' ? 'NULL' : 'NULL') : String(val ?? '')}</td>;
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>

        {/* Drag Handle */}
        <div className={`clean-drag-handle ${dragging ? 'dragging' : ''}`} style={{ left: `${leftWidth}%` }} onMouseDown={() => setDragging(true)}></div>

        {/* RIGHT PANEL - Step Control */}
        <div className="clean-panel-right">
          <div className="clean-rpanel-head">
            <div className="clean-rpanel-title">Step {currentStep} — {STEPS[currentStep-1].name}</div>
            <div className="clean-rpanel-sub">Configure how you'd like to clean this dataset</div>
          </div>
          <div className="clean-rpanel-body">
            
            {/* Step 1 */}
            {currentStep === 1 && (
              <div>
                <div className="clean-stat-row">
                  <div className="clean-stat-mini"><div className="clean-stat-mini-val" style={{color:'var(--red)'}}>{totNulls}</div><div className="clean-stat-mini-lbl">Nulls Found</div></div>
                  <div className="clean-stat-mini"><div className="clean-stat-mini-val">{nullCols.length}</div><div className="clean-stat-mini-lbl">Cols Affected</div></div>
                </div>
                <div className="clean-step-card">
                  <div className="clean-step-card-title">Column Strategies</div>
                  {nullCols.length === 0 ? (
                    <div style={{fontSize:11, color:'var(--green)'}}>No nulls found!</div>
                  ) : (
                    nullCols.map(col => (
                      <div className="clean-col-row" key={col}>
                        <div><div className="clean-col-name">{col}</div><div style={{fontFamily:"'IBM Plex Mono',monospace", fontSize:9, color:'var(--ink3)'}}>{cNulls[col]} nulls</div></div>
                        <div className="clean-col-stat danger">{cNulls[col]} nulls</div>
                        <select className="clean-strategy-sel" value={settings[1][col] || 'Keep as-is'} onChange={e => setSettings({...settings, 1: {...settings[1], [col]: e.target.value}})}>
                          {NULL_STRATEGIES.map(s => <option key={s}>{s}</option>)}
                        </select>
                      </div>
                    ))
                  )}
                </div>
                <div className="clean-skip-bar">
                  <button className="clean-btn clean-btn-ghost clean-skip-btn" onClick={() => setCurrentStep(2)}>Skip This Step</button>
                  <button className="clean-btn clean-btn-ghost clean-skip-btn" style={{color:'var(--purple)', borderColor:'rgba(167,139,250,0.3)'}} onClick={handleAiDecide}>✦ Let AI Decide</button>
                </div>
              </div>
            )}

            {/* Step 2 */}
            {currentStep === 2 && (
              <div>
                <div className="clean-stat-row">
                  <div className="clean-stat-mini"><div className="clean-stat-mini-val" style={{color:'var(--amber)'}}>{totDupes}</div><div className="clean-stat-mini-lbl">Dupes Found</div></div>
                  <div className="clean-stat-mini"><div className="clean-stat-mini-val">{(tableRows.length - totDupes).toLocaleString()}</div><div className="clean-stat-mini-lbl">Unique Rows</div></div>
                </div>
                <div className="clean-step-card">
                  <div className="clean-step-card-title">Duplicate Strategy</div>
                  <div style={{display:'flex', flexDirection:'column', gap:8, marginTop:4}}>
                    {DUPE_STRATEGIES.map(st => {
                      const isActive = settings[2].strategy === st.id;
                      return (
                        <label key={st.id} style={{ display:'flex', alignItems:'center', gap:10, cursor:'pointer', padding:'8px 10px', borderRadius:8, border:`1px solid ${isActive?'var(--accent)':'var(--border)'}`, background:isActive?'var(--accentbg)':'transparent' }}>
                          <input type="radio" name="dupstrat" checked={isActive} onChange={() => setSettings({...settings, 2:{strategy:st.id}})} style={{accentColor:'var(--accent)'}} />
                          <div><div style={{fontSize:12, fontWeight:500, color:'var(--ink)'}}>{st.label}</div><div style={{fontFamily:"'IBM Plex Mono',monospace", fontSize:9, color:'var(--ink3)', marginTop:2}}>{st.desc}</div></div>
                        </label>
                      );
                    })}
                  </div>
                </div>
                <div className="clean-skip-bar">
                  <button className="clean-btn clean-btn-ghost clean-skip-btn" onClick={() => setCurrentStep(3)}>Skip This Step</button>
                  <button className="clean-btn clean-btn-ghost clean-skip-btn" style={{color:'var(--purple)', borderColor:'rgba(167,139,250,0.3)'}} onClick={handleAiDecide}>✦ Let AI Decide</button>
                </div>
              </div>
            )}

            {/* Step 3 */}
            {currentStep === 3 && (
              <div>
                <div className="clean-step-card">
                  <div className="clean-step-card-title">Type Adjustments</div>
                  {showHeaders.slice(0, 10).map(col => {
                    return (
                      <div className="clean-col-row" key={col}>
                        <div><div className="clean-col-name">{col}</div></div>
                        <select className="clean-strategy-sel" value={settings[3][col] || 'Auto-detect'} onChange={e => setSettings({...settings, 3: {...settings[3], [col]: e.target.value}})}>
                          {TYPE_STRATEGIES.map(s => <option key={s}>{s}</option>)}
                        </select>
                      </div>
                    );
                  })}
                </div>
                <div className="clean-skip-bar">
                  <button className="clean-btn clean-btn-ghost clean-skip-btn" onClick={() => setCurrentStep(4)}>Skip This Step</button>
                  <button className="clean-btn clean-btn-ghost clean-skip-btn" style={{color:'var(--purple)', borderColor:'rgba(167,139,250,0.3)'}} onClick={handleAiDecide}>✦ Let AI Decide</button>
                </div>
              </div>
            )}

            {/* Step 4 */}
            {currentStep === 4 && (
              <div>
                <div className="clean-step-card">
                  <div className="clean-step-card-title">Outlier Detect - Numeric Only</div>
                  {numCols.length === 0 ? (
                    <div style={{fontSize:11, color:'var(--ink3)'}}>No numeric columns detected.</div>
                  ) : (
                    numCols.map(col => (
                      <div className="clean-col-row" key={col}>
                        <div><div className="clean-col-name">{col}</div></div>
                        <select className="clean-strategy-sel" value={settings[4][col] || 'Keep as-is'} onChange={e => setSettings({...settings, 4: {...settings[4], [col]: e.target.value}})}>
                          {OUTLIER_STRATEGIES.map(s => <option key={s}>{s}</option>)}
                        </select>
                      </div>
                    ))
                  )}
                </div>
                <div className="clean-skip-bar">
                  <button className="clean-btn clean-btn-ghost clean-skip-btn" onClick={() => setCurrentStep(5)}>Skip This Step</button>
                  <button className="clean-btn clean-btn-ghost clean-skip-btn" style={{color:'var(--purple)', borderColor:'rgba(167,139,250,0.3)'}} onClick={handleAiDecide}>✦ Let AI Decide</button>
                </div>
              </div>
            )}

            {/* Step 5 */}
            {currentStep === 5 && (
              <div>
                {featStreaming && (
                  <div className="feat-loading">
                    <div className="feat-loading-top">
                      <div className="feat-spinner"></div>
                      <div style={{fontFamily:"'IBM Plex Mono',monospace", fontSize:10, color:'var(--purple)'}}>Ollama analyzing dataset schema…</div>
                    </div>
                    <div className="feat-stream">
                      {featStreamText.split('\n').map((line, i) => <div key={i}>{line}</div>)}
                      <span className="clean-cursor-blink"></span>
                    </div>
                  </div>
                )}

                {featDone && (
                  <div>
                    <div style={{fontFamily:"'IBM Plex Mono',monospace", fontSize:9, color:'var(--ink3)', marginBottom:12, padding:'8px 10px', background:'var(--purplebg)', borderRadius:7, border:'1px solid rgba(167,139,250,0.15)'}}>
                      ✦ Ollama suggested {aiSuggestions.length} new features · Accept or reject each individually
                    </div>
                    {aiSuggestions.map(s => {
                      const status = featStatuses[s.id];
                      return (
                        <div className="feat-card" key={s.id} style={{ borderColor: status==='accept'?'rgba(34,211,238,0.5)':'rgba(34,211,238,0.18)', opacity: status==='reject'?0.4:1 }}>
                          <div className="feat-card-header">
                            <span className="feat-col-name">{s.col}</span>
                            <span className="feat-type-tag">{s.type}</span>
                          </div>
                          <div className="feat-formula"><strong>{s.col}</strong> = {s.formula}</div>
                          <div className="feat-desc">{s.desc}</div>
                          <div className="feat-actions">
                            {status !== 'reject' && (
                              <button className="feat-btn feat-accept" style={{ background: status==='accept'?'rgba(34,211,238,0.2)':'var(--tealbg)' }} onClick={() => setFeatStatuses({...featStatuses, [s.id]:'accept'})}>
                                {status === 'accept' ? '✓ Accepted' : '✓ Accept'}
                              </button>
                            )}
                            {status !== 'accept' && (
                              <button className="feat-btn feat-reject" style={{ background: status==='reject'?'rgba(251,113,133,0.2)':'var(--redbg)' }} onClick={() => setFeatStatuses({...featStatuses, [s.id]:'reject'})}>
                                {status === 'reject' ? '✕ Rejected' : '✕ Reject'}
                              </button>
                            )}
                          </div>
                        </div>
                      )
                    })}
                  </div>
                )}

                {!featStreaming && !featDone && (
                  <button className="clean-btn clean-btn-ghost clean-skip-btn" style={{width:'100%', color:'var(--purple)', borderColor:'rgba(167,139,250,0.3)'}} onClick={startFeatStream}>
                    ✦ Start AI Feature Engineering
                  </button>
                )}

                <div className="clean-skip-bar" style={{ display: featDone ? 'flex' : 'none' }}>
                  <button className="clean-btn clean-btn-ghost clean-skip-btn">Skip Feature Eng.</button>
                  <button className="clean-btn clean-btn-green clean-skip-btn" onClick={() => setVerifyOpen(true)}>Finalize Dataset →</button>
                </div>
              </div>
            )}

            <div style={{ marginTop: 32, padding: '16px 0 0 0', borderTop: '1px solid var(--border)' }}>
              <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
                 {currentStep > 1 && (
                   <button 
                     className="clean-btn clean-btn-ghost" 
                     style={{ flex: 1 }} 
                     onClick={() => setCurrentStep(prev => prev - 1)}
                   >
                     ← Previous Step
                   </button>
                 )}
              </div>
              {currentStep < 5 ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                  <div style={{ display: 'flex', gap: 8 }}>
                    <button 
                      className="clean-btn" 
                      style={{ flex: 1, background: 'rgba(167, 139, 250, 0.1)', color: 'var(--purple)', border: '1px solid rgba(167, 139, 250, 0.3)' }}
                      onClick={handleLetAiDecide}
                      disabled={loading}
                    >
                      ✦ Let AI Decide
                    </button>
                    <button 
                      className="clean-btn clean-btn-primary" 
                      style={{ flex: 1 }}
                      onClick={handleApplyManual}
                      disabled={loading}
                    >
                      Apply Changes
                    </button>
                  </div>
                  <button 
                    className="clean-btn clean-btn-ghost" 
                    style={{ width: '100%', fontSize: 12, opacity: 0.8 }}
                    onClick={handleSkip}
                    disabled={loading}
                  >
                    Skip this step
                  </button>
                </div>
              ) : (
                <div style={{ display: 'flex', gap: 8 }}>
                   <button className="clean-btn clean-btn-ghost" style={{flex: 1}} onClick={() => setCurrentStep(prev => Math.max(1, prev - 1))}>← Back</button>
                   <button className="clean-btn clean-btn-primary" style={{flex: 1}} onClick={() => setVerifyOpen(true)}>Finalize Dataset →</button>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Bottom Action Bar */}
      <div className="clean-action-bar">
        <div className="clean-step-indicator">Step {currentStep} of 5 — {STEPS[currentStep-1].name}</div>
        <div style={{ display: 'flex', gap: 8 }}>
          {currentStep > 1 && <button className="clean-btn clean-btn-ghost" onClick={() => setCurrentStep(prev => Math.max(1, prev - 1))}>← Previous</button>}
          {currentStep < 5 ? (
             <button className="clean-btn clean-btn-primary" onClick={handleApplyManual} disabled={loading}>Apply & Next</button>
          ) : (
             <button className="clean-btn clean-btn-primary" onClick={() => setVerifyOpen(true)}>Finalize</button>
          )}
        </div>
      </div>

      {/* VERIFY MODAL */}
      {verifyOpen && (
        <div className="clean-verify-overlay show" onClick={() => { setVerifyOpen(false); setShowFullCleaned(false); }}>
          <div className="clean-verify-modal" onClick={e => e.stopPropagation()}>
            <div className="clean-verify-head">
              <div>
                <div className="clean-verify-title">Verify Cleaned Dataset</div>
                <div style={{fontFamily:"'IBM Plex Mono',monospace", fontSize:10, color:'var(--ink3)', marginTop:2}}>{dsName} · v1 → cleaned · {showFullCleaned ? 'All Rows' : 'Preview: 50 rows'}</div>
              </div>
              <button className="clean-btn clean-btn-ghost clean-btn-sm" onClick={() => { setVerifyOpen(false); setShowFullCleaned(false); }}>✕ Close</button>
            </div>
            
            <div className="clean-verify-stats">
              <div className="clean-vstat"><div className="clean-vstat-val" style={{color:'var(--green)'}}>{activeData.length}</div><div className="clean-vstat-lbl">Rows After Cleaning</div></div>
              <div className="clean-vstat"><div className="clean-vstat-val" style={{color:'var(--red)'}}>{totNulls}</div><div className="clean-vstat-lbl">Nulls Handled</div></div>
              <div className="clean-vstat"><div className="clean-vstat-val" style={{color:'var(--amber)'}}>{totDupes}</div><div className="clean-vstat-lbl">Dupes Handled</div></div>
            </div>
            
            <div className="clean-verify-body">
              <table className="clean-data-table">
                <thead>
                  <tr>
                    <th>#</th>
                    {finalHeaders.map(col => {
                      const isNew = acceptedFeatObj.some(f => f.col === col);
                      return <th key={col} className={isNew ? 'col-new' : ''}>{col}</th>;
                    })}
                  </tr>
                </thead>
                <tbody>
                  {(showFullCleaned ? activeData : activeData.slice(0, 50)).map((row, ri) => (
                    <tr key={ri}>
                      <td className="row-num">{ri + 1}</td>
                      {finalHeaders.map(col => {
                         const isNew = acceptedFeatObj.some(f => f.col === col);
                         return <td key={col} className={isNew ? 'cell-new' : ''}>{String(row[col] ?? '')}</td>;
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            
            <div className="clean-verify-foot" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <button className="clean-btn clean-btn-ghost" onClick={() => { setVerifyOpen(false); setShowFullCleaned(false); }}>← Re-clean</button>
                <span style={{fontFamily:"'IBM Plex Mono',monospace", fontSize:10, color:'var(--ink3)', marginLeft:12}}>
                  {!showFullCleaned ? (
                    <>Showing preview · <span style={{color:'var(--accent2)', cursor:'pointer'}} onClick={() => setShowFullCleaned(true)}>View Full Dataset ↗</span></>
                  ) : (
                    <>Showing full set · <span style={{color:'var(--accent2)', cursor:'pointer'}} onClick={() => setShowFullCleaned(false)}>Collapse ↙</span></>
                  )}
                </span>
              </div>
              <div style={{ display: 'flex', gap: 8 }}>
                <button className="clean-btn clean-btn-primary" onClick={handleDownload}>↓ Download Cleaned CSV</button>
                <button 
                  className="clean-btn clean-btn-green" 
                  disabled={loading}
                  onClick={async () => {
                    setLoading(true);
                    try {
                      // 1. Gather accepted features
                      const acceptedFeatures = aiSuggestions.filter(s => featStatuses[s.id] === 'accept');
                      
                      // 2. If features exist, apply them via the transformation engine first
                      if (acceptedFeatures.length > 0 && !acceptedFeatures.every((feat) => tableHeaders.includes(feat.col))) {
                        const transformConfig = { type: 'feature_eng', params: { features: acceptedFeatures } };
                        const transformRes = await transformDataset(dsId, transformConfig.type, transformConfig.params);
                        if (!transformRes.success) {
                          throw new Error("Failed to generate features: " + transformRes.message);
                        }
                        await syncPreview();
                      }

                      // 3. Move file to finalized (cleaned) directory
                      await finalizeDataset(dsId);
                      navigate(`/employee/visualization?ds=${dsId}&name=${encodeURIComponent(dsName)}`);
                    } catch (err) {
                      setError(err.message || "Failed to finalize dataset.");
                    } finally {
                      setLoading(false);
                    }
                  }}
                >
                  {loading ? 'Finalizing...' : 'Proceed to Visualization →'}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default EmployeeCleaningPage;
