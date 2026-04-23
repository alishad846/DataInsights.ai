import { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Database, CheckCircle, Clock, AlertTriangle } from 'lucide-react';
import { getDatasetStatus } from '../services/api';

const PIPELINE_STEPS = [
  "Validating Schema & Schema Profiles",
  "Cleaning Constraints & Sampling",
  "Detecting Implicit Relationships",
  "Engineering Advanced Features",
  "Evaluating Machine Learning Models",
  "Executing Time-Series Forecasts",
  "Extracting Performance KPIs",
  "Generating Metric Graph Definitions",
  "Discovering Anomalies & Trends",
  "Compiling Visualization Dashboards"
];

const ProcessingPage = () => {
  const { datasetId } = useParams();
  const navigate = useNavigate();
  const [status, setStatus] = useState('processing');
  const [error, setError] = useState('');
  const [simulatedStep, setSimulatedStep] = useState(0);

  // Poll Backend Status
  useEffect(() => {
    if (!datasetId) return;

    const interval = setInterval(async () => {
      try {
        const res = await getDatasetStatus(datasetId);
        console.log(`[POLL] Dataset status response for ${datasetId}:`, res);
        
        if (res && res.status) {
          if (res.status === 'completed') {
            setStatus('completed');
            clearInterval(interval);
            setTimeout(() => {
              navigate(`/dashboard/${datasetId}`);
            }, 1500);
          } else if (res.status === 'failed') {
            setStatus('failed');
            setError(res.error || 'Pipeline execution failed during artifact generation.');
            clearInterval(interval);
          }
        }
      } catch (err) {
        console.error(err);
        // Sometimes polling might fail momentarily, we shouldn't hard-fail immediately 
        // but if it's consistently failing, set error.
      }
    }, 1500); // Poll every 1.5 seconds

    return () => clearInterval(interval);
  }, [datasetId, navigate]);

  // Simulate graphical progress bar text to keep user engaged while ML processes in background
  useEffect(() => {
    if (status !== 'processing') return;
    
    const maxSteps = PIPELINE_STEPS.length;
    // We estimate ML takes around 15-20 seconds total. We step every ~1.5s
    const stepInterval = setInterval(() => {
      setSimulatedStep(prev => {
        if (prev < maxSteps - 1) return prev + 1;
        return prev;
      });
    }, 1800);
    
    return () => clearInterval(stepInterval);
  }, [status]);

  return (
    <div className="view-enter flex-center" style={{ minHeight: '60vh', flexDirection: 'column' }}>
      <div className="glass-panel" style={{ width: '100%', maxWidth: '600px', padding: '3rem' }}>
        
        {status === 'processing' && (
          <div style={{ textAlign: 'center' }}>
            <div className="flex-center" style={{ marginBottom: '2rem' }}>
              <div style={{ position: 'relative' }}>
                <Database size={64} color="var(--primary)" style={{ opacity: 0.5 }} />
                <div style={{ 
                  position: 'absolute', 
                  top: '50%', left: '50%', transform: 'translate(-50%, -50%)',
                  width: '80px', height: '80px', 
                  border: '4px solid transparent',
                  borderTopColor: 'var(--accent)',
                  borderRightColor: 'var(--primary)',
                  borderRadius: '50%',
                  animation: 'spin 1.5s cubic-bezier(0.68, -0.55, 0.265, 1.55) infinite'
                }}></div>
              </div>
            </div>
            
            <h2 style={{ fontSize: '1.8rem', marginBottom: '1rem' }}>Orchestrating AI Pipeline...</h2>
            
            <div style={{
              background: 'rgba(88, 166, 255, 0.1)',
              border: '1px solid var(--primary)',
              padding: '0.75rem',
              borderRadius: '8px',
              color: 'var(--primary)',
              fontWeight: '500',
              display: 'inline-block',
              marginBottom: '1rem'
            }}>
              Current Status: {status}
            </div>

            <div style={{ 
              background: 'rgba(0,0,0,0.3)', 
              borderRadius: '8px', 
              padding: '1.5rem',
              marginTop: '2rem',
              textAlign: 'left'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1rem' }}>
                <Clock size={20} color="var(--primary)" />
                <span style={{ fontSize: '0.9rem', color: 'var(--text-muted)' }}>Executing Module {simulatedStep + 1} of 10</span>
              </div>
              
              <h3 style={{ fontSize: '1.1rem', color: 'var(--text-main)', margin: 0, fontWeight: 500 }}>
                {PIPELINE_STEPS[simulatedStep]}
              </h3>
              
              <div style={{ marginTop: '1.5rem', height: '6px', background: 'rgba(255,255,255,0.1)', borderRadius: '3px', overflow: 'hidden' }}>
                <div style={{ 
                  height: '100%', 
                  background: 'linear-gradient(90deg, var(--secondary), var(--accent))',
                  width: `${((simulatedStep + 1) / PIPELINE_STEPS.length) * 100}%`,
                  transition: 'width 1.5s ease'
                }}></div>
              </div>
            </div>
          </div>
        )}

        {status === 'completed' && (
          <div style={{ textAlign: 'center', animation: 'fadeSlideUp 0.5s ease forwards' }}>
            <CheckCircle size={80} color="var(--success)" style={{ marginBottom: '1.5rem' }} />
            <h2 style={{ marginBottom: '1rem', color: 'var(--success)' }}>Dataset Processed Successfully</h2>
            <p style={{ color: 'var(--text-muted)' }}>Redirecting to visualization dashboard...</p>
          </div>
        )}

        {status === 'failed' && (
          <div style={{ textAlign: 'center', animation: 'fadeSlideUp 0.5s ease forwards' }}>
            <AlertTriangle size={80} color="var(--danger)" style={{ marginBottom: '1.5rem' }} />
            <h2 style={{ marginBottom: '1rem', color: 'var(--danger)' }}>Processing Failed</h2>
            <div style={{ 
              background: 'rgba(248, 81, 73, 0.1)', 
              padding: '1rem', 
              borderRadius: '8px',
              border: '1px solid var(--danger)',
              color: 'var(--text-main)',
              marginBottom: '2rem',
              textAlign: 'left'
            }}>
              <strong>Error Logic:</strong> {error}
            </div>
            <button className="btn-primary" onClick={() => navigate('/')} style={{ background: 'var(--danger)' }}>
              Upload New Dataset
            </button>
          </div>
        )}
        
      </div>

      <style dangerouslySetInnerHTML={{__html: `
        @keyframes spin { 
          0% { transform: translate(-50%, -50%) rotate(0deg); } 
          100% { transform: translate(-50%, -50%) rotate(360deg); } 
        }
      `}} />
    </div>
  );
};

export default ProcessingPage;
