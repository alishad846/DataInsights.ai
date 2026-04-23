import { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { getDashboardConfig } from '../services/api';
import InsightsPanel from '../components/InsightsPanel';
import {
  BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Cell, PieChart, Pie, Legend, AreaChart, Area
} from 'recharts';
import { MessageSquare, Database, Download, BarChart3 } from 'lucide-react';

const COLORS = ['#58a6ff', '#bc8cff', '#3fb950', '#d29922', '#f85149'];

const DashboardPage = () => {
  const { datasetId } = useParams();
  const navigate = useNavigate();

  const [dashboardData, setDashboardData] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState('');

  // DOWNLOAD REPORT FUNCTION
  const handleDownloadReport = async () => {
    try {
      const response = await fetch(`/api/download-report/${datasetId}`);

      if (!response.ok) {
        throw new Error("Download failed");
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);

      const link = document.createElement("a");
      link.href = url;
      link.download = `${datasetId}_report.pdf`;

      document.body.appendChild(link);
      link.click();
      link.remove();

      window.URL.revokeObjectURL(url);
    } catch (err) {
      console.error("Download error:", err);
      alert("Report download failed.");
    }
  };

  useEffect(() => {
    if (!datasetId) return;

    const fetchDashboard = async () => {
      try {
        const config = await getDashboardConfig(datasetId);
        setDashboardData(config);
      } catch (err) {
        console.error("Dashboard Config Error:", err);
        setError("Failed to load dashboard configuration.");
      } finally {
        setIsLoading(false);
      }
    };

    fetchDashboard();
  }, [datasetId]);

  if (isLoading) {
    return (
      <div className="view-enter flex-center"
        style={{ minHeight: '60vh', flexDirection: 'column' }}>
        <div className="spinner"
          style={{
            width: '40px',
            height: '40px',
            border: '4px solid rgba(88,166,255,0.2)',
            borderTopColor: 'var(--primary)',
            borderRadius: '50%',
            animation: 'spin 1s linear infinite'
          }}
        />
        <p style={{ marginTop: '1rem', color: 'var(--text-muted)' }}>
          Synchronizing workspace...
        </p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="view-enter glass-panel"
        style={{ padding: '3rem', textAlign: 'center', marginTop: '4rem' }}>
        <h2 style={{ color: 'var(--warning)', marginBottom: '1rem' }}>
          Dataset Processing
        </h2>

        <p style={{ color: 'var(--text-muted)', marginBottom: '2rem' }}>
          {error}
        </p>

        <button
          className="btn-primary"
          onClick={() => navigate('/datasets')}
        >
          Go to Workspace
        </button>
      </div>
    );
  }

  const renderKPIs = () => {
    if (!dashboardData?.kpis) return null;

    return (
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(240px,1fr))',
          gap: '1.5rem',
          marginBottom: '2rem'
        }}
      >
        {dashboardData.kpis.map((kpi, idx) => (
          <div
            key={idx}
            className="glass-panel"
            style={{
              padding: '1.5rem',
              display: 'flex',
              flexDirection: 'column',
              gap: '0.5rem'
            }}
          >

            <span
              style={{
                fontSize: '0.85rem',
                color: 'var(--text-muted)',
                textTransform: 'uppercase'
              }}
            >
              {kpi.label}
            </span>

            <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.5rem' }}>
              <h2 style={{ fontSize: '2rem', margin: 0 }}>{kpi.value}</h2>

              {kpi.trend && (
                <span
                  style={{
                    fontSize: '0.85rem',
                    color: kpi.trend > 0 ? 'var(--success)' : 'var(--danger)'
                  }}
                >
                  {kpi.trend > 0 ? '+' : ''}{kpi.trend}%
                </span>
              )}
            </div>

          </div>
        ))}
      </div>
    );
  };

  const renderChart = (chartConfig) => {
    const data = chartConfig.data || [];
    const xKey = chartConfig.x;
    const yKey = chartConfig.y;

    const ChartWrapper = ({ children }) => (
      <div
        className="glass-panel"
        style={{
          padding: '1.5rem',
          height: '400px',
          display: 'flex',
          flexDirection: 'column'
        }}
      >
        <h4 style={{ marginBottom: '1rem' }}>
          {chartConfig.title}
        </h4>

        <div style={{ flexGrow: 1 }}>
          {children}
        </div>
      </div>
    );

    if (!data.length) return null;

    switch (chartConfig.type) {

      case 'bar':
        return (
          <ChartWrapper key={chartConfig.id}>
            <ResponsiveContainer>
              <BarChart data={data}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey={xKey} />
                <YAxis />
                <Tooltip />
                <Bar dataKey={yKey}>
                  {data.map((_, index) => (
                    <Cell
                      key={index}
                      fill={COLORS[index % COLORS.length]}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </ChartWrapper>
        );

      case 'line':
        return (
          <ChartWrapper key={chartConfig.id}>
            <ResponsiveContainer>
              <LineChart data={data}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey={xKey} />
                <YAxis />
                <Tooltip />
                <Line type="monotone" dataKey={yKey} stroke="#58a6ff" />
              </LineChart>
            </ResponsiveContainer>
          </ChartWrapper>
        );

      case 'pie':
        return (
          <ChartWrapper key={chartConfig.id}>
            <ResponsiveContainer>
              <PieChart>
                <Pie
                  data={data}
                  dataKey={yKey}
                  nameKey={xKey}
                  outerRadius={120}
                >
                  {data.map((_, index) => (
                    <Cell
                      key={index}
                      fill={COLORS[index % COLORS.length]}
                    />
                  ))}
                </Pie>

                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </ChartWrapper>
        );

      case 'area':
        return (
          <ChartWrapper key={chartConfig.id}>
            <ResponsiveContainer>
              <AreaChart data={data}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey={xKey} />
                <YAxis />
                <Tooltip />
                <Area
                  type="monotone"
                  dataKey={yKey}
                  stroke="#58a6ff"
                  fill="#58a6ff"
                />
              </AreaChart>
            </ResponsiveContainer>
          </ChartWrapper>
        );

      default:
        return null;
    }
  };

  return (
    <div className="view-enter">

      <header
        style={{
          marginBottom: '2rem',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'flex-end'
        }}
      >

        <div>
          <h1 style={{ fontSize: '2.5rem' }}>
            Intelligence Dashboard
          </h1>

          <p style={{ color: 'var(--text-muted)', display: 'flex', gap: '6px' }}>
            <Database size={16} /> {datasetId}
          </p>
        </div>

        <div style={{ display: 'flex', gap: '1rem' }}>

          <button
            className="btn-primary"
            onClick={() => navigate(`/employee/visualization?ds=${datasetId}`)}
          >
            <BarChart3 size={16} />
            Visualize & Filter
          </button>

          <button
            className="btn-primary"
            onClick={handleDownloadReport}
          >
            <Download size={16} />
            Download Report
          </button>

          <button
            className="btn-primary"
            onClick={() => navigate(`/chat/${datasetId}`)}
          >
            <MessageSquare size={16} />
            AI Assistant
          </button>

        </div>

      </header>

      {renderKPIs()}

      {dashboardData && (
        <>
          <InsightsPanel
            summary={dashboardData.executive_summary}
            insights={dashboardData.insights}
          />

          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit,minmax(500px,1fr))',
              gap: '2rem',
              marginTop: '2rem'
            }}
          >
            {dashboardData.charts?.map(chart => renderChart(chart))}
          </div>
        </>
      )}

    </div>
  );
};

export default DashboardPage;

