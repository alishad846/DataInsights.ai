import { useEffect, useState } from 'react';
import { useNavigate, useLocation, Link } from 'react-router-dom';
import { Database, Sparkles, LayoutDashboard, MessageSquare, FileText, LogOut, Upload, BarChart3, Activity } from 'lucide-react';
import { getMe } from '../services/api';
import '../styles/Employee.css';

const navItems = [
  { path: '/employee/dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { path: '/employee/datasets', label: 'Datasets', icon: Database },
  { path: '/employee/cleaning', label: 'Clean', icon: Sparkles },
  { path: '/employee/visualization', label: 'Visualize', icon: BarChart3 },
  { path: '/employee/chat', label: 'Chatbot', icon: MessageSquare },
  { path: '/employee/summary', label: 'Summary', icon: FileText },
];

const EmployeeLayout = ({ children }) => {
  const navigate = useNavigate();
  const location = useLocation();
  const [userName, setUserName] = useState(sessionStorage.getItem('userName') || 'Employee');
  
  // Safe parsing to prevent application crashes if userName gets resolved as null from backend
  const safeName = userName || 'Employee';
  const userInitials = safeName.split(' ').filter(Boolean).map(n => n[0]).join('').toUpperCase().slice(0, 2);

  useEffect(() => {
    getMe().then(user => {
      if (user && (user.full_name || user.name)) {
        setUserName(user.full_name || user.name);
      } else if (user && user.email) {
        setUserName(user.email.split('@')[0]);
      }
    });
  }, []);

  const handleLogout = () => {
    sessionStorage.removeItem('token');
    sessionStorage.removeItem('role');
    sessionStorage.removeItem('userName');
    navigate('/');
  };

  return (
    <div className="emp-layout">
      <aside className="emp-sidebar">
        <div className="emp-sidebar-logo">
          <h2>Data Insights</h2>
          <p>Employee Portal</p>
        </div>

        <nav className="emp-nav">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = location.pathname === item.path;
            return (
              <Link
                key={item.path}
                to={item.path}
                className={`emp-nav-item ${isActive ? 'active' : ''}`}
              >
                <Icon size={18} />
                <span>{item.label}</span>
              </Link>
            );
          })}
        </nav>

        <div className="emp-user-section">
          <div className="emp-user-card">
            <div className="emp-user-avatar">
              {userInitials}
            </div>
            <div className="emp-user-info">
              <p className="emp-user-name">{userName}</p>
              <p className="emp-user-role">Employee</p>
            </div>
            <button
              onClick={handleLogout}
              className="emp-logout-btn"
              title="Logout"
            >
              <LogOut size={18} />
            </button>
          </div>
        </div>
      </aside>

      <main className="emp-main">
        {children}
      </main>
    </div>
  );
};

export default EmployeeLayout;
