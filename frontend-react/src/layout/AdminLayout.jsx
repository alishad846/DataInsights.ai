import { useState, useEffect } from 'react';
import {
  LayoutDashboard, FileText, Shield, Users, Database,
  Settings, LogOut, Search, Bell, Activity,
  Menu, X, UploadCloud
} from 'lucide-react';
import { useNavigate, useLocation, Link } from 'react-router-dom';
import { getBadgeCounts, logout } from '../services/api';


const navItems = [
  {
    label: 'Main', items: [
      { path: '/admin', icon: LayoutDashboard, text: 'Dashboard' },
      { path: '/admin/upload', icon: UploadCloud, text: 'Upload' },
      { path: '/admin/permissions', icon: Shield, text: 'Permissions', badgeKey: 'pending' },
      { path: '/admin/logs', icon: FileText, text: 'Logs' },
    ]
  },
  {
    label: 'Manage', items: [
      { path: '/admin/employees', icon: Users, text: 'Employees' },
      { path: '/datasets', icon: Database, text: 'Datasets' },
    ]
  },
  {
    label: 'Company', items: [
      { path: '/admin', icon: Settings, text: 'Settings' },
    ]
  },
];

export default function AdminLayout({ children, title, subtitle }) {
  const navigate = useNavigate();
  const location = useLocation();
  const [mobileOpen, setMobileOpen] = useState(false);
  const [pendingCount, setPendingCount] = useState(0);
  
  const email = sessionStorage.getItem('email') || sessionStorage.getItem('userEmail') || 'admin@datainsights.ai';
  // Use userName from local storage if available
  const adminName = sessionStorage.getItem('userName') || (email ? email.split('@')[0] : 'Admin');
  const initials = adminName.slice(0, 2).toUpperCase();

  useEffect(() => {
    // Fetch unified pending count (Users + Permissions)
    getBadgeCounts().then(res => {
        if (res && typeof res.total === 'number') {
            setPendingCount(res.total);
        }
    }).catch(err => console.warn('Failed to get unified badge counts', err));
  }, []);


  const handleLogout = async () => {
    try {
      await logout();
    } catch (e) {
      console.warn("Logout log failed:", e.message);
    }
    sessionStorage.removeItem('token');
    sessionStorage.removeItem('role');
    sessionStorage.removeItem('email');
    sessionStorage.removeItem('userEmail');
    sessionStorage.removeItem('userName');
    navigate('/login');
  };


  return (
    <div style={{ display: 'flex', minHeight: '100vh', background: 'var(--bg-color)' }}>
      {/* Mobile overlay */}
      {mobileOpen && (
        <div
          className="admin-sidebar-overlay"
          onClick={() => setMobileOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside className={`admin-sidebar ${mobileOpen ? 'mobile-open' : ''}`}>
        <div className="admin-sidebar-logo">
          <Link to="/admin" style={{ textDecoration: 'none', display: 'flex', alignItems: 'center', gap: '10px' }}>
            <Activity color="var(--primary)" size={20} />
            <div>
              <div className="admin-logo-text">Data<span className="gradient-text">Insights</span></div>
              <div className="admin-logo-sub">Admin Panel</div>
            </div>
          </Link>
        </div>

        <div className="admin-company-card">
          <div className="admin-company-avatar">AC</div>
          <div>
            <div className="admin-company-name">Acme Corp</div>
            <div className="admin-company-role">Administrator</div>
          </div>
        </div>

        <nav className="admin-nav">
          {navItems.map((section, si) => (
            <div key={si}>
              <div className="admin-nav-label">{section.label}</div>
              {section.items.map((item, ii) => {
                const Icon = item.icon;
                const isActive = location.pathname === item.path;
                let badgeVal = item.badgeKey === 'pending' ? pendingCount : item.badge;

                return (
                  <a
                    key={ii}
                    onClick={() => { navigate(item.path); setMobileOpen(false); }}
                    className={`admin-nav-item ${isActive ? 'active' : ''}`}
                    style={{ cursor: 'pointer' }}
                  >
                    <Icon size={16} className="admin-nav-icon" />
                    <span>{item.text}</span>
                    {badgeVal ? <span className="admin-nav-badge">{badgeVal}</span> : null}
                  </a>
                );
              })}
              {si < navItems.length - 1 && <div className="admin-nav-divider" />}
            </div>
          ))}
        </nav>

        <div className="admin-sidebar-bottom">
          <div className="admin-user-card">
            <div className="admin-user-avatar">{initials}</div>
            <div>
              <div className="admin-user-name">{adminName}</div>
              <div className="admin-user-email">{email}</div>
            </div>
            <LogOut size={14} className="admin-logout-btn" onClick={handleLogout} style={{ cursor: 'pointer' }} />
          </div>
        </div>
      </aside>

      {/* Main */}
      <main className="admin-main">
        <header className="admin-topbar">
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <button className="admin-mobile-menu" onClick={() => setMobileOpen(!mobileOpen)}>
              {mobileOpen ? <X size={18} /> : <Menu size={18} />}
            </button>
            <div>
              <div className="admin-topbar-title">{title || 'Dashboard'}</div>
              {subtitle && <div className="admin-topbar-sub">{subtitle}</div>}
            </div>
          </div>
          <div className="admin-topbar-actions">
             {/* Note: Global Search input is pushed to employees page layout context if desired, or can be kept here. */}
             {/* For now, removing the dummy search since dedicated search is requested in the UI. */}
             {/* Or just keep the icon without functionality unless needed globally. */}
            <button className="admin-icon-btn">
              <Bell size={16} />
              <span className="admin-notif-dot" />
            </button>
          </div>
        </header>
        <div className="admin-content">
          {children}
        </div>
      </main>
    </div>
  );
}
