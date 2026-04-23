import { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import axios from "axios";
import "../styles/Auth.css";

const ShieldIcon = () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="roucd nd" strokeLinejoin="round">
        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
        <path d="M9 12l2 2 4-4" />
    </svg>
);

const MailIcon = () => (
    <svg className="field-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <rect x="2" y="4" width="20" height="16" rx="2" />
        <path d="M22 7l-10 6L2 7" />
    </svg>
);

const LockIcon = () => (
    <svg className="field-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <rect x="3" y="11" width="18" height="11" rx="2" ry="2" />
        <path d="M7 11V7a5 5 0 0 1 10 0v4" />
    </svg>
);

const EyeIcon = ({ off: isOff }) => isOff ? (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94" />
        <path d="M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19" />
        <line x1="1" y1="1" x2="23" y2="23" />
    </svg>
) : (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
        <circle cx="12" cy="12" r="3" />
    </svg>
);

const ArrowLeftIcon = () => (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <line x1="19" y1="12" x2="5" y2="12" />
        <polyline points="12 19 5 12 12 5" />
    </svg>
);

const AlertIcon = () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="10" />
        <line x1="12" y1="8" x2="12" y2="12" />
        <line x1="12" y1="16" x2="12.01" y2="16" />
    </svg>
);

const ChartIcon = () => (
    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <line x1="18" y1="20" x2="18" y2="10" />
        <line x1="12" y1="20" x2="12" y2="4" />
        <line x1="6" y1="20" x2="6" y2="14" />
    </svg>
);

const CheckCircleIcon = () => (
    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
        <polyline points="22 4 12 14.01 9 11.01" />
    </svg>
);

export default function AdminLogin() {
    const nav = useNavigate();
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");
    const [showPassword, setShowPassword] = useState(false);
    const [error, setError] = useState("");
    const [loading, setLoading] = useState(false);
    const [loginSuccess, setLoginSuccess] = useState(false);
    const [userData, setUserData] = useState(null);

    const login = async () => {
        if (!email || !password) {
            setError("Please fill in all fields");
            return;
        }
        setError("");
        setLoading(true);
        try {
            const res = await axios.post("http://localhost:5000/api/auth/login", {
                email,
                password,
                role: "admin"
            });

            if (res.data.role !== "admin") {
                setError("This account is not authorized for admin access");
                setLoading(false);
                return;
            }

            sessionStorage.setItem("token", res.data.token);
            sessionStorage.setItem("role", res.data.role);
            sessionStorage.setItem("userName", res.data.name || email.split('@')[0]);
            sessionStorage.setItem("userEmail", res.data.email);
            
            setUserData({
                name: res.data.name || email.split('@')[0],
                email: res.data.email,
                role: res.data.role
            });
            setLoginSuccess(true);
            setLoading(false);
        } catch (err) {
            setError(err.response?.data?.message || "Invalid credentials. Please try again.");
            setLoading(false);
        }
    };

    const handleKeyDown = (e) => {
        if (e.key === "Enter") login();
    };

    return (
        <div className="auth-wrapper">
            <div className="orb-1" />
            <div className="orb-2" />
            <div className="orb-3" />

            <div className="auth-card">
                <Link to="/" className="auth-back-link">
                    <ArrowLeftIcon /> Back to role selection
                </Link>

                <div className="auth-brand">
                    <div className="brand-icon" style={{ background: 'linear-gradient(135deg, #b63d3d 0%, #f85149 100%)' }}>
                        <ChartIcon />
                    </div>
                    <h1>Admin Login</h1>
                    <p className="auth-subtitle">Sign in to manage your organization</p>
                    <div className="auth-role-badge admin">
                        <ShieldIcon />
                        Admin Access
                    </div>
                </div>

                <div className="auth-form" onKeyDown={handleKeyDown}>
                    {error && (
                        <div className="auth-error">
                            <AlertIcon />
                            {error}
                        </div>
                    )}

                    <div className="auth-field">
                        <MailIcon />
                        <input
                            type="email"
                            placeholder="admin@example.com"
                            value={email}
                            onChange={e => setEmail(e.target.value)}
                            autoComplete="email"
                        />
                    </div>

                    <div className="auth-field">
                        <LockIcon />
                        <input
                            type={showPassword ? "text" : "password"}
                            placeholder="Password"
                            value={password}
                            onChange={e => setPassword(e.target.value)}
                            autoComplete="current-password"
                        />
                        <button
                            type="button"
                            className="password-toggle"
                            onClick={() => setShowPassword(!showPassword)}
                        >
                            <EyeIcon off={showPassword} />
                        </button>
                    </div>

                    <button
                        className="auth-btn admin"
                        onClick={login}
                        disabled={loading}
                    >
                        {loading ? (
                            <span className="btn-spinner" />
                        ) : (
                            "Sign In as Admin"
                        )}
                    </button>
                </div>

                <div className="auth-footer">
                    Need an admin account?{" "}
                    <Link to="/signup/admin">Create one</Link>
                </div>
            </div>

            {loginSuccess && userData && (
                <div className="auth-card" style={{ marginTop: 20 }}>
                    <div style={{ textAlign: 'center', padding: '20px 0' }}>
                        <div style={{ 
                            width: 60, height: 60, borderRadius: '50%', 
                            background: 'linear-gradient(135deg, #f85149 0%, #da3633 100%)',
                            display: 'flex', alignItems: 'center', justifyContent: 'center',
                            margin: '0 auto 16px'
                        }}>
                            <CheckCircleIcon />
                        </div>
                        <h2 style={{ marginBottom: 8, color: '#fff' }}>Login Successful</h2>
                        <p style={{ color: 'var(--text-muted)', marginBottom: 24 }}>
                            Welcome, <strong style={{ color: '#fff' }}>{userData.name}</strong>
                        </p>
                        
                        <div style={{ 
                            background: 'rgba(255,255,255,0.05)', borderRadius: 10, 
                            padding: '16px 20px', marginBottom: 24, textAlign: 'left' 
                        }}>
                            <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: 1 }}>
                                Account Details
                            </div>
                            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                                <span style={{ color: 'var(--text-muted)', fontSize: 12 }}>Email</span>
                                <span style={{ color: '#fff', fontSize: 12 }}>{userData.email}</span>
                            </div>
                            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                                <span style={{ color: 'var(--text-muted)', fontSize: 12 }}>Role</span>
                                <span style={{ color: '#f85149', fontSize: 12, textTransform: 'capitalize' }}>{userData.role}</span>
                            </div>
                        </div>

                        <button
                            className="auth-btn admin"
                            onClick={() => nav("/admin")}
                            style={{ width: '100%' }}
                        >
                            Continue to Admin Portal
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
}
