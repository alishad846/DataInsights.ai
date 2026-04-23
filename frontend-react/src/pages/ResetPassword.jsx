import { useState, useEffect } from "react";
import { Link, useSearchParams, useNavigate } from "react-router-dom";
import api from "../services/api";
import "../styles/Auth.css";

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

const AlertIcon = () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="10" />
        <line x1="12" y1="8" x2="12" y2="12" />
        <line x1="12" y1="16" x2="12.01" y2="16" />
    </svg>
);

const CheckCircleIcon = () => (
    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
        <polyline points="22 4 12 14.01 9 11.01" />
    </svg>
);

export default function ResetPassword() {
    const [searchParams] = useSearchParams();
    const navigate = useNavigate();
    const token = searchParams.get("token");

    const [password, setPassword] = useState("");
    const [confirmPassword, setConfirmPassword] = useState("");
    const [showPassword, setShowPassword] = useState(false);
    
    // UI states
    const [isValidating, setIsValidating] = useState(true);
    const [isTokenValid, setIsTokenValid] = useState(false);
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [success, setSuccess] = useState(false);
    const [error, setError] = useState("");
    useEffect(() => {
        if (!token) {
            setIsValidating(false);
            return;
        }

        const validateToken = async () => {
            try {
                await api.get("/auth/validate-reset-token", { params: { token } });
                setIsTokenValid(true);
            } catch (err) {
                setError(err.response?.data?.message || "This password reset link is invalid or has expired.");
                setIsTokenValid(false);
            } finally {
                setIsValidating(false);
            }
        };

        validateToken();
    }, [token]);

    const handleReset = async () => {
        if (!password || !confirmPassword) {
            setError("Please fill in both password fields.");
            return;
        }
        if (password !== confirmPassword) {
            setError("Passwords do not match.");
            return;
        }
        if (password.length < 6) {
            setError("Password must be at least 6 characters.");
            return;
        }

        setError("");
        setIsSubmitting(true);

        try {
            await api.post("/auth/reset-password", { token, newPassword: password });
            setSuccess(true);
        } catch (err) {
            setError(err.response?.data?.message || "Failed to reset password. Please try again.");
        } finally {
            setIsSubmitting(false);
        }
    };

    const handleKeyDown = (e) => {
        if (e.key === "Enter" && isTokenValid && !success) handleReset();
    };

    return (
        <div className="auth-wrapper">
            <div className="orb-1" />
            <div className="orb-2" />
            <div className="orb-3" />

            <div className="auth-card">
                {/* 1. Validation Loading State */}
                {isValidating && (
                    <div style={{ textAlign: "center", padding: "40px 0" }}>
                        <span className="btn-spinner" style={{ borderColor: 'var(--primary)', borderTopColor: 'transparent', width: 32, height: 32 }} />
                        <h2 style={{ marginTop: 24, fontSize: 18 }}>Verifying Secure Link...</h2>
                        <p style={{ color: "var(--text-muted)", marginTop: 8 }}>Please wait while we validate your token.</p>
                    </div>
                )}

                {/* 2. Validation Failed / No Token State */}
                {!isValidating && !isTokenValid && !success && (
                    <div style={{ textAlign: "center", padding: "20px 0" }}>
                        <div style={{
                            width: 60, height: 60, borderRadius: '50%',
                            background: 'rgba(248,81,73,0.1)', border: '1px solid rgba(248,81,73,0.2)',
                            display: 'flex', alignItems: 'center', justifyContent: 'center',
                            margin: '0 auto 16px', color: '#f85149'
                        }}>
                            <AlertIcon />
                        </div>
                        <h2 style={{ marginBottom: 8, color: '#fff' }}>Invalid Link</h2>
                        <p style={{ color: 'var(--text-muted)', marginBottom: 24 }}>
                            {error || "It looks like your password reset link is missing or broken."}
                        </p>
                        <Link to="/forgot-password" style={{ textDecoration: 'none' }}>
                            <button className="auth-btn primary" style={{ width: '100%' }}>
                                Request New Link
                            </button>
                        </Link>
                    </div>
                )}

                {/* 3. Success State */}
                {!isValidating && success && (
                    <div style={{ textAlign: "center", padding: "20px 0" }}>
                        <div style={{
                            width: 60, height: 60, borderRadius: '50%',
                            background: 'linear-gradient(135deg, #3fb950 0%, #2ea043 100%)',
                            display: 'flex', alignItems: 'center', justifyContent: 'center',
                            margin: '0 auto 16px'
                        }}>
                            <CheckCircleIcon />
                        </div>
                        <h2 style={{ marginBottom: 8, color: '#fff' }}>Password Reset</h2>
                        <p style={{ color: 'var(--text-muted)', marginBottom: 24 }}>
                            Your password has been successfully updated. You may now sign in locally.
                        </p>
                        <Link to="/login" style={{ textDecoration: 'none' }}>
                            <button className="auth-btn primary" style={{ width: '100%' }}>
                                Sign In
                            </button>
                        </Link>
                    </div>
                )}

                {/* 4. Valid Token Form */}
                {!isValidating && isTokenValid && !success && (
                    <>
                        <div className="auth-brand">
                            <div className="brand-icon">
                                <LockIcon />
                            </div>
                            <h1>Set New Password</h1>
                            <p className="auth-subtitle">Enter a new secure password for your account</p>
                        </div>

                        <div className="auth-form" onKeyDown={handleKeyDown}>
                            {error && (
                                <div className="auth-error">
                                    <AlertIcon />
                                    {error}
                                </div>
                            )}

                            <div className="auth-field">
                                <LockIcon />
                                <input
                                    type={showPassword ? "text" : "password"}
                                    placeholder="New Password"
                                    value={password}
                                    onChange={(e) => setPassword(e.target.value)}
                                />
                                <button
                                    type="button"
                                    className="password-toggle"
                                    onClick={() => setShowPassword(!showPassword)}
                                >
                                    <EyeIcon off={showPassword} />
                                </button>
                            </div>

                            <div className="auth-field">
                                <LockIcon />
                                <input
                                    type={showPassword ? "text" : "password"}
                                    placeholder="Confirm Password"
                                    value={confirmPassword}
                                    onChange={(e) => setConfirmPassword(e.target.value)}
                                />
                            </div>

                            <button
                                className="auth-btn primary"
                                onClick={handleReset}
                                disabled={isSubmitting}
                            >
                                {isSubmitting ? <span className="btn-spinner" /> : "Reset Password"}
                            </button>
                        </div>
                    </>
                )}
            </div>
        </div>
    );
}
