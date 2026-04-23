import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import "../styles/Auth.css";

const UserIcon = () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
        <circle cx="12" cy="7" r="4" />
    </svg>
);

const ShieldIcon = () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
        <path d="M9 12l2 2 4-4" />
    </svg>
);

const ArrowRightIcon = () => (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <line x1="5" y1="12" x2="19" y2="12" />
        <polyline points="12 5 19 12 12 19" />
    </svg>
);

const ChartIcon = () => (
    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <line x1="18" y1="20" x2="18" y2="10" />
        <line x1="12" y1="20" x2="12" y2="4" />
        <line x1="6" y1="20" x2="6" y2="14" />
    </svg>
);

export default function SelectRole() {
    const [selectedRole, setSelectedRole] = useState("");
    const navigate = useNavigate();

    const handleContinue = () => {
        if (!selectedRole) return;
        sessionStorage.setItem("role", selectedRole);
        if (selectedRole === "admin") {
            navigate("/admin-login");
        } else {
            navigate("/login");
        }
    };

    return (
        <div className="role-selection-wrapper">
            <div className="orb-1" />
            <div className="orb-2" />
            <div className="orb-3" />

            <div className="role-selection-card">
                <div className="auth-brand">
                    <div className="brand-icon">
                        <ChartIcon />
                    </div>
                    <h1>Data Insights</h1>
                    <p className="role-subtitle">Choose how you want to continue</p>
                </div>

                <div className="role-choices">
                    <div
                        className={`role-choice ${selectedRole === "employee" ? "selected" : ""}`}
                        onClick={() => setSelectedRole("employee")}
                    >
                        <div className="role-choice-icon user-icon">
                            <UserIcon />
                        </div>
                        <h3>Employee</h3>
                        <p>Upload datasets, analyze data &amp; generate insights</p>
                    </div>

                    <div
                        className={`role-choice ${selectedRole === "admin" ? "selected admin" : ""}`}
                        onClick={() => setSelectedRole("admin")}
                    >
                        <div className="role-choice-icon admin-icon">
                            <ShieldIcon />
                        </div>
                        <h3>Admin</h3>
                        <p>Manage permissions, monitor logs &amp; system settings</p>
                    </div>
                </div>

                <button
                    onClick={handleContinue}
                    disabled={!selectedRole}
                    className={`role-continue-btn ${selectedRole === "admin" ? "admin" : ""}`}
                >
                    Continue <ArrowRightIcon />
                </button>
            </div>
        </div>
    );
}
