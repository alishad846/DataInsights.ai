import { Navigate } from "react-router-dom";

export default function ProtectedRoute({ children, allowedRoles = [] }) {
    const token = sessionStorage.getItem("token");
    const role = sessionStorage.getItem("role");

    // No token - redirect to login
    if (!token) {
        return <Navigate to="/login" replace />;
    }

    // If allowedRoles specified, check role matches
    if (allowedRoles.length > 0 && !allowedRoles.includes(role)) {
        // Redirect based on role - datasets page only
        if (role === 'admin') {
            return <Navigate to="/admin" replace />;
        } else if (role === 'employee') {
            return <Navigate to="/employee/datasets" replace />;
        } else {
            return <Navigate to="/" replace />;
        }
    }

    return children;
}
