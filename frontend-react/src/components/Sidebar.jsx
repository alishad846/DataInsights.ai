import { Link } from "react-router-dom";

export default function Sidebar() {
    return (
        <div style={styles.sidebar}>
            <h3>Admin Panel</h3>

            <Link to="/admin">Dashboard</Link><br />
            <Link to="/admin/logs">Logs</Link><br />
            <Link to="/admin/permissions">Permissions</Link>
        </div>
    );
}

const styles = {
    sidebar: {
        width: "200px",
        height: "100vh",
        background: "#333",
        color: "#fff",
        padding: "20px",
        position: "fixed"
    }
};
